"""
Review store — 검수 판정(묘비 + 확정) + 감사 로그.

KorAct(gov/desktopgui)가 실증한 필요다: LLM 추출은 틀리고, 검수자가 거절한
노드를 그래프에서 지우는 것만으로는 부족하다 — 같은 문서를 다시 인제스트하면
**같은 오추출이 부활한다**. 그래서 거절은 삭제가 아니라 **묘비(tombstone)**
다: 빌더가 병합 전에 조회해서 부활을 막는 영속 기록이다.

설계 선택과 근거:
- **로그가 원본이다** — 상태(묘비/확정)를 따로 저장하지 않고 append-only
  이벤트 로그를 재생(replay)해서 복원한다 (KorAct ontology_versions 패턴).
  상태 파일을 따로 두면 로그와 상태가 어긋나는 순간 어느 쪽이 진실인지
  알 수 없다.
- **append 모드** — chunk_store 의 원자적 교체(.tmp → rename)를 쓰지 않는
  이유: 이벤트는 절대 다시 쓰이지 않으므로(감사 무결성) 전체 재작성 자체가
  없고, record 시점마다 즉시 append 하면 프로세스가 죽어도 판정이 남는다.
  깨진 줄은 로드에서 건너뛴다 (JSONL 을 고른 이유 — chunk_store 와 동일).
- **나중 판정이 이긴다** — confirm 이 이전 reject 를 뒤집는다. 검수자의
  번복은 정상 워크플로우다 (잘못 거절한 노드를 영영 못 살리면 안 된다).
- **네임스페이스별 싱글턴** — KG 엔진·chunk store 와 같은 수명·같은 경계.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class ReviewEvent:
    """검수 이벤트 한 건 — 판정(reject/confirm)이자 감사 항목이다.

    before/after 는 판정 시점의 attrs 스냅샷(diff) — 무엇이 어떻게 바뀌었는지
    없이는 감사가 아니다 (KorAct ontology_versions 의 {before, after}).
    """
    action: str
    node_id: str
    actor: str = ""
    reason: str = ""
    source: str = ""
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewEvent":
        return cls(
            action=data["action"],
            node_id=data["node_id"],
            actor=data.get("actor", ""),
            reason=data.get("reason", ""),
            source=data.get("source", ""),
            before=data.get("before"),
            after=data.get("after"),
            at=data.get("at", ""),
        )


def _relation_key(subject: Any, predicate: Any, obj: Any) -> tuple:
    """트리플 비교용 키 — '같다'의 정의는 인용 검증(_squash_ws)과 한 벌.

    공백 변형('C50( 유방의 악성 신생물 )' vs 'C50(유방의 악성 신생물)')이
    다른 트리플로 보이면 묘비가 구멍난다. 지연 import 는 순환 회피.
    """
    from .evidence_checker import _squash_ws
    return (_squash_ws(str(subject or "")), _squash_ws(str(predicate or "")),
            _squash_ws(str(obj or "")))


def _gap_key(node_type: Any, name: Any) -> tuple:
    """커버리지 gap 비교용 키 — 승인 경로가 쓰는 신원과 같은 정규화.

    `approve_coverage_gaps` 가 `_squash_ws(name)` 으로 node_id 를 만드므로,
    묘비도 같은 규칙이어야 "기각한 그것"과 "승인하려는 그것"이 같은 것으로
    보인다. 규칙을 두 벌 두면 묘비가 구멍난다.
    """
    from .evidence_checker import _squash_ws
    return (_squash_ws(str(node_type or "")), _squash_ws(str(name or "")))


class ReviewStore:
    """네임스페이스 하나의 검수 판정 저장소."""

    def __init__(self, namespace: str = "default", path=None):
        self.namespace = namespace
        self._path = Path(path) if path else None
        self._events: List[ReviewEvent] = []
        # 최신 판정만 남는 파생 상태 — 로그 replay 로 복원된다
        self._rejected: Dict[str, ReviewEvent] = {}
        self._confirmed: Dict[str, ReviewEvent] = {}
        # 에이전트 추천 (근거대조 등) — 판정이 아니다. is_rejected/is_confirmed
        # 에 절대 영향을 주지 않는다: 추천 권한과 판정 권한의 경계가 이 분리다.
        self._recommended: Dict[str, ReviewEvent] = {}
        # 재분류 재지도 (P-4, 2026-08-03) — 묘비의 자매. old_id → new_id.
        # 재분류는 묘비를 남기지 않으므로("타입이 틀렸다" ≠ "개체가 틀렸다"),
        # 재빌드에서 옛 id 로 재추출된 개체를 차단 대신 새 id 로 **재지도**해
        # 보강(근거 합집합)이 되게 한다. reclassify 이벤트에서 파생 — 로그가
        # 원본이라는 이 저장소의 규율 그대로 replay 로 복원된다.
        self._reclassified: Dict[str, str] = {}
        # 구조 단위로 선언된 타입들 (P-3 approve 가 남긴 type_declared 이벤트
        # 파생). 탐지가 이 타입의 노드를 다시 후보로 올리면 큐가 영원히
        # 마르지 않는다 — 타입명 하드코딩 없이 로그에서 읽는다.
        self._structural_types: set = set()
        # 관계판 묘비 (A1, 2026-08-03) — 기각된 관계 제안의 registry.
        # 실측: 백필 기각 10건 중 8건이 이전 라운드 기각 패턴의 재출현 —
        # 기각이 응답 JSON 으로만 반환되고 어디에도 남지 않았기 때문.
        # 키는 트리플 (subject, predicate, object) 정규화형 — 방향이 있고
        # 술어가 다르면 다른 주장이다.
        self._rejected_relations: Dict[tuple, Dict[str, Any]] = {}
        # 커버리지 gap 기각 (관계판 묘비의 대칭). 키는 (타입, 정규화 이름) —
        # gap 은 아직 노드가 아니라서 node_id 묘비(_rejected)로는 못 막는다.
        # 게다가 reject_node 는 노드를 그래프에서 제거하므로, 다음 라운드
        # known_names 에서 사라져 LLM 이 반드시 다시 제안한다.
        self._rejected_gaps: Dict[tuple, Dict[str, Any]] = {}

    # ─── 경로 ────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """네임스페이스별 저장 파일. 청크·KG 체크포인트와 같은 디렉터리."""
        if self._path is not None:
            return self._path
        return _DEFAULT_DATA_DIR / f"reviews_{self.namespace}.jsonl"

    # ─── 기록 (append-only) ──────────────────────────────────────────

    def record(self, action: str, node_id: str,
               before: Optional[Dict[str, Any]] = None,
               after: Optional[Dict[str, Any]] = None,
               actor: str = "", source: str = "",
               reason: str = "") -> ReviewEvent:
        """감사 이벤트를 추가한다. reject/confirm 이면 파생 상태도 갱신.

        기록 시점에 즉시 디스크에 append 한다 — 판정이 메모리에만 있으면
        프로세스가 죽는 순간 묘비가 사라지고 오추출이 부활한다.
        """
        event = ReviewEvent(action=action, node_id=node_id, actor=actor,
                            reason=reason, source=source,
                            before=before, after=after,
                            at=datetime.now().isoformat())
        self._apply(event)
        self._events.append(event)
        self._append_to_disk(event)
        return event

    def reject(self, node_id: str, reason: str = "", actor: str = "",
               before: Optional[Dict[str, Any]] = None) -> ReviewEvent:
        """묘비 기록 — 판정과 감사가 같은 이벤트다 (따로 적으면 이력이 중복된다)."""
        return self.record("reject", node_id, before=before,
                           actor=actor, reason=reason)

    def confirm(self, node_id: str, actor: str = "",
                before: Optional[Dict[str, Any]] = None) -> ReviewEvent:
        """확정 기록. 기존 묘비가 있으면 걷어낸다 — 나중 판정이 이긴다."""
        return self.record("confirm", node_id, before=before, actor=actor)

    def recommend(self, node_id: str, verdict: str, rationale: str = "",
                  quote: str = "", actor: str = "") -> ReviewEvent:
        """에이전트 추천 기록 — **판정이 아니다**.

        묘비도 확정도 만들지 않는다. 추천은 검수자가 볼 참고 정보이고, 최종
        판정 권한은 인간에게 남는다 (aicoach 오케스트레이터의 "사실 생성 안 함"
        경계와 같은 급). after 에 추천 내용을 실어 감사 이력에도 남긴다 —
        나중에 "에이전트가 뭐라고 했었나"를 추적할 수 있어야 추천 품질을
        평가할 수 있다.
        """
        return self.record("recommend", node_id, actor=actor,
                           reason=rationale,
                           after={"verdict": verdict, "rationale": rationale,
                                  "evidence_quote": quote})

    def _apply(self, event: ReviewEvent) -> None:
        """이벤트 하나를 파생 상태에 반영 — record 와 replay 가 공유하는
        유일한 상태 전이 규칙 (두 경로가 갈라지면 로그≠상태가 된다)."""
        if event.action == "reject":
            self._rejected[event.node_id] = event
            self._confirmed.pop(event.node_id, None)
            # 나중 판정이 이긴다 — 재분류된 옛 id 를 이후에 거절하면
            # 재지도를 걷어낸다 (묘비가 재지도보다 뒤의 판단이다).
            self._reclassified.pop(event.node_id, None)
        elif event.action == "confirm":
            self._confirmed[event.node_id] = event
            self._rejected.pop(event.node_id, None)
        elif event.action == "recommend":
            # 최신 추천이 이긴다 (재검사 허용). 판정 상태는 건드리지 않는다.
            self._recommended[event.node_id] = event
        elif event.action == "relation_reject":
            after = event.after or {}
            key = _relation_key(event.node_id, after.get("predicate"),
                                after.get("target"))
            if all(key):
                self._rejected_relations[key] = {
                    "subject": event.node_id,
                    "predicate": str(after.get("predicate") or ""),
                    "object": str(after.get("target") or ""),
                    "chunk_id": str(after.get("chunk_id") or ""),
                    "scope": str(after.get("scope") or "triple"),
                    "reason": event.reason,
                    "actor": event.actor,
                    "at": event.at,
                }
        elif event.action == "relation_approve":
            # 나중 판정이 이긴다 — 승인이 같은 트리플의 묘비를 걷는다
            # (기존 로그의 approve 는 pop 대상이 없어 no-op — 하위 호환).
            after = event.after or {}
            key = _relation_key(event.node_id, after.get("predicate"),
                                after.get("target"))
            self._rejected_relations.pop(key, None)
        elif event.action == "coverage_reject":
            after = event.after or {}
            key = _gap_key(after.get("type"), after.get("name"))
            if all(key):
                self._rejected_gaps[key] = {
                    "type": str(after.get("type") or ""),
                    "name": str(after.get("name") or ""),
                    "chunk_id": str(after.get("chunk_id") or ""),
                    "scope": str(after.get("scope") or "entity"),
                    "reason": event.reason,
                    "actor": event.actor,
                    "at": event.at,
                }
        elif event.action == "coverage_approve":
            # 나중 판정이 이긴다 — 승인이 같은 gap 의 묘비를 걷는다
            # (기존 로그의 approve 는 pop 대상이 없어 no-op — 하위 호환).
            after = event.after or {}
            self._rejected_gaps.pop(
                _gap_key(after.get("type"), after.get("name")), None)
        elif event.action == "type_declared":
            t = str((event.after or {}).get("type") or "")
            if t:
                self._structural_types.add(t)
        elif event.action == "reclassify":
            after = event.after or {}
            old = str(after.get("old_id") or "")
            new = str(after.get("new_id") or "")
            if old and new and old != new:
                self._reclassified[old] = new

    # ─── 조회 ────────────────────────────────────────────────────────

    def is_rejected(self, node_id: str) -> bool:
        return node_id in self._rejected

    def is_confirmed(self, node_id: str) -> bool:
        return node_id in self._confirmed

    def rejected(self) -> List[Dict[str, Any]]:
        """활성 묘비 목록 (번복된 것은 제외)."""
        return [asdict(e) for e in self._rejected.values()]

    def confirmed_ids(self) -> set:
        """확정된 node_id 집합 — PG 검수 카운트가 그래프 스캔 없이 쓰는 진입점."""
        return set(self._confirmed.keys())

    def rejected_ids(self) -> set:
        """거절(묘비) node_id 집합."""
        return set(self._rejected.keys())

    def reclassify_target(self, node_id: str) -> Optional[str]:
        """재분류 재지도의 종착지 (없으면 None). 체인 전이 + 사이클 가드.

        A→B, B→C 면 A 는 C 로 간다 — 중간 id 로 노드를 만들면 그 다음
        재빌드에서 또 재지도해야 한다. 사이클은 정상 경로로는 못 만들지만
        (타깃 실존 시 rename 거부) 로그 손상에 대비해 한 바퀴에서 멈춘다 —
        무한 루프는 빌드 전체를 죽인다.
        """
        cur = str(node_id)
        visited = {cur}
        while cur in self._reclassified:
            nxt = self._reclassified[cur]
            if nxt in visited:
                logger.warning(
                    f"⚠️ reclassify 사이클 감지 ({nxt}) — 재지도를 여기서 멈춘다")
                break
            visited.add(nxt)
            cur = nxt
        return cur if cur != str(node_id) else None

    def reject_relation(self, subject: str, predicate: str, obj: str, *,
                        chunk_id: str = "", reason: str = "",
                        scope: str = "triple",
                        actor: str = "") -> ReviewEvent:
        """관계 제안 기각 — 관계판 묘비 (relation_approve 와 대칭 이벤트).

        scope: "triple"(기본 — 주장 자체가 거짓, 어느 인용에서 와도 거른다)
             | "evidence"(이 인용만 — 같은 청크 재제안만 거른다).
        """
        if scope not in ("triple", "evidence"):
            scope = "triple"
        return self.record(
            action="relation_reject", node_id=subject,
            reason=reason, actor=actor,
            after={"predicate": predicate, "target": obj,
                   "chunk_id": chunk_id, "scope": scope})

    def relation_rejection(self, subject: str, predicate: str,
                           obj: str) -> Optional[Dict[str, Any]]:
        """이 트리플의 활성 묘비 (없으면 None)."""
        return self._rejected_relations.get(
            _relation_key(subject, predicate, obj))

    def rejected_relations(self) -> List[Dict[str, Any]]:
        """활성 관계 묘비 전량 (감사·보드용 사본)."""
        return [dict(v) for v in self._rejected_relations.values()]

    def reject_gap(self, node_type: str, name: str, *, chunk_id: str = "",
                   reason: str = "", scope: str = "entity",
                   actor: str = "") -> ReviewEvent:
        """커버리지 gap 제안 기각 — coverage_approve 와 대칭 이벤트.

        scope: "entity"(기본 — 이 개체는 gap 이 아니다, 어느 청크에서 와도
               거른다) | "evidence"(이 인용만 — 같은 청크 재제안만 거른다).
        """
        if scope not in ("entity", "evidence"):
            scope = "entity"
        return self.record(
            action="coverage_reject", node_id=f"{node_type}:{name}",
            reason=reason, actor=actor,
            after={"type": node_type, "name": name,
                   "chunk_id": chunk_id, "scope": scope})

    def gap_rejection(self, node_type: str,
                      name: str) -> Optional[Dict[str, Any]]:
        """이 gap 의 활성 묘비 (없으면 None)."""
        return self._rejected_gaps.get(_gap_key(node_type, name))

    def rejected_gaps(self) -> List[Dict[str, Any]]:
        """활성 gap 묘비 전량 (감사·보드용 사본)."""
        return [dict(v) for v in self._rejected_gaps.values()]

    def structural_types(self) -> set:
        """구조 단위로 선언된 타입 집합 (탐지 제외용 사본)."""
        return set(self._structural_types)

    def reclassified_map(self) -> Dict[str, str]:
        """재지도 전량 (보고용 사본)."""
        return dict(self._reclassified)

    def recommendation_for(self, node_id: str) -> Optional[Dict[str, Any]]:
        """이 노드의 최신 에이전트 추천 (없으면 None)."""
        event = self._recommended.get(node_id)
        if event is None or not event.after:
            return None
        return {**event.after, "actor": event.actor, "at": event.at}

    def history(self, node_id: Optional[str] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        """감사 이력, 최신 먼저.

        같은 초 안의 이벤트는 타임스탬프로 구별되지 않으므로 정렬이 아니라
        **로그 순서**를 뒤집는다 — append-only 라 로그 순서가 곧 시간 순서다.
        """
        events = self._events if node_id is None else \
            [e for e in self._events if e.node_id == node_id]
        return [asdict(e) for e in reversed(events[-limit:])]

    # ─── 영속성 ──────────────────────────────────────────────────────

    def _append_to_disk(self, event: ReviewEvent) -> None:
        """이벤트 한 줄 append. 실패해도 판정 자체(메모리)는 유지한다 —
        디스크 오류가 검수 API 를 죽이면 안 된다."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"⚠️ Review store append failed ({self.path}): {e}")

    def load_from_disk(self, path=None) -> bool:
        """로그를 replay 해서 상태를 복원한다. 깨진 줄은 건너뛴다 —
        한 줄 때문에 묘비 전체를 잃으면 오추출이 일제히 부활한다."""
        load_path = Path(path) if path else self.path
        if not load_path.exists():
            logger.info(f"No review store at {load_path} — starting fresh")
            return False
        try:
            # 파생 상태 **전부** clear — _reclassified/_structural_types 가
            # 목록에 빠져 재로드 시 중복 누적되던 결함을 이번에 수리 (2026-08-03).
            self._events.clear()
            self._rejected.clear()
            self._confirmed.clear()
            self._recommended.clear()
            self._reclassified.clear()
            self._structural_types.clear()
            self._rejected_relations.clear()
            self._rejected_gaps.clear()
            skipped = 0
            with open(load_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = ReviewEvent.from_dict(json.loads(line))
                    except Exception:
                        skipped += 1
                        continue
                    self._apply(event)
                    self._events.append(event)
            if skipped:
                logger.warning(f"⚠️ Review store: skipped {skipped} corrupt line(s)")
            logger.info(f"📂 Review store loaded: {len(self._events)} events "
                        f"({len(self._rejected)} tombstones) ← {load_path}")
            return True
        except Exception as e:
            logger.error(f"Review store load failed: {e}")
            return False


# ─── 네임스페이스 싱글턴 ────────────────────────────────────────────

# 네임스페이스-독립 감사 싱크. 네임스페이스 **삭제**처럼 대상과 함께 로그가
# 사라지는 사건을 여기에 남긴다 — `reviews_{ns}.jsonl` 은 삭제 대상에
# 포함되므로 거기 적으면 기록과 증거가 같이 없어진다.
# 앞의 `_` 가 방어선이다: 빌더·업로드가 만드는 네임스페이스 이름은 이 모양이
# 될 수 없고, `list_namespaces` 는 kg_*.json 을 훑으므로 여기 파일만으로는
# 목록에 나타나지 않는다 (유령 네임스페이스를 만들지 않는다).
ADMIN_AUDIT_NAMESPACE = "_admin"

_stores: Dict[str, ReviewStore] = {}


def get_review_store(namespace: str = "default") -> ReviewStore:
    """네임스페이스별 공유 ReviewStore. 첫 호출 시 디스크에서 replay 로드
    (get_chunk_store 와 같은 수명·같은 규약)."""
    if namespace not in _stores:
        store = ReviewStore(namespace=namespace)
        store.load_from_disk()
        _stores[namespace] = store
        logger.info(f"🧾 Review store initialized (namespace={namespace})")
    return _stores[namespace]


def reset_review_stores() -> None:
    """싱글턴 초기화 — 테스트 격리용."""
    _stores.clear()
