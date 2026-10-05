"""
검색 QA (⑤) — 골든셋 하네스: 검색 품질을 감이 아니라 숫자로.

이 코드베이스에는 정직하게 표시해둔 빚이 있다: entry_ratio=0.5(graph_retrieval),
RRF_K=60 은 **측정값이 아니다**. aicoach 는 KII 골든셋으로 BM25 가중치를
측정해 recall@5=0.944 를 만들었다 (rag/search.py:20-22 — "tuned on KII
grounding golden set"). 골든셋 없이는 모든 검색 변경이 장님이다.

역할 분담 (검수 루프와 같은 권한 경계):
- **평가는 결정적, LLM 0콜** — 평가에 LLM 이 끼면 평가 자체가 비결정이 되어
  회귀 검사가 성립하지 않는다. evaluate_cases 는 순수 함수다.
- **LLM(생성기)은 초안 제안까지** — status=draft 로 들어오고 확정은 인간이
  한다. 생성 질의가 정답 노드의 이름을 문자 그대로 포함하면 버린다:
  키워드 매칭만 테스트하는 케이스는 의미 검색을 재지 못한다 (패러프레이즈
  강제 — evidence_checker 의 인용 검증과 대칭인, 반대 방향의 게이트다).
- 케이스는 자산이다 — JSONL append + replay (review_store 규약).
"""

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def _squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class GoldenCase:
    """골든 케이스 하나 — 자연어 질의와 그 정답 노드."""
    case_id: str
    query: str
    expected_node_id: str
    # draft(LLM 초안) | verified(기계 왕복 검증) | confirmed(인간 확정)
    #
    # `verified` 가 따로 있는 이유: 초안은 라벨이 검증되지 않았고 `confirmed` 는
    # **인간**이 한다는 계약이 있어 그 사이가 비어 있었다. 왕복 검증(노드 → 질의
    # → 혼동 후보 중에서 다시 노드)을 통과한 라벨은 초안보다 강하지만 사람이 본
    # 것은 아니다. 그 차이를 status 로 남긴다 — 기본 평가는 confirmed 만 쓰고
    # verified 는 명시해야 들어간다(evaluate_cases 의 statuses).
    status: str = "draft"
    source: str = ""             # hand | generator | ...
    # 시나리오 축(선택) — 축 A 매칭난이도(exact/semantic/graph) × 축 B 의도.
    # eval 이 태그별로 지표를 분해해 "어느 유형이 약한가 → 어느 knob"을 준다.
    tags: List[str] = field(default_factory=list)
    # 추가 정답(relevant set) — 동의어/교차연결로 개념이 여러 노드에 나뉠 때
    # (유방암·유방의 악성 신생물·C50) 어느 것이 top-k 에 와도 맞다고 credit.
    # 정답 집합 = {expected_node_id} ∪ accepted. 옛 케이스는 [] → 단일 정답과 동일.
    accepted: List[str] = field(default_factory=list)
    # 청크(원문 조각) 단위 정답 — 노드 단위와 **별개 축**이다. 노드 채점은
    # semantic 과 retrieve 가 같게 나온다(retrieve 의 노드 랭킹이 semantic 진입
    # 노드에서 파생 — retrieve_result_to_nodes 의 측정 기록 참고). 우리 차별점
    # (확장 질의 + 그래프 채널 + RRF)은 청크 회수에서 작동하므로 그걸 재려면
    # 정답도 청크여야 한다. 옛 케이스는 빈 값 → 청크 채점에서 스킵된다.
    expected_chunk_id: str = ""
    accepted_chunks: List[str] = field(default_factory=list)

    def accepted_ids(self) -> set:
        ids = {self.expected_node_id} if self.expected_node_id else set()
        ids.update(a for a in self.accepted if a)
        return ids

    def accepted_chunk_ids(self) -> set:
        """청크 정답 집합 — accepted_ids 와 대칭. 비어 있으면 청크 채점 대상이 아니다."""
        ids = {self.expected_chunk_id} if self.expected_chunk_id else set()
        ids.update(c for c in self.accepted_chunks if c)
        return ids

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenCase":
        return cls(case_id=data["case_id"], query=data.get("query", ""),
                   expected_node_id=data.get("expected_node_id", ""),
                   status=data.get("status", "draft"),
                   source=data.get("source", ""),
                   tags=list(data.get("tags") or []),      # 옛 레코드엔 없음 → []
                   accepted=list(data.get("accepted") or []),
                   expected_chunk_id=data.get("expected_chunk_id", "") or "",
                   accepted_chunks=list(data.get("accepted_chunks") or []))


class GoldenSet:
    """네임스페이스 하나의 골든셋 — JSONL 이벤트 로그 (add/confirm) + replay."""

    def __init__(self, namespace: str = "default", path=None):
        self.namespace = namespace
        self._path = Path(path) if path else None
        self._cases: Dict[str, GoldenCase] = {}
        self._query_keys: set = set()  # 공백 정규화된 질의 — 중복 방지

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        return _DEFAULT_DATA_DIR / f"goldenset_{self.namespace}.jsonl"

    # ─── 쓰기 (이벤트) ───────────────────────────────────────────────

    def add(self, query: str, expected_node_id: str,
            status: str = "draft", source: str = "",
            tags: Optional[List[str]] = None,
            accepted: Optional[List[str]] = None,
            expected_chunk_id: str = "",
            accepted_chunks: Optional[List[str]] = None) -> Optional[str]:
        """케이스 추가. 같은 질의(공백 무시)는 두 번 넣지 않는다 —
        중복 케이스는 그 질의에 지표를 과가중시킨다. accepted 는 추가 정답
        (relevant set) — 동의어/교차연결 개념을 credit 하기 위한 것.
        expected_chunk_id/accepted_chunks 는 청크 단위 채점용(선택)."""
        key = _squash_ws(query)
        if not key or key in self._query_keys:
            return None
        case_id = hashlib.sha1(
            f"{self.namespace}::{key}".encode("utf-8")).hexdigest()[:12]
        case = GoldenCase(case_id=case_id, query=query.strip(),
                          expected_node_id=expected_node_id,
                          status=status, source=source,
                          tags=list(tags or []),
                          accepted=list(accepted or []),
                          expected_chunk_id=expected_chunk_id or "",
                          accepted_chunks=list(accepted_chunks or []))
        self._cases[case_id] = case
        self._query_keys.add(key)
        self._append({"event": "add", "case": asdict(case)})
        return case_id

    def confirm(self, case_id: str) -> bool:
        """초안 → 확정. 확정만 지표에 들어간다 (evaluate 기본값)."""
        case = self._cases.get(case_id)
        if case is None:
            return False
        case.status = "confirmed"
        self._append({"event": "confirm", "case_id": case_id})
        return True

    def accept(self, case_id: str, node_id: str) -> bool:
        """정답 집합 확장 — 케이스의 `accepted` 에 노드를 더한다.

        **골든셋도 그래프 상태에 종속적이다** (실측): PROJ-A 커버리지 회복 후
        hit@1 이 내려갔는데 검색 회귀가 아니라 **라벨 노후화**였다 — 새로 생긴
        같은 개념(요구서 유래 노드)이 1위에 오는데 라벨(제안서 유래)이 그걸
        몰랐다. 코퍼스가 자라면 정답 집합도 자라야 하고, `accepted` 가 그
        문서화된 용도다.

        판정(status)은 건드리지 않는다 — 정답 확장은 판정 번복이 아니다.
        이미 정답인 노드는 False (집합이 거짓으로 커지면 안 된다).
        """
        case = self._cases.get(case_id)
        node_id = (node_id or "").strip()
        if case is None or not node_id or node_id in case.accepted_ids():
            return False
        case.accepted.append(node_id)
        self._append({"event": "accept", "case_id": case_id,
                      "node_id": node_id})
        return True

    def verify(self, case_id: str, picked: str = "") -> bool:
        """초안 → verified (기계 왕복 검증 통과).

        **confirmed 는 강등하지 않는다** — 인간 판정이 기계보다 강하고, 강등하면
        사람이 한 작업을 조용히 지운다. 그 경우 False 를 돌려준다(실패가 아니라
        "할 일이 없었다"인데, 호출자가 승격 수를 세므로 구분되어야 한다).

        picked 는 왕복에서 LLM 이 고른 node_id — 감사용으로 이벤트에만 남긴다.
        """
        case = self._cases.get(case_id)
        if case is None or case.status == "confirmed":
            return False
        case.status = "verified"
        self._append({"event": "verify", "case_id": case_id,
                      "picked": picked or ""})
        return True

    def relabel_node(self, old_node_id: str, new_node_id: str) -> List[str]:
        """정답 노드 id 를 갈아끼운다 — 노드 병합의 골든셋 쪽 절반.

        **왜 라벨을 고쳐도 되는가**: 라벨이 가리키는 것은 id 문자열이 아니라
        **개념**이고, 병합 후 그 개념은 이긴 노드에 산다. 안 고치면 46건이
        지워진 id 를 가리켜 조용히 전부 오답이 되고, 그러면 "검색이 나빠진 것"과
        "라벨이 깨진 것"을 구별할 수 없다 — 지표가 거짓말을 시작한다.

        **이벤트로 남기는 이유**: 골든셋은 추가전용 로그이고 로그가 원본이다.
        파일을 고쳐 쓰면 그 계약이 깨지고, 무엇이 왜 바뀌었는지도 사라진다.

        돌려주는 것은 손댄 case_id 들. 판정(draft/confirmed)은 건드리지 않는다 —
        병합은 사람의 정답 판단을 번복하는 사건이 아니다.
        """
        old_node_id = (old_node_id or "").strip()
        new_node_id = (new_node_id or "").strip()
        if not old_node_id or not new_node_id or old_node_id == new_node_id:
            return []
        touched = [c.case_id for c in self._cases.values()
                   if old_node_id in c.accepted_ids()]
        if not touched:
            return []
        for case_id in touched:
            self._apply_relabel(self._cases[case_id], old_node_id, new_node_id)
        self._append({"event": "relabel", "old": old_node_id,
                      "new": new_node_id, "case_ids": touched})
        return touched

    @staticmethod
    def _apply_relabel(case: "GoldenCase", old: str, new: str) -> None:
        """한 케이스의 정답 참조를 갈아끼운다 (메모리 · 로드 재생 공용).

        accepted 에서 expected 와 같아진 항목은 지운다 — accepted 는 '추가'
        정답이라 expected 와 겹치면 의미가 없고, 남겨두면 정답 집합 크기가
        부풀어 보인다.
        """
        if case.expected_node_id == old:
            case.expected_node_id = new
        case.accepted = [
            item for item in dict.fromkeys(
                new if a == old else a for a in case.accepted)
            if item and item != case.expected_node_id]

    # ─── 읽기 ────────────────────────────────────────────────────────

    def cases(self, status: Optional[str] = None) -> List[GoldenCase]:
        found = list(self._cases.values())
        if status:
            found = [c for c in found if c.status == status]
        return found

    def __len__(self) -> int:
        return len(self._cases)

    # ─── 영속성 ──────────────────────────────────────────────────────

    def _append(self, event: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"⚠️ Golden set append failed ({self.path}): {e}")

    def load_from_disk(self, path=None) -> bool:
        load_path = Path(path) if path else self.path
        if not load_path.exists():
            return False
        try:
            self._cases.clear()
            self._query_keys.clear()
            skipped = 0
            with open(load_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("event") == "add":
                            case = GoldenCase.from_dict(event["case"])
                            self._cases[case.case_id] = case
                            self._query_keys.add(_squash_ws(case.query))
                        elif event.get("event") == "confirm":
                            case = self._cases.get(event.get("case_id", ""))
                            if case:
                                case.status = "confirmed"
                        elif event.get("event") == "accept":
                            case = self._cases.get(event.get("case_id", ""))
                            nid = (event.get("node_id") or "").strip()
                            if case and nid and nid not in case.accepted_ids():
                                case.accepted.append(nid)
                        elif event.get("event") == "verify":
                            case = self._cases.get(event.get("case_id", ""))
                            # confirmed 를 덮지 않는다 — 로그가 verify → confirm
                            # 순서면 최종은 confirmed 여야 한다(verify 와 같은 규칙).
                            if case and case.status != "confirmed":
                                case.status = "verified"
                        elif event.get("event") == "relabel":
                            # 노드 병합의 재생. case_ids 를 신뢰하지 않고 현재
                            # 상태에서 다시 판정한다 — 로그를 순서대로 재생하는
                            # 중이므로, 앞선 이벤트로 달라진 상태가 진실이다.
                            old = event.get("old", "")
                            new = event.get("new", "")
                            if old and new:
                                for case in self._cases.values():
                                    if old in case.accepted_ids():
                                        self._apply_relabel(case, old, new)
                    except Exception:
                        skipped += 1
            if skipped:
                logger.warning(f"⚠️ Golden set: skipped {skipped} corrupt line(s)")
            logger.info(f"📏 Golden set loaded: {len(self._cases)} cases "
                        f"← {load_path}")
            return True
        except Exception as e:
            logger.error(f"Golden set load failed: {e}")
            return False


# ─── 평가 (결정적 — LLM 0콜) ─────────────────────────────────────────

RankFn = Callable[[str, int], List[str]]  # (query, top_k) → [node_id, ...]


def chunk_hits_to_nodes(hits: List[Dict[str, Any]]) -> List[str]:
    """청크 hit 리스트 → 노드 랭킹 (retrieve 채널 어댑터, **순수 함수**).

    /retrieve 는 청크를 돌려주지만 골든셋은 노드 단위(expected_node_id)로 평가한다.
    각 청크 hit 의 node_ids 를 hit 순서로 펴고 중복을 제거해 '노드가 처음 등장한
    청크의 순위'로 노드 랭킹을 만든다 — 그래프-조건부 검색을 노드 단위로 잴 수 있게."""
    seen: set = set()
    out: List[str] = []
    for hit in hits or []:
        for nid in (hit.get("node_ids") or []):
            if nid and nid not in seen:
                seen.add(nid)
                out.append(nid)
    return out


def chunk_hits_to_ids(hits: Any) -> List[str]:
    """청크 hit → 청크 id 랭킹 (청크 채널 어댑터, **순수 함수**).

    세 가지 hit 모양을 모두 받는다 — 채널마다 반환형이 다른데 그걸 호출부에
    떠넘기면 채널을 나란히 비교하는 evaluate_cases 의 계약이 깨진다:
      · dict           — /retrieve 응답의 hits (chunk_id 키)
      · ChunkHit       — GraphConditionedRetriever.search() (.chunk.chunk_id)
      · (chunk, score) — ChunkIndex.search()
    깨진 항목은 조용히 건너뛴다 — 채널 하나가 죽어 채점 전체가 멈추면 안 된다.
    """
    seen: set = set()
    out: List[str] = []
    for hit in hits or []:
        cid = ""
        if isinstance(hit, dict):
            cid = hit.get("chunk_id") or ""
        elif isinstance(hit, (tuple, list)) and hit:
            cid = getattr(hit[0], "chunk_id", "") or ""
        else:
            cid = getattr(getattr(hit, "chunk", None), "chunk_id", "") or ""
        cid = str(cid)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def retrieve_result_to_nodes(result: Optional[Dict[str, Any]]) -> List[str]:
    """/retrieve 결과 전체 → 노드 랭킹 (retrieve 채널 어댑터, **순수 함수**).

    hits 만 펴면(chunk_hits_to_nodes) 정답 노드가 상위 청크의 여러 node_ids 중
    뒤로 밀려 hit@1 이 구조적으로 0이 됐다. 그래프-조건부 검색은 이미 노드 단위
    신호를 준다 — 그걸 우선한다:
      1) entry_nodes: 질의가 **직접 매칭한 노드**(이름/의미 직결) — 점수 내림차순
      2) expanded_nodes: 그래프 hop 으로 닿은 노드 (온톨로지가 명시한 관계)
      3) 청크 hit 의 node_ids: 위에서 안 나온 나머지(hit 순)
      4) via="propagation" 확장 노드 — **맨 뒤**
    첫 등장 유지로 중복 제거.

    확산(PPR)을 맨 뒤로 보내는 이유는 실측이다. 처음엔 2)에 섞었더니
    **hit@10 0.9375 → 0.8750** 으로 떨어졌다 — 확산 노드 최대 10개가 청크 유래
    노드 앞에 끼어들어 정답을 꼬리 밖으로 밀어냈다. 확산은 1-hop 이 닿지 못하는
    곳을 메우는 **보강**이고, 명시된 관계나 실제 근거를 밀어낼 근거가 없다.
    맨 뒤에 두면 순수하게 더하기만 한다(기존 순위 불변 — 회귀 불가능).

    ⚠️ 2-D 실험 기록: 세 소스를 RRF 로 융합해 coverage 를 회복하려 했으나
    **측정상 더 나빴다**(retrieve hit@5 0.88→0.75, definition/exact/procedure
    후퇴, coverage 는 그대로 0.60). coverage 실패는 랭킹 순서가 아니라 **임베딩
    리콜**(정답 노드가 어느 채널에서도 상위에 없음)이라 RRF 로 못 고친다 —
    eval_sweep 이 가리킨 결론. entry-우선이 국소 최적이라 이 설계를 유지한다.
    다음 레버는 임베더/질의확장이지 노드 랭킹 블렌드가 아니다."""
    res = result or {}
    exp = res.get("expansion") or {}
    ordered: List[str] = []
    for e in sorted(exp.get("entry_nodes") or [],
                    key=lambda x: -(x.get("score") or 0.0)):
        ordered.append(e.get("node_id"))
    propagated: List[str] = []
    for e in exp.get("expanded_nodes") or []:
        if (e.get("via") or "") == "propagation":
            propagated.append(e.get("node_id"))   # 꼬리로 미룬다 (위 설명)
        else:
            ordered.append(e.get("node_id"))
    for hit in res.get("hits") or []:
        ordered.extend(hit.get("node_ids") or [])
    ordered.extend(propagated)
    seen: set = set()
    out: List[str] = []
    for nid in ordered:
        if nid and nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def make_chunk_expander(store) -> Callable[[str], set]:
    """청크 id → 그 청크가 근거인 노드 집합 (target="evidence" 의 확장 함수).

    `evaluate_cases` 는 순수 함수로 남아야 하므로(커널 경계 — ChunkStore 를
    모른다) 이 어댑터가 저장소 의존을 흡수한다.

    **없는 청크·고아 청크·터진 저장소는 모두 빈 집합**이다. 청크가 지워졌는데
    색인에 남아 있는 것은 정상 상태이고(재색인 전), 그것 때문에 측정 전체가
    죽으면 회귀를 볼 수 없다. 빈 집합은 "이 청크는 정답을 근거하지 않는다"로
    안전하게 채점된다 — 없는 것을 맞다고 세지는 않는다.
    """
    def expand(chunk_id: str) -> set:
        try:
            chunk = store.get(chunk_id) if store is not None else None
        except Exception:                     # 저장소 장애 — 채널 하나를 잃을 뿐
            return set()
        return set(getattr(chunk, "node_ids", None) or []) if chunk else set()

    return expand


def evaluate_cases(cases: List[GoldenCase],
                   channels: Dict[str, RankFn],
                   k: int = 5,
                   include_drafts: bool = False,
                   target: str = "node",
                   expand_fn: Optional[Callable[[str], set]] = None,
                   statuses: Optional[set] = None) -> Dict[str, Any]:
    """골든셋 평가 — 채널별 hit@1 / hit@k / MRR + 케이스별 순위.

    순수 함수다. 채널은 (query, top_k) → id 순위 리스트를 돌려주는 함수라면
    무엇이든 — semantic_search, /retrieve, 설정만 다른 같은 검색기 두 개
    (entry_ratio 0.3 vs 0.5)를 나란히 넣어 비교하는 것이 이 구조의 목적이다.

    target 은 **채점 위치**다:
      · "node"     — 정답 = accepted_ids(),        랭킹 = 노드
      · "chunk"    — 정답 = accepted_chunk_ids(),  랭킹 = 청크
      · "evidence" — 정답 = accepted_ids()(노드),  랭킹 = **청크**
    노드 채점은 semantic 과 retrieve 가 같게 나온다(구조상 파생) — 우리 차별점은
    청크 회수에 있으므로 그걸 재려면 채점 위치를 옮겨야 한다.

    **evidence 가 따로 있는 이유** (실측이 요구했다): 확산 청크 채널을 켰을 때
    지표가 0.9375 → 0.8750 으로 떨어졌지만, 그건 청크 채널을 **노드 자로** 잰
    숫자였다(청크 순서 변경 → retrieve_result_to_nodes 의 3번 소스 재배열).
    정작 청크 자(target="chunk")는 골든셋에 청크 라벨이 없어 0 케이스였다.
    게다가 노드 채점은 청크 채널에 **구조적으로** 불리하다 — 1위 청크가 정답의
    근거인데 그 청크의 node_ids 중 정답이 5번째면 rank 5 로 센다.

    evidence 는 라벨을 새로 만들지 않고 **기존 노드 라벨을 청크 위치에 적용**해
    "몇 번째 청크가 정답 노드의 근거인가"를 센다 (supporting-passage recall).
    ⚠️ 편향을 숨기지 않는다: 라벨이 근거 링크(노드↔청크)에서 오므로 그래프를
    아는 채널에 유리하다. 절대 우열이 아니라 **같은 자로 재는 채널 비교**다.

    expand_fn 은 "랭킹 항목 → 정답과 맞춰볼 id 집합". 기본 None 은 항등
    (항목 자신) — 기존 동작과 완전히 같다. evidence 는 청크→노드 확장이
    필수이므로 **없으면 ValueError**: 폴백하면 청크 id ∈ 노드 id = 항상 공집합
    → 전 채널 0.0 인데, 그건 이 함수가 고치려는 바로 그 거짓말이다.

    해당 타깃의 정답이 없는 케이스는 **스킵**하고 skipped 로 보고한다. 오답으로
    세면 "라벨이 없다"가 "검색이 틀렸다"로 뒤바뀌어 지표가 거짓말을 한다.
    같은 이유로 **0 건이면 지표는 0.0 이 아니라 None** 이고 measured=False 다 —
    소비자가 cases 를 같이 봐야만 진실을 아는 구조는 foot-gun 이다(graph_health
    에서 이미 같은 부류의 0/0 → 0.0 오보고를 고쳤다).

    per_case 를 함께 돌려주는 이유: 지표만으로는 못 고친다 — 어느 케이스가
    몇 위였는지가 있어야 회귀의 원인을 찾는다. 미검출은 None 으로 명시된다.
    """
    target = target if target in ("node", "chunk", "evidence") else "node"
    if target == "evidence" and expand_fn is None:
        raise ValueError(
            'target="evidence" 는 expand_fn(청크 id → 노드 집합)이 필요하다. '
            "없으면 청크 id 를 노드 정답과 대조해 전 채널이 0.0 이 되고, "
            "그건 '측정 불가'를 '전부 실패'로 보고하는 거짓말이다.")
    if target == "chunk":
        def _accepted(c):
            return c.accepted_chunk_ids()

        def _expected(c):
            return c.expected_chunk_id
    else:
        def _accepted(c):
            return c.accepted_ids()

        def _expected(c):
            return c.expected_node_id

    # 어느 라벨 집합을 재는가. 기본은 confirmed 만 — verified(기계 왕복)가
    # 조용히 섞이면 지표가 "인간이 확인한 정확도"라는 뜻을 잃는다. 빈 집합은
    # "아무것도 재지 마라"가 아니라 미지정으로 본다(조용한 0건 방지).
    selected = set(statuses) if statuses else None
    if selected is not None:
        eligible = [c for c in cases if c.status in selected]
    else:
        eligible = [c for c in cases
                    if include_drafts or c.status == "confirmed"]
    active = [c for c in eligible if _accepted(c)]
    skipped = len(eligible) - len(active)

    def _blank():
        return {name: {"hit1": 0, "hitk": 0, "rr": 0.0, "n": 0} for name in channels}

    def _metrics(t, n):
        # 0 건이면 None — 0.0 은 "다 틀렸다"로 읽힌다 (docstring 참고).
        return {name: {"cases": n,
                       "hit@1": (v["hit1"] / n) if n else None,
                       f"hit@{k}": (v["hitk"] / n) if n else None,
                       "mrr": (v["rr"] / n) if n else None}
                for name, v in t.items()}

    def _matches(ranked_id, accepted) -> bool:
        if expand_fn is None:
            return ranked_id in accepted        # 항등 — 기존 동작
        return bool(expand_fn(ranked_id) & accepted)

    per_case: List[Dict[str, Any]] = []
    totals = _blank()
    tag_totals: Dict[str, Dict[str, Dict[str, float]]] = {}   # tag → channel → 누적

    for case in active:
        row: Dict[str, Any] = {"case_id": case.case_id, "query": case.query,
                               "expected": _expected(case),
                               "tags": list(getattr(case, "tags", []) or [])}
        accepted = _accepted(case)   # {expected} ∪ accepted — 아무거나 맞으면 credit
        for name, rank_fn in channels.items():
            ranking = list(rank_fn(case.query, k))[:k]
            position = next((i + 1 for i, nid in enumerate(ranking)
                             if _matches(nid, accepted)), None)
            row[f"{name}_rank"] = position
            hit1 = 1 if position == 1 else 0
            hitk = 1 if position is not None else 0
            rr = (1.0 / position) if position is not None else 0.0
            for bucket in (totals[name],
                           *(tag_totals.setdefault(tag, _blank())[name]
                             for tag in row["tags"])):
                bucket["hit1"] += hit1
                bucket["hitk"] += hitk
                bucket["rr"] += rr
                bucket["n"] += 1
        per_case.append(row)

    n = len(active)
    by_tag = {tag: _metrics(t, next(iter(t.values()))["n"] if t else 0)
              for tag, t in sorted(tag_totals.items())}
    out = {"k": k, "target": target, "cases": n, "skipped": skipped,
           "measured": n > 0,
           # 어느 라벨 집합을 쟀는지 — 없으면 두 숫자(confirmed vs +verified)를
           # 헷갈린다.
           "statuses": sorted(selected) if selected is not None
           else (["*"] if include_drafts else ["confirmed"]),
           "channels": _metrics(totals, n),
           "by_tag": by_tag, "per_case": per_case}
    if n == 0:
        # 왜 못 쟀는지가 응답에 있어야 사람이 다음 행동을 안다.
        label = ("청크 정답(expected_chunk_id)" if target == "chunk"
                 else "노드 정답(expected_node_id)")
        out["reason"] = (
            f"채점 대상 0건 — {label} 이 있는 "
            f"{'' if include_drafts else '확정 '}케이스가 없다. "
            "지표는 0 이 아니라 미측정(None)이다.")
    return out


# ─── 생성기 (LLM — 초안까지만) ──────────────────────────────────────

_GENERATE_PROMPT = """당신은 검색 품질 평가용 질의 작성자입니다.
아래 개체를 찾는 자연어 질문을 {count}개 만드세요.

## 규칙 (반드시 지킬 것)
1. 질문에 개체의 이름("{name}")을 **절대 그대로 쓰지 마세요** — 이름이 들어간
   질문은 키워드 매칭 테스트일 뿐, 의미 검색을 재지 못합니다.
   뜻이 같은 다른 표현(패러프레이즈)으로 물어야 합니다.
2. 실제 사용자가 칠 법한 짧고 자연스러운 한국어 질문.
3. JSON 외의 다른 텍스트를 출력하지 마세요.

## 개체
{node}

## 출력 형식
{{"queries": ["...", "..."]}}
"""


def build_generate_prompt(node_view: Dict[str, Any], count: int = 2) -> str:
    return _GENERATE_PROMPT.format(
        count=count, name=node_view.get("name", ""),
        node=json.dumps({k: v for k, v in node_view.items()
                         if k in ("name", "type", "definition") and v},
                        ensure_ascii=False))


def parse_generated_cases(raw: str,
                          node_view: Dict[str, Any]) -> List[Dict[str, str]]:
    """생성기 출력 검증 — LLM 출력은 불신한다.

    - 정답 노드의 이름이 질의에 문자 그대로(공백 무시) 들어 있으면 버린다:
      그 케이스는 키워드 매칭만 테스트한다 (골든셋의 목적에 반함).
    - 빈 질의·중복 질의 드롭. 절대 raise 하지 않는다.
    """
    from ..builder.extractor import parse_llm_json

    parsed = parse_llm_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return []

    name_key = _squash_ws(str(node_view.get("name", ""))).lower()
    node_id = str(node_view.get("node_id", ""))
    seen: set = set()
    cases: List[Dict[str, str]] = []
    for query in parsed.get("queries") or []:
        if not isinstance(query, str):
            continue
        key = _squash_ws(query)
        if not key or key in seen:
            continue
        if name_key and name_key in key.lower().replace(" ", "") \
                or (name_key and name_key in _squash_ws(query.lower())):
            continue  # 이름 포함 — 패러프레이즈가 아니다
        seen.add(key)
        cases.append({"query": query.strip(), "expected_node_id": node_id})
    return cases


_ROUNDTRIP_PROMPT = """다음 질문이 **어느 개체를 묻는지** 고르세요.

## 질문
{query}

## 후보 개체
{candidates}

## 규칙
1. 후보 목록의 node_id 중 **하나만** 고르세요.
2. 어느 것인지 확신할 수 없거나, 질문이 여러 개체에 똑같이 해당하면
   반드시 "none" 을 답하세요. **추측하지 마세요** — 틀린 답보다 none 이 낫습니다.
3. JSON 외의 다른 텍스트를 출력하지 마세요.

## 출력 형식
{{"node_id": "..."}}
"""


def build_roundtrip_prompt(query: str,
                           candidates: List[Dict[str, Any]]) -> str:
    """왕복 검증 프롬프트 — "이 질의가 묻는 개체는?" (순수 함수).

    **후보를 node_id 로 정렬한다.** 호출자가 정답을 먼저 넣는 습관이 있으면
    위치가 정답을 흘리고, 그러면 검증이 아니라 받아쓰기가 된다. 정렬은 입력
    순서를 지우므로 위치에 정보가 없다.

    거부("none")를 허용하는 이유: 강제 선택은 추측을 만들고, 추측이 통과하면
    라벨이 오염된다. 통과율이 낮아지는 것은 이 설계의 비용이 아니라 목적이다.
    """
    lines: List[str] = []
    for cand in sorted(candidates or [],
                       key=lambda c: str(c.get("node_id", ""))):
        node_id = str(cand.get("node_id", "") or "")
        if not node_id:
            continue
        definition = _squash_ws(str(cand.get("definition") or ""))[:160]
        lines.append(f'- {node_id}' + (f' — {definition}' if definition else ""))
    return _ROUNDTRIP_PROMPT.format(
        query=query.strip(),
        candidates="\n".join(lines) if lines else "(없음)")


def parse_roundtrip(raw: str, candidate_ids: set) -> str:
    """왕복 출력 검증 — 고른 node_id 또는 "" (순수 함수, 절대 raise 안 함).

    **후보에 없는 id 는 버린다.** LLM 이 그럴듯한 id 를 지어내는 것은 흔하고,
    통과시키면 라벨이 조용히 오염된다 — 골든셋이 오염되면 그 뒤 모든 측정이
    거짓말을 한다.
    """
    from ..builder.extractor import parse_llm_json

    try:
        parsed = parse_llm_json(raw)
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    picked = _squash_ws(str(parsed.get("node_id") or ""))
    if not picked or picked.lower() == "none":
        return ""
    return picked if picked in (candidate_ids or set()) else ""


class QAGenerator:
    """골든 케이스 초안 생성기 — gemini-3.5-flash 기본, llm_fn 주입 가능."""

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None,
                 llm_provider: str = "google",
                 llm_model: Optional[str] = None,
                 llm_base_url: Optional[str] = None):
        self.llm_fn = llm_fn
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self._provider = None

    async def _call_llm(self, prompt: str) -> str:
        if self.llm_fn is not None:
            return await asyncio.to_thread(self.llm_fn, prompt)
        if self._provider is None:
            from .llm_provider import resolve_provider
            self._provider = resolve_provider(
                self.llm_provider, self.llm_model, base_url=self.llm_base_url)
        return await self._provider.complete(prompt)

    async def generate_for_nodes(self, golden_set: GoldenSet,
                                 node_views: List[Dict[str, Any]],
                                 per_node: int = 2,
                                 tags: Optional[List[str]] = None) -> int:
        """노드마다 패러프레이즈 질의 초안을 생성해 골든셋에 추가한다.

        전부 status=draft — 확정은 인간이 한다. 노드당 LLM 1콜.
        한 노드의 실패는 그 노드만 건너뛴다 (절대 raise 안 함).

        생성 초안은 이름을 금지한 패러프레이즈라 **의미 매칭형(semantic)**이다 —
        기본 태그를 그렇게 달아 eval 태그별 분해에서 semantic 버킷을 채운다.
        (exact/graph 유형은 손 시드 몫 — 생성기로는 안 나온다.)
        """
        tags = tags if tags is not None else ["semantic"]
        added = 0
        for node_view in node_views:
            try:
                raw = await self._call_llm(
                    build_generate_prompt(node_view, count=per_node))
            except Exception as e:
                logger.warning(f"⚠️ QA generation failed "
                               f"({node_view.get('node_id')}): {e}")
                continue
            for case in parse_generated_cases(raw, node_view):
                if golden_set.add(case["query"], case["expected_node_id"],
                                  status="draft", source="generator", tags=tags):
                    added += 1
        if added:
            logger.info(f"📏 Golden set: +{added} draft case(s) generated")
        return added

    async def verify_drafts(self, golden_set: GoldenSet,
                            candidate_fn: Callable[[str, str], List[Dict[str, Any]]],
                            limit: int = 0) -> Dict[str, Any]:
        """초안을 왕복 검증해 통과분을 verified 로 승격한다. 케이스당 LLM 1콜.

        `candidate_fn(query, expected_node_id)` → 후보 노드 뷰 목록. 정답을
        반드시 포함해야 한다(없으면 통과가 불가능하다) — 호출자 책임이고,
        여기서는 후보에 정답이 없으면 `no_candidates` 로 건너뛴다. 그렇지
        않으면 "LLM 이 틀렸다"와 "후보를 잘못 만들었다"가 섞인다.

        승격하지 못한 초안은 **그대로 draft 로 남긴다** — 지우지 않는다.
        나중에 사람이 보거나 더 나은 후보로 다시 시도할 수 있다.

        한 케이스의 실패는 그 케이스만 건너뛴다 (절대 raise 안 함).
        """
        drafts = [c for c in golden_set.cases() if c.status == "draft"]
        if limit:
            drafts = drafts[:limit]
        promoted = 0
        rejected: List[Dict[str, str]] = []
        for case in drafts:
            candidates = list(candidate_fn(case.query, case.expected_node_id)
                              or [])
            ids = {str(c.get("node_id", "")) for c in candidates}
            if case.expected_node_id not in ids:
                rejected.append({"case_id": case.case_id,
                                 "reason": "no_candidates"})
                continue
            try:
                raw = await self._call_llm(
                    build_roundtrip_prompt(case.query, candidates))
            except Exception as e:
                logger.warning(f"⚠️ Roundtrip verify failed "
                               f"({case.case_id}): {e}")
                rejected.append({"case_id": case.case_id, "reason": "llm_error"})
                continue
            picked = parse_roundtrip(raw, ids)
            if picked and picked == case.expected_node_id:
                if golden_set.verify(case.case_id, picked=picked):
                    promoted += 1
            else:
                # 무엇을 골랐는지 남긴다 — "왜 떨어졌나"를 사람이 봐야 한다
                # (혼동 쌍이 드러나면 그 자체가 온톨로지 결함 신호다).
                rejected.append({"case_id": case.case_id,
                                 "reason": "mismatch", "picked": picked or "none"})
        logger.info(f"📏 Golden set roundtrip: +{promoted} verified, "
                    f"{len(rejected)} not promoted")
        return {"checked": len(drafts), "promoted": promoted,
                "rejected": rejected}


# ─── 네임스페이스 싱글턴 ────────────────────────────────────────────

_golden_sets: Dict[str, GoldenSet] = {}


def get_golden_set(namespace: str = "default") -> GoldenSet:
    """네임스페이스별 공유 GoldenSet (chunk/review store 와 같은 규약)."""
    if namespace not in _golden_sets:
        golden = GoldenSet(namespace=namespace)
        golden.load_from_disk()
        _golden_sets[namespace] = golden
    return _golden_sets[namespace]


def reset_golden_sets() -> None:
    """싱글턴 초기화 — 테스트 격리용."""
    _golden_sets.clear()
