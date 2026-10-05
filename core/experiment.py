"""실험 하네스 커널 — 열거·지문·파레토·스토어 (Phase 1).

"어떤 방법 × 어떤 데이터 × 어떤 모델이 최고 효율인가"를 반복 측정 가능하게
하는 순수 부품들. LLM 0콜, 결정적. 설계: docs/experiment-harness-architecture.html.

설계 규정 (전부 이 저장소의 실측이 근거다):
- **레코드에 그래프 상태 지문 필수** — max_terms 스윕이 그래프 상태에 따라
  두 번 뒤집혔다. 지문이 다른 레코드끼리의 비교는 비교가 아니다.
- **골든셋 지문도 필수** — 라벨도 코퍼스 세대에 묶인다 (OCR 재인제스트 실측).
  accept 로 정답이 자라면 hash 가 달라져야 "같은 자로 쟀는가"가 남는다.
- **eval_history 와 분리 저장** — eval_history → latest_quality → /retrieve 의
  품질 블록. 실험 레코드가 섞이면 라이브 응답이 실험 설정의 품질을 운영
  품질로 보고한다. 저장 규약(append-only JSONL·깨진 줄 스킵·기록 실패 무해)은
  eval_history 와 같다.
- **파레토, 단일 스칼라 합성 금지** — 품질과 비용의 가중치를 지어내지 않는다.
  "벡터만"이 품질은 낮아도 비용이 훨씬 싸면 프런티어에 남는다.
"""

import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


# ─── 축 열거 ─────────────────────────────────────────────────────────

def enumerate_combos(axes: Dict[str, List[Any]],
                     max_combos: int = 64) -> List[Dict[str, Any]]:
    """축 격자를 조합 목록으로 편다 — 결정적 (키 정렬, 값은 입력 순서 보존).

    총 조합수가 상한을 넘으면 **소리내어 거부**한다 — 조용히 자르면 "다
    돌았다"로 읽힌다 (격자 폭발 방지의 유일한 관문). 빈 축(값 0개)도 거부:
    곱이 0이 되어 실험 전체가 조용히 사라진다.

    axes 가 빈 dict 면 기준선 1조합([{}]) — "아무 knob 도 안 바꾼 현재 설정"
    도 실험의 한 점이다.
    """
    if not axes:
        return [{}]
    for key, values in axes.items():
        if not isinstance(values, list):
            raise ValueError(f"축 '{key}' 의 값은 리스트여야 한다 "
                             f"(받음: {type(values).__name__})")
        if not values:
            raise ValueError(f"축 '{key}' 가 비어 있다 — 조합 0개는 "
                             "실험이 조용히 사라지는 것이다")
    keys = sorted(axes)
    total = 1
    for key in keys:
        total *= len(axes[key])
    if total > max_combos:
        raise ValueError(f"조합 {total}개가 상한 {max_combos}개를 넘는다 — "
                         "축을 줄이거나 max_combos 를 명시적으로 올려라")
    return [dict(zip(keys, values))
            for values in itertools.product(*(axes[k] for k in keys))]


# ─── 지문 2종 — 결과의 유효 조건을 박제한다 ──────────────────────────

def _short_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def graph_fingerprint(graph, chunks) -> Dict[str, Any]:
    """그래프 상태 지문 — 이 위에서 잰 숫자인가를 남긴다.

    수치는 `graph_health.evidence_gaps` 를 재사용한다 (linked/orphan 의 정의를
    두 벌 두면 지문과 /health 가 갈라진다). hash 는 노드 id 전량 + (s,p,o)
    트리플 전량 — 엣지 하나가 바뀌어도 달라져야 "같은 그래프"가 거짓말을
    못 한다.

    coverage 는 청크 0 이면 None — 0/0 을 숫자로 보고하면 빈 네임스페이스가
    '완벽'으로 보인다 (graph_health 와 같은 계약).
    """
    from .graph_health import evidence_gaps

    node_ids = sorted(str(n) for n in graph.nodes())
    triples = sorted(
        (str(s), str((attrs or {}).get("predicate", "")), str(t))
        for s, t, attrs in graph.edges(data=True))
    gaps = evidence_gaps(node_ids, chunks)
    total_chunks = gaps["chunks"]
    return {
        "nodes": gaps["nodes"],
        "edges": len(triples),
        "chunks": total_chunks,
        "linked_chunks": gaps["linked_chunks"],
        "coverage": (gaps["linked_chunks"] / total_chunks)
        if total_chunks else None,
        "orphan_nodes": len(gaps["orphan_nodes"]),
        "hash": _short_hash([node_ids, triples]),
    }


def _case_field(case, name: str, default=None):
    if isinstance(case, dict):
        return case.get(name, default)
    return getattr(case, name, default)


def golden_fingerprint(cases: Iterable[Any]) -> Dict[str, Any]:
    """골든셋 지문 — "같은 자로 쟀는가"를 레코드에 남긴다.

    hash 에 accepted 까지 넣는 이유: **라벨 노후화 법칙의 박제**다. accept 로
    정답 집합이 자라면(코퍼스가 자라면 정답도 자란다 — 08-02 실측) 케이스 수는
    그대로여도 자가 달라진 것이고, 그 전후의 지표는 비교 대상이 아니다.

    케이스는 GoldenCase 객체든 직렬화된 dict 든 받는다 — 지문은 스토어 밖
    (러너·스크립트)에서도 만들 수 있어야 한다.
    """
    statuses: Dict[str, int] = {}
    rows = []
    for case in cases or ():
        status = str(_case_field(case, "status", "") or "")
        statuses[status] = statuses.get(status, 0) + 1
        rows.append((
            str(_case_field(case, "case_id", "") or ""),
            status,
            str(_case_field(case, "expected_node_id", "") or ""),
            sorted(str(a) for a in (_case_field(case, "accepted", ()) or ())),
        ))
    rows.sort()
    return {"n": len(rows), "statuses": statuses, "hash": _short_hash(rows)}


def sample_warnings(cases: int, min_cases: int = 50) -> List[str]:
    """표본 경고 — 1건=0.02 인 자로 knob 을 확정하지 않는다 (과적합 경고).

    0건은 별도로 표시한다: "작은 표본"과 "잰 적 없음"은 다른 사실이다."""
    warnings: List[str] = []
    if cases == 0:
        warnings.append("no_cases")
    if cases < min_cases:
        warnings.append("small_sample")
    return warnings


# ─── 파레토 프런티어 ─────────────────────────────────────────────────

def _point(record: Dict[str, Any], quality: str, cost: str):
    """(품질, 비용) 좌표 — 못 재는 레코드는 None (프런티어 진입 불가).

    measured=False·지표 None·비수치는 전부 배제한다: 재지 않은 결과가
    지배점이 되면 추천이 허구 위에 선다."""
    if record.get("measured") is False:
        return None
    q = (record.get("metrics") or {}).get(quality)
    c = (record.get("cost") or {}).get(cost)
    if not isinstance(q, (int, float)) or isinstance(q, bool):
        return None
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        return None
    return (float(q), float(c))


def pareto_frontier(records: List[Dict[str, Any]], quality: str = "mrr",
                    cost: str = "latency_ms_p50") -> List[Dict[str, Any]]:
    """비지배 집합 — 품질(높을수록 좋음) × 비용(낮을수록 좋음).

    **단일 스칼라 합성 금지.** 가중치를 지어내면 저품질·저비용 점("벡터만")이
    부당하게 떨어진다 — 그게 정답인 네임스페이스도 있다. 지배의 정의:
    다른 레코드가 (품질 ≥ 이고 비용 ≤, 둘 중 하나는 strict) 이면 탈락.
    동률(같은 품질·같은 비용)은 서로 지배하지 않는다 — 둘 다 남는다.

    반환은 비용 오름차순 (화면의 산점도 x축과 같은 순서).
    """
    scored = [(record, point) for record in records
              if (point := _point(record, quality, cost)) is not None]
    frontier: List[Any] = []
    for record, (q, c) in scored:
        dominated = any(
            oq >= q and oc <= c and (oq > q or oc < c)
            for _, (oq, oc) in scored)
        if not dominated:
            frontier.append((record, q, c))
    frontier.sort(key=lambda item: (item[2], -item[1]))
    return [record for record, _, _ in frontier]


# ─── 실험 레코드 스토어 ──────────────────────────────────────────────

class ExperimentStore:
    """네임스페이스 하나의 실험 레코드 (추가전용 JSONL).

    eval_history 와 **분리**된 파일인 것이 계약이다 — eval_history 는
    latest_quality 를 거쳐 /retrieve 의 운영 품질 블록이 되므로, 실험
    레코드가 섞이면 라이브 응답이 실험 설정의 품질을 운영 품질로 보고한다.
    저장 규약(append-only·깨진 줄 스킵·기록 실패 무해)은 그대로 복제한다.
    """

    def __init__(self, namespace: str = "default", path=None):
        self.namespace = namespace
        self._path = Path(path) if path else None
        self._entries: List[Dict[str, Any]] = []

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        return _DEFAULT_DATA_DIR / f"experiments_{self.namespace}.jsonl"

    def record(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """실험 레코드 한 건. `at` 은 자동 — 사람이 기억해서 넣는 필드는
        지켜지지 않는다."""
        row = dict(entry or {})
        row.setdefault("at", datetime.now().isoformat(timespec="seconds"))
        self._entries.append(row)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            # 측정이 본체다 — 기록 실패로 실험을 죽이지 않는다.
            logger.warning(f"⚠️ 실험 레코드 기록 실패 ({self.path}): {e}")
        return row

    def entries(self, limit: int = 50, layer: Optional[str] = None
                ) -> List[Dict[str, Any]]:
        """최신 먼저. 추가전용이라 로그 순서가 곧 시간 순서다 (eval_history
        와 같은 이유 — 같은 초 안의 기록은 타임스탬프로 갈리지 않는다).
        limit=0 은 전량, layer 는 Tier 필터 (층이 다르면 파레토도 층
        안에서만 — 설계 문서 4장)."""
        rows = list(reversed(self._entries))
        if layer is not None:
            rows = [row for row in rows if row.get("layer") == layer]
        return rows[:limit] if limit else rows

    def load_from_disk(self, path=None) -> bool:
        load_path = Path(path) if path else self.path
        if not load_path.exists():
            return False
        try:
            self._entries.clear()
            skipped = 0
            with open(load_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._entries.append(json.loads(line))
                    except Exception:
                        skipped += 1
            if skipped:
                logger.warning(f"⚠️ 실험 레코드: 깨진 줄 {skipped}개 건너뜀")
            return True
        except Exception as e:
            logger.error(f"실험 레코드 로드 실패: {e}")
            return False


_stores: Dict[str, ExperimentStore] = {}


def get_experiment_store(namespace: str = "default") -> ExperimentStore:
    if namespace not in _stores:
        store = ExperimentStore(namespace=namespace)
        store.load_from_disk()
        _stores[namespace] = store
    return _stores[namespace]


def reset_experiment_stores() -> None:
    _stores.clear()
