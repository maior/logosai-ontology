"""계획 회귀 판정 — 플래너를 바꾼 전후에 같은 쿼리가 같은 계획을 받는가.

플래너를 SDK 로 옮기는 작업(orchestrator-unify P2)의 판정 기준이다. 실행은
scripts/plan_regression.py 가 하고, 여기는 LLM 을 부르지 않는 순수 판정만 둔다.

판정은 '한 번의 계획'이 아니라 '반복 실행의 분포'로 한다. 플래너는 LLM 이라
같은 쿼리에도 흔들릴 수 있다 — 2026-10-04 실측으로 명확한 쿼리 4개는 5회 모두
같았지만, 기준선이 흔들리는 시나리오는 drift 로 단정하지 않고 따로 표시한다.

표준 라이브러리만 쓴다 (패키징 규칙: orchestrator 모듈은 모듈 레벨에서 logosai 를
끌어오지 않는다).
"""
import hashlib
import json
from collections import Counter
from typing import Any, Dict, Iterable, List

#: 안정 시나리오의 조건 — 기준선 반복 MIN_STABLE_RUNS 회 이상, 최빈 비율 STABLE_SHARE 이상.
#: 안정 시나리오만 엄격히 판정한다. 첫 판(N=3, 처음 보는 모양=실패)은 코드를 안 바꾼
#: 재실행에서 '회귀 5건'을 냈다 — 플래너에는 원래 가끔 나오는 다른 계획이 있다.
#: 값은 잡음 재실행(코드 무변경)으로 정했다 — N=5·80%·60% 는 매 실행 거짓 회귀 2건을 냈다.
MIN_STABLE_RUNS = 8
STABLE_SHARE = 7 / 8
#: 안정 시나리오에서 기준선 최빈의 현재 비율이 이 아래로 떨어지면 drift.
#: 진짜 회귀(프롬프트 변화)는 계통적이라 여러 시나리오에서 크게 드러난다 — 개별은 관대하게.
KEEP_SHARE = 0.5
#: 실패로 치는 판정
FAILING = ("drift", "missing")

#: 이 값이 다르면 계획이 달라도 플래너 탓이 아니다 — 비교를 거부한다
_STRICT_CONFIG_KEYS = ("registry", "selector")


class ConfigMismatch(ValueError):
    """기준선과 현재 실행의 조건(레지스트리·선택기)이 다르다."""


def plan_shape(plan_or_error: Any) -> str:
    """계획의 모양 — stage 순서·종류·에이전트(병렬 안은 정렬)·gap 여부.

    sub_query 문구는 매번 달라 넣지 않는다. 예외는 `error:<타입>`.
    """
    if isinstance(plan_or_error, BaseException):
        return f"error:{type(plan_or_error).__name__}"
    parts = []
    for stage in plan_or_error.stages or []:
        ids = [t.agent_id for t in (stage.agents or [])]
        if stage.execution_type == "parallel":
            ids = sorted(ids)            # 병렬 안의 순서는 의미가 없다
        parts.append(f"{'P' if stage.execution_type == 'parallel' else 'S'}[{','.join(ids)}]")
    shape = " → ".join(parts) or "∅"
    gap = getattr(plan_or_error, "capability_gap", None)
    if isinstance(gap, dict) and gap.get("detected"):
        shape += " | gap"
    return shape


def at_level(shape: str, level: str = "plan") -> str:
    """비교 수준에 맞게 모양을 줄인다.

    plan — 그대로.
    gap  — gap 선언 여부만 ("gap" / "no-gap"). FORGE 생성 요청에서 라우팅을 정하는 건
           gap 선언이고, 함께 붙는 단계 구성은 잡음이다 (잡음 재실행에서 E1~E3 가 매번 흔들렸다).
    예외 모양(error:*)은 어느 수준에서도 그대로 둔다.
    """
    if level != "gap" or shape.startswith("error:"):
        return shape
    return "gap" if shape.endswith("| gap") else "no-gap"


def summarize(shapes: Iterable[str]) -> Dict[str, Any]:
    """반복 실행 결과 → {n, counts, modal}. 동률이면 먼저 나온 모양이 최빈."""
    shapes = list(shapes)
    counts = Counter(shapes)
    modal = max(counts, key=lambda s: (counts[s], -shapes.index(s))) if shapes else None
    return {"n": len(shapes), "counts": dict(counts), "modal": modal}


def registry_fingerprint(agents: List[Dict[str, Any]]) -> str:
    """레지스트리 지문 — 에이전트 순서는 무시하고 내용(설명·능력·태그)은 반영한다."""
    canon = sorted(
        (json.dumps({k: a.get(k) for k in ("agent_id", "name", "description",
                                            "capabilities", "tags")},
                    ensure_ascii=False, sort_keys=True)
         for a in agents),
    )
    return hashlib.sha256("\n".join(canon).encode("utf-8")).hexdigest()[:16]


def compare(baseline: Dict[str, Any], current: Dict[str, Any]) -> List[Dict[str, Any]]:
    """시나리오별 판정 목록. 조건이 다르면 ConfigMismatch.

    안정 시나리오(기준선 N≥MIN_STABLE_RUNS, 최빈 ≥STABLE_SHARE):
      same      기준선 최빈의 현재 비율이 KEEP_SHARE 이상
      drift     기준선 최빈이 현재 절반 미만으로 사라졌다 (실패)
    그 밖의 시나리오 — 판정하지 않고 정보로 보고한다:
      variable  기준선이 흔들려 비교 근거가 약하다
      shifted   기준선에 한 번도 없던 계획이 현재 최빈이 됐다 (눈여겨볼 것)
    missing     현재 실행에 그 시나리오가 없다 (실패)

    드문 계획이 한 번 섞이는 것은 실패가 아니다 — `unseen` 으로 정보만 남긴다.
    """
    b_cfg, c_cfg = baseline.get("config", {}), current.get("config", {})
    for key in _STRICT_CONFIG_KEYS:
        if b_cfg.get(key) != c_cfg.get(key):
            raise ConfigMismatch(
                f"{key} 가 다르다: 기준선 {b_cfg.get(key)!r} ↔ 현재 {c_cfg.get(key)!r}")

    findings = []
    for sid, base in baseline.get("scenarios", {}).items():
        cur = current.get("scenarios", {}).get(sid)
        if cur is None:
            findings.append({"id": sid, "status": "missing", "baseline": base["modal"]})
            continue
        n_base = max(base["n"], 1)
        stable = base["n"] >= MIN_STABLE_RUNS and \
            base["counts"].get(base["modal"], 0) / n_base >= STABLE_SHARE
        kept = cur["counts"].get(base["modal"], 0) / max(cur["n"], 1)
        unseen = sorted(set(cur["counts"]) - set(base["counts"]))
        if stable:
            status = "same" if kept >= KEEP_SHARE else "drift"
        elif cur["modal"] not in base["counts"]:
            status = "shifted"
        else:
            status = "variable"
        findings.append({"id": sid, "status": status, "stable": stable,
                         "baseline": base["modal"], "current": cur["modal"],
                         "kept_share": round(kept, 2), "unseen": unseen,
                         "baseline_counts": base["counts"], "current_counts": cur["counts"]})
    return findings
