"""Tier 2 라우팅 실험 — 라벨 구성 × 학습 방식의 파레토 (2026-08-03).

설계: docs/experiment-harness-architecture.html §4 (Tier 2) · §3 (레코드 계약).
Tier 0(검색층) 러너와 **같은 뼈대**다 — 축 열거 → 조합마다 측정 → 지문 붙인
레코드를 ExperimentStore 에 쌓고 파레토로 비교. 다른 것은 자와 비용뿐:

| | Tier 0 (retrieval) | Tier 2 (routing) |
|---|---|---|
| 자 | evaluate_cases (hit@k·MRR) | 홀드아웃 top-1 + SNIPS (training_loop) |
| 비용 | 질의 지연 p50/p95 | 학습 시간 (CPU 수 분) |
| 축 | 채널·knob | **라벨 소스 · 가중 BC vs 비가중 · off_policy 포함** |

층이 다르면 파레토도 층 안에서만 (설계 문서 4장) — `layer="routing"` 이 그
경계다.

**세 가지 계약**:

1. **실험은 정책을 바꾸지 않는다.** Tier 0 가 라이브 retrieval_config 를 안
   바꾸는 것과 같다. 여기서는 정책 가중치가 그 대상이라 함수 진입 시
   `state_dict_all()` 전체를 스냅샷하고, 조합마다·종료 시 복원한다
   (에이전트 인덱스 맵까지 — 실험 중 등록한 에이전트도 되돌린다).
   배포는 `training_loop.retrain` 의 게이트 경로가 한다.
2. **에이전트 등록은 루프 밖에서 한 번.** 조합마다 등록하면 라벨 구성에 따라
   인덱스가 달라져 조합 간 holdout_acc 가 같은 자로 잰 숫자가 아니게 된다.
3. **없는 수를 지어내지 않는다.** 라벨 부족 조합은 레코드 없이 `skipped`
   목록으로만 나가고, 홀드아웃 표본 0 이면 `measured=False`.

LLM 0콜. 임베딩 추론만 (버퍼 info.query 재임베딩 — P0-5 스키마 계약).
"""

from __future__ import annotations

import copy
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from ..core.experiment import (enumerate_combos, get_experiment_store,
                               pareto_frontier, sample_warnings)
from .training_loop import (_runtime_logged, _states_for, collect_labels,
                            holdout_accuracy, snips_estimate, split_holdout)

# 기본 격자 = 라벨 소스 3 × 가중 2 = 6 조합.
# "부트스트랩 정본만 vs 실피드백만 vs 전부"가 Tier 2 의 1급 축이다 — 정본
# 361건은 2026-02 세대라 노후화 위험이 있고(라벨 노후화 법칙), 실피드백은
# 현재 세대지만 소량이다. 어느 쪽이 홀드아웃에서 이기는지는 측정 문제다.
DEFAULT_AXES: Dict[str, List[Any]] = {
    "sources": [["kg_checkpoint"], ["runtime"], None],
    "weighted": [True, False],
}

_AXIS_KEYS = ("sources", "weighted", "include_off_policy")

# 홀드아웃 표본 경고 하한. 검색층의 50(골든셋 케이스)보다 낮은 이유: 라우팅
# 홀드아웃은 라벨의 20% 라 같은 하한을 쓰면 라벨 250건 미만이 전부 경고가 되어
# 경고가 정보를 잃는다. 20건 = 1건이 0.05 인 자 — 그래도 작다는 뜻이다.
MIN_HOLDOUT_CASES = 20


def _sources_label(sources: Optional[List[str]]) -> str:
    """소스 축 값 → 레코드용 스칼라 라벨.

    **enumerate_combos 는 리스트 값을 그대로 처리한다** (검증 완료: 축의 값
    목록이 리스트면 되고, 그 원소가 리스트/None 이어도 itertools.product 가
    문제없이 편다). 그래서 조합 생성은 원값(list|None)으로 돈다.

    다만 **레코드의 `axes` 에는 문자열 라벨로 넣는다** — axes 는 화면에서
    조합을 묶고 비교하는 키인데, 리스트는 해시 불가이고 순서에 따라 같은
    조합이 다른 키로 갈린다(["a","b"] vs ["b","a"]). None("전부")도 이름이
    있어야 표에 뜬다. 원값은 `golden.sources` 에 그대로 남으므로 재현에
    필요한 정보는 잃지 않는다.
    """
    if sources is None:
        return "all"
    items = sorted(str(s) for s in sources)
    return "+".join(items) if items else "none"


def _validate_axes(axes: Dict[str, List[Any]]) -> Optional[str]:
    """축 검증 — 소리내는 거부 (오타 축이 조용히 무시되면 "그 축을 돌았다"로
    오독한다. Tier 0 러너와 같은 계약)."""
    for key, values in axes.items():
        if key not in _AXIS_KEYS:
            return f"알 수 없는 축 {key!r} — 허용: {list(_AXIS_KEYS)}"
        if not isinstance(values, list) or not values:
            return f"axis {key!r}: 비어있지 않은 리스트여야 한다"
        if key == "sources":
            for v in values:
                if v is None:
                    continue
                if not isinstance(v, (list, tuple)) or not v:
                    return ("sources 축 값은 None(전부) 또는 비어있지 않은 "
                            f"소스 목록 — 받음: {v!r}")
                if not all(isinstance(s, str) and s for s in v):
                    return f"sources 축 값의 원소는 문자열 — 받음: {v!r}"
        else:
            if not all(isinstance(v, bool) for v in values):
                return f"{key} 축은 bool 값만 — 받음: {values!r}"
    return None


def _fit_combo(selector, train: List[Dict[str, Any]], mask: torch.Tensor,
               weighted: bool, epochs: int):
    """한 조합의 학습 — (fit, train_seconds, 학습 행 수).

    시간은 상태 구성(재임베딩) + imitate 를 함께 잰다: 라벨이 많을수록
    임베딩도 비싸고, 그게 이 축의 실제 비용이다. 홀드아웃 채점은 측정이지
    학습이 아니므로 뺀다.
    """
    rows = [l for l in train if l["agent"] in selector.rl_policy._agent_to_idx]
    if not rows:
        return {"epochs": 0, "final_loss": 0.0, "train_acc": 0.0}, 0.0, 0
    t0 = time.perf_counter()
    states = _states_for(selector, [l["query"] for l in rows])
    actions = torch.tensor([selector.rl_policy._agent_to_idx[l["agent"]]
                            for l in rows], dtype=torch.long)
    rewards = torch.tensor([l["reward"] for l in rows], dtype=torch.float32)
    masks = mask.unsqueeze(0).expand(len(rows), -1)
    fit = selector.rl_policy.imitate(
        states=states, actions=actions, masks=masks, rewards=rewards,
        weights=rewards if weighted else None, epochs=epochs)
    return fit, round(time.perf_counter() - t0, 3), len(rows)


def run_routing_experiments(selector, namespace: str,
                            axes: Optional[Dict[str, List[Any]]] = None,
                            holdout_ratio: float = 0.2,
                            epochs: int = 100,
                            min_labels: int = 30,
                            actor: str = "manual",
                            max_combos: int = 32) -> Dict[str, Any]:
    """Tier 2 러너 — 라벨 구성 조합 × 홀드아웃/SNIPS → 레코드 + 파레토.

    조합마다: 라벨 수집(축 적용) → 결정적 홀드아웃 분할 → 보상 가중/비가중
    행동복제 → 홀드아웃 top-1 + SNIPS → **스냅샷 복원** → 레코드 기록.

    홀드아웃 분할 비율은 조합 간 고정이고 질의 해시 기반이라, 같은 질의는
    어느 조합에서도 같은 쪽에 떨어진다 — 조합 비교가 성립하는 조건이다.

    SNIPS 는 라벨 소스와 무관하게 **runtime 로그**로 잰다 (OPE 의 정의:
    행동 정책이 남긴 로그로 신정책을 평가한다). 부트스트랩 라벨만으로
    학습한 정책도 실주행 로그에서 어떻게 보이는지가 배포 판단의 재료다.
    """
    axes = axes if axes is not None else {k: list(v)
                                          for k, v in DEFAULT_AXES.items()}
    detail = _validate_axes(axes)
    if detail:
        return {"error": "invalid", "detail": detail}
    try:
        combos = enumerate_combos(axes, max_combos=max_combos)
    except ValueError as e:
        return {"error": "invalid", "detail": str(e)}

    buffer = selector.experience_buffer
    policy = selector.rl_policy

    # 계약 1: 함수 진입 상태 전체를 박제 (가중치 + 에이전트 인덱스 맵).
    entry_snapshot = copy.deepcopy(policy.state_dict_all())

    try:
        # 계약 2: 등록은 루프 밖에서 한 번 — 전 소스 라벨의 합집합.
        universe = collect_labels(buffer)["labels"]
        all_agents = list(dict.fromkeys(
            list(policy._agent_to_idx.keys())
            + sorted({l["agent"] for l in universe})))
        if all_agents:
            policy.register_agents(all_agents)
        mask = policy.build_available_mask(all_agents)
        baseline = copy.deepcopy(policy.state_dict_all())

        run_id = f"rexp-{datetime.now():%Y%m%d%H%M%S}"
        store = get_experiment_store(namespace)
        cfg = selector.config
        base_config = {
            "embedder": getattr(cfg, "embedding_model", None),
            "state_dim": cfg.rl.state_dim,
            "max_agents": cfg.rl.max_agents,
            "buffer_size": buffer.size,
            "epochs": epochs,
        }

        records: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        for combo in combos:
            sources = combo.get("sources")
            weighted = bool(combo.get("weighted", True))
            include_off = bool(combo.get("include_off_policy", True))
            axes_label = {"sources": _sources_label(sources),
                          "weighted": weighted,
                          "include_off_policy": include_off}

            collected = collect_labels(buffer, sources=sources,
                                       include_off_policy=include_off)
            labels = collected["labels"]
            if len(labels) < min_labels:
                skipped.append({
                    "axes": axes_label, "labels": len(labels),
                    "sources": sorted(sources) if sources is not None else None,
                    "reason": f"labels {len(labels)} < min {min_labels}"})
                continue

            train, hold = split_holdout(labels, holdout_ratio)
            if not train or not hold:
                skipped.append({
                    "axes": axes_label, "labels": len(labels),
                    "sources": sorted(sources) if sources is not None else None,
                    "reason": "holdout empty — 자 없이 학습하지 않는다"})
                continue

            try:
                fit, seconds, n_train = _fit_combo(selector, train, mask,
                                                   weighted, epochs)
                acc = holdout_accuracy(selector, hold, mask)
                snips = snips_estimate(_runtime_logged(selector, buffer, mask))
            finally:
                # 계약 1: 조합의 학습은 다음 조합·라이브로 새지 않는다.
                policy.load_state_dict_all(baseline)

            entry = {
                "run_id": run_id, "namespace": namespace,
                "layer": "routing", "actor": actor,
                "axes": axes_label,
                "config": base_config,
                "golden": {
                    "labels": len(labels), "train": n_train,
                    "holdout": len(hold),
                    "sources": sorted(sources) if sources is not None else None,
                    "skipped_source": collected["skipped_source"],
                    "skipped_off_policy": collected["skipped_off_policy"],
                    "skipped_failure": collected["skipped_failure"],
                    "skipped_no_query": collected["skipped_no_query"],
                },
                "metrics": {
                    "holdout_acc": acc,
                    "snips": snips,
                    "train_acc": fit.get("train_acc"),
                    "measured": acc is not None,
                },
                "cost": {"train_seconds": seconds, "epochs": epochs,
                         "labels": n_train},
                "warnings": sample_warnings(len(hold),
                                            min_cases=MIN_HOLDOUT_CASES),
            }
            store.record(entry)
            records.append(entry)
    finally:
        # 진입 시점으로 완전 복원 — 실험이 남기는 것은 레코드뿐이다.
        policy.load_state_dict_all(entry_snapshot)

    if skipped:
        logger.info(f"run_routing_experiments: {len(skipped)}개 조합 skip "
                    f"(라벨 부족) — 없는 수를 지어내지 않는다")

    return {"namespace": namespace, "run_id": run_id,
            "combos": len(records), "records": records,
            "skipped": skipped,
            "pareto": pareto_frontier(records, quality="holdout_acc",
                                      cost="train_seconds")}
