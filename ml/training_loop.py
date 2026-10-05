"""RL 학습 루프 — "언제 무엇으로 배우고, 좋아졌는지 어떻게 알고 배포하나"의 답.

문제 정식화 (2026-08-03, 문헌 대조):
이 시스템의 결정은 **단발 contextual bandit** 이다 — 질의 하나에 에이전트
하나를 고르고 끝난다 (버퍼의 모든 경험이 done=True, next_state=zeros —
순차 신용 배분이 없는데 PPO+GAE+γ 를 얹은 것이 원래 설계의 형식 오류였다).
2025-26 라우팅 문헌도 같은 정식화다: Online Multi-LLM Selection via
Contextual Bandits (AAAI'26), PILOT (오프라인 선호 → 온라인 bandit 정련),
BaRP (bandit 피드백 학습, 오프라인 라우터 대비 +12%).

세 부품:

1. **학습 = 보상 가중 행동복제** (`retrain`) — bandit 피드백 학습의 가장
   강건한 형태. 성공한 (질의→에이전트) 를 보상 가중으로 흉내낸다. P0-5 실측
   (PPO top-1 3% vs imitation 100%)과 정합. 실패 경험은 CE 라벨로 쓸 수 없어
   제외하되 **세어 보고한다** (무엇을 하지 말라는 정보는 다음 단계 —
   dueling/negative 학습 — 의 재료로 남긴다).

2. **자 = 라우팅 홀드아웃** — 학습에 쓰지 않은 (질의→에이전트) 정답의
   top-1 재현율. 질의 해시 기반 결정적 분할 (같은 데이터면 같은 분할 —
   재현 가능한 측정, golden_sampling 과 같은 이유).

3. **배포 게이트 = SNIPS (self-normalized IPS)** — 실주행 로그(행동 정책의
   log_prob 이 경험에 남아 있다)만으로 신정책의 기대 보상을 배포 전에
   추정한다. "측정 없이 배포하지 않는다"의 RL 판. 홀드아웃 정확도와 SNIPS
   중 하나라도 퇴화하면 **롤백** — 이전 가중치로 복원하고 저장하지 않는다.

이 모듈은 selector 를 인자로 받는 함수들이다 (순수에 가깝게 — 상태 변경은
retrain 의 정책 갱신뿐). LLM 0콜. 임베딩 추론만 (질의 재임베딩 — 버퍼
info.query 를 보존한 P0-5 스키마 계약이 여기서 회수된다).
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Collection, Dict, List, Optional, Tuple

import torch
from loguru import logger


def split_holdout(labels: List[Dict[str, Any]],
                  holdout_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """질의 해시 기반 결정적 분할 → (train, holdout).

    무작위가 아닌 이유: 같은 데이터에서 두 번 재면 같은 홀드아웃이어야
    측정이 재현된다. 같은 질의 텍스트는 항상 같은 쪽에 떨어진다 —
    train/holdout 간 질의 누수가 없다.
    """
    train: List[Dict] = []
    hold: List[Dict] = []
    threshold = int(holdout_ratio * 0xFFFF)
    for lab in labels:
        h = int(hashlib.sha1(str(lab.get("query", "")).encode("utf-8"))
                .hexdigest()[:4], 16)
        (hold if h < threshold else train).append(lab)
    return train, hold


def collect_labels(buffer, min_reward: float = 0.0,
                   sources: Optional[Collection[str]] = None,
                   include_off_policy: bool = True) -> Dict[str, Any]:
    """버퍼 → 학습 라벨. 성공(보상>min_reward)한 (질의, 에이전트, 보상)만.

    실패·질의 없음은 버리되 **센다** — 조용한 절단 금지. (질의 없음 = P0-5
    이전 구세대 경험. 실패 = CE 라벨 불가 — "무엇이 옳았나"의 정보가 없다.)

    **라벨 구성 축** (Tier 2 실험, 2026-08-03) — 기본값은 현행 동작이다
    (인자 없이 부르면 전부, 회귀 0):

    - `sources`: `info["source"]` 화이트리스트. `{"kg_checkpoint"}` = 부트스트랩
      정본만, `{"runtime"}` = 실피드백만, None = 전부. 두 소스는 성질이 다르다
      — 정본은 2026-02 세대의 지도 라벨(노후화 위험), 실피드백은 소량이지만
      현재 세대다. 어느 구성이 홀드아웃에서 더 나은가는 **측정 문제**이고,
      그 측정을 가능하게 하는 것이 이 인자다.
    - `include_off_policy`: 실행 에이전트 ≠ 샘플 에이전트로 귀속된 경험
      (`info["off_policy"] == "executed_fallback"`)의 포함 여부. 이 라벨의
      행동 정책은 KG+LLM 이지 이 정책이 아니다 — 배울 가치가 있는지가 축이다.

    필터는 **well-formed 성공 라벨에만** 적용하고 그 수를 따로 센다
    (`skipped_source` / `skipped_off_policy`) — "이 축이 뺀 라벨 수"가 축의
    효과 크기다. 실패·질의없음을 여기 섞으면 축과 무관한 수가 들어온다.
    """
    allowed = {str(s) for s in sources} if sources is not None else None
    labels: List[Dict[str, Any]] = []
    skipped_failure = 0
    skipped_no_query = 0
    skipped_source = 0
    skipped_off_policy = 0
    seen: set = set()
    for exp in buffer.all_experiences():
        info = exp.info or {}
        query = info.get("query")
        if not query:
            skipped_no_query += 1
            continue
        if exp.reward <= min_reward:
            skipped_failure += 1
            continue
        agent = str(info.get("agent_id") or "")
        if not agent:
            skipped_no_query += 1
            continue
        source = str(info.get("source") or "")
        if allowed is not None and source not in allowed:
            skipped_source += 1
            continue
        if not include_off_policy and info.get("off_policy") == "executed_fallback":
            skipped_off_policy += 1
            continue
        key = (str(query), agent)
        if key in seen:
            continue
        seen.add(key)
        labels.append({"query": str(query), "agent": agent,
                       "reward": float(exp.reward),
                       "source": source})
    return {"labels": labels, "skipped_failure": skipped_failure,
            "skipped_no_query": skipped_no_query,
            "skipped_source": skipped_source,
            "skipped_off_policy": skipped_off_policy}


def snips_estimate(logged: List[Dict[str, Any]]) -> Optional[float]:
    """SNIPS (self-normalized inverse propensity scoring) 기대 보상 추정.

    logged 항목: {"reward", "logged_prob"(행동 정책의 그 액션 확률),
    "new_prob"(신정책의 그 액션 확률)}. 반환 None = 추정 불가 (표본 0 또는
    유효 가중치 0) — 0.0 으로 오보고하지 않는다 (측정 안 됨 ≠ 나쁨).

    IPS 대신 SNIPS 인 이유: 가중치 합으로 정규화해 분산이 훨씬 작고 보상
    평행이동에 불변 — 소표본 로그에서 IPS 는 한 건의 큰 가중치가 추정을
    지배한다 (Swaminathan & Joachims 의 self-normalized estimator).
    """
    num = 0.0
    den = 0.0
    for item in logged:
        p_log = float(item.get("logged_prob") or 0.0)
        if p_log <= 1e-8:
            continue  # propensity 0 — 보정 불가 항목은 제외 (폭주 방지)
        w = float(item.get("new_prob") or 0.0) / p_log
        num += w * float(item.get("reward") or 0.0)
        den += w
    if den <= 1e-8:
        return None
    return num / den


def _states_for(selector, queries: List[str]) -> torch.Tensor:
    """질의 → 상태 텐서 (그래프·이력 0 — bootstrap_from_mappings 와 같은
    결정: 기록 시점 컨텍스트는 없고, 지어낸 컨텍스트는 잡음이다)."""
    zeros_g = torch.zeros(selector.config.rl.graph_embedding_dim)
    zeros_h = torch.zeros(selector.config.rl.history_dim)
    states = [torch.cat([selector._embed_query(q).cpu().float(),
                         zeros_g, zeros_h], dim=0) for q in queries]
    return torch.stack(states)


def _policy_probs(selector, states: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = selector.rl_policy.feature_extractor(states)
        return selector.rl_policy.actor(
            features, mask.unsqueeze(0).expand(states.size(0), -1))


def holdout_accuracy(selector, holdout: List[Dict[str, Any]],
                     mask: torch.Tensor) -> Optional[float]:
    """홀드아웃 (질의→에이전트) top-1 재현율. 표본 0 → None (정직 보고)."""
    rows = [l for l in holdout
            if l["agent"] in selector.rl_policy._agent_to_idx]
    if not rows:
        return None
    states = _states_for(selector, [l["query"] for l in rows])
    probs = _policy_probs(selector, states, mask)
    pred = probs.argmax(dim=-1)
    want = torch.tensor([selector.rl_policy._agent_to_idx[l["agent"]]
                         for l in rows])
    return float((pred == want).float().mean())


def _runtime_logged(selector, buffer, mask: torch.Tensor,
                    limit: int = 500) -> List[Dict[str, Any]]:
    """SNIPS 용 실주행 로그 — runtime 경험(질의·log_prob 보유)만.

    bootstrap 라벨은 로그가 아니라 지도 라벨이라 OPE 대상이 아니다.
    """
    rows = []
    for exp in buffer.all_experiences():
        info = exp.info or {}
        if info.get("source") != "runtime" or not info.get("query"):
            continue
        rows.append(exp)
    rows = rows[-limit:]
    if not rows:
        return []
    states = _states_for(selector, [e.info["query"] for e in rows])
    probs = _policy_probs(selector, states, mask)
    out = []
    for i, exp in enumerate(rows):
        out.append({
            "reward": float(exp.reward),
            "logged_prob": math.exp(float(exp.info.get("log_prob", 0.0))),
            "new_prob": float(probs[i, exp.action]),
        })
    return out


def retrain(selector,
            holdout_ratio: float = 0.2,
            epochs: int = 100,
            min_labels: int = 30,
            regression_eps: float = 0.02,
            sources: Optional[Collection[str]] = None,
            include_off_policy: bool = True) -> Dict[str, Any]:
    """주기 재학습 — 수집 → 학습 → 이중 자 → 게이트 → 저장 or 롤백.

    게이트: 홀드아웃 정확도와 SNIPS(실주행 로그가 있을 때) 둘 다
    `이전 − eps` 이상이어야 채택. 하나라도 퇴화하면 **이전 가중치로 롤백**
    하고 저장하지 않는다. 자가 없으면(홀드아웃 0) 학습하지 않는다 —
    측정 없는 자동화 금지.

    `sources`/`include_off_policy` 는 collect_labels 로 그대로 전달된다 —
    Tier 2 실험(experiment_routing)이 고른 라벨 구성을 **운영 재학습에도
    같은 인자로** 걸 수 있어야 실험 결과가 배포로 이어진다. 기본값은
    현행 동작(전 소스·off_policy 포함).
    """
    collected = collect_labels(selector.experience_buffer, sources=sources,
                               include_off_policy=include_off_policy)
    labels = collected["labels"]
    report: Dict[str, Any] = {
        "labels": len(labels),
        "skipped_failure": collected["skipped_failure"],
        "skipped_no_query": collected["skipped_no_query"],
        "skipped_source": collected["skipped_source"],
        "skipped_off_policy": collected["skipped_off_policy"],
    }
    if len(labels) < min_labels:
        report["status"] = "skipped"
        report["reason"] = f"labels {len(labels)} < min {min_labels}"
        return report

    # 라벨의 에이전트가 전부 등록돼 있게 (기존 등록 순서 보존 — 인덱스 안정)
    all_agents = list(dict.fromkeys(
        list(selector.rl_policy._agent_to_idx.keys())
        + sorted({l["agent"] for l in labels})))
    selector.rl_policy.register_agents(all_agents)
    mask = selector.rl_policy.build_available_mask(all_agents)

    train, hold = split_holdout(labels, holdout_ratio)
    if not hold or not train:
        report["status"] = "skipped"
        report["reason"] = "holdout empty — 자 없이 학습하지 않는다"
        return report

    # 이전 자 (게이트 기준선)
    acc_before = holdout_accuracy(selector, hold, mask)
    logged = _runtime_logged(selector, selector.experience_buffer, mask)
    snips_before = snips_estimate(logged)
    snapshot = copy.deepcopy(selector.rl_policy.state_dict_all())

    # 학습 — 보상 가중 행동복제
    train_rows = [l for l in train
                  if l["agent"] in selector.rl_policy._agent_to_idx]
    states = _states_for(selector, [l["query"] for l in train_rows])
    actions = torch.tensor([selector.rl_policy._agent_to_idx[l["agent"]]
                            for l in train_rows], dtype=torch.long)
    rewards = torch.tensor([l["reward"] for l in train_rows],
                           dtype=torch.float32)
    masks = mask.unsqueeze(0).expand(len(train_rows), -1)
    fit = selector.rl_policy.imitate(
        states=states, actions=actions, masks=masks,
        rewards=rewards, weights=rewards, epochs=epochs)

    # 새 자 + 게이트
    acc_after = holdout_accuracy(selector, hold, mask)
    logged_after = _runtime_logged(selector, selector.experience_buffer, mask)
    snips_after = snips_estimate(logged_after)

    acc_ok = (acc_before is None or acc_after is None
              or acc_after >= acc_before - regression_eps)
    snips_ok = (snips_before is None or snips_after is None
                or snips_after >= snips_before - regression_eps)

    report.update({
        "train": len(train_rows), "holdout": len(hold),
        "fit": fit,
        "holdout_acc_before": acc_before, "holdout_acc_after": acc_after,
        "snips_logged_n": len(logged),
        "snips_before": snips_before, "snips_after": snips_after,
    })

    if acc_ok and snips_ok:
        selector.save_models()
        report["status"] = "accepted"
    else:
        selector.rl_policy.load_state_dict_all(snapshot)
        report["status"] = "rolled_back"
        report["reason"] = (f"gate 미달 — acc {acc_before}→{acc_after}, "
                            f"snips {snips_before}→{snips_after}")
        logger.warning(f"retrain rolled back: {report['reason']}")
    return report
