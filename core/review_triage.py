"""검수 트리아지 결합기 — 렌즈 신호를 밴드로 묶는 순수 함수 (C1, 2026-08-03).

설계 원문: docs/review-collaboration-architecture.html §5. 검수 파이프라인의
렌즈들(근거대조 LLM · 중복 · 일관성 · 구조단위 · 관계인용)은 전부 기존
부품이고, **신설은 이 결합기 하나뿐이다**. LLM 을 호출하지 않는다 — 판정
신호는 호출자(server/service)가 만들어 넘기고, 여기서는 결정적 규칙으로
밴드만 정한다. 같은 입력이면 같은 밴드다 — 트리아지가 실행마다 흔들리면
검수를 재현할 수 없다.

밴드 규칙 (설계 §5 그대로):
- strong_confirm : LLM confirm ∧ 근거 ≥1 ∧ 중복 비소속 ∧ 일관성 0 ∧
                   구조단위 아님. 하나라도 걸리면 강등 — 특히 dup_member ·
                   structural_candidate 는 confirm 추천이 있어도 강등한다
                   (중복은 병합 판단이, 구조단위는 재분류 판단이 먼저다).
- strong_reject  : LLM reject ∧ 사유 있음. **결정적 신호만으로는 만들지
                   않는다** — evidence_count==0 은 relink 실패일 수 있다
                   (2026-07-31 실측: 고아 27 중 25 가 빌더의 링크 유실).
- borderline     : 그 외 전부 → 사람. unsure / no_evidence / 판정 없음 포함.

밴드는 **판정이 아니라 선반이다** — 최종 confirm/reject 는 인간(judge_batch
경유)만 만든다. 이 모듈은 review_store 조차 건드리지 않는다 (기록은 호출자).

커널 경계: stdlib + loguru 만. 절대 raise 하지 않는다 — 트리아지가 죽어서
검수 자체가 막히면 안 된다 (evidence_checker 의 같은 규율).
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

BAND_STRONG_CONFIRM = "strong_confirm"
BAND_STRONG_REJECT = "strong_reject"
BAND_BORDERLINE = "borderline"

BANDS = (BAND_STRONG_CONFIRM, BAND_STRONG_REJECT, BAND_BORDERLINE)


def _squash(text: Any) -> str:
    """공백 정규화 — evidence_checker._squash_ws 와 같은 규칙.

    import 하지 않고 재정의한 이유: 이 모듈의 계약이 "stdlib+loguru 만"이고,
    세 줄짜리 규칙 하나 때문에 의존 방향을 늘리지 않는다. 규칙이 갈라지면
    관계 트리플 매칭이 묘비(_relation_key)와 어긋나므로, 규칙 변경 시 양쪽을
    같이 바꿔야 한다 (둘 다 \\s+ → ' ' + strip).
    """
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def triage_node(item: Optional[Dict[str, Any]],
                verdict: Optional[Dict[str, Any]],
                signals: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """검수 큐 항목 하나 → 밴드 + 사유 (LLM 0콜, 결정적, never raise).

    Args:
        item: get_review_queue 항목 {node_id, evidence_count, ...}
        verdict: EvidenceChecker 결과 {verdict, rationale, evidence_quote}
                 또는 None (LLM 콜 생략 — 그 사유는 호출자가 reasons 에 덧붙인다)
        signals: {"dup_member": bool, "consistency_findings": int,
                  "structural_candidate": bool} — 전부 결정적 렌즈의 산물

    Returns:
        {"band": strong_confirm|strong_reject|borderline, "reasons": [str, ...]}
    """
    try:
        item = item if isinstance(item, dict) else {}
        signals = signals if isinstance(signals, dict) else {}
        verdict = verdict if isinstance(verdict, dict) else None

        dup_member = bool(signals.get("dup_member"))
        structural = bool(signals.get("structural_candidate"))
        findings = _as_int(signals.get("consistency_findings"))
        evidence = _as_int(item.get("evidence_count"))

        v = _squash((verdict or {}).get("verdict")).lower()
        rationale = _squash((verdict or {}).get("rationale"))

        # strong_reject — LLM reject + 사유. 결정적 신호는 여기 관여하지 않는다:
        # dup/structural 은 confirm 을 강등하는 신호다("이 개체가 맞긴 한데
        # 정리가 먼저") — "이 개체가 틀렸다"는 reject 판단을 뒤집을 근거가
        # 아니다. 이유 없는 reject 는 borderline — 인간이 판정할 근거가 없다.
        if v == "reject":
            if rationale:
                return {"band": BAND_STRONG_REJECT,
                        "reasons": [f"LLM reject + 사유: {rationale}"]}
            return {"band": BAND_BORDERLINE,
                    "reasons": ["LLM reject 인데 사유 없음 — 판정 근거 부족"]}

        if v == "confirm":
            demote: List[str] = []
            if evidence < 1:
                # confirm 인데 근거 0 — LLM 이 원문 없이 확정했거나 링크가
                # 유실됐다. 어느 쪽이든 사람이 봐야 한다.
                demote.append("근거 청크 0 — relink 실패 가능, 확정 불가")
            if dup_member:
                demote.append("중복 클러스터 소속: 병합 판단이 먼저")
            if findings > 0:
                demote.append(f"일관성 발견 {findings}건: 모순 정리가 먼저")
            if structural:
                demote.append("구조 단위 후보: 재분류 판단이 먼저")
            if not demote:
                return {"band": BAND_STRONG_CONFIRM,
                        "reasons": ["LLM confirm + 근거 실재 + 결정적 신호 청정"]}
            return {"band": BAND_BORDERLINE, "reasons": demote}

        # unsure / no_evidence / 미지 값 / 판정 없음 → 전부 borderline
        reasons: List[str] = []
        if verdict is None:
            reasons.append("LLM 사전판정 없음")
        elif v == "no_evidence":
            reasons.append("근거 청크 없음(no_evidence) — 대조 불가")
        elif v == "unsure":
            reasons.append(f"LLM unsure: {rationale}" if rationale
                           else "LLM unsure")
        else:
            reasons.append(f"미지의 verdict '{v}'")
        if dup_member:
            reasons.append("중복 클러스터 소속: 병합 판단이 먼저")
        if structural:
            reasons.append("구조 단위 후보: 재분류 판단이 먼저")
        if findings > 0:
            reasons.append(f"일관성 발견 {findings}건")
        return {"band": BAND_BORDERLINE, "reasons": reasons}
    except Exception as e:  # pragma: no cover — 계약: 절대 안 던진다
        logger.warning(f"⚠️ triage_node failed ({e}) — borderline 로 degrade")
        return {"band": BAND_BORDERLINE, "reasons": [f"triage_error: {e}"]}


def triage_relation(proposal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """관계 제안 하나 → 밴드 + 사유 (LLM 0콜, 결정적, never raise).

    **관계에는 strong_reject 가 없다 — 2분류(confirm/borderline)로 시작한다.**
    근거는 실측이다: 타입 시그니처 미관측 30/36건(83%) 중 다수가 정상 관계였다
    (relation_backfill.type_signatures docstring — 조항·문서가 InsuranceTerm
    으로 잡혀 있어 옳은 관계가 새 패턴으로 보인다). 미관측을 기각 신호로 쓰면
    정상 관계까지 기계가 버린다 — 그래서 약한 쪽은 전부 사람에게 간다.

    strong_confirm 조건 (전부 결정적 신호):
    - signature_seen        : 기존 그래프에 같은 (타입,술어,타입) 패턴 실재
    - names_in_quote == 2   : 인용 안에 양끝 이름이 둘 다 있다 (강한 근거 —
                              parse_relations 가 판정엔 안 쓰고 신호로만 준다)
    - previously_rejected 가 None : 과거 기각 이력 없음. 기각 이력이 있으면
      어떤 인용이 와도 기계가 다시 올리면 안 된다 — 번복은 사람의 명시적
      override 로만 (approve_relations 의 override_rejected 와 같은 규율).
    """
    try:
        p = proposal if isinstance(proposal, dict) else {}
        signature_seen = bool(p.get("signature_seen"))
        names = _as_int(p.get("names_in_quote"))
        prev = p.get("previously_rejected")

        weak: List[str] = []
        if not signature_seen:
            weak.append("타입 시그니처 미관측 — 새 패턴인지 오용인지 사람이 가린다")
        if names != 2:
            weak.append(f"인용 속 양끝 이름 {names}/2 — 대명사 지시 가능, 약한 근거")
        if prev is not None:
            reason = ""
            if isinstance(prev, dict):
                reason = _squash(prev.get("reason"))
            weak.append(f"과거 기각 이력 있음: {reason or '(사유 미기재)'} — "
                        "번복은 사람의 override 로만")
        if not weak:
            return {"band": BAND_STRONG_CONFIRM,
                    "reasons": ["시그니처 관측 + 인용에 양끝 이름 + 기각 이력 없음"]}
        return {"band": BAND_BORDERLINE, "reasons": weak}
    except Exception as e:  # pragma: no cover
        logger.warning(f"⚠️ triage_relation failed ({e}) — borderline 로 degrade")
        return {"band": BAND_BORDERLINE, "reasons": [f"triage_error: {e}"]}


def _normalize_verdict(value: Any) -> str:
    """추천 verdict 와 판정 action 을 같은 축(confirm/reject)으로 —
    관계 승인(approve/relation_approve)은 confirm 과 같은 판정이다."""
    v = _squash(value).lower()
    if v in ("approve", "relation_approve"):
        return "confirm"
    if v == "relation_reject":
        return "reject"
    return v


def recommendation_agreement(events: Optional[List[Dict[str, Any]]]
                             ) -> Dict[str, Any]:
    """일치율 자 — recommend ↔ 이후 사람 판정의 대조 (로그 replay 파생, 공짜).

    입력 순서를 신뢰하지 않는다: review_store.history() 는 최신-먼저를,
    _events 는 로그 순서(오래된 것 먼저)를 준다. **정렬 키는 `at`(ISO 문자열,
    사전순 == 시간순) 오름차순**이고 Python 의 stable sort 라 같은 `at` 은
    입력 순서를 유지한다 — 같은 초 안의 이벤트 순서까지 보존하려면 호출자가
    로그 순서(_events)를 넘기는 것이 정확하다 (service 가 그렇게 한다).

    대조 규칙:
    - recommend(노드: after 에 predicate 없음) → 그 노드의 **다음**
      confirm/reject 와 대조.
    - recommend(관계: after 에 predicate/target 있음) → 같은 트리플의 다음
      relation_approve/relation_reject 와 대조 (트리플 비교는 공백 정규화 —
      _relation_key 와 같은 규칙).
    - 판정이 아직 없는 recommend 는 n 에서 제외 — **pending ≠ disagree**.
    - verdict 가 빈 recommend(LLM 생략 기록 등)도 제외 — 예측이 없는 것의
      일치율은 셀 수 없다 (0건을 0.0 으로 오보고하지 않는 규율의 동형).
    - unsure 는 n 에 포함되고 절대 agree 가 못 된다 — 보수적 방향. 렌즈별
      실질 적중률은 by_verdict 의 confirm/reject 행으로 읽는다.

    Returns:
        {"per_actor": {actor: {"n","agree","rate","by_verdict":
                               {verdict: {"n","agree","rate"}}}},
         "overall": {"n","agree","rate"}}   # n==0 이면 rate 는 None
    """
    try:
        rows = [e for e in (events or []) if isinstance(e, dict)]
        rows.sort(key=lambda e: str(e.get("at") or ""))

        # 판정 대기 중인 추천들. 노드는 node_id 로, 관계는 정규화 트리플로 묶는다.
        pending_nodes: Dict[str, List[Dict[str, Any]]] = {}
        pending_rels: Dict[tuple, List[Dict[str, Any]]] = {}
        scored: List[Dict[str, Any]] = []   # {actor, verdict, agree}

        def _score(rec: Dict[str, Any], judgment: str) -> None:
            scored.append({
                "actor": str(rec.get("actor") or ""),
                "verdict": rec["verdict"],
                "agree": rec["verdict"] == judgment,
            })

        for event in rows:
            action = str(event.get("action") or "")
            node_id = str(event.get("node_id") or "")
            after = event.get("after") if isinstance(event.get("after"), dict) \
                else {}

            if action == "recommend":
                verdict = _normalize_verdict(after.get("verdict"))
                if not verdict:
                    continue    # 예측 없는 추천 — 자로 잴 것이 없다
                rec = {"actor": event.get("actor"), "verdict": verdict}
                predicate = _squash(after.get("predicate"))
                if predicate:
                    key = (_squash(node_id), predicate,
                           _squash(after.get("target")))
                    pending_rels.setdefault(key, []).append(rec)
                else:
                    pending_nodes.setdefault(node_id, []).append(rec)

            elif action in ("confirm", "reject"):
                judgment = _normalize_verdict(action)
                for rec in pending_nodes.pop(node_id, []):
                    _score(rec, judgment)

            elif action in ("relation_approve", "relation_reject"):
                judgment = _normalize_verdict(action)
                key = (_squash(node_id), _squash(after.get("predicate")),
                       _squash(after.get("target")))
                for rec in pending_rels.pop(key, []):
                    _score(rec, judgment)

        def _rate(n: int, agree: int) -> Optional[float]:
            return (agree / n) if n else None   # 0건은 None — 0.0 오보고 금지

        per_actor: Dict[str, Dict[str, Any]] = {}
        overall_n = 0
        overall_agree = 0
        for s in scored:
            overall_n += 1
            overall_agree += 1 if s["agree"] else 0
            actor = per_actor.setdefault(
                s["actor"], {"n": 0, "agree": 0, "by_verdict": {}})
            actor["n"] += 1
            actor["agree"] += 1 if s["agree"] else 0
            bv = actor["by_verdict"].setdefault(
                s["verdict"], {"n": 0, "agree": 0})
            bv["n"] += 1
            bv["agree"] += 1 if s["agree"] else 0

        for actor in per_actor.values():
            actor["rate"] = _rate(actor["n"], actor["agree"])
            for bv in actor["by_verdict"].values():
                bv["rate"] = _rate(bv["n"], bv["agree"])

        return {"per_actor": per_actor,
                "overall": {"n": overall_n, "agree": overall_agree,
                            "rate": _rate(overall_n, overall_agree)}}
    except Exception as e:  # pragma: no cover
        logger.warning(f"⚠️ recommendation_agreement failed ({e})")
        return {"per_actor": {},
                "overall": {"n": 0, "agree": 0, "rate": None}}
