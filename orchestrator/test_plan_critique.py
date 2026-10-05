"""Plan Critique 관문 (Agentic Upgrade Phase 5).

멀티스테이지 계획 생성 직후 좁은 판정 1콜로 "불필요 스테이지 / 끊긴 입력 사슬"을
검증한다 (배제 관문 패턴의 일반화 — 계획 전체를 프롬프트 규칙으로 시키면 flash-lite
가 무시하지만 좁은 판정 질문은 안정적, 2026-07-11 실측).

실측 근거: 직접 데이터 차트 요청에 플래너가 불필요한 analysis 를 삽입해 그 단계가
데이터를 오염시킨 사례 (2026-07-14).

안전핀: ① 단일 에이전트 계획은 발동 안 함(콜 0) ② 마지막(최종 표현) 스테이지는
제거 금지 ③ 제거 후 계획이 비면 원본 유지 ④ LLM 에러/비JSON → fail-open.

직접 실행: .venv/bin/python ontology/orchestrator/test_plan_critique.py
"""

import asyncio
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ontology"))

from orchestrator.query_planner import critique_plan  # noqa: E402

DESCS = {
    "internet_agent": "웹 검색으로 최신 정보를 조회",
    "analysis_agent": "수치 데이터 통계·추세 분석",
    "data_visualization_agent": "데이터를 차트로 시각화",
}


def _plan(*stage_agents):
    return {"stages": [
        {"stage_id": i + 1, "agents": [{"agent_id": a, "sub_query": f"{a} 작업"} for a in agents]}
        for i, agents in enumerate(stage_agents)
    ]}


def _llm(payload):
    calls = []

    async def invoke(prompt):
        calls.append(prompt)
        if isinstance(payload, Exception):
            raise payload
        return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    invoke.calls = calls
    return invoke


def agents_of(plan):
    return [[a["agent_id"] for a in st["agents"]] for st in plan.get("stages", [])]


def main():
    fails = []

    def t(name, cond):
        print(("PASS  " if cond else "FAIL  ") + name)
        if not cond:
            fails.append(name)

    # C-1 불필요 중간 스테이지 지목 → 제거
    llm = _llm({"unnecessary_agents": ["analysis_agent"], "broken_chain": False})
    plan, verdict = asyncio.run(critique_plan(
        "다음 데이터를 차트로: 1월 5, 2월 7",
        _plan(["analysis_agent"], ["data_visualization_agent"]), DESCS, llm))
    t("C-1 불필요 중간 스테이지 제거",
      agents_of(plan) == [["data_visualization_agent"]] and verdict.get("unnecessary_agents"))

    # C-2 문제 없는 계획 → 무변경 (과교정 방지)
    llm2 = _llm({"unnecessary_agents": [], "broken_chain": False})
    plan2, _ = asyncio.run(critique_plan(
        "GDP 조사해서 차트로",
        _plan(["internet_agent"], ["analysis_agent"], ["data_visualization_agent"]), DESCS, llm2))
    t("C-2 정상 계획 무변경",
      agents_of(plan2) == [["internet_agent"], ["analysis_agent"], ["data_visualization_agent"]])

    # C-3 마지막(최종 표현) 스테이지 지목 → 무시 (결정적 가드)
    llm3 = _llm({"unnecessary_agents": ["data_visualization_agent"], "broken_chain": False})
    plan3, _ = asyncio.run(critique_plan(
        "차트로 보여줘", _plan(["internet_agent"], ["data_visualization_agent"]), DESCS, llm3))
    t("C-3 마지막 스테이지 제거 금지",
      agents_of(plan3) == [["internet_agent"], ["data_visualization_agent"]])

    # C-4 전부 지목 → 원계획 유지 (fail-open, 빈 계획 금지)
    llm4 = _llm({"unnecessary_agents": ["internet_agent", "analysis_agent",
                                        "data_visualization_agent"], "broken_chain": False})
    plan4, _ = asyncio.run(critique_plan(
        "차트", _plan(["internet_agent"], ["analysis_agent"], ["data_visualization_agent"]), DESCS, llm4))
    t("C-4 과잉 제거로 빈 계획이면 원본 유지", len(plan4.get("stages", [])) == 3)

    # C-5 LLM 예외 → 원계획 + 빈 verdict (fail-open)
    plan5, v5 = asyncio.run(critique_plan(
        "차트", _plan(["internet_agent"], ["data_visualization_agent"]), DESCS,
        _llm(RuntimeError("503"))))
    t("C-5 LLM 에러 fail-open", len(plan5["stages"]) == 2 and v5 == {})

    # C-6 단일 에이전트 계획 → 발동 안 함 (LLM 콜 0)
    llm6 = _llm({"unnecessary_agents": [], "broken_chain": False})
    plan6, v6 = asyncio.run(critique_plan("날씨", _plan(["internet_agent"]), DESCS, llm6))
    t("C-6 단일 계획 미발동 (콜 0)", len(llm6.calls) == 0 and v6 == {})

    # C-7 broken_chain → verdict 로 전달 (재계획 판단은 호출측)
    llm7 = _llm({"unnecessary_agents": [], "broken_chain": True, "reason": "2단계 입력 없음"})
    plan7, v7 = asyncio.run(critique_plan(
        "x", _plan(["internet_agent"], ["analysis_agent"]), DESCS, llm7))
    t("C-7 broken_chain verdict 전달", v7.get("broken_chain") is True
      and len(plan7["stages"]) == 2)

    # C-8 비 JSON 응답 → fail-open
    plan8, v8 = asyncio.run(critique_plan(
        "x", _plan(["internet_agent"], ["analysis_agent"]), DESCS, _llm("판단 불가입니다.")))
    t("C-8 비 JSON fail-open", len(plan8["stages"]) == 2 and v8 == {})

    print("\nRESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
