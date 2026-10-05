"""산출물 관문 — 사용자가 명시한 산출물을 낼 수 있는 에이전트가 계획에 있는가 (2026-08-11).

**태그 부분문자열 매칭을 버린 이유 (실측)**:
첫 판은 "쿼리에 등장하는 고유 태그의 주인이 계획에 없으면 누락"으로 판정했다.
실제 레지스트리(106 에이전트, tags 보유 74)로 돌리니 정상 계획조차 통과 못 했다:

    "…도시별 여행 추천 한 줄을 붙여서 엑셀 표로…"
      → restaurant_finder_agent 누락 (근거 tag '여행')   ← 오탐
      → currency_exchange_agent 누락 (근거 tag '환율')   ← 애매

'여행 추천 한 줄'의 '여행'은 맛집 검색 요구가 아니다. **문자열은 문맥을 못 본다** —
이 프로젝트가 키워드 매칭을 금지하는 이유 그대로다(ontology/CLAUDE.md 핵심 원칙 1).
그래서 판정을 LLM 으로 옮기고, 질문을 **산출물 한정**으로 좁힌다.

패턴은 기존 `enforce_exclusion_gate`(2026-07-11)를 따른다 — 순수 함수 ·
`llm_invoke` 주입 · 좁은 단일 질문 · **fail-open**(관문 실패가 계획을 막지 않는다).

실행: cd ontology && ../.venv/bin/python -m pytest tests/test_artifact_gate.py -q
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from orchestrator.plan_validator import check_artifact_capability  # noqa: E402


AGENTS = {
    "xlsx_generator_agent": "표·수치를 실제 계산되는 Excel 워크북(.xlsx)으로 만듭니다.",
    "pptx_generator_agent": "발표용 PowerPoint 슬라이드를 생성합니다.",
    "analysis_agent": "숫자·텍스트 데이터를 분석하는 범용 분석 에이전트입니다.",
    "weather_agent": "도시별 날씨를 조회합니다.",
    "restaurant_finder_agent": "맛집을 검색해 리뷰·평점·주소를 추천합니다.",
}

QUERY = ("서울과 부산과 제주의 현재 날씨를 각각 조사하고, 원달러 환율도 확인한 다음, "
         "도시별 여행 추천 한 줄을 붙여서 엑셀 표로 만들어줘")


def _llm(payload):
    """지정한 JSON 을 돌려주는 가짜 LLM."""
    async def _fn(prompt: str) -> str:
        return json.dumps(payload, ensure_ascii=False)
    return _fn


def _boom(exc=RuntimeError("LLM down")):
    async def _fn(prompt: str) -> str:
        raise exc
    return _fn


# ── 핵심 판정 ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_flags_missing_artifact_agent():
    """엑셀을 명시했는데 만들 에이전트가 계획에 없으면 잡는다 (실측 오배정)."""
    errs = await check_artifact_capability(
        QUERY, planned={"weather_agent", "analysis_agent"}, agents_info=AGENTS,
        llm_invoke=_llm({"artifact": "엑셀 파일", "missing_agent": "xlsx_generator_agent"}))
    assert errs, "산출물 누락을 잡지 못했다"
    joined = " ".join(errs)
    assert "xlsx_generator_agent" in joined and "엑셀" in joined, errs


@pytest.mark.asyncio
async def test_passes_when_artifact_agent_planned():
    """전문가가 이미 계획에 있으면 통과 (LLM 이 missing 없음으로 답한다)."""
    errs = await check_artifact_capability(
        QUERY, planned={"weather_agent", "xlsx_generator_agent"}, agents_info=AGENTS,
        llm_invoke=_llm({"artifact": "엑셀 파일", "missing_agent": None}))
    assert errs == [], errs


@pytest.mark.asyncio
async def test_no_artifact_no_error():
    """산출물 요구가 없으면 아무것도 막지 않는다.

    ⚠️ 태그 방식이 여기서 '여행'·'환율' 로 오탐했다 — 그 회귀를 고정한다.
    """
    errs = await check_artifact_capability(
        "서울 날씨 알려주고 맛집도 추천해줘", planned={"weather_agent"}, agents_info=AGENTS,
        llm_invoke=_llm({"artifact": "none", "missing_agent": None}))
    assert errs == [], errs


# ── fail-open · 환각 방어 ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_failure_is_fail_open():
    """관문 실패가 계획을 막으면 LLM 장애가 서비스 장애가 된다."""
    assert await check_artifact_capability(
        QUERY, planned={"weather_agent"}, agents_info=AGENTS, llm_invoke=_boom()) == []


@pytest.mark.asyncio
async def test_unparseable_answer_is_fail_open():
    async def _junk(prompt: str) -> str:
        return "네, 엑셀이 필요합니다."
    assert await check_artifact_capability(
        QUERY, planned={"weather_agent"}, agents_info=AGENTS, llm_invoke=_junk) == []


@pytest.mark.asyncio
async def test_hallucinated_agent_id_ignored():
    """LLM 이 없는 에이전트를 지목하면 무시한다 — 존재하지 않는 것을 요구할 수 없다."""
    errs = await check_artifact_capability(
        QUERY, planned={"weather_agent"}, agents_info=AGENTS,
        llm_invoke=_llm({"artifact": "PDF", "missing_agent": "pdf_maker_agent"}))
    assert errs == [], errs


@pytest.mark.asyncio
async def test_already_planned_agent_not_flagged():
    """LLM 이 계획에 있는 에이전트를 누락이라 해도 무시 (자기모순 방어)."""
    errs = await check_artifact_capability(
        QUERY, planned={"xlsx_generator_agent"}, agents_info=AGENTS,
        llm_invoke=_llm({"artifact": "엑셀", "missing_agent": "xlsx_generator_agent"}))
    assert errs == [], errs


@pytest.mark.asyncio
async def test_no_llm_means_no_gate():
    """llm_invoke 가 없으면 관문은 침묵한다 (문자열 매칭으로 되돌아가지 않는다)."""
    assert await check_artifact_capability(
        QUERY, planned={"weather_agent"}, agents_info=AGENTS, llm_invoke=None) == []


# ── 프롬프트 계약 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prompt_carries_query_and_agents():
    """LLM 이 판단하려면 쿼리와 후보를 봐야 한다."""
    seen = {}

    async def _capture(prompt: str) -> str:
        seen["p"] = prompt
        return json.dumps({"artifact": "none", "missing_agent": None})

    await check_artifact_capability(QUERY, planned={"weather_agent"},
                                    agents_info=AGENTS, llm_invoke=_capture)
    p = seen["p"]
    assert QUERY[:20] in p, "쿼리가 프롬프트에 없다"
    assert "xlsx_generator_agent" in p, "후보 에이전트가 프롬프트에 없다"
    assert "weather_agent" in p, "계획된 에이전트가 프롬프트에 없다"


def test_no_substring_matching_in_source():
    """판정을 다시 문자열 매칭으로 되돌리지 못하게.

    `tag in query` 류가 돌아오면 '여행 추천'을 맛집 요구로 읽는 그 오탐이 재발한다.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(check_artifact_capability)))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body = fn.body[1:]
    code = ast.unparse(fn)
    assert "in query" not in code and "in q" not in code, \
        "쿼리 부분문자열 매칭이 돌아왔다 — 판정은 LLM 이 한다"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
