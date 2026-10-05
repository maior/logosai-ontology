"""ExecutionEngine — 병렬 동일 에이전트 결과가 다음 stage 로 전부 전달되는가.

2026-08-10 C1 실측: "서울·부산·제주 날씨 + 원달러 환율 → 엑셀 표" 워크플로우의
stage1 이 `[weather_agent×3 + currency_exchange_agent]` 병렬이었는데,
`_agent_results` 의 키가 `f"stage_{stage_id}.{agent_id}"` 라서 **같은 stage 의
같은 agent_id 3건이 한 칸을 두고 덮어썼다**. 프로브 실측:

    stage results: 4
    _agent_results keys: ['stage_1.currency_exchange_agent', 'stage_1.weather_agent']
    previous_results['weather_agent'] = {'location': '제주', ...}   # 서울·부산 소실

`_call_agent` 가 이 dict 에서 `previous_results` 를 만들므로(구조화 채널),
하류 에이전트는 3개 도시 중 1개의 구조화 결과만 본다. 나머지는 2000자 절단
산문(`input_data`)으로만 흘러 재파싱 대상이 된다.

직접 실행: .venv/bin/python -m pytest ontology/tests/test_execution_engine_structured_handoff.py
"""

import asyncio

from ontology.orchestrator.execution_engine import ExecutionEngine
from ontology.orchestrator.models import AgentTask, ExecutionStage


def _weather(city):
    return {"answer": f"# {city} 현재 날씨 🌈\n\n온도 정보", "location": city}


CURRENCY = {
    "result": "# 💱 환율\n\n1 USD = 1,418.00 KRW",
    "raw_data": [{"from_currency": "USD", "to_currency": "KRW", "rate": 1418.0}],
}


def _run_c1_workflow():
    """C1 워크플로우 재현 — stage1 병렬 4건 → stage2 소비자 1건."""
    seen = {}

    async def executor(agent_id, query, context):
        if agent_id == "xlsx_generator_agent":
            seen["context"] = dict(context or {})
            seen["query"] = query
            return {"answer": "워크북 생성 완료"}
        if agent_id == "currency_exchange_agent":
            return CURRENCY
        return _weather(query)

    engine = ExecutionEngine(agent_executor=executor)
    engine._current_user_query = "서울·부산·제주 날씨와 원달러 환율을 엑셀 표로"

    stage1 = ExecutionStage(stage_id=1, execution_type="parallel", agents=[
        AgentTask(agent_id="weather_agent", sub_query="서울"),
        AgentTask(agent_id="weather_agent", sub_query="부산"),
        AgentTask(agent_id="weather_agent", sub_query="제주"),
        AgentTask(agent_id="currency_exchange_agent", sub_query="원달러 환율"),
    ])
    r1 = asyncio.run(engine._execute_stage(stage=stage1, previous_output=None, context={}))

    stage2 = ExecutionStage(stage_id=2, execution_type="sequential", agents=[
        AgentTask(agent_id="xlsx_generator_agent", sub_query="엑셀 표로 만들어줘"),
    ])
    asyncio.run(engine._execute_stage(
        stage=stage2, previous_output=r1.aggregated_output, context={}))
    return engine, r1, seen


# ── ① 병렬 동일 에이전트 결과 전량 전달 ────────────────────────────────

def test_parallel_duplicate_agents_all_reach_next_stage():
    """weather_agent 3건이 previous_results 에 3건으로 남는다 (C1 결함의 뿌리)."""
    _, _, seen = _run_c1_workflow()
    prev = seen["context"].get("previous_results") or {}
    cities = {v.get("location") for v in prev.values() if isinstance(v, dict)}
    assert {"서울", "부산", "제주"} <= cities, f"소실된 도시가 있다: {cities}"


def test_all_stage_results_are_kept():
    """stage 결과 4건이 4칸을 차지한다 (키 충돌로 2건이 되지 않는다)."""
    engine, r1, _ = _run_c1_workflow()
    assert len(r1.results) == 4
    stage1_keys = [k for k in engine._agent_results if k.startswith("stage_1.")]
    assert len(stage1_keys) == 4, stage1_keys


def test_duplicate_labels_are_deterministic_and_ordered():
    """중복 라벨은 stage 정의 순서를 따른다 — 병렬 완료 순서에 흔들리면 안 된다."""
    engine, _, _ = _run_c1_workflow()
    assert engine._agent_results["stage_1.weather_agent"].data["location"] == "서울"
    assert engine._agent_results["stage_1.weather_agent#2"].data["location"] == "부산"
    assert engine._agent_results["stage_1.weather_agent#3"].data["location"] == "제주"


# ── ② 구조화 값이 산문 재파싱 없이 닿는다 ──────────────────────────────

def test_structured_value_reaches_consumer_without_reparsing_prose():
    """환율 1418 이 구조화된 채로 하류 context 에 도착한다 (F열 ₩0 의 반례)."""
    _, _, seen = _run_c1_workflow()
    prev = seen["context"]["previous_results"]
    assert prev["currency_exchange_agent"]["raw_data"][0]["rate"] == 1418.0


def test_consumer_side_contract_is_reachable_via_handoff():
    """소비자(logosai HandoffEnvelope)가 그 값을 실제로 꺼낼 수 있다.

    engine 이 구조화 데이터를 실었는데 SDK 접근자가 없으면 계약은 반쪽이다 —
    C1 이 정확히 그 상태였다 (실려는 있었고, 꺼낼 방법이 없었다).
    """
    from logosai.handoff import HandoffEnvelope
    _, _, seen = _run_c1_workflow()
    env = HandoffEnvelope.from_context(seen["query"], seen["context"])
    assert env.find_value("rate") == 1418.0
    locations = {i["data"].get("location") for i in env.structured()
                 if isinstance(i["data"], dict)}
    assert {"서울", "부산", "제주"} <= locations


# ── ③ 기존 문자열 경로 하위 호환 ───────────────────────────────────────

def test_string_enrichment_path_unchanged():
    """산문 핸드오프(enriched_query)는 종전 형식 그대로 유지된다."""
    _, _, seen = _run_c1_workflow()
    assert seen["query"].startswith("[이전 단계 결과]")
    assert "[요청]\n엑셀 표로 만들어줘" in seen["query"]


def test_original_query_and_caller_context_preserved():
    _, _, seen = _run_c1_workflow()
    assert seen["context"]["original_query"] == "서울·부산·제주 날씨와 원달러 환율을 엑셀 표로"
    assert "input_data" in seen["context"]


def test_get_agent_result_still_resolves_by_agent_id():
    """기존 조회 API 는 agent_id 로 계속 결과를 돌려준다 (하위 호환)."""
    engine, _, _ = _run_c1_workflow()
    r = engine.get_agent_result(1, "weather_agent")
    assert r is not None and r.success
    assert engine.get_agent_result(1, "currency_exchange_agent") is not None


def test_single_agent_per_stage_keys_unchanged():
    """중복이 없으면 키에 접미사가 붙지 않는다 (기존 소비자 무영향)."""
    engine, _, _ = _run_c1_workflow()
    assert "stage_1.currency_exchange_agent" in engine._agent_results
    assert "stage_2.xlsx_generator_agent" in engine._agent_results
    assert not any(k.endswith("#2") for k in engine._agent_results
                   if "currency" in k or "xlsx" in k)
