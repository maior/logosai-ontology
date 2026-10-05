"""Gap 미발화 수정 — 코드 생성 에이전트의 기능 쿼리 흡수 방지 (2026-07-10).

배경: "ISBN-13 번호가 유효한지 체크섬으로 검증해줘" 같은 **결과·판정을 원하는**
질문을 플래너가 code_generation_agent 로 배정 → 답이 코드 덤프가 되고
capability_gap 이 선언되지 않아 FORGE 협상 경로가 발화하지 않았다 (라이브 실측).
프롬프트의 약한 매칭 금지(시그널 C)가 internet/analysis/llm_search 만 지목해
"산출물이 코드인 에이전트"가 검사를 빠져나가는 구멍.

계약:
  - _build_planning_prompt 는 순수 함수 → __new__ + stub registry 로 결정론 테스트
  - 수정 후: 시그널 C-2 (결과를 원하는 질문에 코드 생성 매칭 금지) 텍스트 존재
  - 특정 agent_id 를 규칙에 하드코딩하지 않음 (역할 서술만 — CLAUDE.md 원칙)
  - 기존 시그널 A/B 와 명시적 코드 요청의 정상 경로는 보존 (과교정 방지)

실행: .venv/bin/python ontology/orchestrator/test_query_planner_code_absorption.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ONTOLOGY = os.path.join(_ROOT, "ontology")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ONTOLOGY)


class _StubRegistry:
    def build_prompt_context(self, include_schema: bool = False) -> str:
        return "## code_generation_agent\n- 설명: 코드 생성\n"


def _planner_without_init():
    from ontology.orchestrator.query_planner import QueryPlanner

    planner = QueryPlanner.__new__(QueryPlanner)
    planner.registry = _StubRegistry()
    return planner


def test_prompt_contains_code_absorption_guardrail():
    """RED(수정 전): 시그널 C-2 부재. GREEN(수정 후): '결과를 원하는 질문에
    코드 생성 매칭 금지' 규칙이 프롬프트에 존재."""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt("ISBN 번호가 유효한지 검증해줘")

    assert "코드 산출물" in prompt, "시그널 C-2 헤드라인(코드 산출물 명시 요청 조건) 부재"
    assert "코드 덤프" in prompt, "'답이 코드 덤프가 된다'는 근거 서술 부재"
    assert "유효한지 검증해줘" in prompt, "결과·판정형 질문 안티패턴 예시 부재"


def test_rule_does_not_hardcode_agent_id():
    """규칙 신설분에 특정 agent_id 하드코딩 금지 — 역할 서술로만 표현.
    (시그널 C-2 블록 안에 code_generation_agent 같은 id 가 없어야 함)"""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt("아무 쿼리")
    # C-2 블록만 도려내 검사 (기존 시그널 C 는 generic 3종을 이미 지목 — 대상 아님)
    start = prompt.find("코드 산출물")
    assert start != -1, "시그널 C-2 자체가 없음 (선행 테스트가 먼저 실패해야 정상)"
    block = prompt[start:start + 600]
    assert "code_generation_agent" not in block, "C-2 규칙에 agent_id 하드코딩됨"


def test_explicit_code_request_path_preserved():
    """과교정 방지: 사용자가 코드 작성을 명시 요청하는 정상 경로는 규칙에 문서화."""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt("퀵소트 파이썬 코드 짜줘")
    assert "코드를 작성" in prompt or "코드 작성을 명시" in prompt, \
        "명시적 코드 요청은 여전히 허용된다는 서술 부재"


def test_prompt_contains_exclusion_respect_rule():
    """시그널 C-3: description 의 배제 조건('~는 대상이 아닙니다')을 존중.
    RED(수정 전): 규칙 부재 → 배제 명시에도 '점자'가 모스 에이전트로 배정됨
    (라이브 실측 — 데이터 수정만으로는 LLM 이 무시). GREEN: 규칙 존재."""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt("점자로 변환해줘")
    assert "대상이 아닙니다" in prompt, "배제 조건 존중 규칙(C-3) 부재"
    assert "배제" in prompt and "존중" in prompt, "배제 존중 서술 부재"


def test_gnn_recommendation_yields_to_exclusion():
    """GNN+RL '강력 추천' 섹션이 description 배제 조건에 우선하면 안 된다.
    RED(수정 전): '반드시 첫 번째 단계에서 사용' 지시가 C-3 를 눌러 점자→모스
    오배정 지속 (라이브 실측: hybrid 추천 주입 확인). GREEN: 추천 섹션에
    배제 조건 우선 기각 단서 존재."""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt(
        "점자로 변환해줘", None, "some_agent",
        {"confidence": 0.9, "selection_source": "kg_assisted", "is_hint": False})
    start = prompt.find("GNN+RL 추천 에이전트")
    assert start != -1, "강추천 섹션 자체가 없음 (테스트 셋업 오류)"
    block = prompt[start:start + 900]
    assert "배제" in block and ("기각" in block or "무시" in block), \
        "강추천 블록에 배제 조건 우선(추천 기각) 단서 부재"


def test_gnn_hint_yields_to_exclusion_too():
    """저신뢰 힌트 블록에도 동일한 배제 우선 단서."""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt(
        "점자로 변환해줘", None, "some_agent",
        {"confidence": 0.3, "selection_source": "kg_assisted", "is_hint": True})
    start = prompt.find("GNN+RL 추천 에이전트")
    assert start != -1
    block = prompt[start:start + 900]
    assert "배제" in block and ("기각" in block or "무시" in block), \
        "힌트 블록에 배제 조건 우선 단서 부재"


def test_existing_gap_signals_preserved():
    """기존 시그널 A/B/C 텍스트는 훼손 금지 (additive 계약)."""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt("아무 쿼리")
    assert "명시적 에이전트 생성 요청" in prompt, "시그널 A 유실"
    assert "외부 API 전용성" in prompt, "시그널 B 유실"
    assert "약한 매칭 금지" in prompt, "시그널 C 유실"


def main():
    fails = []
    for fn in (
        test_prompt_contains_code_absorption_guardrail,
        test_rule_does_not_hardcode_agent_id,
        test_explicit_code_request_path_preserved,
        test_prompt_contains_exclusion_respect_rule,
        test_gnn_recommendation_yields_to_exclusion,
        test_gnn_hint_yields_to_exclusion_too,
        test_existing_gap_signals_preserved,
    ):
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            print("FAIL", fn.__name__, "→", e)
            fails.append(fn.__name__)
        except Exception as e:
            print("ERROR", fn.__name__, "→", type(e).__name__, str(e)[:80])
            fails.append(fn.__name__)
    print("\nRESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
