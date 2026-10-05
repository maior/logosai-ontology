"""B4 — planner 시간범위 보존 가드레일 검증 (2026-07-09).

배경: query_planner 프롬프트가 "명시 안 한 조건 추가 금지"(rule 5, line 848-851)는
있으나 "명시한 조건(특히 시간범위) 삭제 금지"의 대칭 규칙이 없어, LLM 이
"이번주 날씨" 를 "현재 날씨" 로 좁히는 사례 발생. 게다가 주식 예시(465/633/844)가
'현재' 주입을 학습시켜 상충.

계약:
  - _build_planning_prompt 는 순수 함수(self.registry 만 사용) → LLM/API 키 없이
    __new__ + stub registry 로 결정론적 테스트 가능
  - 수정 후 프롬프트에 시간범위 보존 가드레일(rule 6) 텍스트가 존재해야 함
  - 기존 주식 '현재' 추론 예시는 그대로 보존(과교정 방지 회귀 가드)

실행: .venv/bin/python ontology/orchestrator/test_query_planner_timerange.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ONTOLOGY = os.path.join(_ROOT, "ontology")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ONTOLOGY)  # query_planner 내부 `from core.models` 해석용


class _StubRegistry:
    """_build_planning_prompt 가 쓰는 유일한 의존 — LLM/API 키 없이 실행."""

    def build_prompt_context(self, include_schema: bool = False) -> str:
        return "## weather_agent\n- 설명: 날씨 조회\n"


def _planner_without_init():
    """__init__(GOOGLE_API_KEY + genai.Client) 우회 — 메서드는 self.registry 만 사용."""
    from ontology.orchestrator.query_planner import QueryPlanner

    planner = QueryPlanner.__new__(QueryPlanner)
    planner.registry = _StubRegistry()
    return planner


def test_prompt_contains_timerange_preservation_guardrail():
    """RED(수정 전): 시간범위 보존 규칙(rule 6) 부재. GREEN(수정 후): 존재."""
    planner = _planner_without_init()
    # 시간 토큰 없는 중립 쿼리 — 앵커는 가드레일 상수에서만 나와야 함
    prompt = planner._build_planning_prompt("테슬라 주가 알려줘")

    assert "삭제" in prompt and "축소" in prompt, "rule 6 헤드라인(삭제·축소 금지) 부재"
    assert "이번주 날씨" in prompt, "시간범위 보존 긍정 예시 부재"
    assert "현재 날씨" in prompt, "'현재 날씨'로 좁히는 안티패턴 예시 부재"


def test_stock_current_inference_preserved():
    """과교정 방지: 맥락 추론 '현재'(주식 후속질문)는 여전히 허용/문서화돼야 함."""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt("삼성전자 주가")
    assert "삼성전자 현재 주가" in prompt, "주식 '현재' 추론 예시(line 465/844)가 유실됨"


def test_add_condition_rule_still_present():
    """기존 rule 5(명시 안 한 조건 추가 금지)는 그대로 유지."""
    planner = _planner_without_init()
    prompt = planner._build_planning_prompt("파일 찾아줘")
    assert "명시하지 않은 조건" in prompt, "rule 5(조건 추가 금지)가 훼손됨"


def main():
    fails = []
    for fn in (
        test_prompt_contains_timerange_preservation_guardrail,
        test_stock_current_inference_preserved,
        test_add_condition_rule_still_present,
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
