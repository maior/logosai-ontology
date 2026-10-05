"""TDD: query_planner.py 의 capability_gap 인식.

3 단계 검증:
  1. ExecutionPlan 에 capability_gap 필드 존재
  2. query_planner 가 명시적 에이전트 생성 패턴 (코드 safety net) 을 capability_gap 으로 감지
  3. query_planner 가 LLM 응답의 capability_gap 필드를 ExecutionPlan 에 전파

logos_api 측 fallback 동작 (Step 4) 은 E2E 시나리오 5 로 검증.
"""
import os
import sys
import unittest

# sys.path 두 곳 추가:
#   - Logos 루트 (ontology 패키지 접근용)
#   - ontology/orchestrator (relative import 우회용)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # Logos/
sys.path.insert(0, _HERE)  # ontology/orchestrator/


class TestExecutionPlanHasCapabilityGap(unittest.TestCase):

    def test_execution_plan_has_capability_gap_field(self):
        """ExecutionPlan dataclass 에 capability_gap 필드 존재."""
        from models import ExecutionPlan
        plan = ExecutionPlan(query="x", workflow_strategy="sequential")
        self.assertTrue(hasattr(plan, "capability_gap"),
                        "ExecutionPlan 에 capability_gap 필드 없음")
        self.assertIsNone(plan.capability_gap, "기본값은 None")

    def test_to_dict_includes_capability_gap(self):
        """to_dict() 에 capability_gap 직렬화."""
        from models import ExecutionPlan
        plan = ExecutionPlan(query="x", workflow_strategy="sequential")
        plan.capability_gap = {"detected": True, "missing_capabilities": ["foo"]}
        d = plan.to_dict()
        self.assertIn("capability_gap", d, "to_dict 에 capability_gap 누락")
        self.assertEqual(d["capability_gap"]["detected"], True)


class TestQueryPlannerPromptCapabilityGap(unittest.TestCase):

    def _build_prompt(self):
        from query_planner import QueryPlanner
        planner = QueryPlanner()
        return planner._build_planning_prompt("test query", None, None, None)

    def test_prompt_mentions_capability_gap_field(self):
        """LLM prompt 에 capability_gap 출력 필드 명시."""
        prompt = self._build_prompt()
        self.assertIn("capability_gap", prompt,
                      "prompt 에 capability_gap 필드 안내 없음")

    def test_prompt_explicit_creation_signal(self):
        """prompt 에 명시적 에이전트 생성 요청 시그널 명시."""
        prompt = self._build_prompt()
        signals = ["에이전트 만들", "에이전트 생성", "에이전트를 만들",
                   "build agent", "create agent"]
        found = any(s in prompt for s in signals)
        self.assertTrue(found, f"prompt 에 명시적 생성 시그널 없음. 검색: {signals}")

    def test_prompt_external_api_specificity(self):
        """prompt 에 외부 API specificity 시그널 명시."""
        prompt = self._build_prompt()
        signals = ["외부 API", "external API", "전용 API", "특정 API",
                   "specialized service", "internet_agent 의 일반 검색", "internet_agent 로 대체"]
        found = any(s in prompt for s in signals)
        self.assertTrue(found, f"prompt 에 외부 API specificity 시그널 없음. 검색: {signals}")


class TestPostProcessingSafetyNet(unittest.TestCase):
    """LLM 이 capability_gap 누락해도 명시적 패턴 키워드면 강제 trigger."""

    def test_safety_net_function_exists(self):
        from query_planner import detect_explicit_capability_gap
        self.assertTrue(callable(detect_explicit_capability_gap))

    def test_explicit_korean_pattern_triggers(self):
        from query_planner import detect_explicit_capability_gap
        gap = detect_explicit_capability_gap(
            "Mastodon API 로 toot 가져와서 분석하는 에이전트 만들어줘"
        )
        self.assertIsNotNone(gap, "Korean 'X 에이전트 만들어줘' 패턴 미감지")
        self.assertTrue(gap.get("detected"))

    def test_explicit_english_pattern_triggers(self):
        from query_planner import detect_explicit_capability_gap
        gap = detect_explicit_capability_gap(
            "Build an agent that scrapes my Mastodon timeline"
        )
        self.assertIsNotNone(gap, "English 'build an agent' 패턴 미감지")
        self.assertTrue(gap.get("detected"))

    def test_normal_query_does_not_trigger(self):
        from query_planner import detect_explicit_capability_gap
        gap = detect_explicit_capability_gap("오늘 날씨 알려줘")
        self.assertIsNone(gap, "일반 쿼리에서 false positive")

    def test_returns_missing_capabilities(self):
        from query_planner import detect_explicit_capability_gap
        gap = detect_explicit_capability_gap("Discord webhook 보내는 에이전트 만들어줘")
        self.assertIn("missing_capabilities", gap)
        self.assertIn("suggested_agent_description", gap)


class TestBuildExecutionPlanPropagatesGap(unittest.TestCase):
    """_build_execution_plan 이 LLM 응답의 capability_gap 을 ExecutionPlan 에 전파."""

    def test_llm_capability_gap_propagated(self):
        from query_planner import QueryPlanner
        planner = QueryPlanner()
        plan_data = {
            "workflow_strategy": "sequential",
            "stages": [],
            "capability_gap": {
                "detected": True,
                "missing_capabilities": ["mastodon_api"],
                "suggested_agent_description": "Mastodon agent",
                "reason": "no mastodon agent registered",
            },
            "reasoning": "LLM detected gap",
        }
        plan = planner._build_execution_plan("query", plan_data, "plan_id_x")
        self.assertIsNotNone(plan.capability_gap)
        self.assertTrue(plan.capability_gap.get("detected"))
        self.assertEqual(plan.capability_gap.get("missing_capabilities"), ["mastodon_api"])

    def test_safety_net_overrides_when_llm_missed(self):
        """LLM 이 capability_gap 누락했어도, query 에 명시 패턴 있으면 safety net 이 채움."""
        from query_planner import QueryPlanner
        planner = QueryPlanner()
        plan_data = {
            "workflow_strategy": "sequential",
            "stages": [],
            "reasoning": "...",
            # capability_gap 없음
        }
        # query 에 명시 패턴
        plan = planner._build_execution_plan(
            "Mastodon API 로 toot 분석하는 에이전트 만들어줘",
            plan_data,
            "plan_id_y",
        )
        self.assertIsNotNone(plan.capability_gap, "safety net 미작동")
        self.assertTrue(plan.capability_gap.get("detected"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
