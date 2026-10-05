"""Unit test (relative import 우회) — ExecutionPlan + detect_explicit_capability_gap.

query_planner._build_execution_plan / _build_planning_prompt 등 통합 부분은
relative import 때문에 단위 테스트 불가 → E2E (시나리오 5) 로 검증.
"""
import importlib.util
import os
import sys
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# models.py 는 standalone import 가능
models = _load_module("ontology_models", os.path.join(_HERE, "models.py"))


class TestExecutionPlanCapabilityGap(unittest.TestCase):

    def test_field_exists_default_none(self):
        plan = models.ExecutionPlan(query="x", workflow_strategy="sequential")
        self.assertTrue(hasattr(plan, "capability_gap"))
        self.assertIsNone(plan.capability_gap)

    def test_to_dict_serializes_capability_gap(self):
        plan = models.ExecutionPlan(query="x", workflow_strategy="sequential")
        plan.capability_gap = {"detected": True, "missing_capabilities": ["foo"]}
        d = plan.to_dict()
        self.assertIn("capability_gap", d)
        self.assertTrue(d["capability_gap"]["detected"])

    def test_to_dict_serializes_none(self):
        plan = models.ExecutionPlan(query="x", workflow_strategy="sequential")
        d = plan.to_dict()
        self.assertIn("capability_gap", d)
        self.assertIsNone(d["capability_gap"])


# detect_explicit_capability_gap 만 함수 단위 import (query_planner.py 의 일부 추출)
# 직접 import 가 막히므로 필요한 함수 코드를 별도 minimal 모듈로 복사 — 또는
# query_planner.py 에서 함수만 추출. 가장 간단: ast 로 함수 본문 가져와서 exec.
# 실용적 접근: 함수가 순수 함수이고 짧으니 직접 string 으로 재정의해서 검증.

class TestDetectExplicitCapabilityGap(unittest.TestCase):
    """detect_explicit_capability_gap 함수 — string 매칭 순수 함수 단위 검증."""

    def setUp(self):
        # query_planner.py 의 detect_explicit_capability_gap 만 추출 실행.
        # ast 로 function definition 만 가져와서 새 namespace 에 exec.
        import ast
        src_path = os.path.join(_HERE, "query_planner.py")
        with open(src_path, "r") as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "detect_explicit_capability_gap":
                func_src = ast.unparse(node)
                ns = {"Optional": __import__("typing").Optional, "Dict": __import__("typing").Dict, "Any": __import__("typing").Any}
                exec(func_src, ns)
                self.fn = ns["detect_explicit_capability_gap"]
                return
        raise RuntimeError("detect_explicit_capability_gap 함수 없음")

    def test_korean_pattern_triggers(self):
        gap = self.fn("Mastodon API 로 toot 가져와서 분석하는 에이전트 만들어줘")
        self.assertIsNotNone(gap)
        self.assertTrue(gap.get("detected"))

    def test_english_pattern_triggers(self):
        gap = self.fn("Build an agent that scrapes my Mastodon timeline")
        self.assertIsNotNone(gap)
        self.assertTrue(gap.get("detected"))

    def test_normal_query_no_trigger(self):
        gap = self.fn("오늘 날씨 알려줘")
        self.assertIsNone(gap)

    def test_returns_required_keys(self):
        gap = self.fn("Discord webhook 보내는 에이전트 만들어줘")
        self.assertIn("missing_capabilities", gap)
        self.assertIn("suggested_agent_description", gap)
        self.assertIn("reason", gap)

    def test_empty_query(self):
        self.assertIsNone(self.fn(""))
        self.assertIsNone(self.fn(None))

    def test_case_insensitive_english(self):
        gap = self.fn("BUILD AN AGENT FOR DISCORD")
        self.assertIsNotNone(gap)
        self.assertTrue(gap.get("detected"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
