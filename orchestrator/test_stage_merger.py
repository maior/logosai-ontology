"""TDD: 의존성 없는 인접 1-agent stages 자동 병합 (parallel 안전망).

LLM 이 prompt 무시하고 1-agent-per-stage 로 만들어도 후처리에서 자동 정정.

규칙:
- 인접한 stages 들이 모두 input_from=null 이고 1 agent 만 가지면 → 같은 stage 로 병합 (parallel)
- 데이터 의존성 있는 stage 는 건드리지 않음
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestStageMerger(unittest.TestCase):

    def _stages_data(self, *stages):
        return list(stages)

    def test_module_has_merge_function(self):
        from ontology.orchestrator.query_planner import merge_independent_stages
        self.assertTrue(callable(merge_independent_stages))

    def test_no_merge_when_dependencies_exist(self):
        """input_from 이 있으면 병합 안 됨."""
        from ontology.orchestrator.query_planner import merge_independent_stages
        stages = [
            {"stage_id": 1, "execution_type": "sequential", "agents": [{"agent_id": "a1", "input_from": None}]},
            {"stage_id": 2, "execution_type": "sequential", "agents": [{"agent_id": "a2", "input_from": ["stage_1.a1"]}]},
        ]
        merged = merge_independent_stages(stages)
        self.assertEqual(len(merged), 2, "stages with dependencies must not merge")

    def test_merge_independent_singletons(self):
        """3개 의존성 없는 1-agent stages → 1개 parallel stage."""
        from ontology.orchestrator.query_planner import merge_independent_stages
        stages = [
            {"stage_id": 1, "execution_type": "sequential", "agents": [{"agent_id": "weather", "input_from": None}]},
            {"stage_id": 2, "execution_type": "sequential", "agents": [{"agent_id": "currency", "input_from": None}]},
            {"stage_id": 3, "execution_type": "sequential", "agents": [{"agent_id": "internet", "input_from": None}]},
            {"stage_id": 4, "execution_type": "sequential", "agents": [{"agent_id": "llm_search", "input_from": ["stage_1", "stage_2", "stage_3"]}]},
        ]
        merged = merge_independent_stages(stages)
        self.assertEqual(len(merged), 2, f"expected 2 stages after merge, got {len(merged)}")
        self.assertEqual(merged[0]["execution_type"], "parallel")
        self.assertEqual(len(merged[0]["agents"]), 3)
        self.assertEqual(merged[1]["agents"][0]["agent_id"], "llm_search")

    def test_no_merge_for_single_stage(self):
        """1 stage 면 그대로 유지."""
        from ontology.orchestrator.query_planner import merge_independent_stages
        stages = [
            {"stage_id": 1, "execution_type": "sequential", "agents": [{"agent_id": "weather", "input_from": None}]},
        ]
        merged = merge_independent_stages(stages)
        self.assertEqual(len(merged), 1)

    def test_already_parallel_unchanged(self):
        """이미 parallel 인 stage 는 그대로."""
        from ontology.orchestrator.query_planner import merge_independent_stages
        stages = [
            {"stage_id": 1, "execution_type": "parallel", "agents": [
                {"agent_id": "a", "input_from": None},
                {"agent_id": "b", "input_from": None},
            ]},
        ]
        merged = merge_independent_stages(stages)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["agents"]), 2)
        self.assertEqual(merged[0]["execution_type"], "parallel")

    def test_partial_merge_keeps_dependency_chain(self):
        """앞 2개는 독립, 마지막 2개는 의존성 → 앞 2개만 병합."""
        from ontology.orchestrator.query_planner import merge_independent_stages
        stages = [
            {"stage_id": 1, "execution_type": "sequential", "agents": [{"agent_id": "a", "input_from": None}]},
            {"stage_id": 2, "execution_type": "sequential", "agents": [{"agent_id": "b", "input_from": None}]},
            {"stage_id": 3, "execution_type": "sequential", "agents": [{"agent_id": "c", "input_from": ["stage_2.b"]}]},
            {"stage_id": 4, "execution_type": "sequential", "agents": [{"agent_id": "d", "input_from": ["stage_3.c"]}]},
        ]
        merged = merge_independent_stages(stages)
        self.assertEqual(len(merged), 3, f"expected 3 stages (merged a+b, c, d), got {len(merged)}")
        self.assertEqual(merged[0]["execution_type"], "parallel")
        self.assertEqual(len(merged[0]["agents"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
