"""
검색 QA (⑤) — 골든셋 하네스: 검색 품질을 감이 아니라 숫자로.

이 세션 내내 정직하게 표시해온 빚이 있다: entry_ratio=0.5, RRF_K=60 은
측정값이 아니라 지어낸 숫자다. aicoach 는 KII 골든셋으로 BM25 가중치를
측정해 recall@5=0.944 를 만들었다 (rag/search.py:20-22). 골든셋 없이는
모든 검색 변경이 장님이다 — 좋아졌는지 나빠졌는지 알 수 없다.

고정하는 계약:
1. **평가는 결정적, LLM 0콜** — 평가에 LLM 이 끼면 평가 자체가 비결정이
   되어 회귀 검사가 성립하지 않는다.
2. LLM(생성기)의 역할은 **초안 케이스 제안**까지 — status=draft 로 들어오고,
   확정(confirmed)은 인간이 한다 (검수 루프와 같은 권한 경계).
3. 생성된 질의가 정답 노드의 이름을 **문자 그대로 포함하면 버린다** —
   키워드 매칭만 테스트하는 케이스는 골든셋의 목적(의미 검색 측정)에
   반한다. 패러프레이즈여야 잰다.
4. 저장은 JSONL append + replay (review_store 규약) — 케이스는 자산이다.
"""

import asyncio
import json

import pytest

from ontology.core.search_qa import (
    GoldenCase,
    GoldenSet,
    QAGenerator,
    evaluate_cases,
    parse_generated_cases,
    reset_golden_sets,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean():
    reset_golden_sets()
    yield
    reset_golden_sets()


# ─── 1. 골든셋 저장소 ────────────────────────────────────────────────

class TestGoldenSetStore:
    def test_add_and_list(self, tmp_path):
        gs = GoldenSet(namespace="g", path=tmp_path / "g.jsonl")
        case_id = gs.add(query="옛날 무덤", expected_node_id="HeritageClass:무덤",
                         status="confirmed", source="hand")
        assert len(gs.cases()) == 1
        assert gs.cases()[0].case_id == case_id
        assert gs.cases()[0].status == "confirmed"

    def test_roundtrip_persistence(self, tmp_path):
        path = tmp_path / "g.jsonl"
        gs = GoldenSet(namespace="g", path=path)
        gs.add(query="q1", expected_node_id="N:1", status="draft")

        restored = GoldenSet(namespace="g", path=path)
        restored.load_from_disk()
        assert len(restored.cases()) == 1
        assert restored.cases()[0].query == "q1"

    def test_confirm_changes_status(self, tmp_path):
        gs = GoldenSet(namespace="g", path=tmp_path / "g.jsonl")
        case_id = gs.add(query="q", expected_node_id="N:1", status="draft")
        assert gs.confirm(case_id) is True
        assert gs.cases()[0].status == "confirmed"

    def test_confirm_unknown_id_is_false(self, tmp_path):
        gs = GoldenSet(namespace="g", path=tmp_path / "g.jsonl")
        assert gs.confirm("없는id") is False

    def test_confirm_survives_replay(self, tmp_path):
        """확정도 이벤트다 — replay 후에도 상태가 유지돼야 자산이 된다."""
        path = tmp_path / "g.jsonl"
        gs = GoldenSet(namespace="g", path=path)
        case_id = gs.add(query="q", expected_node_id="N:1", status="draft")
        gs.confirm(case_id)

        restored = GoldenSet(namespace="g", path=path)
        restored.load_from_disk()
        assert restored.cases()[0].status == "confirmed"

    def test_duplicate_query_is_not_added_twice(self, tmp_path):
        """같은 질의를 두 번 넣으면 골든셋 지표가 그 케이스에 과가중된다."""
        gs = GoldenSet(namespace="g", path=tmp_path / "g.jsonl")
        gs.add(query="같은 질의", expected_node_id="N:1")
        gs.add(query="같은  질의", expected_node_id="N:2")  # 공백만 다름
        assert len(gs.cases()) == 1

    def test_corrupt_line_is_skipped(self, tmp_path):
        path = tmp_path / "g.jsonl"
        path.write_text('{"event": "add", "case": {"case_id": "a", "query": "q",'
                        ' "expected_node_id": "N:1", "status": "draft",'
                        ' "source": ""}}\nnot json\n', encoding="utf-8")
        gs = GoldenSet(namespace="g", path=path)
        gs.load_from_disk()
        assert len(gs.cases()) == 1


# ─── 2. 평가 (결정적, LLM 0콜) ───────────────────────────────────────

def make_ranker(ranking):
    """결정적 가짜 검색 채널 — {query: [node_id, ...]} 순위표."""
    def rank(query, top_k):
        return ranking.get(query, [])[:top_k]
    return rank


class TestEvaluate:
    CASES = [
        GoldenCase(case_id="1", query="옛날 무덤", expected_node_id="N:무덤",
                   status="confirmed"),
        GoldenCase(case_id="2", query="부처님 절", expected_node_id="N:사찰",
                   status="confirmed"),
        GoldenCase(case_id="3", query="성곽 문", expected_node_id="N:성문",
                   status="confirmed"),
    ]

    def test_perfect_channel_scores_one(self):
        rank = make_ranker({"옛날 무덤": ["N:무덤"], "부처님 절": ["N:사찰"],
                            "성곽 문": ["N:성문"]})
        metrics = evaluate_cases(self.CASES, {"semantic": rank}, k=5)
        m = metrics["channels"]["semantic"]
        assert m["hit@1"] == 1.0
        assert m["hit@5"] == 1.0
        assert m["mrr"] == 1.0
        assert m["cases"] == 3

    def test_rank_position_shapes_mrr(self):
        """1위 1건 + 2위 1건 + 미검출 1건 → MRR = (1 + 0.5 + 0)/3."""
        rank = make_ranker({"옛날 무덤": ["N:무덤"],
                            "부처님 절": ["N:다른것", "N:사찰"],
                            "성곽 문": ["N:엉뚱"]})
        m = evaluate_cases(self.CASES, {"semantic": rank}, k=5)["channels"]["semantic"]
        assert m["hit@1"] == pytest.approx(1 / 3)
        assert m["hit@5"] == pytest.approx(2 / 3)
        assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)

    def test_per_case_results_name_the_failures(self):
        """지표만으로는 못 고친다 — 어느 케이스가 몇 위였는지가 있어야
        회귀의 원인을 찾는다."""
        rank = make_ranker({"옛날 무덤": ["N:무덤"]})
        result = evaluate_cases(self.CASES, {"semantic": rank}, k=5)
        by_id = {r["case_id"]: r for r in result["per_case"]}
        assert by_id["1"]["semantic_rank"] == 1
        assert by_id["2"]["semantic_rank"] is None  # 미검출이 명시된다

    def test_draft_cases_are_excluded_by_default(self):
        """미확정 초안이 지표를 오염시키면 안 된다 — 확정만 잰다."""
        cases = self.CASES + [GoldenCase(case_id="d", query="초안",
                                         expected_node_id="N:x", status="draft")]
        rank = make_ranker({})
        m = evaluate_cases(cases, {"semantic": rank}, k=5)["channels"]["semantic"]
        assert m["cases"] == 3

    def test_include_drafts_flag(self):
        cases = [GoldenCase(case_id="d", query="초안", expected_node_id="N:x",
                            status="draft")]
        m = evaluate_cases(cases, {"c": make_ranker({})}, k=5,
                           include_drafts=True)["channels"]["c"]
        assert m["cases"] == 1

    def test_two_channels_are_compared_side_by_side(self):
        """골든셋의 목적: 채널(설정) 간 비교 — entry_ratio 0.3 vs 0.5 를
        나란히 재는 것이 이 구조다."""
        good = make_ranker({c.query: [c.expected_node_id] for c in self.CASES})
        bad = make_ranker({})
        result = evaluate_cases(self.CASES, {"A": good, "B": bad}, k=5)
        assert result["channels"]["A"]["hit@5"] == 1.0
        assert result["channels"]["B"]["hit@5"] == 0.0

    def test_empty_cases_do_not_divide_by_zero(self):
        """0 나눗셈으로 죽지 않고, **0.0 으로 거짓말하지도 않는다.**

        예전 계약은 0.0 을 돌려주고 소비자가 cases 를 같이 보게 했다. 그건
        foot-gun 이다 — 0.0 은 "다 틀렸다"로 읽힌다. None + measured=False 로
        미측정임을 자체적으로 드러낸다.
        """
        result = evaluate_cases([], {"c": make_ranker({})}, k=5)
        m = result["channels"]["c"]
        assert m["cases"] == 0 and m["mrr"] is None
        assert result["measured"] is False


# ─── 3. 생성기 검증 (LLM 출력 불신) ──────────────────────────────────

class TestParseGeneratedCases:
    NODE = {"node_id": "HeritageClass:무덤", "name": "무덤"}

    def test_paraphrase_case_passes(self):
        raw = json.dumps({"queries": ["옛날 사람들이 묻힌 곳은?"]},
                         ensure_ascii=False)
        cases = parse_generated_cases(raw, self.NODE)
        assert len(cases) == 1
        assert cases[0]["expected_node_id"] == "HeritageClass:무덤"

    def test_query_containing_the_name_is_dropped(self):
        """계약 3 — 이름이 그대로 든 질의는 키워드 매칭 테스트일 뿐이다."""
        raw = json.dumps({"queries": ["무덤이 뭐야?", "옛날 매장지는?"]},
                         ensure_ascii=False)
        cases = parse_generated_cases(raw, self.NODE)
        assert len(cases) == 1
        assert "무덤" not in cases[0]["query"]

    def test_blank_and_duplicate_queries_dropped(self):
        raw = json.dumps({"queries": ["", "  ", "매장 유적?", "매장  유적?"]},
                         ensure_ascii=False)
        assert len(parse_generated_cases(raw, self.NODE)) == 1

    def test_garbage_returns_empty(self):
        assert parse_generated_cases("json 아님", self.NODE) == []


class TestGenerator:
    def test_generates_draft_cases(self, tmp_path):
        def fake_llm(prompt):
            return json.dumps({"queries": ["옛날 매장 유적은 어디에?"]},
                              ensure_ascii=False)

        gs = GoldenSet(namespace="g", path=tmp_path / "g.jsonl")
        generator = QAGenerator(llm_fn=fake_llm)
        added = run(generator.generate_for_nodes(
            gs, [{"node_id": "HeritageClass:무덤", "name": "무덤",
                  "definition": "옛 사람을 묻은 곳"}]))
        assert added == 1
        assert gs.cases()[0].status == "draft"  # 계약 2 — 확정은 인간이

    def test_llm_failure_adds_nothing_never_raises(self, tmp_path):
        def broken(prompt):
            raise RuntimeError("down")

        gs = GoldenSet(namespace="g", path=tmp_path / "g.jsonl")
        generator = QAGenerator(llm_fn=broken)
        assert run(generator.generate_for_nodes(
            gs, [{"node_id": "N:1", "name": "x"}])) == 0


# ─── 태그(시나리오 축) + 태그별 분해 + retrieve 채널 어댑터 (Phase 2) ───

class TestTagsAndRetrieveChannel:
    def test_add_and_persist_tags(self, tmp_path):
        from ontology.core.search_qa import GoldenSet
        gs = GoldenSet(namespace="tg", path=tmp_path / "g.jsonl")
        gs.add("청약을 무를 수 있나?", "Clause:제19조", status="confirmed",
               tags=["semantic", "procedure"])
        assert gs.cases()[0].tags == ["semantic", "procedure"]
        # JSONL replay 로 태그 보존
        gs2 = GoldenSet(namespace="tg", path=tmp_path / "g.jsonl")
        assert gs2.load_from_disk()
        assert gs2.cases()[0].tags == ["semantic", "procedure"]

    def test_legacy_case_without_tags_defaults_empty(self):
        from ontology.core.search_qa import GoldenCase
        c = GoldenCase.from_dict({"case_id": "x", "query": "q",
                                  "expected_node_id": "n"})
        assert c.tags == []

    def test_evaluate_by_tag_breakdown(self):
        from ontology.core.search_qa import GoldenCase, evaluate_cases
        cases = [
            GoldenCase("a", "q1", "N1", status="confirmed", tags=["exact"]),
            GoldenCase("b", "q2", "N2", status="confirmed", tags=["graph"]),
            GoldenCase("c", "q3", "N3", status="confirmed", tags=["exact"]),
        ]
        ranks = {"q1": ["N1"], "q2": ["Z"], "q3": ["N3"]}  # exact 2/2, graph 0/1
        res = evaluate_cases(cases, {"ch": lambda q, k: ranks[q]}, k=5)
        assert res["by_tag"]["exact"]["ch"]["cases"] == 2
        assert res["by_tag"]["exact"]["ch"]["hit@1"] == 1.0
        assert res["by_tag"]["graph"]["ch"]["hit@1"] == 0.0

    def test_chunk_hits_to_nodes_orders_and_dedups(self):
        from ontology.core.search_qa import chunk_hits_to_nodes
        hits = [{"node_ids": ["A", "B"]}, {"node_ids": ["B", "C"]},
                {"node_ids": []}, {"node_ids": ["D"]}]
        assert chunk_hits_to_nodes(hits) == ["A", "B", "C", "D"]

    def test_chunk_hits_to_nodes_handles_missing_key(self):
        from ontology.core.search_qa import chunk_hits_to_nodes
        assert chunk_hits_to_nodes([{}, {"node_ids": None}, {"node_ids": ["X"]}]) == ["X"]

    def test_retrieve_result_to_nodes_entry_first_then_expanded_then_hits(self):
        from ontology.core.search_qa import retrieve_result_to_nodes
        result = {
            "expansion": {
                "entry_nodes": [{"node_id": "E2", "score": 0.4},
                                {"node_id": "E1", "score": 0.9}],
                "expanded_nodes": [{"node_id": "X1"}, {"node_id": "E1"}],  # E1 중복
            },
            "hits": [{"node_ids": ["H1", "E2"]}, {"node_ids": ["H2"]}],
        }
        # entry 점수순(E1,E2) → expanded(X1; E1 중복 skip) → hits(H1; E2 중복, H2)
        assert retrieve_result_to_nodes(result) == ["E1", "E2", "X1", "H1", "H2"]

    def test_retrieve_result_to_nodes_none_and_hits_only(self):
        from ontology.core.search_qa import retrieve_result_to_nodes
        assert retrieve_result_to_nodes(None) == []
        assert retrieve_result_to_nodes({"hits": [{"node_ids": ["A", "A"]}]}) == ["A"]


class TestGeneratedTagging:
    """생성 초안은 패러프레이즈(semantic 매칭형) — semantic 태그 자동 부여."""

    def test_generated_drafts_tagged_semantic(self, tmp_path):
        import asyncio
        from ontology.core.search_qa import GoldenSet, QAGenerator

        def fake(prompt):
            return json.dumps({"queries": ["계약을 무를 수 있나", "환불 받을 수 있나"]},
                              ensure_ascii=False)
        gs = GoldenSet(namespace="gt", path=tmp_path / "g.jsonl")
        gen = QAGenerator(llm_fn=fake)
        nvs = [{"node_id": "Clause:청약철회", "name": "청약철회",
                "definition": "계약 철회권"}]
        added = asyncio.run(gen.generate_for_nodes(gs, nvs, per_node=2))
        assert added >= 1
        cases = gs.cases()
        assert all("semantic" in c.tags for c in cases)
        assert all(c.status == "draft" and c.source == "generator" for c in cases)


class TestRelevantSet:
    """골든 케이스가 복수 정답(relevant set)을 허용 — 동의어/교차연결 개념을
    credit (다-1에서 드러난 단일 expected 한계 해소, 병렬 C)."""

    def test_accepted_ids_union_with_primary(self):
        from ontology.core.search_qa import GoldenCase
        c = GoldenCase("x", "q", "N1", accepted=["N2", "N3"])
        assert c.accepted_ids() == {"N1", "N2", "N3"}

    def test_legacy_case_defaults_to_primary_only(self):
        from ontology.core.search_qa import GoldenCase
        c = GoldenCase.from_dict({"case_id": "x", "query": "q",
                                  "expected_node_id": "N1"})
        assert c.accepted == [] and c.accepted_ids() == {"N1"}

    def test_add_and_roundtrip_accepted(self, tmp_path):
        from ontology.core.search_qa import GoldenSet
        gs = GoldenSet(namespace="rs", path=tmp_path / "g.jsonl")
        gs.add("유방암 보장?", "Disease:유방의 악성 신생물", status="confirmed",
               accepted=["Disease:유방암", "Disease:C50"])
        assert gs.cases()[0].accepted_ids() == {
            "Disease:유방의 악성 신생물", "Disease:유방암", "Disease:C50"}
        gs2 = GoldenSet(namespace="rs", path=tmp_path / "g.jsonl")
        assert gs2.load_from_disk()
        assert len(gs2.cases()[0].accepted) == 2   # replay 보존

    def test_evaluate_credits_any_accepted_node(self):
        from ontology.core.search_qa import GoldenCase, evaluate_cases
        c = GoldenCase("x", "q", "N1", status="confirmed", accepted=["N2"])
        # 랭킹에 primary(N1) 없이 accepted(N2)만 1위 → hit@1 로 credit
        res = evaluate_cases([c], {"ch": lambda q, k: ["N2", "N9"]}, k=5)
        assert res["channels"]["ch"]["hit@1"] == 1.0
