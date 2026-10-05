"""청크 단위 골든셋 — 정답을 청크(원문 조각)로 잡고 채점한다.

**왜 청크 단위인가**: 노드 단위 채점은 semantic 과 retrieve 가 소수점까지 같게
나온다 — retrieve 의 노드 랭킹이 semantic 진입 노드에서 파생되기 때문이며, 그
설계는 이미 측정으로 국소 최적임이 확인됐다(retrieve_result_to_nodes 주석의 2-D
실험 기록: 노드 랭킹 RRF 융합은 hit@5 0.88→0.75 로 더 나빴다). 우리 차별점
(확장 질의 + 그래프 채널 + RRF)은 **청크 회수**에서 작동하므로, 그걸 재려면
정답도 청크여야 한다.

**정직성 계약**: 청크 정답이 없는 케이스는 target="chunk" 채점에서 **스킵**한다.
오답으로 세면 "라벨이 없다"가 "검색이 틀렸다"로 뒤바뀐다 — 자가 거짓말을 한다.
"""
import pytest

from ontology.core.search_qa import (
    GoldenCase,
    GoldenSet,
    chunk_hits_to_ids,
    evaluate_cases,
)


# ─── GoldenCase — 청크 정답 집합 ─────────────────────────────────────

class TestChunkAcceptedIds:
    def test_chunk_ids_include_expected_and_accepted(self):
        case = GoldenCase(case_id="1", query="청약 철회", expected_node_id="",
                          expected_chunk_id="c1", accepted_chunks=["c2", "c3"])
        assert case.accepted_chunk_ids() == {"c1", "c2", "c3"}

    def test_chunk_ids_empty_when_no_chunk_expectation(self):
        """노드만 라벨된 옛 케이스 — 청크 정답 집합은 비어 있다(스킵 대상)."""
        case = GoldenCase(case_id="1", query="q", expected_node_id="N:1")
        assert case.accepted_chunk_ids() == set()

    def test_node_ids_unaffected_by_chunk_fields(self):
        """하위호환: 청크 필드가 노드 정답 집합을 오염시키지 않는다."""
        case = GoldenCase(case_id="1", query="q", expected_node_id="N:1",
                          accepted=["N:2"], expected_chunk_id="c1",
                          accepted_chunks=["c2"])
        assert case.accepted_ids() == {"N:1", "N:2"}
        assert case.accepted_chunk_ids() == {"c1", "c2"}

    def test_blank_entries_dropped(self):
        case = GoldenCase(case_id="1", query="q", expected_node_id="",
                          expected_chunk_id="", accepted_chunks=["", "c9"])
        assert case.accepted_chunk_ids() == {"c9"}

    def test_from_dict_reads_chunk_fields(self):
        case = GoldenCase.from_dict({
            "case_id": "1", "query": "q", "expected_node_id": "N:1",
            "expected_chunk_id": "c1", "accepted_chunks": ["c2"],
        })
        assert case.expected_chunk_id == "c1"
        assert case.accepted_chunks == ["c2"]

    def test_from_dict_old_record_without_chunk_fields(self):
        """디스크의 옛 레코드엔 청크 필드가 없다 — 기본값으로 흡수."""
        case = GoldenCase.from_dict({
            "case_id": "1", "query": "q", "expected_node_id": "N:1",
        })
        assert case.expected_chunk_id == ""
        assert case.accepted_chunks == []
        assert case.accepted_chunk_ids() == set()


# ─── GoldenSet.add — 청크 정답 저장 ──────────────────────────────────

class TestGoldenSetAddChunk:
    def test_add_with_chunk_expectation(self, tmp_path):
        gs = GoldenSet(namespace="t", path=tmp_path / "g.jsonl")
        cid = gs.add(query="청약 철회 기간", expected_node_id="",
                     expected_chunk_id="chunk-19", accepted_chunks=["chunk-20"])
        assert cid
        case = gs.cases()[0]
        assert case.expected_chunk_id == "chunk-19"
        assert case.accepted_chunk_ids() == {"chunk-19", "chunk-20"}

    def test_chunk_fields_survive_replay(self, tmp_path):
        """JSONL 이벤트 로그 → replay 후에도 청크 정답이 살아 있어야 한다."""
        path = tmp_path / "g.jsonl"
        gs = GoldenSet(namespace="t", path=path)
        gs.add(query="q", expected_node_id="N:1", expected_chunk_id="c1",
               accepted_chunks=["c2"])
        reloaded = GoldenSet(namespace="t", path=path)
        assert reloaded.load_from_disk() is True
        case = reloaded.cases()[0]
        assert case.expected_chunk_id == "c1"
        assert case.accepted_chunks == ["c2"]

    def test_add_without_chunk_still_works(self, tmp_path):
        """기존 호출부(노드만) 무변경 — 청크 인자는 선택이다."""
        gs = GoldenSet(namespace="t", path=tmp_path / "g.jsonl")
        assert gs.add(query="q", expected_node_id="N:1")
        assert gs.cases()[0].expected_chunk_id == ""


# ─── chunk_hits_to_ids — 여러 hit 모양에서 청크 id 추출 ──────────────

class _FakeChunk:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id


class _FakeChunkHit:
    """graph_retrieval.ChunkHit 모양 — .chunk.chunk_id"""
    def __init__(self, chunk_id):
        self.chunk = _FakeChunk(chunk_id)


class TestChunkHitsToIds:
    def test_dict_hits(self):
        hits = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        assert chunk_hits_to_ids(hits) == ["c1", "c2"]

    def test_chunk_hit_objects(self):
        """GraphConditionedRetriever.search() 반환형."""
        assert chunk_hits_to_ids([_FakeChunkHit("c1"), _FakeChunkHit("c2")]) == ["c1", "c2"]

    def test_tuple_hits(self):
        """ChunkIndex.search() 반환형 — [(StoredChunk, score)]."""
        hits = [(_FakeChunk("c1"), 0.9), (_FakeChunk("c2"), 0.5)]
        assert chunk_hits_to_ids(hits) == ["c1", "c2"]

    def test_order_preserved_and_deduped(self):
        hits = [{"chunk_id": "c1"}, {"chunk_id": "c1"}, {"chunk_id": "c2"}]
        assert chunk_hits_to_ids(hits) == ["c1", "c2"]

    def test_empty_and_garbage_safe(self):
        assert chunk_hits_to_ids([]) == []
        assert chunk_hits_to_ids(None) == []
        assert chunk_hits_to_ids([None, {}, {"chunk_id": ""}, 42]) == []

    def test_hit_with_missing_chunk_object(self):
        """.chunk 가 None 인 hit — 채널 함수가 죽으면 채점 전체가 멈춘다."""
        class _Broken:
            chunk = None
        assert chunk_hits_to_ids([_Broken(), _FakeChunkHit("c1")]) == ["c1"]


# ─── evaluate_cases(target=...) — 채점 위치 전환 ─────────────────────

def _cases():
    return [
        # 청크·노드 둘 다 라벨된 케이스
        GoldenCase(case_id="a", query="qa", expected_node_id="N:1",
                   expected_chunk_id="c1", status="confirmed", tags=["exact"]),
        # 청크만 라벨
        GoldenCase(case_id="b", query="qb", expected_node_id="",
                   expected_chunk_id="c2", status="confirmed", tags=["semantic"]),
        # 노드만 라벨 — target=chunk 에서는 스킵되어야 한다
        GoldenCase(case_id="c", query="qc", expected_node_id="N:3",
                   status="confirmed", tags=["exact"]),
    ]


class TestEvaluateTarget:
    def test_default_target_is_node_backward_compatible(self):
        """target 인자를 주지 않으면 예전처럼 노드로 채점한다."""
        result = evaluate_cases(_cases(), {"ch": lambda q, k: ["N:1", "N:3"]}, k=5)
        assert result["target"] == "node"
        # 노드 정답이 있는 케이스는 a, c 두 개 → b(노드 정답 없음)는 스킵
        assert result["cases"] == 2
        assert result["skipped"] == 1
        assert result["channels"]["ch"]["hit@1"] == 0.5   # a 는 1위, c 는 2위

    def test_chunk_target_scores_chunk_ids(self):
        result = evaluate_cases(_cases(), {"ch": lambda q, k: ["c2", "c1"]},
                                k=5, target="chunk")
        assert result["target"] == "chunk"
        assert result["cases"] == 2          # a, b — c 는 청크 정답 없어 스킵
        assert result["skipped"] == 1
        assert result["channels"]["ch"]["hit@1"] == 0.5   # b 1위, a 2위
        assert result["channels"]["ch"]["hit@5"] == 1.0

    def test_chunk_target_expected_field_is_chunk_id(self):
        """per_case 의 expected 가 타깃에 맞게 바뀌어야 화면이 헷갈리지 않는다."""
        result = evaluate_cases(_cases(), {"ch": lambda q, k: []},
                                k=5, target="chunk")
        expected = {r["case_id"]: r["expected"] for r in result["per_case"]}
        assert expected == {"a": "c1", "b": "c2"}

    def test_skipped_cases_absent_from_per_case(self):
        """스킵은 오답이 아니다 — 표에 0점으로 나타나서도 안 된다."""
        result = evaluate_cases(_cases(), {"ch": lambda q, k: []},
                                k=5, target="chunk")
        assert [r["case_id"] for r in result["per_case"]] == ["a", "b"]

    def test_no_chunk_labels_at_all(self):
        """청크 라벨이 하나도 없으면 채점 0건 — 지표는 **None**(미측정)이다.

        실제 골든셋(ins_cancer_demo)이 이 상태였고, 예전 계약은 0.0 을 내면서
        cases=0/skipped=n 으로만 진실을 알렸다. 그 조합은 "hit@1 0%" 로 읽혀
        청크 채널이 나쁘다는 오해를 만들었다 — 라벨이 없었을 뿐이다.
        자세한 계약은 test_search_qa_evidence.TestZeroCaseHonesty.
        """
        nodes_only = [GoldenCase(case_id="x", query="q", expected_node_id="N:1",
                                 status="confirmed")]
        result = evaluate_cases(nodes_only, {"ch": lambda q, k: ["c1"]},
                                k=5, target="chunk")
        assert result["cases"] == 0
        assert result["skipped"] == 1
        assert result["channels"]["ch"]["hit@1"] is None
        assert result["measured"] is False

    def test_accepted_chunks_credit(self):
        """추가 정답(동의 청크) 중 아무거나 상위면 맞다."""
        case = GoldenCase(case_id="a", query="q", expected_node_id="",
                          expected_chunk_id="c1", accepted_chunks=["c9"],
                          status="confirmed")
        result = evaluate_cases([case], {"ch": lambda q, k: ["c9"]},
                                k=5, target="chunk")
        assert result["channels"]["ch"]["hit@1"] == 1.0

    def test_tags_still_decompose_under_chunk_target(self):
        result = evaluate_cases(_cases(), {"ch": lambda q, k: ["c1", "c2"]},
                                k=5, target="chunk")
        assert set(result["by_tag"]) == {"exact", "semantic"}
        assert result["by_tag"]["semantic"]["ch"]["cases"] == 1   # b 만

    def test_drafts_excluded_by_default_under_chunk_target(self):
        draft = GoldenCase(case_id="d", query="q", expected_node_id="",
                           expected_chunk_id="c1", status="draft")
        assert evaluate_cases([draft], {"ch": lambda q, k: ["c1"]},
                              k=5, target="chunk")["cases"] == 0
        assert evaluate_cases([draft], {"ch": lambda q, k: ["c1"]}, k=5,
                              target="chunk", include_drafts=True)["cases"] == 1

    def test_two_channels_can_differ_under_chunk_target(self):
        """이 작업의 목적 — 청크 단위에서는 두 채널이 실제로 갈린다."""
        cases = [GoldenCase(case_id="a", query="q", expected_node_id="",
                            expected_chunk_id="c5", status="confirmed")]
        result = evaluate_cases(cases, {
            "chunk": lambda q, k: ["c1", "c2", "c5"],     # 3위
            "retrieve": lambda q, k: ["c5", "c1"],        # 1위
        }, k=5, target="chunk")
        assert result["channels"]["chunk"]["hit@1"] == 0.0
        assert result["channels"]["retrieve"]["hit@1"] == 1.0
        assert result["channels"]["chunk"]["hit@5"] == 1.0

    def test_unknown_target_falls_back_to_node(self):
        """오타/미래 값이 조용히 빈 결과를 내지 않게 — 노드로 폴백."""
        result = evaluate_cases(_cases(), {"ch": lambda q, k: ["N:1"]},
                                k=5, target="nope")
        assert result["target"] == "node"
        assert result["cases"] == 2
