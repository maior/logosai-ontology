"""그래프 건강 진단 — 근거 없는 노드 · 추출 안 된 청크 · 중복 노드 후보.

**왜 이게 필요한가 (실측 동기)**: ins_cancer_demo 에서 청크 92개 중 27개(29%)만
노드를 냈고, 노드 106개 중 34개(32%)는 근거 청크가 없었다. /retrieve 의 그래프
채널은 노드→청크를 걷는데, 코퍼스의 71% 가 그 경로에서 안 보인다는 뜻이다 —
우리 차별점이 3분의 1 데이터 위에서 돌고 있었다. 게다가 골든셋 정답
`Disease:유방의 악성 신생물` 은 저장된 `Disease:C50( 유방의 악성 신생물 )` 과
다른 id 라 근거를 못 찾았다(같은 것의 서로 다른 이름).

이 모듈이 하는 일은 **판정이 아니라 제시**다. 무엇을 합칠지는 사람이 정한다 —
자동 병합은 되돌릴 수 없고, 잘못 합치면 인용이 엉뚱한 조항을 가리킨다.

**하드코딩 금지 준수**: 정규화는 구조(타입 접두사·괄호·공백·문장부호)만 본다.
'암'·'조항' 같은 도메인 어휘나 ICD 코드 패턴(C50 등)을 넣지 않는다 — 넣는 순간
보험 문서에서만 동작하는 커널이 된다.
"""

import pytest

from ontology.core.graph_health import (
    duplicate_clusters,
    evidence_gaps,
    health_report,
    identity_keys,
    normalize_name,
)


class _C:
    """StoredChunk 최소 형태 — 진단이 읽는 속성만."""

    def __init__(self, chunk_id, text="", node_ids=None, source="d.pdf"):
        self.chunk_id = chunk_id
        self.text = text
        self.node_ids = list(node_ids or [])
        self.source = source
        self.index = 0
        self.section = ""


# ── 정규화 ──────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_strips_type_prefix(self):
        assert normalize_name("Disease:유방암") == normalize_name("InsuranceTerm:유방암")

    def test_strips_whitespace_and_punctuation(self):
        assert normalize_name("InsuranceTerm:제6조(보험금의 지급사유 )") == \
               normalize_name("InsuranceTerm:제6조(보험금의 지급사유)")

    def test_casefolds(self):
        assert normalize_name("Thing:Cancer") == normalize_name("Thing:cancer")

    def test_empty_is_empty(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""

    def test_prefixless_id_survives(self):
        """타입 접두사가 없는 id 도 있다 — 통째로 날리면 안 된다."""
        assert normalize_name("유방암") == normalize_name("Disease:유방암")

    def test_distinct_names_stay_distinct(self):
        """정규화가 너무 세면 다른 개념이 합쳐진다 — 과합침이 미합침보다 위험."""
        assert normalize_name("Disease:위암") != normalize_name("Disease:간암")


class TestIdentityKeys:
    def test_includes_full_and_outside_parens(self):
        keys = identity_keys("InsuranceTerm:납입최고 (독촉 )")
        assert normalize_name("납입최고") in keys

    def test_includes_inside_parens(self):
        """`C50( 유방의 악성 신생물 )` 이 `유방의 악성 신생물` 과 만나는 지점.

        괄호 안을 버리면 코드만 남아 이름과 영영 못 만난다. 괄호는 문서에서
        '같은 것의 다른 표기'를 적는 자리라 양쪽 다 신원 후보로 둔다."""
        keys = identity_keys("Disease:C50( 유방의 악성 신생물 )")
        assert normalize_name("유방의 악성 신생물") in keys

    def test_no_parens_gives_single_key(self):
        assert identity_keys("Disease:위암") == {normalize_name("위암")}

    def test_empty_id_gives_no_keys(self):
        assert identity_keys("") == set()

    def test_short_fragments_are_dropped(self):
        """1글자 조각으로 묶으면 무관한 노드가 무더기로 붙는다."""
        keys = identity_keys("Thing:암(A)")
        assert normalize_name("a") not in keys


# ── 중복 군집 ────────────────────────────────────────────────────────

class TestDuplicateClusters:
    def test_finds_whitespace_variants(self):
        ids = ["InsuranceTerm:제6조(보험금의 지급사유 )",
               "InsuranceTerm:제6조(보험금의 지급사유)"]
        assert [sorted(c) for c in duplicate_clusters(ids)] == [sorted(ids)]

    def test_finds_paren_annotation_variant(self):
        ids = ["Disease:유방의 악성 신생물", "Disease:C50( 유방의 악성 신생물 )"]
        clusters = duplicate_clusters(ids)
        assert len(clusters) == 1 and set(clusters[0]) == set(ids)

    def test_finds_cross_type_same_name(self):
        """타입이 달라도 이름이 같으면 사람이 볼 후보다 (Disease:장해 vs Term:장해)."""
        ids = ["InsuranceTerm:장해", "Disease:장해"]
        assert len(duplicate_clusters(ids)) == 1

    def test_transitive_merge(self):
        """A~B, B~C 면 한 군집이어야 한다 — 쪼개지면 사람이 두 번 판단한다."""
        ids = ["Term:납입최고", "Term:납입최고 (독촉 )", "Term:납입최고(독촉)"]
        assert len(duplicate_clusters(ids)) == 1
        assert len(duplicate_clusters(ids)[0]) == 3

    def test_singletons_are_not_clusters(self):
        assert duplicate_clusters(["Disease:위암", "Disease:간암"]) == []

    def test_empty_input(self):
        assert duplicate_clusters([]) == []

    def test_duplicate_ids_are_not_a_cluster(self):
        """같은 id 가 두 번 들어와도 '중복 노드'가 아니다 (노드는 하나다)."""
        assert duplicate_clusters(["Disease:위암", "Disease:위암"]) == []

    def test_deterministic_order(self):
        ids = ["Term:b", "Term:b ", "Term:a", "Term:a "]
        assert duplicate_clusters(ids) == duplicate_clusters(list(reversed(ids)))


# ── 근거 공백 ────────────────────────────────────────────────────────

class TestEvidenceGaps:
    def test_orphan_node_has_no_chunk(self):
        chunks = [_C("c1", "본문", ["Disease:위암"])]
        gaps = evidence_gaps(["Disease:위암", "Disease:간암"], chunks)
        assert gaps["orphan_nodes"] == ["Disease:간암"]

    def test_unlinked_chunk_produced_no_node(self):
        chunks = [_C("c1", "본문" * 200, ["Disease:위암"]), _C("c2", "본문" * 200, [])]
        gaps = evidence_gaps(["Disease:위암"], chunks)
        assert gaps["unlinked_chunks"] == ["c2"]

    def test_short_unlinked_chunk_is_separated(self):
        """머리말·페이지번호 같은 짧은 조각까지 '추출 실패'로 세면 비율이 거짓말한다."""
        chunks = [_C("c1", "짧음", []), _C("c2", "본" * 300, [])]
        gaps = evidence_gaps([], chunks, min_chunk_len=200)
        assert gaps["unlinked_chunks"] == ["c2"]
        assert gaps["unlinked_trivial"] == ["c1"]

    def test_counts_are_reported(self):
        chunks = [_C("c1", "본" * 300, ["N:a"]), _C("c2", "본" * 300, [])]
        gaps = evidence_gaps(["N:a", "N:b"], chunks)
        assert gaps["nodes"] == 2 and gaps["chunks"] == 2
        assert gaps["linked_chunks"] == 1

    def test_node_referenced_by_chunk_but_absent_from_graph(self):
        """청크는 가리키는데 그래프엔 없는 노드 — 삭제/개명의 흔적, 끊긴 인용."""
        chunks = [_C("c1", "본" * 300, ["N:ghost"])]
        gaps = evidence_gaps(["N:a"], chunks)
        assert gaps["dangling_node_refs"] == ["N:ghost"]

    def test_empty_inputs_do_not_crash(self):
        gaps = evidence_gaps([], [])
        assert gaps["nodes"] == 0 and gaps["orphan_nodes"] == []


# ── 종합 리포트 ──────────────────────────────────────────────────────

class TestHealthReport:
    def _fixture(self):
        chunks = [_C("c1", "본" * 300, ["Term:납입최고"]),
                  _C("c2", "본" * 300, [])]
        nodes = ["Term:납입최고", "Term:납입최고 (독촉 )", "Term:고아"]
        return nodes, chunks

    def test_reports_all_three_signals(self):
        nodes, chunks = self._fixture()
        rep = health_report(nodes, chunks)
        assert rep["orphan_node_count"] == 2          # 납입최고(독촉), 고아
        assert rep["unlinked_chunk_count"] == 1
        assert rep["duplicate_cluster_count"] == 1

    def test_extraction_coverage_is_a_ratio(self):
        nodes, chunks = self._fixture()
        rep = health_report(nodes, chunks)
        assert rep["extraction_coverage"] == pytest.approx(0.5)

    def test_coverage_is_none_when_no_chunks(self):
        """0/0 을 100% 로 보고하면 '건강함'으로 읽힌다 — 모르는 건 모른다고."""
        assert health_report(["N:a"], [])["extraction_coverage"] is None

    def test_samples_are_capped_but_counts_are_not(self):
        chunks = [_C(f"c{i}", "본" * 300, []) for i in range(40)]
        rep = health_report([], chunks, sample=5)
        assert rep["unlinked_chunk_count"] == 40
        assert len(rep["unlinked_chunk_sample"]) == 5

    def test_duplicate_clusters_are_included_for_review(self):
        nodes, chunks = self._fixture()
        rep = health_report(nodes, chunks)
        assert any(len(c) == 2 for c in rep["duplicate_clusters"])

    def test_report_is_json_serializable(self):
        import json
        nodes, chunks = self._fixture()
        json.dumps(health_report(nodes, chunks), ensure_ascii=False)
