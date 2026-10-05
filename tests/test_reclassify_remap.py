"""
재빌드 부활 통제 — 묘비의 자매: 재지도(reclassify registry). 로드맵 4 P-4.

재분류(P-2)는 묘비를 남기지 않는다 — 재분류는 "타입이 틀렸다"지 "개체가
틀렸다"가 아니고, 묘비면 재빌드에서 근거가 통째로 버려진다. 대신 **재지도**:
LLM 이 다음 재빌드에서 같은 개체를 옛 타입으로 다시 추출하면, 차단이 아니라
새 id 로 **보강**한다 (근거 합집합 — 옛 타입 중복이 되살아나지 않는다).

계약:
- 재지도는 reclassify 감사 이벤트에서 파생된다 (로그가 원본 — replay 복원).
- 체인 전이 (A→B, B→C ⇒ A→C) + 사이클 가드.
- 나중 판정이 이긴다: 옛 id 의 이후 reject 는 재지도를 걷어낸다.
- 파이프라인 3곳(_merge 개체 · records 본체 · records 관계 타깃) 모두 재지도.
- 재지도 결과가 묘비면 묘비가 이긴다 (재지도 후 검사 순서).
"""

import asyncio
import json

import pytest

from ontology.builder import BuilderSchema, OntologyBuilder
from ontology.core.chunk_store import ChunkStore, reset_chunk_stores
from ontology.core.review_store import ReviewStore, reset_review_stores


def run(coro):
    return asyncio.run(coro) \
        if False else asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_singletons():
    reset_chunk_stores()
    reset_review_stores()
    yield
    reset_chunk_stores()
    reset_review_stores()


def _reclassify(store, old, new, actor="tester"):
    """P-2 rename_node 가 남기는 것과 같은 모양의 감사 이벤트."""
    store.record(action="reclassify", node_id=new,
                 before={"old_id": old},
                 after={"old_id": old, "new_id": new, "new_type": new.split(":")[0]},
                 actor=actor)


# ─── ReviewStore: 재지도 파생 상태 ───────────────────────────────────


class TestReclassifyRegistry:
    @pytest.fixture
    def store(self, tmp_path):
        return ReviewStore(namespace="t", path=tmp_path / "r.jsonl")

    def test_maps_old_to_new(self, store):
        _reclassify(store, "InsuranceTerm:제6조", "Clause:제6조")
        assert store.reclassify_target("InsuranceTerm:제6조") == "Clause:제6조"
        assert store.reclassify_target("Clause:제6조") is None
        assert store.reclassify_target("없는것") is None

    def test_replay_restores_registry(self, tmp_path):
        s1 = ReviewStore(namespace="t", path=tmp_path / "r.jsonl")
        _reclassify(s1, "A:x", "B:x")
        s2 = ReviewStore(namespace="t", path=tmp_path / "r.jsonl")
        s2.load_from_disk()
        assert s2.reclassify_target("A:x") == "B:x"

    def test_chain_transitivity(self, store):
        _reclassify(store, "A:x", "B:x")
        _reclassify(store, "B:x", "C:x")
        assert store.reclassify_target("A:x") == "C:x"

    def test_cycle_guard_terminates(self, store):
        # 정상 경로로는 못 만들지만(타깃 존재 시 rename 거부) 로그 손상·수동
        # 편집에 대비한다 — 무한 루프는 빌드 전체를 죽인다.
        _reclassify(store, "A:x", "B:x")
        _reclassify(store, "B:x", "A:x")
        assert store.reclassify_target("A:x") == "B:x"  # 한 바퀴에서 멈춘다

    def test_later_reject_wins_over_remap(self, store):
        """나중 판정이 이긴다 — 재분류 후 옛 id 를 거절하면 재지도가 걷힌다."""
        _reclassify(store, "A:x", "B:x")
        store.reject("A:x", reason="역시 오추출")
        assert store.reclassify_target("A:x") is None
        assert store.is_rejected("A:x")

    def test_malformed_event_is_ignored(self, store):
        store.record(action="reclassify", node_id="B:x", after={})  # 필드 없음
        assert store.reclassify_target("B:x") is None
        assert store.reclassified_map() == {}


# ─── 파이프라인 관통 — 재추출이 새 id 의 보강이 된다 ─────────────────


SAMPLE_MD = """## 제1조 청약철회
계약자는 15일 이내에 청약을 철회할 수 있습니다. 금융소비자 보호에 관한 법률을 따릅니다.
"""

SCHEMA = BuilderSchema(
    node_types=["Clause", "Regulation"],
    predicates={"citesRegulation": ("Clause", "Regulation")})


def fake_llm(prompt: str) -> str:
    return json.dumps({
        "entities": [
            {"name": "청약철회", "type": "Clause", "attrs": {}},
            {"name": "금융소비자 보호에 관한 법률", "type": "Regulation", "attrs": {}},
        ],
        "relations": [
            {"subject": "청약철회", "predicate": "citesRegulation",
             "object": "금융소비자 보호에 관한 법률"},
        ],
    })


@pytest.fixture
def build_env(tmp_path):
    from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine

    kg = KnowledgeGraphEngine(fast_mode=True, namespace="remap_build")
    review = ReviewStore(namespace="remap_build", path=tmp_path / "reviews.jsonl")
    chunks = ChunkStore(namespace="remap_build", path=tmp_path / "chunks.jsonl")
    builder = OntologyBuilder(schema=SCHEMA, kg=kg, llm_fn=fake_llm,
                              chunk_store=chunks, review_store=review,
                              auto_save=False)
    return builder, kg, review


class TestRemapThroughPipeline:
    def test_reextraction_reinforces_new_id(self, build_env):
        """재분류 후 재빌드: 옛 타입이 부활하지 않고 새 id 가 보강된다.
        관계의 주어도 새 id 로 걸린다."""
        builder, kg, review = build_env
        run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert "Clause:청약철회" in kg.graph

        # P-2 가 하는 일: 개명 + 감사 (여기서는 그래프 쪽을 직접 흉내)
        _reclassify(review, "Clause:청약철회", "Article:청약철회")
        attrs = dict(kg.graph.nodes["Clause:청약철회"])
        kg.graph.remove_node("Clause:청약철회")
        kg.graph.add_node("Article:청약철회", **{**attrs, "type": "Article"})

        report = run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert "Clause:청약철회" not in kg.graph          # 부활 없음
        assert "Article:청약철회" in kg.graph
        assert kg.graph.nodes["Article:청약철회"]["type"] == "Article"
        assert report.entities_remapped >= 1
        edges = [(s, t, d.get("predicate"))
                 for s, t, d in kg.graph.out_edges("Article:청약철회", data=True)]
        assert ("Article:청약철회", "Regulation:금융소비자 보호에 관한 법률",
                "citesRegulation") in edges

    def test_remap_then_tombstone_wins(self, build_env):
        """재지도의 종착지가 묘비면 묘비가 이긴다 (재지도 → 묘비 검사 순서)."""
        builder, kg, review = build_env
        _reclassify(review, "Clause:청약철회", "Article:청약철회")
        review.reject("Article:청약철회", reason="이것도 오추출")

        report = run(builder.build_from_text(SAMPLE_MD, source="약관.md"))
        assert "Clause:청약철회" not in kg.graph
        assert "Article:청약철회" not in kg.graph
        assert report.entities_rejected >= 1

    def test_records_path_remaps(self, build_env):
        builder, kg, review = build_env
        _reclassify(review, "Site:숭례문", "Landmark:숭례문")
        report = run(builder.build_from_records(
            [{"이름": "숭례문"}], {"node_type": "Site", "name_field": "이름"},
            source="w"))
        assert "Site:숭례문" not in kg.graph
        assert "Landmark:숭례문" in kg.graph
        assert kg.graph.nodes["Landmark:숭례문"]["type"] == "Landmark"
        assert report.entities_remapped == 1

    def test_records_relation_target_remaps(self, build_env):
        builder, kg, review = build_env
        _reclassify(review, "City:서울", "Capital:서울")
        run(builder.build_from_records(
            [{"이름": "숭례문", "도시": "서울"}],
            {"node_type": "Site", "name_field": "이름",
             "relations": [{"field": "도시", "predicate": "locatedIn",
                            "target_type": "City"}]},
            source="w"))
        assert "City:서울" not in kg.graph
        assert "Capital:서울" in kg.graph
        edges = [(s, t) for s, t, d in kg.graph.edges(data=True)
                 if d.get("predicate") == "locatedIn"]
        assert ("Site:숭례문", "Capital:서울") in edges
