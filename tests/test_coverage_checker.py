"""
커버리지 에이전트 (Coverage Checker) — 추출이 놓친 개체 탐지.

③번 검수 보조 에이전트. ②일관성 감시가 "그래프 안의 모순"(결정 가능 →
LLM 0콜)을 본다면, 커버리지는 "원문에는 있는데 그래프에 없는 것"을 본다 —
재현율(recall) 판단이라 LLM 이 필요하다 (evidence_checker 와 같은 주입
패턴, gemini-3.5-flash 기본).

고정하는 계약 (①근거대조와 같은 급의 LLM 출력 불신):
1. **원문에 문자 그대로 없는 후보는 버린다** — 공백 정규화 비교
   (evidence_checker._quote_in_chunks 재사용). 원문 밖 개체를 "놓쳤다"고
   보고하면 커버리지가 아니라 환각 주입이다.
2. 이미 아는 개체(이름·별칭, 대소문자/공백 무시)는 버린다 — 있는 것을
   "놓쳤다"고 하면 검수자가 도구를 불신하게 된다.
3. 허용 타입 밖의 후보는 버린다 — 스키마 열지 않기.
4. 빈 원문이면 LLM 을 부르지 않는다 / LLM 실패는 [] (never raise).
5. 발견(gap)은 review_store 감사 로그에 **남긴다** (action=coverage_gap) —
   일관성 findings 와 달리 재실행 비용이 LLM 이라 비싸고, "무엇을 놓쳤었나"
   자체가 사건이다.
"""

import asyncio
import json
import uuid

import pytest

from ontology.builder.models import Chunk
from ontology.core.chunk_store import ChunkStore, StoredChunk, reset_chunk_stores
from ontology.core.coverage_checker import (
    CoverageChecker,
    build_coverage_prompt,
    parse_coverage,
)
from ontology.core.review_store import ReviewStore, reset_review_stores


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean():
    reset_chunk_stores()
    reset_review_stores()
    yield
    reset_chunk_stores()
    reset_review_stores()


CHUNK_TEXT = ("청약철회권은 보험증권을 받은 날부터 15일 이내에 행사할 수 있다. "
              "암진단비는 최초 1회에 한하여 지급하며 ISBN 규격을 따른다.")
KNOWN = ["청약철회권"]
TYPES = ["Concept", "Clause"]


def stored_chunk(text=CHUNK_TEXT):
    return StoredChunk(chunk_id="c1", text=text, source="약관.md")


# ─── 1. 프롬프트 (grounded) ──────────────────────────────────────────

class TestPrompt:
    def test_prompt_contains_chunk_and_known_names(self):
        prompt = build_coverage_prompt(CHUNK_TEXT, KNOWN, TYPES)
        assert "암진단비" in prompt          # 원문이 통째로 들어간다
        assert "청약철회권" in prompt        # 기지 개체 목록도 들어간다
        assert "Concept" in prompt

    def test_prompt_demands_literal_and_new_only(self):
        """원문에 문자 그대로 + 기지 목록에 없는 것만 — 이 두 지시가
        없으면 LLM 은 의역·기지 개체를 섞어 낸다."""
        prompt = build_coverage_prompt(CHUNK_TEXT, KNOWN, TYPES)
        assert "그대로" in prompt
        assert "제외" in prompt or "없는" in prompt


# ─── 2. 검증 (LLM 출력 불신 — 순수 함수) ─────────────────────────────

def raw_candidates(*cands):
    return json.dumps({"candidates": list(cands)}, ensure_ascii=False)


class TestParseCoverage:
    def test_valid_new_candidate_is_kept(self):
        raw = raw_candidates({"name": "암진단비", "type": "Concept"})
        assert parse_coverage(raw, CHUNK_TEXT, KNOWN, TYPES) == \
            [{"name": "암진단비", "type": "Concept"}]

    def test_hallucinated_candidate_is_dropped(self):
        """계약 1 — 원문에 없는 개체를 '놓쳤다'고 보고하면 환각 주입이다."""
        raw = raw_candidates({"name": "화성기지조항", "type": "Concept"})
        assert parse_coverage(raw, CHUNK_TEXT, KNOWN, TYPES) == []

    def test_already_known_is_dropped_case_and_ws_insensitive(self):
        """계약 2 — 기지 개체 비교는 대소문자/공백 무시.
        (LLM 은 'isbn' 을 'ISBN' 으로, 공백을 붙여 되돌려주는 일이 잦다)"""
        raw = raw_candidates({"name": "ISBN", "type": "Concept"})
        assert parse_coverage(raw, CHUNK_TEXT, ["  isbn  "], TYPES) == []

    def test_unknown_type_is_dropped(self):
        raw = raw_candidates({"name": "암진단비", "type": "Alien"})
        assert parse_coverage(raw, CHUNK_TEXT, KNOWN, TYPES) == []

    def test_violations_drop_per_candidate_not_whole_batch(self):
        """한 후보의 위반이 배치 전체를 죽이면 정상 후보까지 잃는다."""
        raw = raw_candidates({"name": "화성기지조항", "type": "Concept"},
                             {"name": "암진단비", "type": "Concept"})
        assert parse_coverage(raw, CHUNK_TEXT, KNOWN, TYPES) == \
            [{"name": "암진단비", "type": "Concept"}]

    def test_garbage_returns_empty(self):
        assert parse_coverage("json 아님", CHUNK_TEXT, KNOWN, TYPES) == []


# ─── 3. 체커 흐름 ────────────────────────────────────────────────────

def gap_llm(prompt: str) -> str:
    return raw_candidates({"name": "암진단비", "type": "Concept"})


class TestCheckerFlow:
    def test_finding_carries_chunk_provenance(self):
        """gap 은 어느 청크·어느 출처에서 나왔는지 들고 나온다 —
        provenance 없는 발견은 검수자가 확인할 수 없다."""
        checker = CoverageChecker(llm_fn=gap_llm)
        gaps = run(checker.check_chunk(stored_chunk(), KNOWN, TYPES))
        assert gaps == [{"name": "암진단비", "type": "Concept",
                         "chunk_id": "c1", "source": "약관.md",
                         "quote": "암진단비"}]

    def test_empty_chunk_text_means_no_llm_call(self):
        """계약 4 — 원문 없이 물으면 지어낸 후보가 돌아온다 (0콜)."""
        calls = {"n": 0}

        def counting(prompt):
            calls["n"] += 1
            return gap_llm(prompt)

        checker = CoverageChecker(llm_fn=counting)
        gaps = run(checker.check_chunk(stored_chunk(text="  "), KNOWN, TYPES))
        assert calls["n"] == 0
        assert gaps == []

    def test_llm_failure_returns_empty_not_raise(self):
        def broken(prompt):
            raise RuntimeError("LLM down")

        checker = CoverageChecker(llm_fn=broken)
        assert run(checker.check_chunk(stored_chunk(), KNOWN, TYPES)) == []


# ─── 4. 서비스 + API 관통 ────────────────────────────────────────────

def service_llm(prompt: str) -> str:
    # 기지 개체(청약철회)와 미기지 개체(암진단비)를 함께 돌려준다 —
    # 서비스가 기지 개체를 걸러내는지 관통 검증하기 위해서다.
    return raw_candidates({"name": "청약철회", "type": "Concept"},
                          {"name": "암진단비", "type": "Concept"})


@pytest.fixture
def cov_ns(tmp_path):
    from ontology.engines.knowledge_graph_clean import (KnowledgeGraphEngine,
                                                        _kg_instances)
    import ontology.core.chunk_store as cs
    import ontology.core.review_store as rs
    from ontology.server.service import OntologyBuilderService

    ns = f"covns_{uuid.uuid4().hex[:8]}"
    engine = KnowledgeGraphEngine(fast_mode=True, namespace=ns)
    _kg_instances[ns] = engine
    # 그래프에는 청약철회만 있다 — 암진단비가 "놓친 것"이다
    engine.graph.add_node("Concept:청약철회", name="청약철회", type="Concept",
                          source="약관.md")

    store = ChunkStore(namespace=ns, path=tmp_path / "c.jsonl")
    store.add(Chunk(text="청약철회권은 15일 이내에 행사할 수 있다. "
                         "암진단비는 최초 1회에 한하여 지급한다.",
                    source="약관.md", index=0), ["Concept:청약철회"])
    store.add(Chunk(text="   ", source="약관.md", index=1))  # 빈 청크 — skip 대상
    cs._stores[ns] = store
    rs._stores[ns] = ReviewStore(namespace=ns, path=tmp_path / "r.jsonl")

    service = OntologyBuilderService(data_dir=tmp_path / "ds",
                                     llm_fn=service_llm)
    yield service, ns
    _kg_instances.pop(ns, None)


class TestServiceCoverage:
    def test_gap_is_reported_and_known_entity_is_not(self, cov_ns):
        """관통 계약 — 그래프에 이미 있는 개체(청약철회)는 LLM 이 후보로
        내놔도 gap 이 아니다. 놓친 것(암진단비)만 보고된다."""
        service, ns = cov_ns
        result = run(service.check_coverage(ns))
        names = [g["name"] for g in result["gaps"]]
        assert names == ["암진단비"]
        assert result["namespace"] == ns

    def test_empty_chunks_are_skipped_from_count(self, cov_ns):
        """chunks_checked 는 LLM 을 실제로 부른 청크 수 — 비용의 단위다."""
        service, ns = cov_ns
        result = run(service.check_coverage(ns))
        assert result["chunks_checked"] == 1  # 빈 청크는 세지 않는다

    def test_gaps_are_recorded_as_audit_events(self, cov_ns):
        """계약 5 — gap 발견은 사건이고 재실행 비용(LLM)이 비싸다.
        coverage_gap 이벤트로 감사 로그에 남는다 (판정 상태는 무영향)."""
        service, ns = cov_ns
        from ontology.core.review_store import get_review_store
        run(service.check_coverage(ns))
        reviews = get_review_store(ns)
        events = [e for e in reviews.history() if e["action"] == "coverage_gap"]
        assert len(events) == 1
        event = events[0]
        assert event["node_id"] == "Concept:암진단비"
        assert event["actor"] == "coverage_checker"
        assert event["source"] == "약관.md"
        assert event["after"]["name"] == "암진단비"
        # gap 은 추천도 판정도 아니다 — 검수 상태를 건드리지 않는다
        assert reviews.is_rejected("Concept:암진단비") is False
        assert reviews.is_confirmed("Concept:암진단비") is False

    def test_limit_caps_llm_cost(self, cov_ns, tmp_path):
        """청크당 LLM 1콜 — limit 이 비용 상한이다."""
        service, ns = cov_ns
        import ontology.core.chunk_store as cs
        store = cs._stores[ns]
        for i in range(2, 6):
            store.add(Chunk(text=f"제{i}조 암진단비는 최초 1회에 한하여 지급한다.",
                            source="약관.md", index=i))
        result = run(service.check_coverage(ns, limit=2))
        assert result["chunks_checked"] == 2

    def test_route_happy_path(self, cov_ns):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ontology.server import router as server_router

        service, ns = cov_ns
        app = FastAPI()
        app.include_router(server_router.router, prefix="/api/v1/ontology")
        app.dependency_overrides[server_router.get_ontology_service] = \
            lambda: service
        client = TestClient(app)

        response = client.post(
            f"/api/v1/ontology/graphs/{ns}/review/coverage",
            json={"limit": 5})
        assert response.status_code == 200
        body = response.json()
        assert body["chunks_checked"] == 1
        assert [g["name"] for g in body["gaps"]] == ["암진단비"]
