"""커버리지 검사를 **추출이 실패한 청크**로 겨냥한다.

실측 동기(graph_health): ins_cancer_demo 의 청크 92 중 65 는 노드를 하나도 내지
못했고, 그중 30 은 본문이 200자 이상이었다 — 머리말이 아니라 실제 조문이다.
그런데 check_coverage 는 chunks.all() 을 **순서대로** 훑어, 이미 노드가 잘 나온
청크에 LLM 콜을 쓴다. 청크당 1콜이고 상한이 30 이므로, 정작 비어 있는 청크에는
예산이 닿지 않는다.

`only_unlinked=True` 는 그 예산을 공백으로 돌린다. 짧은 조각(페이지번호·머리말)은
건너뛴다 — 거기서 개체가 안 나온 것은 결함이 아니라 정상이고, 콜만 태운다.

이 테스트가 고정하는 것은 **어디에 LLM 콜을 쓰는가**다. 기본값은 바꾸지 않는다:
기존 호출자가 조용히 다른 대상을 검사하게 되면 그게 회귀다.
"""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ontology.core.chunk_index import reset_chunk_indices
from ontology.core.chunk_store import reset_chunk_stores
from ontology.core.review_store import reset_review_stores
from ontology.core.search_qa import reset_golden_sets

API = "/api/v1/ontology"
LONG = "제28조 【보험료의 납입이 연체되는 경우 납입최고와 계약의 해지】 " + "본문 " * 80
SHORT = "제2조 【용어의 정의】"


@pytest.fixture(autouse=True)
def _clean():
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()
    yield
    reset_chunk_stores(); reset_review_stores()
    reset_chunk_indices(); reset_golden_sets()


class _Calls:
    """LLM 콜을 세는 가짜 — 무엇을 검사했는지 본문으로 기록한다."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"missing": []}'


@pytest.fixture()
def env(tmp_path):
    from ontology.builder.models import Chunk
    from ontology.core.chunk_store import get_chunk_store
    from ontology.engines.knowledge_graph_clean import get_knowledge_graph_engine
    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService

    calls = _Calls()
    service = OntologyBuilderService(data_dir=tmp_path, llm_fn=calls)
    app = FastAPI()
    app.include_router(server_router.router, prefix=API)
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    ns = f"covns_{uuid.uuid4().hex[:8]}"

    engine = get_knowledge_graph_engine(ns)
    engine.graph.add_node("InsuranceTerm:해지", type="InsuranceTerm", name="해지")

    store = get_chunk_store(ns)
    # 0번: 노드가 나온 청크 (검사할 필요 없다)
    store.add(Chunk(text="LINKED " + LONG, source="약관.pdf", index=0,
                    char_start=0, char_end=10), node_ids=["InsuranceTerm:해지"])
    # 1번: 본문인데 노드 0 — 진짜 공백
    store.add(Chunk(text="UNLINKED " + LONG, source="약관.pdf", index=1,
                    char_start=10, char_end=20))
    # 2번: 짧은 조각 — 노드가 없는 게 정상
    store.add(Chunk(text=SHORT, source="약관.pdf", index=2,
                    char_start=20, char_end=30))
    return TestClient(app), ns, calls


def _post(client, ns, **body):
    return client.post(f"{API}/graphs/{ns}/review/coverage", json=body)


class TestDefaultUnchanged:
    def test_default_checks_linked_chunks_too(self, env):
        """기본값을 바꾸면 기존 호출자가 조용히 다른 대상을 검사한다."""
        client, ns, calls = env
        assert _post(client, ns, limit=10).status_code == 200
        assert any("LINKED" in p for p in calls.prompts)

    def test_default_flag_is_false(self, env):
        client, ns, calls = env
        body = _post(client, ns, limit=10).json()
        assert body["only_unlinked"] is False


class TestOnlyUnlinked:
    def test_skips_chunks_that_already_produced_nodes(self, env):
        client, ns, calls = env
        assert _post(client, ns, limit=10, only_unlinked=True).status_code == 200
        assert not any("LINKED " in p and "UNLINKED" not in p
                       for p in calls.prompts)

    def test_checks_the_unlinked_body_chunk(self, env):
        client, ns, calls = env
        _post(client, ns, limit=10, only_unlinked=True)
        assert any("UNLINKED" in p for p in calls.prompts)

    def test_skips_short_fragments(self, env):
        """머리말에서 개체가 안 나온 것은 결함이 아니다 — 콜만 태운다."""
        client, ns, calls = env
        _post(client, ns, limit=10, only_unlinked=True)
        assert not any(SHORT in p for p in calls.prompts)

    def test_spends_exactly_one_call_here(self, env):
        client, ns, calls = env
        _post(client, ns, limit=10, only_unlinked=True)
        assert len(calls.prompts) == 1

    def test_limit_still_caps_calls(self, env):
        client, ns, calls = env
        _post(client, ns, limit=1, only_unlinked=True)
        assert len(calls.prompts) <= 1

    def test_reports_how_many_were_eligible(self, env):
        """검사한 수만 보고하면 '다 봤다'로 읽힌다 — 남은 후보 수가 보여야 한다."""
        client, ns, _ = env
        body = _post(client, ns, limit=1, only_unlinked=True).json()
        assert body["eligible"] == 1
        assert body["chunks_checked"] == 1

    def test_min_len_is_configurable(self, env):
        """상수는 측정값이 아니다 — 호출부가 덮을 수 있어야 한다."""
        client, ns, calls = env
        _post(client, ns, limit=10, only_unlinked=True, min_len=1)
        assert any(SHORT in p for p in calls.prompts)
