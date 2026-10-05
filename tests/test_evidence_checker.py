"""
근거대조 에이전트 (Evidence Checker) — 검수 큐 사전판정.

aicoach ACP 협업 6에이전트에서 확인된 패턴을 검수에 적용한다:
**역할 하나 + grounded 도구 하나, 추천까지만, 사실은 인프라에서만.**
- 역할: 추출된 사실이 원문 청크(span)에 근거하는지 판단
- grounded 도구: ChunkStore (이 노드가 추출된 원문)
- 권한: 검수 큐에 confirm/reject **추천**을 다는 것까지. 최종 판정(묘비)은
  인간만 만든다 — 에이전트 추천은 is_rejected/is_confirmed 에 영향이 없다.

고정하는 계약:
1. **인용이 가짜면 추천도 가짜다** — evidence_quote 가 원문 청크의 실제
   부분문자열이 아니면 verdict 를 unsure 로 강등한다. 근거를 지어내는
   추천은 검수를 돕는 게 아니라 오염시킨다 (aicoach Advisor 화이트리스트
   필터와 같은 원리: LLM 출력은 검증 가능한 것만 채택).
2. 근거 청크가 없으면 **LLM 을 부르지 않는다** — 원문 없이 LLM 에게 물으면
   지어낸 판단이 돌아온다 (verdict=no_evidence, 0콜).
3. LLM 이 죽어도 raise 하지 않는다 — unsure + 이유.
4. 추천은 감사 이력에 남는다 (action=recommend) — 로그가 원본 원칙 그대로.
"""

import asyncio
import json

import pytest

from ontology.builder.models import Chunk
from ontology.core.chunk_store import ChunkStore, reset_chunk_stores
from ontology.core.evidence_checker import (
    EvidenceChecker,
    build_check_prompt,
    parse_check_verdict,
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


CHUNK_TEXTS = [
    "청약철회권은 보험증권을 받은 날부터 15일 이내에 행사할 수 있다.",
    "암진단비는 최초 1회에 한하여 지급한다.",
]

NODE = {"node_id": "Clause:청약철회", "name": "청약철회", "type": "Clause",
        "definition": "15일 이내 청약을 무르는 권리", "source": "약관.md"}


# ─── 1. 검증 (LLM 출력 불신 — 순수 함수) ─────────────────────────────

class TestParseVerdict:
    def test_confirm_with_real_quote_passes(self):
        raw = json.dumps({"verdict": "confirm", "rationale": "원문에 명시됨",
                          "evidence_quote": "15일 이내에 행사할 수 있다"},
                         ensure_ascii=False)
        rec = parse_check_verdict(raw, CHUNK_TEXTS)
        assert rec["verdict"] == "confirm"
        assert rec["evidence_quote"] == "15일 이내에 행사할 수 있다"

    def test_hallucinated_quote_downgrades_to_unsure(self):
        """계약 1 — 인용이 가짜면 추천도 가짜다."""
        raw = json.dumps({"verdict": "confirm", "rationale": "확인됨",
                          "evidence_quote": "원문 어디에도 없는 문장이다"},
                         ensure_ascii=False)
        rec = parse_check_verdict(raw, CHUNK_TEXTS)
        assert rec["verdict"] == "unsure"
        assert "인용" in rec["rationale"]  # 왜 강등됐는지 사람이 읽을 수 있다

    def test_reject_without_rationale_downgrades(self):
        """이유 없는 거절 추천으로는 인간이 판정할 수 없다."""
        raw = json.dumps({"verdict": "reject", "rationale": "",
                          "evidence_quote": "암진단비는 최초 1회에 한하여"},
                         ensure_ascii=False)
        assert parse_check_verdict(raw, CHUNK_TEXTS)["verdict"] == "unsure"

    def test_whitespace_differences_in_quote_are_tolerated(self):
        """LLM 은 공백·개행을 바꿔 인용하는 일이 잦다 — 공백 정규화 비교.
        (내용이 같은데 공백 때문에 강등되면 추천이 무의미하게 비어진다)"""
        raw = json.dumps({"verdict": "confirm", "rationale": "명시됨",
                          "evidence_quote": "15일  이내에\n행사할 수 있다"},
                         ensure_ascii=False)
        assert parse_check_verdict(raw, CHUNK_TEXTS)["verdict"] == "confirm"

    def test_unknown_verdict_becomes_unsure(self):
        raw = json.dumps({"verdict": "maybe", "rationale": "?"})
        assert parse_check_verdict(raw, CHUNK_TEXTS)["verdict"] == "unsure"

    def test_garbage_becomes_unsure(self):
        assert parse_check_verdict("json 아님", CHUNK_TEXTS)["verdict"] == "unsure"

    def test_unsure_needs_no_quote(self):
        raw = json.dumps({"verdict": "unsure", "rationale": "원문만으로 판단 불가"})
        assert parse_check_verdict(raw, CHUNK_TEXTS)["verdict"] == "unsure"


# ─── 2. 프롬프트 (grounded) ──────────────────────────────────────────

class TestPrompt:
    def test_prompt_contains_fact_and_chunks(self):
        prompt = build_check_prompt(NODE, CHUNK_TEXTS)
        assert "청약철회" in prompt
        assert "15일 이내에 행사할 수 있다" in prompt

    def test_prompt_forbids_outside_knowledge(self):
        """aicoach _QaAgent 원칙 — '원문 안에서만'."""
        prompt = build_check_prompt(NODE, CHUNK_TEXTS)
        assert "원문" in prompt and "금지" in prompt


# ─── 3. 체커 흐름 ────────────────────────────────────────────────────

def confirm_llm(prompt: str) -> str:
    return json.dumps({"verdict": "confirm", "rationale": "원문에 명시",
                       "evidence_quote": "15일 이내에 행사할 수 있다"},
                      ensure_ascii=False)


class TestCheckerFlow:
    def test_supported_fact_gets_confirm(self):
        checker = EvidenceChecker(llm_fn=confirm_llm)
        rec = run(checker.check(NODE, CHUNK_TEXTS))
        assert rec["verdict"] == "confirm"
        assert rec["evidence_quote"]

    def test_no_chunks_means_no_llm_call(self):
        """계약 2 — 원문 없이 LLM 에게 물으면 지어낸 판단이 돌아온다."""
        calls = {"n": 0}

        def counting(prompt):
            calls["n"] += 1
            return confirm_llm(prompt)

        checker = EvidenceChecker(llm_fn=counting)
        rec = run(checker.check(NODE, []))
        assert calls["n"] == 0
        assert rec["verdict"] == "no_evidence"

    def test_llm_failure_is_unsure_not_raise(self):
        def broken(prompt):
            raise RuntimeError("LLM down")

        checker = EvidenceChecker(llm_fn=broken)
        rec = run(checker.check(NODE, CHUNK_TEXTS))
        assert rec["verdict"] == "unsure"
        assert "LLM down" in rec["rationale"]


# ─── 4. ReviewStore 추천 이벤트 ──────────────────────────────────────

class TestRecommendationEvents:
    def test_recommend_and_lookup(self, tmp_path):
        store = ReviewStore(namespace="r", path=tmp_path / "r.jsonl")
        store.recommend("Clause:청약철회", verdict="confirm",
                        rationale="원문 명시", quote="15일 이내",
                        actor="evidence_checker")
        rec = store.recommendation_for("Clause:청약철회")
        assert rec["verdict"] == "confirm"
        assert rec["actor"] == "evidence_checker"

    def test_recommendation_does_not_judge(self, tmp_path):
        """핵심 권한 경계 — 추천은 판정이 아니다. 묘비도 확정도 안 만든다."""
        store = ReviewStore(namespace="r", path=tmp_path / "r.jsonl")
        store.recommend("N:1", verdict="reject", rationale="모순", quote="")
        assert store.is_rejected("N:1") is False
        assert store.is_confirmed("N:1") is False

    def test_latest_recommendation_wins(self, tmp_path):
        store = ReviewStore(namespace="r", path=tmp_path / "r.jsonl")
        store.recommend("N:1", verdict="unsure", rationale="1차")
        store.recommend("N:1", verdict="confirm", rationale="재검사", quote="q")
        assert store.recommendation_for("N:1")["verdict"] == "confirm"

    def test_recommendations_survive_replay(self, tmp_path):
        path = tmp_path / "r.jsonl"
        store = ReviewStore(namespace="r", path=path)
        store.recommend("N:1", verdict="confirm", rationale="근거", quote="q")

        restored = ReviewStore(namespace="r", path=path)
        restored.load_from_disk()
        assert restored.recommendation_for("N:1")["verdict"] == "confirm"

    def test_recommendation_appears_in_history(self, tmp_path):
        store = ReviewStore(namespace="r", path=tmp_path / "r.jsonl")
        store.recommend("N:1", verdict="confirm", rationale="근거")
        assert store.history()[0]["action"] == "recommend"

    def test_missing_recommendation_is_none(self, tmp_path):
        store = ReviewStore(namespace="r", path=tmp_path / "r.jsonl")
        assert store.recommendation_for("없음") is None


# ─── 5. 서비스 + API 관통 ────────────────────────────────────────────

def extract_llm(prompt: str) -> str:
    if "지식을 추출" in prompt:
        return json.dumps({"entities": [
            {"name": "청약철회", "type": "Concept",
             "attrs": {"definition": "15일 이내 무르는 권리"}},
            {"name": "화성기지약관", "type": "Concept",
             "attrs": {"definition": "화성 이주민 전용 조항"}}],  # 환각
            "relations": []}, ensure_ascii=False)
    if "검수 보조" in prompt or "근거" in prompt:
        # 근거대조: 원문에 있는 것만 confirm
        if "청약철회권은" in prompt and '"청약철회"' in prompt:
            return json.dumps({"verdict": "confirm", "rationale": "원문 명시",
                               "evidence_quote": "청약철회권은 보험증권을 받은 날부터"},
                              ensure_ascii=False)
        return json.dumps({"verdict": "reject",
                           "rationale": "원문 어디에도 없는 사실",
                           "evidence_quote": "청약철회권은 보험증권을 받은 날부터"},
                          ensure_ascii=False)
    return "{}"


@pytest.fixture
def built(tmp_path):
    from ontology.engines.knowledge_graph_clean import (KnowledgeGraphEngine,
                                                        _kg_instances)
    from ontology.server.service import OntologyBuilderService

    ns = "precheck_ns"
    _kg_instances[ns] = KnowledgeGraphEngine(fast_mode=True, namespace=ns)
    import ontology.core.chunk_store as cs
    import ontology.core.review_store as rs
    cs._stores[ns] = ChunkStore(namespace=ns, path=tmp_path / "c.jsonl")
    rs._stores[ns] = ReviewStore(namespace=ns, path=tmp_path / "r.jsonl")

    service = OntologyBuilderService(data_dir=tmp_path / "ds",
                                     llm_fn=extract_llm)
    saved = service.save_dataset([
        ("약관.md", "청약철회권은 보험증권을 받은 날부터 15일 이내에 행사할 수 있다.".encode())])
    job_id = service.create_job(ns)
    run(service.run_ingest(job_id, saved["dataset_id"], ns,
                           [{"filename": "약관.md", "route": "prose"}],
                           save=False, schema_mode="custom",
                           custom_schema={"node_types": ["Concept"],
                                          "predicates": {}}))
    yield service, ns
    _kg_instances.pop(ns, None)


class TestServicePrecheck:
    def test_precheck_attaches_recommendations(self, built):
        service, ns = built
        result = run(service.precheck_review_queue(ns))
        assert result["checked"] == 2
        by_node = {r["node_id"]: r for r in result["recommendations"]}
        assert by_node["Concept:청약철회"]["verdict"] == "confirm"
        assert by_node["Concept:화성기지약관"]["verdict"] == "reject"

    def test_queue_carries_recommendation_after_precheck(self, built):
        service, ns = built
        run(service.precheck_review_queue(ns))
        queue = service.get_review_queue(ns)
        by_node = {i["node_id"]: i for i in queue["items"]}
        assert by_node["Concept:청약철회"]["recommendation"]["verdict"] == "confirm"

    def test_queue_recommendation_is_none_before_precheck(self, built):
        service, ns = built
        queue = service.get_review_queue(ns)
        assert all(i["recommendation"] is None for i in queue["items"])

    def test_judged_nodes_are_not_prechecked(self, built):
        """이미 인간이 판정한 노드에 LLM 콜을 쓰지 않는다."""
        service, ns = built
        service.confirm_node(ns, "Concept:청약철회", actor="human")
        result = run(service.precheck_review_queue(ns))
        assert result["checked"] == 1  # 화성기지약관만

    def test_final_judgment_stays_human(self, built):
        """추천이 reject 여도 그래프는 그대로다 — 판정은 인간의 것."""
        service, ns = built
        from ontology.engines.knowledge_graph_clean import _kg_instances
        run(service.precheck_review_queue(ns))
        assert "Concept:화성기지약관" in _kg_instances[ns].graph
