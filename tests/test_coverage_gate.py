"""
커버리지 expectation 게이트 — B2 배선 회귀 계약.

- 게이트와 health 탭은 **같은 자**(graph_health)를 쓴다.
- 임계 미설정 = unconfigured (경고 아님 — 임계는 운영자가 명시적으로 건다).
- 경고 게이트 — 빌드/잡을 실패시키지 않는다. auto_coverage_check 는
  옵트인·기본 off + 예산 상한.
"""

import asyncio

import pytest

from ontology.builder.models import Chunk


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    import ontology.core.chunk_store as cs
    import ontology.core.coverage_expectations as ce
    import ontology.core.review_store as rs
    from ontology.core.chunk_store import reset_chunk_stores
    from ontology.core.coverage_expectations import reset_coverage_expectations
    from ontology.core.review_store import reset_review_stores
    from ontology.engines import knowledge_graph_clean as kgc
    from ontology.server.service import OntologyBuilderService

    monkeypatch.setattr(cs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(rs, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "_DEFAULT_DATA_DIR", tmp_path)
    reset_chunk_stores()
    reset_review_stores()
    reset_coverage_expectations()
    kgc._kg_instances.pop("gatens", None)

    NS = "gatens"
    service = OntologyBuilderService()
    engine = kgc.get_knowledge_graph_engine(NS)
    engine.graph.clear()
    engine.graph.add_node("T:a", type="T", name="a")
    engine.graph.add_node("T:b", type="T", name="b")   # 고아 (링크 없음)

    store = cs.get_chunk_store(NS)
    store.clear()
    c1 = store.add(Chunk(text="a 가 나오는 본문", source="w.pdf", index=0,
                         char_start=0, char_end=10), node_ids=["T:a"])
    store.add(Chunk(text="아무 개체도 없는 본문", source="w.pdf", index=1,
                    char_start=10, char_end=25))       # 미연결

    monkeypatch.setattr(service, "_namespace_exists", lambda ns: ns == NS)
    return service, NS


class TestCoverageGate:
    def test_unconfigured_reports_metrics_only(self, svc):
        service, NS = svc
        gate = service.coverage_gate(NS)
        assert gate["status"] == "unconfigured"
        # 커버리지 1/2 — health 와 같은 자
        assert gate["metrics"]["extraction_coverage"] == pytest.approx(0.5)

    def test_warn_on_breach_and_ok_on_pass(self, svc):
        service, NS = svc
        service.set_coverage_expectations_settings(
            NS, {"min_extraction_coverage": 0.6})
        gate = service.coverage_gate(NS)
        assert gate["status"] == "warn"
        assert gate["warnings"][0]["key"] == "min_extraction_coverage"
        assert gate["thresholds"] == {"min_extraction_coverage": 0.6}

        service.set_coverage_expectations_settings(
            NS, {"min_extraction_coverage": 0.4})
        assert service.coverage_gate(NS)["status"] == "ok"

    def test_settings_validate_and_audit(self, svc):
        service, NS = svc
        res = service.set_coverage_expectations_settings(NS, {"오타키": 1})
        assert res["error"] == "invalid"

        service.set_coverage_expectations_settings(
            NS, {"max_orphan_ratio": 0.1}, actor="tester")
        from ontology.core.review_store import get_review_store
        history = get_review_store(NS).history()
        assert any(h["action"] == "expectations_change" for h in history)

    def test_job_gate_is_best_effort_and_auto_off_by_default(self, svc, monkeypatch):
        """warn 이어도 auto_check 는 기본 off — LLM 0콜."""
        service, NS = svc
        service.set_coverage_expectations_settings(
            NS, {"min_extraction_coverage": 0.9})
        called = []

        async def _fake_check(namespace, limit=10, only_unlinked=False, **kw):
            called.append((namespace, limit, only_unlinked))
            return {"chunks_checked": limit, "gaps": [1, 2]}

        monkeypatch.setattr(service, "check_coverage", _fake_check)
        gate = asyncio.run(service._coverage_gate_for_job(NS))
        assert gate["status"] == "warn"
        assert gate["auto_check"] is None
        assert called == []

    def test_auto_check_optin_respects_budget(self, svc, monkeypatch):
        service, NS = svc
        service.set_coverage_expectations_settings(
            NS, {"min_extraction_coverage": 0.9,
                 "auto_coverage_check": True, "auto_coverage_limit": 7})
        called = []

        async def _fake_check(namespace, limit=10, only_unlinked=False, **kw):
            called.append((namespace, limit, only_unlinked))
            return {"chunks_checked": limit, "gaps": [1, 2, 3]}

        monkeypatch.setattr(service, "check_coverage", _fake_check)
        gate = asyncio.run(service._coverage_gate_for_job(NS))
        assert gate["auto_check"] == {"chunks_checked": 7, "gaps": 3, "limit": 7}
        assert called == [(NS, 7, True)]   # only_unlinked 고정 — 기존 예산 장치

    def test_auto_check_failure_does_not_break_gate(self, svc, monkeypatch):
        """best-effort — 자동 검사 실패가 잡을 깨뜨리면 안 된다."""
        service, NS = svc
        service.set_coverage_expectations_settings(
            NS, {"min_extraction_coverage": 0.9, "auto_coverage_check": True})

        async def _boom(*a, **kw):
            raise RuntimeError("llm down")

        monkeypatch.setattr(service, "check_coverage", _boom)
        gate = asyncio.run(service._coverage_gate_for_job(NS))
        assert gate["status"] == "warn"          # 게이트 자체는 성립
        assert "error" in gate["auto_check"]
