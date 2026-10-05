"""품질 게이트 배선 — 인제스트에서 근거 아닌 조각을 trust 강등한다.

**게이트가 기본 off 인 이유**: 임계값(TOC_RATIO 0.03 · GARBLED_RATIO 0.35)은
데이터 관찰에서 고른 값이고 **골든셋으로 측정된 값이 아니다**. 측정 없이 기본
동작을 바꾸면 이 저장소가 스스로 금지한 짓("감으로 정한 상수를 강제")을 하는
것이다. 청크 단위 골든셋으로 전후를 잴 수 있게 된 뒤에 기본값을 논한다.

**강등이지 삭제가 아니다**: guard_attrs_by_trust 가 "요약서만 아는 사실은
유효하다 — 금지는 덮어쓰기지 기여가 아니다"라고 규정했다. 목차·잡음 청크도
저장은 되고, 등급만 낮아져 사실을 덮어쓰지 못한다.
"""
import pytest

from ontology.builder.chunk_quality import DEMOTED_TRUST
from ontology.builder.models import BuildReport
from ontology.builder.pipeline import OntologyBuilder


class _FakeChunk:
    def __init__(self, text, source="s.txt", index=0):
        self.text = text
        self.source = source
        self.index = index
        self.section = ""
        self.char_start = 0
        self.char_end = len(text)


PROSE = ("보험계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 "
         "있습니다. 다만 진단계약의 경우에는 예외가 적용됩니다.")
TOC = ("제1장 총칙 ……………………………… 3\n"
       "제2장 보험금의 지급 …………………… 12\n"
       "제3장 계약의 성립과 유지 ………………… 25\n")


def _builder(**kw):
    """LLM·KG 없이 게이트 로직만 보려고 최소 구성으로 만든다.
    schema 는 첫 위치 인자이며 None = auto 가 허용된다(생성자 계약)."""
    class _KG:
        namespace = "t"
    return OntologyBuilder(None, kg=_KG(), store_chunks=False, **kw)


class TestGateDefaultOff:
    def test_gate_off_by_default(self):
        assert _builder().quality_gate is False

    def test_gate_off_leaves_trust_untouched(self):
        b = _builder()
        report = BuildReport(namespace="t")
        assert b.resolve_chunk_trust(_FakeChunk(TOC), "authoritative", report) \
            == "authoritative"
        assert report.chunks_demoted == 0

    def test_env_can_enable(self, monkeypatch):
        monkeypatch.setenv("ONTOLOGY_CHUNK_QUALITY_GATE", "true")
        assert _builder().quality_gate is True

    def test_explicit_param_beats_env(self, monkeypatch):
        monkeypatch.setenv("ONTOLOGY_CHUNK_QUALITY_GATE", "true")
        assert _builder(quality_gate=False).quality_gate is False


class TestGateOn:
    def test_prose_keeps_incoming_trust(self):
        b = _builder(quality_gate=True)
        report = BuildReport(namespace="t")
        assert b.resolve_chunk_trust(_FakeChunk(PROSE), "authoritative", report) \
            == "authoritative"
        assert report.chunks_demoted == 0

    def test_toc_is_demoted_and_counted(self):
        b = _builder(quality_gate=True)
        report = BuildReport(namespace="t")
        assert b.resolve_chunk_trust(_FakeChunk(TOC), "authoritative", report) \
            == DEMOTED_TRUST
        assert report.chunks_demoted == 1
        assert report.demote_reasons == {"toc": 1}

    def test_reasons_accumulate_per_reason(self):
        b = _builder(quality_gate=True)
        report = BuildReport(namespace="t")
        garbled = ("a) 은 olnt zl 오 Pal 중 e a 1 l m x n o p q r "
                   "s t u v w z l o 은 중 오 a b c d e f g h")
        b.resolve_chunk_trust(_FakeChunk(TOC), "", report)
        b.resolve_chunk_trust(_FakeChunk(TOC), "", report)
        b.resolve_chunk_trust(_FakeChunk(garbled), "", report)
        assert report.chunks_demoted == 3
        assert report.demote_reasons == {"toc": 2, "garbled": 1}

    def test_already_demoted_not_double_counted(self):
        """들어온 등급이 이미 summary 면 강등이 아니다 — 집계가 부풀면 안 된다."""
        b = _builder(quality_gate=True)
        report = BuildReport(namespace="t")
        assert b.resolve_chunk_trust(_FakeChunk(TOC), DEMOTED_TRUST, report) \
            == DEMOTED_TRUST
        assert report.chunks_demoted == 0

    def test_unspecified_trust_can_be_demoted(self):
        """미지정("")은 rank 1 이므로 summary(0)로 내려가는 것이 강등이다."""
        b = _builder(quality_gate=True)
        report = BuildReport(namespace="t")
        assert b.resolve_chunk_trust(_FakeChunk(TOC), "", report) == DEMOTED_TRUST
        assert report.chunks_demoted == 1

    def test_assess_failure_does_not_break_ingest(self):
        """판정이 죽어도 인제스트는 계속돼야 한다 — 들어온 등급을 그대로 쓴다."""
        b = _builder(quality_gate=True)
        report = BuildReport(namespace="t")

        class _Broken:
            source = "s"
            index = 0
            @property
            def text(self):
                raise RuntimeError("boom")

        assert b.resolve_chunk_trust(_Broken(), "authoritative", report) \
            == "authoritative"
        assert report.chunks_demoted == 0


class TestStoreChunkWiring:
    """실제 저장 경로에 게이트가 걸리는가.

    exhaustive 모드는 청크를 _store_chunk 로만 저장한다(topic 모드처럼 미리
    전량 저장하지 않는다) — 여기 배선이 빠지면 게이트가 기본 경로에서 무력하다.
    ChunkStore.add 는 "trust 는 비어 있을 때만 채운다"(첫 쓰기 우선)이므로
    한 번 강등되면 나중 쓰기가 되돌리지 못한다.
    """

    class _SpyStore:
        def __init__(self):
            self.calls = []

        def add(self, chunk, node_ids=(), trust="", **kw):
            self.calls.append(trust)
            return "cid"

    def _wired(self, **kw):
        class _KG:
            namespace = "t"
        b = OntologyBuilder(None, kg=_KG(), store_chunks=True,
                            chunk_store=self._SpyStore(), **kw)
        return b, b.chunk_store

    def test_gate_on_demotes_in_store_path(self):
        b, store = self._wired(quality_gate=True)
        report = BuildReport(namespace="t")
        b._store_chunk(_FakeChunk(TOC), [], trust="authoritative", report=report)
        assert store.calls == [DEMOTED_TRUST]
        assert report.chunks_demoted == 1

    def test_gate_off_passes_trust_through(self):
        b, store = self._wired(quality_gate=False)
        report = BuildReport(namespace="t")
        b._store_chunk(_FakeChunk(TOC), [], trust="authoritative", report=report)
        assert store.calls == ["authoritative"]
        assert report.chunks_demoted == 0

    def test_prose_unaffected_in_store_path(self):
        b, store = self._wired(quality_gate=True)
        report = BuildReport(namespace="t")
        b._store_chunk(_FakeChunk(PROSE), [], trust="authoritative", report=report)
        assert store.calls == ["authoritative"]

    def test_without_report_gate_is_skipped_not_silent_demote(self):
        """report 가 없으면 강등을 집계할 곳이 없다 — 조용히 강등하지 않는다."""
        b, store = self._wired(quality_gate=True)
        b._store_chunk(_FakeChunk(TOC), [], trust="authoritative")
        assert store.calls == ["authoritative"]


class TestReportFields:
    def test_report_has_demote_fields_defaulting_empty(self):
        report = BuildReport(namespace="t")
        assert report.chunks_demoted == 0
        assert report.demote_reasons == {}

    def test_reports_are_independent(self):
        """가변 기본값 공유 사고 방지 — dataclass field(default_factory) 계약."""
        a, b = BuildReport(namespace="a"), BuildReport(namespace="b")
        a.demote_reasons["toc"] = 1
        assert b.demote_reasons == {}
