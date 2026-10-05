"""이미지 배선 — build_from_file 이 PDF 이미지를 청크에 실어 보내는가.

**게이트가 기본 off 인 이유**: 이미지 추출은 PDF 를 한 번 더 열고(pdfplumber),
캡션 밴드마다 crop→extract_text 를 돌린다. 인제스트 비용을 측정 없이 전 사용자에게
물릴 수 없다 — 품질 게이트(ONTOLOGY_CHUNK_QUALITY_GATE)와 같은 원칙이다.

여기서 검증하는 것은 **배관**이다: 게이트 on 이면 (a) 추출이 호출되고
(b) 캡션 청크의 meta 에 이미지가 실리고 (c) 리포트에 집계가 남는가.
추출기 자체는 test_pdf_image_extraction.py 가 실제 PDF 로 검증한다.
"""
import pytest

from ontology.builder.models import BuildReport
from ontology.builder.pipeline import OntologyBuilder

TEXT_WITH_CAPTION = (
    "1. Overview\n"
    "This project standardises the data model.\n"
    "Figure 3: System Architecture\n"
    "Each component goes through the API gateway.\n"
)
CAPTION = "Figure 3: System Architecture"


class _FakeAsset:
    """ImageAsset 최소 형태 — proxy_meta 가 읽는 속성만."""
    def __init__(self, caption=CAPTION, sha="a" * 64, page=1):
        self.source = "doc.pdf"
        self.page = page
        self.bbox = (72.0, 400.0, 472.0, 700.0)
        self.sha256 = sha
        self.width = 400
        self.height = 300
        self.caption = caption
        self.meta = {"page_height": 841.89}

    @property
    def asset_id(self):
        from ontology.builder.image_assets import asset_id_for
        return asset_id_for(self.sha256, self.source, self.page, self.bbox)


def _builder(**kw):
    class _KG:
        namespace = "t"
    return OntologyBuilder(None, kg=_KG(), store_chunks=False, **kw)


class TestGateDefault:
    def test_image_assets_off_by_default(self):
        assert _builder().image_assets is False

    def test_env_enables(self, monkeypatch):
        monkeypatch.setenv("ONTOLOGY_IMAGE_ASSETS", "true")
        assert _builder().image_assets is True

    def test_explicit_param_beats_env(self, monkeypatch):
        monkeypatch.setenv("ONTOLOGY_IMAGE_ASSETS", "true")
        assert _builder(image_assets=False).image_assets is False


class TestExtractionCall:
    def test_gate_off_does_not_extract(self, monkeypatch):
        """비용을 물리지 않는다 — 추출기를 부르지도 않아야 한다."""
        calls = []
        monkeypatch.setattr("ontology.builder.readers.extract_pdf_images",
                            lambda *a, **k: calls.append(a) or [])
        assert _builder(image_assets=False).collect_images("doc.pdf") == []
        assert calls == []

    def test_gate_on_extracts(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "ontology.builder.readers.extract_pdf_images",
            lambda path, **k: calls.append(path) or [_FakeAsset()])
        assets = _builder(image_assets=True).collect_images("doc.pdf")
        assert calls == ["doc.pdf"] and len(assets) == 1

    def test_extraction_failure_does_not_break_ingest(self, monkeypatch):
        """이미지 추출 실패로 문서 인제스트를 잃으면 안 된다."""
        def boom(*a, **k):
            raise RuntimeError("broken pdf")
        monkeypatch.setattr("ontology.builder.readers.extract_pdf_images", boom)
        assert _builder(image_assets=True).collect_images("doc.pdf") == []

    def test_no_path_returns_empty(self):
        assert _builder(image_assets=True).collect_images(None) == []


class TestAttachIntoChunks:
    def test_attaches_and_reports(self):
        from ontology.builder.models import Chunk
        b = _builder(image_assets=True)
        report = BuildReport(namespace="t")
        chunks = [Chunk(text=TEXT_WITH_CAPTION, source="doc.pdf", index=0,
                        char_start=0, char_end=len(TEXT_WITH_CAPTION))]
        b.apply_images(chunks, TEXT_WITH_CAPTION, [_FakeAsset()], report)
        assert report.images_attached == 1
        assert report.images_unanchored == 0
        assert chunks[0].meta["images"][0]["caption"] == CAPTION

    def test_unanchored_and_no_caption_are_reported(self):
        from ontology.builder.models import Chunk
        b = _builder(image_assets=True)
        report = BuildReport(namespace="t")
        chunks = [Chunk(text=TEXT_WITH_CAPTION, source="doc.pdf", index=0,
                        char_start=0, char_end=len(TEXT_WITH_CAPTION))]
        b.apply_images(chunks, TEXT_WITH_CAPTION,
                       [_FakeAsset(caption=""),
                        _FakeAsset(caption="Figure 99: Missing", sha="b" * 64)],
                       report)
        assert report.images_attached == 0
        # 캡션 없음 + 원문에 없음 = 둘 다 '검색 불가'로 집계된다
        assert report.images_unanchored == 2

    def test_gate_off_skips_attach(self):
        from ontology.builder.models import Chunk
        b = _builder(image_assets=False)
        report = BuildReport(namespace="t")
        chunks = [Chunk(text=TEXT_WITH_CAPTION, source="doc.pdf", index=0,
                        char_start=0, char_end=len(TEXT_WITH_CAPTION))]
        b.apply_images(chunks, TEXT_WITH_CAPTION, [_FakeAsset()], report)
        assert report.images_attached == 0
        assert not chunks[0].meta.get("images")

    def test_empty_assets_is_noop(self):
        b = _builder(image_assets=True)
        report = BuildReport(namespace="t")
        b.apply_images([], TEXT_WITH_CAPTION, [], report)
        assert report.images_attached == 0


class TestReportFields:
    def test_defaults_are_zero(self):
        r = BuildReport(namespace="t")
        assert r.images_attached == 0 and r.images_unanchored == 0


class TestBuildFromFileEndToEnd:
    """build_from_file → 청크 meta 까지 실제로 이어지는가 (가짜 LLM + 실제 PDF).

    collect_images/apply_images 를 따로 테스트해도 **호출부가 빠지면** 아무 일도
    일어나지 않는다 — 이 사이클 시작 시 extract_pdf_images 가 정확히 그 상태였다
    (구현됐지만 호출부가 테스트뿐). 그래서 경로 자체를 검증한다.
    """

    @pytest.fixture
    def pdf_with_figure(self, tmp_path):
        pytest.importorskip("reportlab")
        pytest.importorskip("pdfplumber")
        from PIL import Image
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas

        path = tmp_path / "doc.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        c.drawString(72, 760, "1. Overview")
        c.drawImage(ImageReader(Image.new("RGB", (400, 300), (200, 60, 60))),
                    72, 400, width=400, height=300)
        c.drawString(72, 380, CAPTION)
        c.showPage()
        c.save()
        return str(path)

    def _kg(self):
        # 기존 빌더 테스트와 같은 픽스처 규약 (fast_mode = 임베딩 없이)
        from ontology.engines.knowledge_graph_clean import KnowledgeGraphEngine
        return KnowledgeGraphEngine(fast_mode=True)

    def _run(self, coro):
        import asyncio
        return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)

    def test_gate_on_attaches_image_to_a_stored_chunk(self, pdf_with_figure):
        from ontology.builder.models import BuilderSchema
        from ontology.core.chunk_store import ChunkStore

        store = ChunkStore(namespace="img_wiring_test",
                           path=pdf_with_figure + ".chunks.jsonl")
        builder = OntologyBuilder(
            BuilderSchema(node_types=["Thing"], predicates={}),
            llm_fn=lambda prompt: '{"entities": [], "relations": []}',
            kg=self._kg(), auto_save=False, chunk_store=store,
            image_assets=True)
        report = self._run(builder.build_from_file(pdf_with_figure))

        assert report.images_attached == 1, (
            f"이미지가 청크에 붙지 않았다 (unanchored={report.images_unanchored})")
        with_images = [c for c in store.all() if (c.meta or {}).get("images")]
        assert len(with_images) == 1
        img = with_images[0].meta["images"][0]
        assert img["page"] == 1 and img["caption"] == CAPTION
        assert len(img["sha256"]) == 64

    def test_gate_off_stores_no_image_meta(self, pdf_with_figure):
        from ontology.builder.models import BuilderSchema
        from ontology.core.chunk_store import ChunkStore

        store = ChunkStore(namespace="img_wiring_test2",
                           path=pdf_with_figure + ".off.jsonl")
        builder = OntologyBuilder(
            BuilderSchema(node_types=["Thing"], predicates={}),
            llm_fn=lambda prompt: '{"entities": [], "relations": []}',
            kg=self._kg(), auto_save=False, chunk_store=store,
            image_assets=False)
        report = self._run(builder.build_from_file(pdf_with_figure))

        assert report.images_attached == 0
        assert all(not (c.meta or {}).get("images") for c in store.all())
