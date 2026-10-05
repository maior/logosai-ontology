"""PDF 이미지 추출 어댑터 — 실제 PDF 로 검증한다.

픽스처를 **생성**하는 이유: 저장소에 PDF 가 없고(원천은 업로드분), 무엇보다
**알려진 좌표**가 있어야 bbox 가 맞는지 판정할 수 있다. reportlab 으로 (72,400)에
400×300 도표를 그려 넣고, 추출된 bbox 가 정확히 그 값인지 본다.

캡션을 영문으로 쓰는 이유: reportlab 기본 폰트에 한글 글리프가 없어 생성 PDF 의
한글이 깨진다(2026-07-27 실측: '그림 3. 시스템 구성도' → 'nn 3. nnn nnn').
이건 픽스처의 한계이지 추출기의 문제가 아니다 — 한글 캡션 판정은
test_image_assets.py 가 find_caption 으로 직접 검증한다. 여기서는 **배관**
(좌표·해시·캡션 밴드·장식 배제·degrade)을 본다.

좌표 규약: bbox 는 **PDF 원좌표 (x0,y0,x1,y1)** — y 는 페이지 아래 기준.
pdfplumber 의 crop 은 위 기준이라 meta["page_height"] 로 top = page_height - y1 을
유도할 수 있어야 한다. 이게 "재크롭해서 대조 가능"이라는 주장의 실체다.
"""
import sys

import pytest

pdfplumber = pytest.importorskip("pdfplumber", reason="images extra 미설치")
pytest.importorskip("reportlab", reason="테스트 픽스처 생성용")

from ontology.builder.readers import extract_pdf_images  # noqa: E402

# 픽스처에 심는 알려진 값
FIG_BBOX = (72.0, 400.0, 472.0, 700.0)      # drawImage(72, 400, 400x300)
FIG_SRCSIZE = (400, 300)
FIG_CAPTION = "Figure 3: System Architecture"
LOGO_BBOX = (20.0, 20.0, 40.0, 40.0)        # 20x20 로고 — 장식으로 배제돼야 한다
TABLE_CAPTION = "Table 2: Cost Breakdown"


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    """알려진 좌표의 2쪽 PDF.
    p1: 도표 + 아래 캡션 + 작은 로고 / p2: 위 캡션 + 도표(표 캡션은 위가 관례)
    """
    from PIL import Image
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    big = Image.new("RGB", FIG_SRCSIZE, (200, 60, 60))
    other = Image.new("RGB", (350, 260), (60, 140, 90))
    logo = Image.new("RGB", (20, 20), (60, 60, 200))

    c = canvas.Canvas(str(path), pagesize=A4)          # A4 = 595.28 x 841.89 pt
    c.drawString(72, 760, "1. Overview")
    c.drawImage(ImageReader(big), 72, 400, width=400, height=300)
    c.drawString(72, 380, FIG_CAPTION)                 # 이미지 **아래** 캡션
    c.drawString(72, 360, "Each component goes through the API gateway.")
    c.drawImage(ImageReader(logo), 20, 20, width=20, height=20)   # 장식
    c.showPage()
    c.drawString(72, 700, TABLE_CAPTION)               # 이미지 **위** 캡션
    c.drawImage(ImageReader(other), 72, 400, width=350, height=260)
    c.showPage()
    c.save()
    return str(path)


@pytest.fixture(scope="module")
def assets(sample_pdf):
    return extract_pdf_images(sample_pdf)


# ─── 주소 = 좌표가 정확한가 (인용의 근거) ────────────────────────────

class TestCoordinates:
    def test_decorative_logo_is_excluded(self, assets):
        """작은 로고는 근거가 아니다 — 배제돼야 한다."""
        boxes = [a.bbox for a in assets]
        assert LOGO_BBOX not in boxes

    def test_two_real_figures_found(self, assets):
        """p1 도표 + p2 도표 = 2개 (로고 제외)."""
        assert len(assets) == 2

    def test_bbox_matches_drawn_position_exactly(self, assets):
        """반올림·좌표계 혼동이 있으면 인용이 엉뚱한 곳을 가리킨다."""
        first = [a for a in assets if a.page == 1][0]
        assert first.bbox == FIG_BBOX

    def test_page_is_one_based(self, assets):
        assert sorted(a.page for a in assets) == [1, 2]

    def test_source_is_recorded(self, assets, sample_pdf):
        assert all(a.source == sample_pdf for a in assets)

    def test_srcsize_is_the_pixel_size(self, assets):
        """width/height 는 원본 픽셀 — 표시 크기는 이미 bbox 에 있다."""
        first = [a for a in assets if a.page == 1][0]
        assert (first.width, first.height) == FIG_SRCSIZE


# ─── 재크롭 대조가 가능한가 (검증 가능성의 실체) ─────────────────────

class TestRecropDerivable:
    def test_page_height_is_in_meta(self, assets):
        first = [a for a in assets if a.page == 1][0]
        assert first.meta["page_height"] == pytest.approx(841.89, abs=0.01)

    def test_top_based_box_is_derivable_and_matches_pdfplumber(self, assets,
                                                               sample_pdf):
        """bbox(아래 기준) + page_height → top 기준 좌표. pdfplumber 가
        보고한 top/bottom 과 일치해야 재크롭이 같은 영역을 가리킨다."""
        first = [a for a in assets if a.page == 1][0]
        derived_top = first.meta["page_height"] - first.bbox[3]
        with pdfplumber.open(sample_pdf) as pdf:
            real = [im for im in pdf.pages[0].images
                    if im["srcsize"] == FIG_SRCSIZE][0]
        assert derived_top == pytest.approx(real["top"], abs=0.01)

    def test_recropping_the_stored_box_returns_the_image(self, assets,
                                                         sample_pdf):
        """저장된 좌표로 실제 크롭하면 그 이미지가 그 영역에 있어야 한다 —
        '재크롭 대조 가능'이 말뿐이 아님을 보인다."""
        first = [a for a in assets if a.page == 1][0]
        ph = first.meta["page_height"]
        x0, y0, x1, y1 = first.bbox
        with pdfplumber.open(sample_pdf) as pdf:
            region = pdf.pages[0].crop((x0, ph - y1, x1, ph - y0))
            assert len(region.images) >= 1


# ─── 내용 해시 (재추출 멱등) ─────────────────────────────────────────

class TestContentHash:
    def test_sha256_is_hex64(self, assets):
        first = [a for a in assets if a.page == 1][0]
        assert len(first.sha256) == 64
        int(first.sha256, 16)          # hex 여야 한다

    def test_reextraction_yields_same_ids(self, sample_pdf, assets):
        """재추출해도 asset_id 가 같아야 중복이 쌓이지 않는다(청크 규약)."""
        again = extract_pdf_images(sample_pdf)
        assert [a.asset_id for a in again] == [a.asset_id for a in assets]

    def test_different_figures_have_different_hashes(self, assets):
        assert len({a.sha256 for a in assets}) == 2


# ─── 캡션 (대리 텍스트의 씨앗) ───────────────────────────────────────

class TestCaption:
    def test_caption_below_image_is_found(self, assets):
        first = [a for a in assets if a.page == 1][0]
        assert first.caption == FIG_CAPTION

    def test_caption_above_image_is_found(self, assets):
        """표 캡션은 이미지 위에 오는 것이 관례다 — 위 밴드도 봐야 한다."""
        second = [a for a in assets if a.page == 2][0]
        assert second.caption == TABLE_CAPTION

    def test_body_prose_is_not_taken_as_caption(self, assets):
        """캡션 밴드 안의 본문 문장('Each component goes...')은 캡션이 아니다."""
        first = [a for a in assets if a.page == 1][0]
        assert "Each component" not in first.caption


# ─── degrade · 실패 경로 (여기서 죽으면 인제스트가 죽는다) ───────────

class TestDegrade:
    def test_missing_pdfplumber_degrades_to_empty(self, monkeypatch,
                                                  sample_pdf):
        """images extra 없이도 인제스트는 돌아야 한다 — 빈 목록으로 degrade."""
        monkeypatch.setitem(sys.modules, "pdfplumber", None)
        assert extract_pdf_images(sample_pdf) == []

    def test_missing_file_returns_empty(self, tmp_path):
        assert extract_pdf_images(str(tmp_path / "nope.pdf")) == []

    def test_non_pdf_returns_empty(self, tmp_path):
        txt = tmp_path / "a.txt"
        txt.write_text("not a pdf", encoding="utf-8")
        assert extract_pdf_images(str(txt)) == []

    def test_none_path_returns_empty(self):
        assert extract_pdf_images(None) == []

    def test_text_only_pdf_returns_empty(self, tmp_path):
        """우리 코퍼스의 대다수는 이미지 없는 PDF다 — 빈 목록이어야 하고
        예외를 던져 인제스트를 멈춰선 안 된다."""
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        p = tmp_path / "text_only.pdf"
        c = canvas.Canvas(str(p), pagesize=A4)
        c.drawString(72, 700, "No images here, only prose.")
        c.showPage()
        c.save()
        assert extract_pdf_images(str(p)) == []


# ─── 임계값 주입 (측정되지 않은 상수는 덮을 수 있어야) ───────────────

class TestThresholds:
    def test_lowering_thresholds_includes_the_logo(self, sample_pdf):
        """임계값이 측정값이 아니므로 호출부가 바꿀 수 있어야 한다."""
        loose = extract_pdf_images(sample_pdf, min_area=1.0, min_side=1.0)
        assert LOGO_BBOX in [a.bbox for a in loose]

    def test_raising_thresholds_excludes_everything(self, sample_pdf):
        assert extract_pdf_images(sample_pdf, min_area=10 ** 9) == []
