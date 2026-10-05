"""이미지 주소화 (①) + 대리 텍스트 (L1) — 순수 로직.

**왜 이미지에 "주소"가 먼저인가**: 온톨로지의 출처 사슬 불변식은
`original[char_start:char_end] == chunk.text` 다. 이미지엔 문자 오프셋이 없어
그 계약이 성립하지 않는다. 그래서 이미지의 출처 좌표를 **(page, bbox)** 로 둔다 —
재크롭해서 대조할 수 있으므로 검증 가능성(falsifiability)이 유지된다.

**왜 대리 텍스트인가 (L1)**: OCR·VLM 없이도 캡션("그림 3. 시스템 구성도")과 주변
텍스트를 이미지의 대리물로 인덱싱하면 검색·인용이 가능해진다. 비용 0, 환각 0 —
대리 텍스트는 **원문에서 그대로 가져온 문자열**이다. OCR(L2)·VLM(L3)은 환각
위험이 있어 trust 강등과 측정이 전제이며 이 사이클 범위 밖이다.

**어휘 하드코딩 금지 준수**: 캡션 탐지는 「라벨 + 번호」라는 **구조**만 본다
(그림/표/Figure/Table 은 문서 구조 표지이지 도메인 어휘가 아니다). 장식 이미지
배제도 어휘가 아니라 **크기**로 한다.
"""
import pytest

from ontology.builder.image_assets import (
    ImageAsset,
    asset_id_for,
    bbox_area,
    find_caption,
    is_decorative,
    proxy_meta,
    proxy_text,
)


def _asset(**kw):
    base = dict(source="rfp.pdf", page=3, bbox=(72.0, 100.0, 472.0, 400.0),
                sha256="a" * 64, width=400, height=300)
    base.update(kw)
    return ImageAsset(**base)


# ─── 주소 = (page, bbox) ─────────────────────────────────────────────

class TestAssetIdentity:
    def test_id_is_stable_for_same_content_and_place(self):
        """내용 해시 기반 — 재추출해도 중복이 쌓이지 않는다(청크 규약과 동일)."""
        a = asset_id_for("a" * 64, "rfp.pdf", 3, (72.0, 100.0, 472.0, 400.0))
        b = asset_id_for("a" * 64, "rfp.pdf", 3, (72.0, 100.0, 472.0, 400.0))
        assert a == b and len(a) == 12

    def test_same_image_on_different_page_is_different_asset(self):
        """같은 로고가 여러 페이지에 있으면 **다른 출처**다 — 인용 좌표가 다르다."""
        a = asset_id_for("a" * 64, "rfp.pdf", 3, (0.0, 0.0, 10.0, 10.0))
        b = asset_id_for("a" * 64, "rfp.pdf", 4, (0.0, 0.0, 10.0, 10.0))
        assert a != b

    def test_different_content_same_place_is_different_asset(self):
        a = asset_id_for("a" * 64, "rfp.pdf", 3, (0.0, 0.0, 10.0, 10.0))
        b = asset_id_for("b" * 64, "rfp.pdf", 3, (0.0, 0.0, 10.0, 10.0))
        assert a != b

    def test_asset_exposes_its_own_id(self):
        assert _asset().asset_id == asset_id_for(
            "a" * 64, "rfp.pdf", 3, (72.0, 100.0, 472.0, 400.0))

    def test_bbox_is_kept_verbatim(self):
        """좌표가 인용의 근거다 — 반올림·정규화로 흔들면 재크롭 대조가 깨진다."""
        assert _asset().bbox == (72.0, 100.0, 472.0, 400.0)


# ─── 장식 이미지 배제 (크기 = 구조 신호) ─────────────────────────────

class TestDecorative:
    def test_area_computed_from_bbox(self):
        assert bbox_area((0.0, 0.0, 10.0, 20.0)) == 200.0

    def test_negative_or_inverted_bbox_is_zero_area(self):
        assert bbox_area((10.0, 10.0, 5.0, 5.0)) == 0.0
        assert bbox_area(None) == 0.0

    def test_large_figure_is_not_decorative(self):
        assert is_decorative(_asset()) is False

    def test_tiny_logo_is_decorative(self):
        assert is_decorative(_asset(bbox=(0.0, 0.0, 20.0, 20.0),
                                    width=20, height=20)) is True

    def test_thin_rule_line_is_decorative(self):
        """구분선: 면적은 커도 한 변이 극단적으로 얇다 — 종횡비가 아니라 최소 변.

        표본의 면적(1200×8=9600)이 **min_area(5000)를 넘는다** — 그래야 면적
        규칙이 아니라 min_side 규칙이 실제로 검증된다. 변이 검사(2026-07-27)에서
        면적이 작은 표본은 min_side 를 지워도 통과해 규칙을 못 지켰다.
        """
        wide_thin = _asset(bbox=(0.0, 0.0, 1200.0, 8.0), width=1200, height=8)
        assert bbox_area(wide_thin.bbox) > 5000.0        # 면적으론 안 걸린다
        assert is_decorative(wide_thin) is True          # min_side 가 잡는다

    def test_thresholds_are_overridable(self):
        """임계값은 측정된 값이 아니다 — 호출부가 바꿀 수 있어야 한다."""
        small = _asset(bbox=(0.0, 0.0, 20.0, 20.0), width=20, height=20)
        assert is_decorative(small, min_area=100.0, min_side=10.0) is False


# ─── 캡션 탐지 (라벨+번호 구조) ──────────────────────────────────────

class TestFindCaption:
    def test_korean_figure_caption(self):
        text = ("아래 그림은 전체 구조를 보여준다.\n"
                "그림 3. 시스템 구성도\n"
                "각 구성요소는 다음과 같다.")
        assert find_caption(text) == "그림 3. 시스템 구성도"

    def test_korean_table_caption(self):
        assert find_caption("표 12. 연도별 예산 집행 내역") == "표 12. 연도별 예산 집행 내역"

    def test_english_figure_caption(self):
        assert find_caption("Figure 4: Deployment topology") == \
            "Figure 4: Deployment topology"

    def test_english_table_caption(self):
        assert find_caption("Table 2 — Cost breakdown") == "Table 2 — Cost breakdown"

    def test_angle_bracket_label(self):
        """공공문서에서 흔한 <그림 1> 형태."""
        assert find_caption("<그림 1> 추진 체계") == "<그림 1> 추진 체계"

    def test_first_caption_wins_when_multiple(self):
        text = "그림 1. 첫째\n본문\n그림 2. 둘째"
        assert find_caption(text) == "그림 1. 첫째"

    def test_no_caption_returns_empty(self):
        assert find_caption("이 문단에는 캡션이 없습니다.") == ""
        assert find_caption("") == ""
        assert find_caption(None) == ""

    def test_reference_in_prose_is_not_a_caption(self):
        """본문 속 '그림 3 참조' 는 캡션이 아니다 — 줄머리 라벨만 인정한다."""
        assert find_caption("자세한 내용은 그림 3 참조 바랍니다.") == ""

    def test_label_without_number_is_not_a_caption(self):
        assert find_caption("그림 설명이 이어집니다") == ""

    def test_caption_is_length_capped(self):
        long_tail = "가" * 300
        got = find_caption(f"그림 1. {long_tail}")
        assert got.startswith("그림 1. ") and len(got) <= 160


# ─── 대리 텍스트 · meta ──────────────────────────────────────────────

class TestProxyText:
    def test_caption_leads_the_proxy_text(self):
        out = proxy_text(_asset(), caption="그림 3. 시스템 구성도",
                         nearby="각 구성요소는 API 게이트웨이를 경유한다.")
        assert out.startswith("그림 3. 시스템 구성도")
        assert "API 게이트웨이" in out

    def test_page_is_included_so_the_hit_is_locatable(self):
        out = proxy_text(_asset(), caption="그림 3. 구성도", nearby="")
        assert "3" in out           # page 3

    def test_no_caption_falls_back_to_nearby(self):
        out = proxy_text(_asset(), caption="", nearby="본문 설명 문장")
        assert "본문 설명 문장" in out

    def test_empty_signals_yield_empty_proxy(self):
        """캡션도 주변 텍스트도 없으면 대리 텍스트를 지어내지 않는다 —
        내용 없는 청크를 인덱스에 넣으면 검색 품질만 떨어진다."""
        assert proxy_text(_asset(), caption="", nearby="") == ""

    def test_nearby_is_length_capped(self):
        out = proxy_text(_asset(), caption="그림 1. 표제", nearby="나" * 2000)
        assert len(out) <= 700


class TestProxyMeta:
    def test_meta_is_namespaced_under_image(self):
        meta = proxy_meta(_asset())
        assert set(meta) == {"image"}

    def test_meta_carries_the_citation_coordinates(self):
        img = proxy_meta(_asset())["image"]
        assert img["page"] == 3
        assert img["bbox"] == [72.0, 100.0, 472.0, 400.0]   # JSON 직렬화 가능
        assert img["sha256"] == "a" * 64
        assert img["asset_id"] == _asset().asset_id
        assert img["source"] == "rfp.pdf"

    def test_meta_is_json_serializable(self):
        import json
        json.dumps(proxy_meta(_asset()))   # 던지지 않아야 한다

    def test_meta_has_no_char_offsets(self):
        """이미지는 문자 오프셋을 갖지 않는다 — 0 으로 채우면 원문 0~0 을
        가리키는 거짓 인용이 된다. (page,bbox) 가 그 자리를 대신한다."""
        img = proxy_meta(_asset())["image"]
        assert "char_start" not in img and "char_end" not in img
