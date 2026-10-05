"""이미지 → 청크 부착 (L1 배선) — 새 청크를 만들지 않는다.

**왜 새 청크를 만들지 않는가**: 대리 텍스트(캡션+페이지+주변문장 합성)를 청크로
저장하면 `original[char_start:char_end] == chunk.text` 불변식이 깨진다 — 합성
문자열은 원문의 부분 문자열이 아니다. Chunk docstring 이 규정한 대로, 이 불변식이
깨지면 인용이 엉뚱한 위치를 가리켜 span provenance 전체가 무의미해진다.

대신 **캡션은 이미 어떤 청크 안에 있다**는 사실을 쓴다:
  이미지 → 캡션 → 원문 offset → 그 offset 을 덮는 청크 → chunk.meta["images"]
새 청크 0, 불변식 무관, chunk_id 불변(= 벡터 캐시 무효화 0 — chunk_store.add 의
meta 병합 주석이 규정한 이점). 주변 문장은 이미 정규 청크로 인덱싱돼 있으므로
중복 저장할 이유가 없다.

**캡션 없는 이미지**: 검색 가능하게 만들 텍스트 앵커가 없다. 페이지→오프셋 추정은
근거가 없어 하지 않는다 — 세 갈래(attached / unanchored / no_caption)로 **집계해
보고**한다. 조용히 버리면 "이미지가 왜 안 나오나"를 나중에 설명할 수 없다.
"""
import pytest

from ontology.builder.image_assets import ImageAsset, attach_images_to_chunks
from ontology.builder.models import Chunk

TEXT = (
    "1. 사업 개요\n"                       # 0-11
    "본 사업은 데이터 표준화를 목표로 한다.\n"
    "그림 3. 시스템 구성도\n"
    "각 구성요소는 API 게이트웨이를 경유한다.\n"
    "2. 추진 체계\n"
    "표 2. 연도별 예산\n"
    "예산은 3개년에 걸쳐 집행된다.\n"
)
CAP_FIG = "그림 3. 시스템 구성도"
CAP_TABLE = "표 2. 연도별 예산"


def _chunks(text=TEXT, size=60):
    """원문을 고정 길이로 잘라 오프셋 불변식을 만족하는 청크들을 만든다."""
    out = []
    for i, start in enumerate(range(0, len(text), size)):
        piece = text[start:start + size]
        out.append(Chunk(text=piece, source="rfp.pdf", index=i,
                         char_start=start, char_end=start + len(piece)))
    # 픽스처 자체가 불변식을 지키는지 확인 — 안 지키면 테스트가 거짓이 된다
    for c in out:
        assert text[c.char_start:c.char_end] == c.text
    return out


def _asset(caption, page=1, bbox=(72.0, 400.0, 472.0, 700.0), sha="a" * 64):
    return ImageAsset(source="rfp.pdf", page=page, bbox=bbox, sha256=sha,
                      width=400, height=300, caption=caption,
                      meta={"page_height": 841.89})


# ─── 부착 (정상 경로) ────────────────────────────────────────────────

class TestAttach:
    def test_image_attaches_to_the_chunk_containing_its_caption(self):
        chunks = _chunks()
        report = attach_images_to_chunks(chunks, TEXT, [_asset(CAP_FIG)])
        assert report["attached"] == 1
        holders = [c for c in chunks if c.meta.get("images")]
        assert len(holders) == 1
        assert CAP_FIG in TEXT[holders[0].char_start:holders[0].char_end]

    def test_attached_payload_carries_citation_coordinates(self):
        chunks = _chunks()
        attach_images_to_chunks(chunks, TEXT, [_asset(CAP_FIG)])
        img = [c for c in chunks if c.meta.get("images")][0].meta["images"][0]
        assert img["page"] == 1
        assert img["bbox"] == [72.0, 400.0, 472.0, 700.0]
        assert img["sha256"] == "a" * 64
        assert "asset_id" in img
        assert img["caption"] == CAP_FIG

    def test_payload_has_no_char_offsets(self):
        """이미지엔 문자 오프셋이 없다 — 0 으로 채우면 거짓 인용이 된다."""
        chunks = _chunks()
        attach_images_to_chunks(chunks, TEXT, [_asset(CAP_FIG)])
        img = [c for c in chunks if c.meta.get("images")][0].meta["images"][0]
        assert "char_start" not in img and "char_end" not in img

    def test_two_images_attach_to_their_own_chunks(self):
        chunks = _chunks()
        report = attach_images_to_chunks(
            chunks, TEXT, [_asset(CAP_FIG), _asset(CAP_TABLE, page=2, sha="b" * 64)])
        assert report["attached"] == 2
        holders = [c for c in chunks if c.meta.get("images")]
        assert len(holders) == 2

    def test_two_images_sharing_a_chunk_are_both_listed(self):
        """한 청크에 그림·표가 함께 있으면 둘 다 실려야 한다(덮어쓰기 금지)."""
        text = f"{CAP_FIG}\n{CAP_TABLE}\n"
        chunks = _chunks(text, size=500)          # 한 청크에 둘 다
        report = attach_images_to_chunks(
            chunks, text, [_asset(CAP_FIG), _asset(CAP_TABLE, sha="b" * 64)])
        assert report["attached"] == 2
        assert len(chunks[0].meta["images"]) == 2

    def test_existing_meta_keys_are_preserved(self):
        chunks = _chunks()
        for c in chunks:
            c.meta["layer"] = "L1"
        attach_images_to_chunks(chunks, TEXT, [_asset(CAP_FIG)])
        holder = [c for c in chunks if c.meta.get("images")][0]
        assert holder.meta["layer"] == "L1"

    def test_idempotent_reattach_does_not_duplicate(self):
        """같은 자산을 두 번 붙이면 목록이 부풀어 '그림 4개'처럼 보인다."""
        chunks = _chunks()
        attach_images_to_chunks(chunks, TEXT, [_asset(CAP_FIG)])
        attach_images_to_chunks(chunks, TEXT, [_asset(CAP_FIG)])
        holder = [c for c in chunks if c.meta.get("images")][0]
        assert len(holder.meta["images"]) == 1


# ─── 앵커 실패 (세 갈래를 구분해 집계) ───────────────────────────────

class TestUnanchored:
    def test_no_caption_is_counted_separately(self):
        """캡션이 없으면 텍스트 앵커가 없다 — OCR/VLM 이 있어야 검색 가능."""
        chunks = _chunks()
        report = attach_images_to_chunks(chunks, TEXT, [_asset("")])
        assert report["no_caption"] == 1
        assert report["attached"] == 0
        assert all(not c.meta.get("images") for c in chunks)

    def test_caption_absent_from_text_is_unanchored(self):
        """clean_pdf_pages 가 원문을 변형해 캡션이 안 남는 경우 —
        조용히 버리지 않고 unanchored 로 보고한다."""
        chunks = _chunks()
        report = attach_images_to_chunks(
            chunks, TEXT, [_asset("그림 99. 존재하지 않는 캡션")])
        assert report["unanchored"] == 1
        assert report["attached"] == 0

    def test_report_counts_sum_to_input(self):
        chunks = _chunks()
        report = attach_images_to_chunks(chunks, TEXT, [
            _asset(CAP_FIG), _asset(""), _asset("그림 99. 없음", sha="c" * 64)])
        assert (report["attached"] + report["no_caption"]
                + report["unanchored"]) == 3


# ─── 경계·안전 ───────────────────────────────────────────────────────

class TestEdges:
    def test_empty_inputs_are_safe(self):
        assert attach_images_to_chunks([], "", []) == {
            "attached": 0, "unanchored": 0, "no_caption": 0}
        assert attach_images_to_chunks(None, None, None)["attached"] == 0

    def test_first_occurrence_wins_when_caption_repeats(self):
        """캡션이 두 번 나오면 첫 출현 청크에만 붙는다 — 양쪽에 붙이면
        같은 이미지가 두 곳에서 인용돼 개수가 거짓이 된다."""
        text = f"{CAP_FIG}\n" + "가" * 200 + f"\n{CAP_FIG}\n"
        chunks = _chunks(text, size=60)
        report = attach_images_to_chunks(chunks, text, [_asset(CAP_FIG)])
        assert report["attached"] == 1
        assert len([c for c in chunks if c.meta.get("images")]) == 1
        assert chunks[0].meta.get("images")          # 첫 청크

    def test_offset_outside_any_chunk_is_unanchored(self):
        """세그먼터가 버린 구간(목차 등)에 캡션이 있으면 덮는 청크가 없다."""
        text = TEXT
        tail_only = [Chunk(text=text[-30:], source="s", index=0,
                           char_start=len(text) - 30, char_end=len(text))]
        report = attach_images_to_chunks(tail_only, text, [_asset(CAP_FIG)])
        assert report["unanchored"] == 1

    def test_chunks_without_meta_attribute_do_not_crash(self):
        """meta 가 없는 청크 유사 객체도 들어올 수 있다(외부 호출부)."""
        class _Bare:
            text = CAP_FIG
            char_start = TEXT.index(CAP_FIG)
            char_end = char_start + len(CAP_FIG)
        report = attach_images_to_chunks([_Bare()], TEXT, [_asset(CAP_FIG)])
        assert report["attached"] + report["unanchored"] == 1   # 죽지 않는다
