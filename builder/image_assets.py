"""이미지 주소화 + 대리 텍스트 — 의존성 0 순수 로직.

출처 사슬 불변식(`original[char_start:char_end] == chunk.text`)은 이미지에 성립하지
않는다 — 문자 오프셋이 없다. 그래서 이미지의 출처 좌표를 **(page, bbox)** 로 둔다:
재크롭해 대조할 수 있으므로 검증 가능성이 유지된다.

이미지 자체는 청크가 아니다. **대리 텍스트 청크**(캡션 + 주변 문장)가 meta.image 로
이미지를 가리킨다 — 대리 텍스트는 원문에서 그대로 가져온 문자열이라 불변식을
깨지 않고 환각도 없다. OCR(L2)·VLM(L3)은 환각 위험이 있어 trust 강등과 측정이
전제이며 여기 없다.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

Bbox = Tuple[float, float, float, float]

# 장식(로고·구분선) 배제 기준. **어휘가 아니라 크기** — 도메인·언어 무관.
# 측정된 값이 아니라 관찰에서 고른 출발점이므로 호출부가 덮을 수 있어야 한다.
DEFAULT_MIN_AREA = 5000.0     # pt² — A4 본문 폭(≈450pt) × 11pt 정도가 하한
DEFAULT_MIN_SIDE = 24.0       # pt — 이보다 얇으면 구분선/괘선

CAPTION_MAX_LEN = 160
PROXY_MAX_LEN = 700

# 캡션 = 「라벨 + 번호」라는 **구조**. 그림/표/Figure/Table 은 문서 구조 표지이지
# 도메인 어휘가 아니다. 줄머리만 인정한다 — 본문 속 "그림 3 참조"는 캡션이 아니다.
_CAPTION = re.compile(
    r"^\s*[<\[【(]?\s*(?:그림|사진|도표|표|Figure|Fig\.?|Table)\s*"
    r"\d+(?:[-.]\d+)*\s*[>\]】)]?\s*[.:：—-]?\s*\S.*$",
    re.IGNORECASE)


def asset_id_for(sha256: str, source: str, page: int,
                 bbox: Optional[Bbox]) -> str:
    """이미지 자산 id — 내용 해시 + 출처 좌표.

    좌표를 넣는 이유: 같은 로고가 여러 페이지에 있으면 **다른 출처**다. 인용은
    좌표를 가리키므로 내용만으로 합치면 엉뚱한 페이지를 지목한다.
    (내용 해시 기반이라 재추출해도 중복이 쌓이지 않는다 — 청크 규약과 동일)
    """
    coords = ",".join(f"{float(v):.2f}" for v in (bbox or ()))
    raw = f"{sha256}::{source}::{page}::{coords}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def bbox_area(bbox: Optional[Sequence[float]]) -> float:
    """bbox 면적. 뒤집힌/불량 좌표는 0 — 없는 면적을 지어내지 않는다."""
    if not bbox or len(tuple(bbox)) != 4:
        return 0.0
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class ImageAsset:
    """문서 안 이미지 하나 — 주소(page, bbox)를 가진 인용 대상."""
    source: str
    page: int
    bbox: Bbox
    sha256: str
    width: int = 0
    height: int = 0
    caption: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def asset_id(self) -> str:
        return asset_id_for(self.sha256, self.source, self.page, self.bbox)


def is_decorative(asset: ImageAsset,
                  min_area: float = DEFAULT_MIN_AREA,
                  min_side: float = DEFAULT_MIN_SIDE) -> bool:
    """로고·구분선처럼 근거가 아닌 이미지인가 — 크기만 본다.

    최소 변을 함께 보는 이유: 구분선은 면적이 커도 한 변이 극단적으로 얇다.
    종횡비로 재면 정당한 와이드 도표(구성도)까지 걸린다.
    """
    bbox = tuple(asset.bbox or ())
    if len(bbox) != 4:
        return True                       # 좌표를 모르면 인용할 수 없다
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
    if min(w, h) < min_side:
        return True
    return bbox_area(bbox) < min_area


def find_caption(text: Optional[str]) -> str:
    """캡션 줄을 찾아 돌려준다. 없으면 "" — 지어내지 않는다.

    줄머리 라벨만 인정하고, 번호가 없으면 캡션이 아니다("그림 설명이 이어집니다").
    여러 개면 첫 번째 — 이미지 바로 앞/뒤 문맥에서 호출되는 것이 전제다.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _CAPTION.match(stripped):
            return stripped[:CAPTION_MAX_LEN]
    return ""


def proxy_text(asset: ImageAsset, caption: str = "",
               nearby: str = "") -> str:
    """이미지의 대리 텍스트 — 캡션이 앞, 주변 문장이 뒤.

    캡션을 앞세우는 이유: 캡션은 사람이 그 그림에 붙인 요약이라 가장 강한 신호다.
    둘 다 없으면 **빈 문자열** — 내용 없는 청크를 인덱스에 넣으면 검색만 흐려진다.
    """
    cap = (caption or "").strip()
    near = " ".join((nearby or "").split())
    if not cap and not near:
        return ""
    head = cap or ""
    page_tag = f"(p.{asset.page})" if asset and asset.page else ""
    parts = [p for p in (head, page_tag, near) if p]
    return " ".join(parts)[:PROXY_MAX_LEN]


def _chunk_covering(chunks, offset: int):
    """이 오프셋을 덮는 청크. 없으면 None (세그먼터가 버린 구간일 수 있다)."""
    for chunk in chunks or ():
        start = int(getattr(chunk, "char_start", 0) or 0)
        end = int(getattr(chunk, "char_end", 0) or 0)
        if start <= offset < end:
            return chunk
    return None


def attach_images_to_chunks(chunks, text, assets) -> Dict[str, int]:
    """이미지를 **캡션이 들어 있는 청크**에 붙인다 (제자리 수정).

    새 청크를 만들지 않는 이유: 대리 텍스트(캡션+페이지+주변문장 합성)를 청크로
    저장하면 `original[char_start:char_end] == chunk.text` 불변식이 깨진다 —
    합성 문자열은 원문의 부분 문자열이 아니다. 캡션은 **이미 어떤 청크 안에**
    있으므로 그 청크에 meta 로 달면 불변식·chunk_id·벡터 캐시가 모두 그대로다
    (chunk_store.add 의 meta 병합이 노린 이점).

    세 갈래로 집계한다 — 조용히 버리면 "이미지가 왜 안 나오나"를 설명할 수 없다:
      · attached    — 캡션 청크에 붙음(검색·인용 가능)
      · no_caption  — 캡션 자체가 없음 → 텍스트 앵커 없음(OCR/VLM 필요)
      · unanchored  — 캡션이 원문에 없거나 덮는 청크가 없음
    """
    report = {"attached": 0, "unanchored": 0, "no_caption": 0}
    body = text if isinstance(text, str) else ""
    for asset in assets or ():
        caption = (getattr(asset, "caption", "") or "").strip()
        if not caption:
            report["no_caption"] += 1
            continue
        offset = body.find(caption)
        if offset < 0:
            # clean_pdf_pages 가 원문을 정규화해 캡션이 그대로 안 남는 경우.
            report["unanchored"] += 1
            continue
        # 첫 출현만 쓴다 — 반복되는 캡션에 양쪽으로 붙이면 같은 이미지가 두 곳에서
        # 인용돼 "그림 N개"가 거짓이 된다.
        holder = _chunk_covering(chunks, offset)
        if holder is None:
            report["unanchored"] += 1
            continue
        payload = proxy_meta(asset)["image"]
        payload["caption"] = caption
        meta = getattr(holder, "meta", None)
        if meta is None:
            # meta 를 못 붙이는 객체(외부 호출부의 청크 유사물)는 세지만 죽지 않는다.
            report["unanchored"] += 1
            continue
        images = meta.setdefault("images", [])
        if any(existing.get("asset_id") == payload["asset_id"]
               for existing in images):
            report["attached"] += 1     # 이미 붙어 있다 — 멱등, 중복 추가 안 함
            continue
        images.append(payload)
        report["attached"] += 1
    return report


def proxy_meta(asset: ImageAsset) -> Dict[str, Any]:
    """대리 텍스트 청크에 실을 meta — 단일 키 image 아래로만.

    char_start/end 를 넣지 않는다: 이미지엔 문자 오프셋이 없고 0 으로 채우면
    원문 0~0 을 가리키는 **거짓 인용**이 된다. (page, bbox)가 그 자리를 대신한다.
    bbox 는 list 로 — JSON 직렬화(청크 JSONL) 가능해야 한다.
    """
    return {"image": {
        "asset_id": asset.asset_id,
        "source": asset.source,
        "page": asset.page,
        "bbox": [float(v) for v in (asset.bbox or ())],
        "sha256": asset.sha256,
        "width": asset.width,
        "height": asset.height,
    }}
