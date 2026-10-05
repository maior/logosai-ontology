"""
Segmenter — structure-aware text splitting.

heading mode: split on markdown headings / 조문(제N조) headings so one
chunk holds one coherent provision (aicoach legal.py의 일반화).
window mode: sentence-boundary sliding window (aicoach chunk_text 이식).
auto: headings when the document has them, window otherwise.

모든 조각은 원본 텍스트 기준 문자 오프셋(char_start/char_end)을 함께
돌려준다 — original[char_start:char_end] == chunk.text 가 불변식이다(축 2).
내부 헬퍼들이 (piece, start, end) 스팬을 다루는 것은 그 때문이다: strip 으로
잘려나간 공백만큼 오프셋을 보정하지 않으면 인용이 통째로 밀린다.
"""

import re
from typing import List, Tuple

from .models import Chunk

# markdown headings, 제N조 article headings, "1. " style numbered headings.
# (coarse — auto-mode 판정 · sniffer.detect_text_species 가 findall 로 쓴다)
_HEADING = re.compile(
    r"(?m)^(?:#{1,6}\s+.+|제\s?\d+\s?조.*|\d+\.\s+\S.*)$")

# ─── 구조-인지 조문 헤딩(약관·법령) — aicoach legal.py 이식 ───────────────
# 줄머리 + 제목 괄호를 요구해 본문 속 "제21조에 따라" 같은 *참조*가 새 조항으로
# 오분할되지 않게 한다. 라벨(제N조(제목))은 인용 근거로 chunk.section 에 실린다.
_MD_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ART_PAREN = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*\(([^)]+)\)")
_ART_LOOSE = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s+(.{1,40})$")
_NUM_HEAD = re.compile(r"^(\d+)\.\s+(\S.*)$")
_REF_IN_TITLE = re.compile(r"제\s*\d+\s*[조항]|준용|에\s*따라|규정")
_REF_AFTER_JOSA = re.compile(r"^(의|에서|에게|에|을|를|이|가|은|는|와|과|으로|로)")
_REF_AFTER = re.compile(r'^\s*(제\s*\d+\s*[조항]|참조|[")”」])')
# 러닝헤더/쪽번호("12 / 313", 순수 숫자 줄)
_RUNNING = re.compile(r"^(\d{1,4}\s*/\s*\d{1,4}|\d{1,4})$")


def _is_reference_heading(title: str, after: str) -> bool:
    """조 헤딩 후보가 실은 본문 속 외부 조문 참조이면 True(→ 헤딩으로 잡지 않음)."""
    t = (title or "").strip()
    if t.endswith("설립") or t.endswith("설립 등") or "의 설립" in t:
        return True
    a = after or ""
    if a and not a[:1].isspace() and _REF_AFTER_JOSA.match(a):  # ')의'·')을' 직접 부착
        return True
    if _REF_AFTER.match(a):                                     # ') 제1항'·') 참조'·') "
        return True
    return False


def _heading_label(line: str):
    """line 이 헤딩이면 인용 라벨을, 아니면 None. 본문 참조는 배제."""
    s = line.strip()
    if not s:
        return None
    m = _MD_HEAD.match(s)
    if m:
        return re.sub(r"\s+", " ", m.group(2)).strip()
    m = _ART_PAREN.match(s)
    if m:
        ttl = re.sub(r"\s+", " ", m.group(3)).strip()
        if _is_reference_heading(ttl, s[m.end():]):
            return None
        num = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        return f"{num}({ttl})"
    m = _ART_LOOSE.match(s)
    if m:
        ttl = re.sub(r"\s+", " ", m.group(3)).strip()
        if _REF_IN_TITLE.search(ttl) or _is_reference_heading(ttl, ""):
            return None
        num = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        return f"{num} {ttl}"
    m = _NUM_HEAD.match(s)
    if m:
        return re.sub(r"\s+", " ", m.group(2)).strip()[:60]
    return None


def _is_noise_chunk(piece: str) -> bool:
    """목차(점 leader 다수)·러닝헤더/쪽번호뿐인 조각인가 → 통째로 버린다.
    텍스트를 변형하지 않고 조각만 버리므로 남는 조각의 오프셋 불변식은 유지된다."""
    t = (piece or "").strip()
    if not t:
        return True
    if (t.count("…") + t.count("‥") + t.count("·")) >= 10:   # 목차 점 leader(특수문자)
        return True
    if len(re.findall(r"\.{4,}", t)) >= 5:                    # 목차 점선 leader(온점 반복)
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines and all(_RUNNING.match(ln) for ln in lines):     # 쪽번호/러닝헤더뿐
        return True
    return False


Span = Tuple[str, int, int]  # (조각 텍스트, 시작 오프셋, 끝 오프셋)
LabeledSpan = Tuple[str, int, int, str]  # + 인용 라벨(section)


def _strip_span(text: str, start: int, end: int) -> Span:
    """text[start:end] 를 strip 하고, 그 결과의 실제 오프셋을 함께 돌려준다."""
    raw = text[start:end]
    piece = raw.strip()
    if not piece:
        return ("", start, start)
    offset = start + (len(raw) - len(raw.lstrip()))
    return (piece, offset, offset + len(piece))


def _window_chunks(text: str, chunk_size: int, overlap: int) -> List[Span]:
    """Sliding window with sentence-boundary preference."""
    spans: List[Span] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # prefer to cut on a sentence/newline boundary in the tail half
            cut = max(text.rfind(". ", start, end),
                      text.rfind("다. ", start, end),
                      text.rfind("\n", start, end))
            if cut > start + chunk_size // 2:
                end = cut + 1
        piece, piece_start, piece_end = _strip_span(text, start, end)
        if piece:
            spans.append((piece, piece_start, piece_end))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return spans


def _heading_chunks(text: str) -> List[LabeledSpan]:
    """헤딩 위치에서 분할 — 각 조각 = 헤딩 + 본문, 인용 라벨 동봉.

    줄머리 헤딩만 인정(_heading_label 이 본문 속 조문 참조를 배제)하고, 목차·
    러닝헤더 조각은 버린다(_is_noise_chunk). 텍스트를 변형하지 않고 조각만
    버리므로 원본 오프셋 불변식은 그대로다."""
    # 줄 오프셋을 추적하며 헤딩 줄을 찾는다 (라벨과 시작 위치를 함께)
    heads: List[Tuple[int, str]] = []   # (줄 시작 오프셋, 라벨)
    pos = 0
    for line in text.splitlines(keepends=True):
        label = _heading_label(line)
        if label is not None:
            heads.append((pos, label))
        pos += len(line)

    if not heads:
        piece, s, e = _strip_span(text, 0, len(text))
        return [(piece, s, e, "")] if piece and not _is_noise_chunk(piece) else []

    out: List[LabeledSpan] = []
    # 첫 헤딩 이전 도입부 (목차/가이드면 버려진다)
    pre, ps, pe = _strip_span(text, 0, heads[0][0])
    if pre and not _is_noise_chunk(pre):
        out.append((pre, ps, pe, ""))
    for i, (start, label) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        piece, s, e = _strip_span(text, start, end)
        if piece and not _is_noise_chunk(piece):
            out.append((piece, s, e, label))
    return out


def segment(text: str, mode: str = "auto", chunk_size: int = 800,
            overlap: int = 120, source: str = "") -> List[Chunk]:
    """Split text into Chunks with provenance (source + index + char span)."""
    if not text or not text.strip():
        return []

    # 오프셋은 원본 text 기준이어야 하므로, strip 대신 스팬으로 다룬다.
    body_text, body_start, body_end = _strip_span(text, 0, len(text))

    if mode == "auto":
        heading_count = len(_HEADING.findall(body_text))
        if heading_count >= 2:
            mode = "heading"
        elif len(body_text) <= chunk_size:
            return [Chunk(text=body_text, source=source, index=0,
                          char_start=body_start, char_end=body_end)]
        else:
            mode = "window"

    if mode == "heading":
        labeled = _heading_chunks(body_text)
        # 큰 조는 추출 입력이 넘치지 않게 윈도우로 쪼갠다(라벨은 하위 조각에 상속)
        bounded: List[LabeledSpan] = []
        for piece, start, end, label in labeled:
            if len(piece) > chunk_size * 2:
                bounded.extend(
                    (sub, start + sub_start, start + sub_end, label)
                    for sub, sub_start, sub_end in _window_chunks(
                        piece, chunk_size, overlap))
            else:
                bounded.append((piece, start, end, label))
        labeled_spans = bounded
    else:
        labeled_spans = [(p, s, e, "")
                         for p, s, e in _window_chunks(body_text, chunk_size, overlap)]

    # spans 는 body_text 기준 — 원본 기준으로 되돌린다
    return [Chunk(text=piece, source=source, index=i, section=label,
                  char_start=body_start + start, char_end=body_start + end)
            for i, (piece, start, end, label) in enumerate(labeled_spans)]
