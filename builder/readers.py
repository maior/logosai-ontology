"""
Data readers — turn heterogeneous input files into plain text.

Each format gets a small deterministic reader (no LLM here). Unsupported
formats raise UnsupportedFormatError so callers can decide to skip or
surface the error; read_folder skips them silently by design.
"""

import csv
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonld", ".csv", ".pdf",
                        ".docx", ".doc", ".hwp"}


class UnsupportedFormatError(ValueError):
    """File extension has no registered reader."""


def extract_pdf_images(path, *, min_area=None, min_side=None,
                       caption_band: float = 60.0) -> List:
    """PDF 안 이미지를 ImageAsset(주소 = page + bbox)으로 뽑는다.

    **왜 pdfplumber(MIT)인가**: bbox 를 주는 리더가 필요하다. pypdf 의
    `page.images` 는 (name, data) 뿐이라 좌표가 없고, PyMuPDF(fitz)는 좌표를
    주지만 **AGPL-3.0** 이라 9274 를 네트워크 서비스로 제공하는 이 저장소에
    전염 위험이 있다. 그래서 fitz 는 쓰지 않는다.

    좌표 규약: bbox 는 **PDF 원좌표 (x0,y0,x1,y1)** — y 는 페이지 아래 기준.
    pdfplumber 의 crop 은 위 기준이므로 meta["page_height"] 를 함께 실어
    `top = page_height - y1` 을 유도할 수 있게 한다. 이게 "재크롭해서 대조
    가능"이라는 주장의 실체다(테스트가 실제로 크롭해 확인한다).

    지연 import + 빈 목록 degrade: `_read_pdf` 가 pypdf 를 지연 import 하는
    것과 같은 관례다. images extra 없이도 인제스트는 돌아야 한다.
    """
    from .image_assets import (DEFAULT_MIN_AREA, DEFAULT_MIN_SIDE, ImageAsset,
                               find_caption, is_decorative)

    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        return []
    try:
        import pdfplumber
        if pdfplumber is None:      # sys.modules 에 None → ImportError 와 동등
            raise ImportError("pdfplumber unavailable")
    except ImportError:
        logger.warning("⚠️ pdfplumber 없음 — 이미지 추출 건너뜀 "
                       "(pip install 'ontology[images]')")
        return []

    area = DEFAULT_MIN_AREA if min_area is None else min_area
    side = DEFAULT_MIN_SIDE if min_side is None else min_side
    assets: List[ImageAsset] = []
    skipped_decorative = 0
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                page_height = float(page.height)
                for raw in (page.images or []):
                    asset = _image_asset_from(raw, str(file_path), page,
                                              page_height, caption_band,
                                              ImageAsset, find_caption)
                    if asset is None:
                        continue
                    if is_decorative(asset, min_area=area, min_side=side):
                        skipped_decorative += 1
                        continue
                    assets.append(asset)
    except Exception as e:
        # 손상 PDF 하나가 빌드를 죽이면 안 된다 — 텍스트는 이미 읽혔을 수 있다.
        logger.warning(f"⚠️ 이미지 추출 실패 ({file_path.name}): {e}")
        return assets
    if skipped_decorative:
        # 조용한 탈락 금지 — 몇 개를 왜 뺐는지 남긴다.
        logger.info(f"🖼️ {file_path.name}: 이미지 {len(assets)}개 추출, "
                    f"장식 {skipped_decorative}개 제외")
    return assets


def _image_asset_from(raw: dict, source: str, page, page_height: float,
                      caption_band: float, ImageAsset, find_caption):
    """pdfplumber image dict → ImageAsset. 실패한 항목은 None (건너뛴다)."""
    try:
        x0, y0, x1, y1 = (float(raw["x0"]), float(raw["y0"]),
                          float(raw["x1"]), float(raw["y1"]))
    except (KeyError, TypeError, ValueError):
        return None

    # 내용 해시 — 재추출 멱등성의 근거. rawdata 는 None 으로 오는 경우가 있어
    # get_data() 를 쓴다(실측). 실패하면 **빈 문자열** — 없는 해시를 지어내지
    # 않는다. 좌표만으로도 인용은 가능하므로 자산 자체는 살린다.
    sha = ""
    meta = {"page_height": page_height, "srcsize": list(raw.get("srcsize") or ())}
    try:
        data = raw["stream"].get_data()
        sha = hashlib.sha256(data).hexdigest()
    except Exception:
        meta["sha_unavailable"] = True

    src_w, src_h = (raw.get("srcsize") or (0, 0))[:2] or (0, 0)
    return ImageAsset(
        source=source, page=int(raw.get("page_number") or page.page_number),
        bbox=(x0, y0, x1, y1), sha256=sha,
        width=int(src_w or 0), height=int(src_h or 0),
        caption=_caption_near(page, raw, page_height, caption_band,
                              find_caption),
        meta=meta,
    )


def _caption_near(page, raw: dict, page_height: float, band: float,
                  find_caption) -> str:
    """이미지 아래 → 위 밴드에서 캡션 줄을 찾는다.

    아래를 먼저 보는 이유: 그림 캡션은 아래가 관례다. 표 캡션은 위에 오므로
    아래에서 못 찾으면 위를 본다. crop 은 **위 기준** 좌표를 받는다.
    """
    top = page_height - float(raw["y1"])
    bottom = page_height - float(raw["y0"])
    x0, x1 = float(raw["x0"]), float(raw["x1"])
    for lo, hi in ((bottom, min(page_height, bottom + band)),
                   (max(0.0, top - band), top)):
        if hi - lo <= 0:
            continue
        try:
            text = page.crop((x0, lo, x1, hi)).extract_text() or ""
        except Exception:
            continue
        caption = find_caption(text)
        if caption:
            return caption
    return ""


def _read_json(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # pretty text keeps keys next to values so the extractor sees context
    return json.dumps(data, ensure_ascii=False, indent=2)


def _read_csv(path: Path) -> str:
    lines: List[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ""
    header = rows[0]
    lines.append(" | ".join(header))
    for row in rows[1:]:
        pairs = [f"{col}: {val}" for col, val in zip(header, row)]
        lines.append(" | ".join(pairs))
    return "\n".join(lines)


_PDF_PAGENUM = re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$|^\d{1,4}$")


def clean_pdf_pages(pages: List[str]) -> str:
    """페이지 텍스트 → 러닝헤더·꼬리말·쪽번호 제거 + 공백 정리한 본문(**순수 함수**).

    러닝헤더는 페이지마다 쪽번호만 달라지므로 숫자를 '#'로 정규화한 형태의
    출현 빈도로 검출한다(여러 페이지 반복 + 짧은 줄). 정규화된 추출 텍스트를
    '원본'으로 저장하는 전략(옵션 a) — 청크·오프셋·인용이 이 텍스트 기준으로
    일관된다. 단어 내 공백(한글 PDF 자간)은 건드리지 않는다: 잘못 붙이면 의미가
    깨지고, cjk 검색은 자간이 있어도 동작한다(aicoach 실증)."""
    per_page = [[ln.strip() for ln in (p or "").splitlines()] for p in pages]
    n = len(per_page)
    thresh = max(3, int(n * 0.4))
    # (1) 페이지마다 반복되는 '독립 줄' 러닝헤더 (숫자만 다른 것도 정규화로 묶음)
    freq: Counter = Counter()
    # (2) 우측정렬 꼬리말이 줄 앞에 "쪽번호  <꼬리말>  <본문>" 으로 붙는 흔한
    #     아티팩트 — 쪽번호와 본문 사이 반복 문구를 검출해 그 접두만 벗긴다.
    lead: Counter = Counter()
    _LEAD = re.compile(r"^\d{1,4}\s{2,}(.{3,40}?)\s{2,}\S")
    for lines in per_page:
        for key in {re.sub(r"\d+", "#", ln) for ln in lines if ln}:
            freq[key] += 1
        for ln in lines:
            m = _LEAD.match(ln)
            if m:
                lead[m.group(1).strip()] += 1
    running = {k for k, c in freq.items() if c >= thresh and len(k) <= 60}
    footers = {p for p, c in lead.items() if c >= thresh}
    foot_re = (re.compile(r"^\d{1,4}\s{2,}(?:" +
                          "|".join(re.escape(f) for f in footers) + r")\s{2,}")
               if footers else None)
    out: List[str] = []
    for lines in per_page:
        kept = []
        for ln in lines:
            if not ln or _PDF_PAGENUM.match(ln):
                continue
            if re.sub(r"\d+", "#", ln) in running:
                continue
            if foot_re is not None:
                ln = foot_re.sub("", ln).strip()   # 접두 꼬리말 제거, 본문 유지
            if ln:
                kept.append(ln)
        if kept:
            out.append("\n".join(kept))
    text = "\n\n".join(out)
    text = re.sub(r"[ \t]{2,}", " ", text)      # 과한 가로 공백만 축약
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # lazy: only needed for pdf inputs
    reader = PdfReader(str(path))
    return clean_pdf_pages([(p.extract_text() or "") for p in reader.pages])


def _read_docx(path: Path) -> str:
    """python-docx — 문단 + 표. 표는 행 단위 'a | b' 로 편다 (CSV reader 와
    같은 규약: 셀 인접성이 보존돼야 추출 LLM 이 관계를 읽을 수 있다)."""
    import docx  # lazy: only needed for docx inputs

    document = docx.Document(str(path))
    parts: List[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _read_doc(path: Path) -> str:
    """구형 바이너리 .doc — macOS textutil 변환. 없으면 명확한 에러.

    .doc 파서를 직접 들이는 것은 의존성 비용 대비 손해다 (포맷이 사실상
    소멸 중). 변환 도구가 없으면 뭘 하라는지 말하고 실패한다."""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("textutil"):  # macOS
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / (path.stem + ".txt")
            subprocess.run(
                ["textutil", "-convert", "txt", "-output", str(out), str(path)],
                check=True, capture_output=True, timeout=60)
            return out.read_text(encoding="utf-8")
    raise UnsupportedFormatError(
        f"'.doc' needs textutil (macOS) — convert {path.name} to .docx instead")


# ─── HWP 5.0 (한글) ──────────────────────────────────────────────────
# OLE 복합 파일: FileHeader(서명+압축 플래그) + BodyText/Section{n}(레코드
# 스트림, 보통 zlib raw 압축). 레코드 헤더 = 4바이트 LE DWORD:
# tag(10bit) | level(10bit) | size(12bit); size==0xFFF 면 다음 4바이트가 실크기.
# 본문 레코드는 HWPTAG_PARA_TEXT(67), UTF-16LE.

_HWP_PARA_TEXT_TAG = 67
_HWP_SIGNATURE = b"HWP Document File"
# 32 미만 제어문자 중 인라인/확장 컨트롤 — WCHAR 8개(16바이트)를 통째로 차지
# 한다. 1개만 걷어내면 나머지 7개가 쓰레기 글자로 새어 나온다 (명세 §4.2).
_HWP_EXTENDED_CONTROLS = frozenset(
    {1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23})


def _decode_hwp_text(payload: bytes) -> str:
    chars: List[str] = []
    i = 0
    end = len(payload) - len(payload) % 2
    while i < end:
        (code,) = struct.unpack_from("<H", payload, i)
        i += 2
        if code < 32:
            if code in _HWP_EXTENDED_CONTROLS:
                i += 14  # 확장 컨트롤의 나머지 7 WCHAR
            elif code in (10, 13):
                chars.append("\n")
            # 그 외 제어문자는 버린다
        else:
            chars.append(chr(code))
    return "".join(chars)


def _parse_hwp_section(data: bytes) -> str:
    """BodyText 섹션의 레코드를 걸어가며 본문 텍스트만 모은다.
    잘린 데이터에서도 죽지 않는다 — 읽은 데까지 돌려준다."""
    texts: List[str] = []
    pos = 0
    n = len(data)
    while pos + 4 <= n:
        (dword,) = struct.unpack_from("<I", data, pos)
        pos += 4
        tag = dword & 0x3FF
        size = (dword >> 20) & 0xFFF
        if size == 0xFFF:
            if pos + 4 > n:
                break
            (size,) = struct.unpack_from("<I", data, pos)
            pos += 4
        chunk = data[pos:pos + size]
        if tag == _HWP_PARA_TEXT_TAG and chunk:
            text = _decode_hwp_text(chunk)
            if text.strip():
                texts.append(text)
        pos += size
    return "\n".join(texts)


def _read_hwp(path: Path) -> str:
    import zlib

    import olefile  # lazy: only needed for hwp inputs

    ole = olefile.OleFileIO(str(path))
    try:
        header = ole.openstream("FileHeader").read()
        if not header.startswith(_HWP_SIGNATURE):
            raise UnsupportedFormatError(
                f"not an HWP 5.0 file: {path.name}")
        compressed = bool(header[36] & 1)

        sections = sorted(
            (entry for entry in ole.listdir()
             if len(entry) == 2 and entry[0] == "BodyText"),
            key=lambda entry: int(entry[1].replace("Section", "") or 0))
        texts: List[str] = []
        for entry in sections:
            data = ole.openstream(entry).read()
            if compressed:
                try:
                    data = zlib.decompress(data, -15)
                except zlib.error:
                    continue  # 한 섹션이 깨져도 나머지는 계속
            section_text = _parse_hwp_section(data)
            if section_text.strip():
                texts.append(section_text)
        if texts:
            return "\n\n".join(texts)

        # 본문 파싱이 비면 미리보기 텍스트로 폴백 (앞부분만이라도)
        if ole.exists("PrvText"):
            return ole.openstream("PrvText").read().decode("utf-16-le", "ignore")
        return ""
    finally:
        ole.close()


def read_file(path) -> str:
    """Read a single file into plain text. Raises FileNotFoundError or
    UnsupportedFormatError — never returns a silent empty result for
    an unknown format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    if suffix in (".json", ".jsonld"):
        return _read_json(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".doc":
        return _read_doc(path)
    if suffix == ".hwp":
        return _read_hwp(path)
    raise UnsupportedFormatError(f"no reader for '{suffix}' ({path.name})")


def read_folder(path, patterns: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Read every supported file under a folder (recursive).

    Returns [(source_path, text)] sorted by path. Unsupported extensions
    are skipped; read errors are logged and skipped so one bad file
    never aborts a whole dataset.
    """
    root = Path(path)
    documents: List[Tuple[str, str]] = []

    candidates = sorted(root.rglob("*"))
    for file_path in candidates:
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if patterns and not any(file_path.match(p) for p in patterns):
            continue
        try:
            documents.append((str(file_path), read_file(file_path)))
        except Exception as e:
            logger.warning(f"⚠️ Reader skipped {file_path.name}: {e}")
    return documents
