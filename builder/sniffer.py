"""
감식기(Sniffer) — 업로드 파일의 종(種) 판별 + 인제스트 설계 제안.

실데이터 조사(2026-07-17, aicoach 36 PDF · KorAct ontology.jsonld · ontology
샘플 5종)에서 나온 결론: 업로드 파일은 4종이고 종마다 올바른 경로가 다르다.
이전에는 전부 LLM 추출로 직행해서, wikidata 1,999 레코드 같은 정형 데이터가
~2,000 LLM 콜을 낭비하며 환각 위험까지 졌다 (올바른 경로: 매핑 1콜 + 결정적
인제스트 — aicoach premiums 원칙 "이미 구조화된 데이터에 LLM 은 비용·오염
양쪽 손해").

    records        정형 레코드 (wikidata JSON, CSV)  → 매핑 제안 + build_from_records
    articled       조문형 문서 (약관·법령·규정)       → heading 분할 + LLM 추출
    prose          자유 산문                         → window 분할 + LLM 추출
    seed_ontology  이미 온톨로지인 파일 (JSON-LD)     → 멱등 upsert (추출 아님)

역할 분담 — 결정적인 것과 판단이 필요한 것을 가른다:
- **결정적 (LLM 0콜)**: 종 판별(JSON 구조·heading 밀도), 카디널리티 통계,
  노드/속성 역할 제안. distinct/total 은 계산이지 판단이 아니다 —
  "찾을 것인가 읽을 것인가"를 데이터로 측정한다.
- **LLM (gemini-3.5-flash)**: 이름 짓기와 의미 해석만. 술어명 제안(매핑),
  파일명 메타데이터(문서종류·엔티티·버전 — 분야가 다양하므로 "약관"·"요약서"
  같은 어휘를 하드코딩할 수 없다, 프로젝트 절대 원칙).
- **LLM 출력은 전부 불신**: 실제 필드명과 대조해 검증하고, 지어낸 필드는
  버린다 (aicoach clean_extraction 원칙).
"""

import asyncio
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from .extractor import parse_llm_json
from .segmenter import _HEADING, segment

# ─── 판별 임계값 (통계적 규칙 — 도메인 무관) ─────────────────────────

# 레코드 배열 인정: 항목들의 키 집합이 첫 항목과 평균 이 비율 이상 겹칠 때.
# 동질성이 낮으면 "키가 제각각인 JSON" 이고 그건 산문 경로가 맞다.
KEY_OVERLAP_THRESHOLD = 0.6

# 시드 온톨로지 마커 — JSON-LD/SKOS 계열의 구조 필드 (도메인 어휘가 아니라
# 표준 어휘라 하드코딩이 아니다)
SEED_MARKERS = ("prefLabel", "@id", "@type", "definition")

# 노드/속성 역할 제안 임계값
NUMERIC_RATIO_ATTR = 0.9      # 값의 90%+ 가 숫자면 리터럴 → 속성
SHARED_VALUE_RATIO = 0.5      # distinct/present ≤ 0.5 면 공유값 → 노드 후보
IDENTITY_RATIO = 0.95         # distinct/present ≥ 0.95 인 문자열 → 정체성 후보

# 매핑 제안에 LLM 에게 보여줄 샘플 레코드 수 — 전체를 보낼 필요가 없다.
# 필드 구조는 통계가 이미 요약했고, 샘플은 이름 짓기의 맥락일 뿐이다.
MAPPING_SAMPLE_SIZE = 5


# ─── 결과 모델 ───────────────────────────────────────────────────────

@dataclass
class FileProfile:
    """파일 하나의 감식 결과 — 확인 게이트에 그대로 표시되는 단위."""
    path: str
    filename: str
    format: str
    species: str                      # records | articled | prose | seed_ontology | unknown
    record_count: int = 0
    records_path: str = ""            # 레코드가 있던 JSON 키 경로
    fields: List[str] = field(default_factory=list)
    field_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    field_roles: Dict[str, str] = field(default_factory=dict)
    hierarchy_count: int = 0          # 계층 sidecar 항목 수 (is_a 재료)
    mapping_proposal: Optional[Dict[str, Any]] = None
    hierarchy_proposal: Optional[Dict[str, Any]] = None
    mapping_error: str = ""
    chunk_count: int = 0
    estimated_llm_calls: int = 0
    filename_meta: Optional[Dict[str, Any]] = None
    error: str = ""


@dataclass
class AnalysisReport:
    """데이터셋 전체의 감식 결과 + 비용 견적."""
    files: List[FileProfile] = field(default_factory=list)
    total_estimated_llm_calls: int = 0


# ─── 1. JSON 구조 분석 (결정적) ──────────────────────────────────────

def _is_homogeneous_records(items: Any) -> bool:
    """dict 리스트이고 키 집합이 서로 충분히 겹치는가."""
    if not isinstance(items, list) or len(items) < 2:
        return False
    sample = items[:50]
    if not all(isinstance(item, dict) for item in sample):
        return False
    first_keys = set(sample[0].keys())
    if not first_keys:
        return False
    overlaps = []
    for item in sample[1:]:
        keys = set(item.keys())
        union = first_keys | keys
        overlaps.append(len(first_keys & keys) / len(union) if union else 0.0)
    return (sum(overlaps) / len(overlaps)) >= KEY_OVERLAP_THRESHOLD


def _looks_like_seed(items: List[Dict[str, Any]]) -> bool:
    """항목 과반이 JSON-LD/SKOS 구조 필드를 가지면 시드 온톨로지다."""
    if not items:
        return False
    sample = items[:50]
    marked = sum(1 for item in sample
                 if isinstance(item, dict)
                 and any(m in item for m in SEED_MARKERS))
    return marked / len(sample) > 0.5


def analyze_json_structure(data: Any) -> Tuple[str, List[Dict[str, Any]], str]:
    """JSON 값 → (species, records, records_path).

    시드 판별이 레코드 판별보다 **먼저**다 — 시드 항목들도 동질 dict 배열이라
    순서를 바꾸면 온톨로지 파일이 레코드로 오인돼 upsert 대신 추출 경로를 탄다.
    """
    # dict 꼴
    if isinstance(data, dict):
        graph = data.get("@graph")
        if isinstance(graph, list) and _looks_like_seed(graph):
            return ("seed_ontology", [i for i in graph if isinstance(i, dict)], "@graph")

        # 안쪽 키들 중 가장 큰 동질 레코드 배열을 찾는다
        # (korean_heritage.json 꼴: {dataset, 설명, items: [...]})
        best_key, best_items = "", []
        for key, value in data.items():
            if _is_homogeneous_records(value) and len(value) > len(best_items):
                best_key, best_items = key, value
        if best_items:
            if _looks_like_seed(best_items):
                return ("seed_ontology", best_items, best_key)
            return ("records", best_items, best_key)
        return ("prose", [], "")

    # list 꼴
    if isinstance(data, list):
        if _looks_like_seed([i for i in data if isinstance(i, dict)]):
            return ("seed_ontology", [i for i in data if isinstance(i, dict)], "")
        if _is_homogeneous_records(data):
            return ("records", data, "")
        return ("prose", [], "")

    return ("prose", [], "")


def find_sidecar_lists(data: Any,
                       main_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """주 레코드 외의 동질 dict 배열(예: wikidata 의 hierarchy)을 찾는다.

    이건 is_a 간선의 재료다 — 놓치면 계층 롤업 추론(+554 실측)이 통째로
    빠진 온톨로지가 만들어진다. 키 이름("hierarchy")을 검사하지 않는 이유:
    도메인마다 이름이 다르다. 모양(동질 dict 배열)으로만 판별한다.
    """
    if not isinstance(data, dict):
        return {}
    sidecars: Dict[str, List[Dict[str, Any]]] = {}
    for key, value in data.items():
        if key == main_path:
            continue
        if isinstance(value, list) and value and all(
                isinstance(i, dict) for i in value[:50]):
            sidecars[key] = value
    return sidecars


# ─── 2. 텍스트 종 판별 (결정적 — segmenter 와 같은 신호) ─────────────

def detect_text_species(text: str) -> str:
    """heading(제N조·마크다운·번호 목차) 2개 이상이면 조문형.

    segmenter 의 auto 모드와 같은 규칙을 쓴다 — 판별과 분할이 다른 기준을
    쓰면 '조문형으로 판별해놓고 window 로 자르는' 불일치가 생긴다.
    """
    return "articled" if len(_HEADING.findall(text)) >= 2 else "prose"


# ─── 3. 카디널리티 통계 → 역할 제안 (결정적) ─────────────────────────

def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.replace(",", ""))
            return True
        except ValueError:
            return False
    return False


def field_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """필드별 카디널리티 통계. 순수 계산 — LLM 0콜."""
    total = len(records)
    all_fields: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for key in record:
            all_fields.setdefault(key, {"values": [], "numeric": 0})

    for record in records:
        for key, slot in all_fields.items():
            value = record.get(key)
            if value is None:
                continue
            slot["values"].append(str(value))
            if _is_numeric(value):
                slot["numeric"] += 1

    stats: Dict[str, Dict[str, Any]] = {}
    for key, slot in all_fields.items():
        present = len(slot["values"])
        stats[key] = {
            "total": total,
            "present": present,
            "distinct": len(set(slot["values"])),
            "numeric_ratio": (slot["numeric"] / present) if present else 0.0,
            "avg_len": (sum(len(v) for v in slot["values"]) / present) if present else 0.0,
        }
    return stats


def suggest_roles(stats: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """필드 → identity | node_candidate | attribute.

    "찾을 것인가, 읽을 것인가"를 통계로 측정한 것이다:
    - 공유값(distinct/present 낮음)은 여러 개체가 수렴하는 질의 축 → 노드 후보
      (heritage 실측: hasDesignation 엣지 1,992개가 국보/보물 노드로 수렴)
    - 숫자·고유값 리터럴은 읽는 값 → 속성 (날짜 노드 수천 개는 아무 질의의
      축도 되지 않는다)
    - 전부 고유한 문자열 → 정체성(name_field 후보)
    도메인 어휘를 전혀 보지 않는다 — 보험이든 유산이든 상품이든 같은 규칙.
    """
    roles: Dict[str, str] = {}
    for key, s in stats.items():
        present = s["present"] or 1
        distinct_ratio = s["distinct"] / present
        if s["numeric_ratio"] >= NUMERIC_RATIO_ATTR:
            roles[key] = "attribute"
        elif distinct_ratio >= IDENTITY_RATIO:
            roles[key] = "identity"
        elif distinct_ratio <= SHARED_VALUE_RATIO:
            roles[key] = "node_candidate"
        else:
            roles[key] = "attribute"  # 애매하면 속성 — 노드 오염이 더 비싸다
    return roles


# ─── 4. LLM 프롬프트 + 검증 ─────────────────────────────────────────

_MAPPING_PROMPT = """당신은 지식그래프 인제스트 설계자입니다.
아래 필드 통계와 샘플을 보고 build_from_records 매핑을 JSON으로만 제안하세요.

## 노드 vs 속성 원칙
- role=node_candidate 필드(여러 레코드가 공유하는 값)만 relations 로 만드세요.
  공유값이 노드여야 "국보인 것 전부" 같은 역방향 질의가 가능합니다.
- role=attribute 필드(숫자·고유 리터럴)는 relations 에 넣지 마세요 —
  자동으로 노드 속성으로 보존됩니다.
- name_field 는 role=identity 필드 중에서 고르세요.
- predicate 는 영문 camelCase, target_type/node_type 은 영문 PascalCase 로
  데이터의 의미에 맞게 지으세요.
- 실제 존재하는 필드명만 사용하세요. 지어내지 마세요.

## 필드 통계 (role 은 카디널리티 기반 제안)
{stats_block}

## 샘플 레코드 ({sample_count}개)
{samples}

{sidecar_block}## 출력 형식 (JSON only)
{{"node_type": "...", "name_field": "...", "type_field": null 또는 "필드명",
  "relations": [{{"field": "...", "predicate": "...", "target_type": "..."}}]{hierarchy_shape}}}
"""

_SIDECAR_BLOCK = """## 부속 목록 (주 레코드 외의 배열)
{sidecar_samples}
위 부속 목록이 상하위 계층(예: 하위개념→상위개념)으로 보이면 hierarchy 를
함께 제안하세요. child_field=하위 이름 필드, parent_field=상위 이름 필드,
node_type=계층 노드의 타입(관련 relation 의 target_type 과 맞추세요).
계층이 아니면 hierarchy 를 넣지 마세요.

"""

_HIERARCHY_SHAPE = (',\n  "hierarchy": {"path": "...", "child_field": "...", '
                    '"parent_field": "...", "node_type": "..."}')

_FILENAME_PROMPT = """다음 파일명들을 분석하세요. 파일명에는 흔히 문서종류·대상
엔티티·버전(날짜)이 들어 있습니다 (예: "통합약관_X보험_20260101.pdf" —
어휘는 분야마다 다르므로 패턴을 가정하지 말고 의미로 판단하세요).

각 파일에 대해 JSON으로만 답하세요:
- doc_kind: 문서의 종류 (파일명에서 읽히는 대로, 없으면 null)
- entity: 문서가 다루는 대상 (없으면 null)
- version: 날짜/버전 (ISO 형식으로, 없으면 null)
- trust: authoritative(약관·법령·원본) | summary(요약·소개) | unknown
  — 요약 문서에서 뽑은 사실이 원본 문서의 사실을 덮어쓰면 안 되므로 중요합니다.

## 파일명 목록
{filenames}

## 출력 형식 (JSON only)
{{"files": [{{"filename": "...", "doc_kind": ..., "entity": ..., "version": ..., "trust": "..."}}]}}
"""


def build_mapping_prompt(stats: Dict[str, Dict[str, Any]],
                         roles: Dict[str, str],
                         samples: List[Dict[str, Any]],
                         sidecars: Optional[Dict[str, List[Dict[str, Any]]]] = None
                         ) -> str:
    lines = []
    for key, s in stats.items():
        lines.append(f"- {key}: distinct {s['distinct']}/{s['present']}, "
                     f"numeric {s['numeric_ratio']:.0%}, role={roles.get(key)}")

    sidecar_block = ""
    hierarchy_shape = ""
    if sidecars:
        previews = [f"- '{key}' ({len(items)}개): "
                    + json.dumps(items[:3], ensure_ascii=False)
                    for key, items in sidecars.items()]
        sidecar_block = _SIDECAR_BLOCK.format(sidecar_samples="\n".join(previews))
        hierarchy_shape = _HIERARCHY_SHAPE

    return _MAPPING_PROMPT.format(
        stats_block="\n".join(lines),
        sample_count=len(samples),
        samples=json.dumps(samples, ensure_ascii=False, indent=1)[:2000],
        sidecar_block=sidecar_block,
        hierarchy_shape=hierarchy_shape)


def build_filename_prompt(filenames: List[str]) -> str:
    return _FILENAME_PROMPT.format(
        filenames="\n".join(f"- {name}" for name in filenames))


def parse_mapping_proposal(raw: str,
                           fields: List[str]) -> Optional[Dict[str, Any]]:
    """LLM 매핑 제안 검증 — 출력은 불신한다.

    - name_field 가 실제 필드가 아니면 **전체 무효** (정체성 없는 인제스트 금지)
    - 지어낸 필드를 참조하는 relation 은 그것만 버린다 (전체를 죽이지 않는다)
    - type_field 가 가짜면 None 으로 강등
    """
    parsed = parse_llm_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return None

    known = set(fields)
    name_field = parsed.get("name_field")
    if name_field not in known:
        return None

    type_field = parsed.get("type_field")
    if type_field is not None and type_field not in known:
        type_field = None

    relations = []
    for rel in parsed.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        if rel.get("field") in known and rel.get("predicate") and rel.get("target_type"):
            relations.append({"field": rel["field"],
                              "predicate": str(rel["predicate"]),
                              "target_type": str(rel["target_type"])})

    node_type = str(parsed.get("node_type") or "Record")
    return {"node_type": node_type, "name_field": name_field,
            "type_field": type_field, "relations": relations}


def parse_hierarchy_proposal(raw: str,
                             sidecars: Dict[str, List[Dict[str, Any]]]
                             ) -> Optional[Dict[str, Any]]:
    """LLM 의 hierarchy 제안 검증 — 출력은 불신한다.

    path 는 실제 sidecar 목록이어야 하고 child/parent 필드는 그 항목들에
    실재해야 한다. 하나라도 가짜면 전체 무효 — 틀린 계층은 없는 계층보다
    나쁘다 (롤업 추론이 조용히 엉뚱한 집계를 낸다).
    """
    parsed = parse_llm_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return None
    spec = parsed.get("hierarchy")
    if not isinstance(spec, dict):
        return None

    path = spec.get("path")
    items = sidecars.get(path)
    if not items:
        return None
    item_keys = set()
    for item in items[:10]:
        item_keys |= set(item.keys())
    child_field = spec.get("child_field")
    parent_field = spec.get("parent_field")
    if child_field not in item_keys or parent_field not in item_keys:
        return None
    if child_field == parent_field:
        return None

    return {"path": path, "child_field": child_field,
            "parent_field": parent_field,
            "node_type": str(spec.get("node_type") or "Class")}


# ─── 5. DatasetAnalyzer ──────────────────────────────────────────────

# 감식기가 직접 JSON 을 파싱하는 확장자 (readers 는 텍스트로 펴버린다)
_JSON_EXTENSIONS = {".json", ".jsonld"}
_RECORD_TEXT_EXTENSIONS = {".csv"}


class DatasetAnalyzer:
    """폴더 → 파일별 FileProfile + 비용 견적.

    LLM 은 주입 가능(테스트는 결정적 가짜). 프로덕션 기본은
    core.llm_provider 경유 gemini-3.5-flash (ONTOLOGY_EXTRACTION_MODEL) —
    빌더의 추출 모델과 같은 기본을 공유한다.
    """

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None,
                 llm_provider: str = "google",
                 llm_model: Optional[str] = None,
                 llm_base_url: Optional[str] = None,
                 chunk_size: int = 800, overlap: int = 120):
        self.llm_fn = llm_fn
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._provider = None

    async def _call_llm(self, prompt: str) -> str:
        if self.llm_fn is not None:
            return await asyncio.to_thread(self.llm_fn, prompt)
        if self._provider is None:
            from ..core.llm_provider import resolve_provider
            self._provider = resolve_provider(
                self.llm_provider, self.llm_model, base_url=self.llm_base_url)
        return await self._provider.complete(prompt)

    # ── 파일 하나 ────────────────────────────────────────────────────

    def _profile_json(self, path: Path, profile: FileProfile) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        species, records, records_path = analyze_json_structure(data)
        profile.species = species
        profile.records_path = records_path
        if species == "records":
            self._fill_records(profile, records)
            sidecars = find_sidecar_lists(data, records_path)
            profile.hierarchy_count = sum(len(v) for v in sidecars.values())
            profile._sidecars = sidecars  # type: ignore[attr-defined]
        elif species == "seed_ontology":
            profile.record_count = len(records)
            profile.estimated_llm_calls = 0  # upsert — 추출 없음
        else:
            # 레코드도 시드도 아닌 JSON → 텍스트로 펴서 산문 경로
            from . import readers
            self._profile_text(readers.read_file(path), profile)

    def _profile_csv(self, path: Path, profile: FileProfile) -> None:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) >= 2:
            profile.species = "records"
            self._fill_records(profile, rows)
        else:
            profile.species = "prose"

    def _fill_records(self, profile: FileProfile,
                      records: List[Dict[str, Any]]) -> None:
        profile.record_count = len(records)
        profile.field_stats = field_stats(records)
        profile.fields = list(profile.field_stats)
        profile.field_roles = suggest_roles(profile.field_stats)
        # 레코드 수와 무관하게 매핑 제안 1콜 — 이것이 이 감식기의 존재 이유다
        # (1,999 레코드 = 1콜, LLM 추출 경로였다면 ~1,999콜)
        profile.estimated_llm_calls = 1
        profile._records_sample = records[:MAPPING_SAMPLE_SIZE]  # type: ignore[attr-defined]

    def _profile_text(self, text: str, profile: FileProfile) -> None:
        profile.species = detect_text_species(text)
        chunks = segment(text, mode="auto", chunk_size=self.chunk_size,
                         overlap=self.overlap)
        profile.chunk_count = len(chunks)
        profile.estimated_llm_calls = max(len(chunks), 1)

    def sniff_file(self, path) -> FileProfile:
        """파일 하나의 결정적 감식 (LLM 0콜)."""
        path = Path(path)
        profile = FileProfile(path=str(path), filename=path.name,
                              format=path.suffix.lower().lstrip("."),
                              species="unknown")
        try:
            suffix = path.suffix.lower()
            if suffix in _JSON_EXTENSIONS:
                self._profile_json(path, profile)
            elif suffix in _RECORD_TEXT_EXTENSIONS:
                self._profile_csv(path, profile)
            else:
                from . import readers
                self._profile_text(readers.read_file(path), profile)
        except Exception as e:
            profile.error = str(e)
            logger.warning(f"⚠️ Sniff failed for {path.name}: {e}")
        return profile

    # ── 데이터셋 전체 ────────────────────────────────────────────────

    async def analyze(self, folder) -> AnalysisReport:
        from .readers import SUPPORTED_EXTENSIONS

        allowed = SUPPORTED_EXTENSIONS | _JSON_EXTENSIONS
        root = Path(folder)
        report = AnalysisReport()

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed:
                continue
            report.files.append(self.sniff_file(file_path))

        # LLM 단계 1 — 레코드 파일별 매핑(+계층) 제안 (파일당 1콜)
        for profile in report.files:
            if profile.species != "records":
                continue
            samples = getattr(profile, "_records_sample", [])
            sidecars = getattr(profile, "_sidecars", {})
            try:
                raw = await self._call_llm(build_mapping_prompt(
                    profile.field_stats, profile.field_roles, samples,
                    sidecars=sidecars))
                profile.mapping_proposal = parse_mapping_proposal(
                    raw, profile.fields)
                if sidecars:
                    profile.hierarchy_proposal = parse_hierarchy_proposal(
                        raw, sidecars)
                if profile.mapping_proposal is None:
                    profile.mapping_error = "LLM 제안이 검증을 통과하지 못함"
            except Exception as e:
                # LLM 이 죽어도 감식은 산다 — 종·통계·역할은 결정적이므로.
                # 매핑만 비고, 사용자는 통계를 보고 손으로 쓸 수 있다.
                profile.mapping_error = str(e)
                logger.warning(f"⚠️ Mapping proposal failed ({profile.filename}): {e}")

        # LLM 단계 2 — 파일명 메타데이터, 전체 파일 **한 번에** 1콜
        # (파일명 의미는 분야마다 달라 하드코딩 불가 — LLM 판단 영역)
        if report.files:
            try:
                raw = await self._call_llm(build_filename_prompt(
                    [p.filename for p in report.files]))
                parsed = parse_llm_json(raw) or {}
                by_name = {f.get("filename"): f
                           for f in parsed.get("files", [])
                           if isinstance(f, dict)}
                for profile in report.files:
                    profile.filename_meta = by_name.get(profile.filename)
            except Exception as e:
                logger.warning(f"⚠️ Filename metadata failed: {e}")

        report.total_estimated_llm_calls = sum(
            p.estimated_llm_calls for p in report.files)
        return report
