"""
Ontology Builder data models.

BuilderSchema is the closed vocabulary an extraction run is allowed to
produce — the LLM prompt injects it and the validator enforces it
(aicoach clean_extraction pattern). Schemas are data, not code: presets
ship here, custom ones arrive via from_dict (JSON/YAML upload).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Chunk:
    """A segment of source text with provenance.

    char_start/char_end 는 **원본 텍스트 기준** 문자 오프셋으로,
    text == original[char_start:char_end] 가 항상 성립한다 (축 2). 이 불변식이
    깨지면 인용이 엉뚱한 위치를 가리키므로 span provenance 전체가 무의미해진다
    — aicoach 가 rag/legal.py 를 따로 만든 이유가 정확히 이것이다(인용이
    p.47 이 아니라 제21조를 가리켜야 한다).
    """
    text: str
    source: str = ""
    index: int = 0
    section: str = ""
    char_start: int = 0
    char_end: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Extraction:
    """Validated LLM extraction result for one chunk."""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BuilderSchema:
    """Closed extraction vocabulary: allowed node types and predicates."""
    node_types: List[str]
    predicates: Dict[str, Tuple[str, str]]  # predicate → (domain, range)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuilderSchema":
        predicates = {
            name: tuple(dom_range)
            for name, dom_range in (data.get("predicates") or {}).items()
        }
        return cls(node_types=list(data.get("node_types") or []),
                   predicates=predicates)

    @classmethod
    def from_graph(cls, graph) -> Optional["BuilderSchema"]:
        """기존 그래프에서 스키마를 뽑는다 — 재인제스트 스키마 고정(다-1, 재현성).

        auto 모드로 같은 문서를 다시 넣으면 LLM 이 매번 타입을 새로 지어(Disease
        vs DefinedTerm) node_id 가 어긋나 골든셋·인용이 깨진다. 기존 그래프가
        있으면 그 타입·술어를 폐쇄 어휘로 재사용해 추출을 고정한다. 그래프가
        비어 있으면(신규 네임스페이스) None → 호출부는 auto 로 새 스키마 제안."""
        node_types = sorted({(a.get("type") or "").strip()
                             for _, a in graph.nodes(data=True)
                             if (a.get("type") or "").strip()})
        if not node_types:
            return None
        preds: Dict[str, Tuple[str, str]] = {}
        for s, t, a in graph.edges(data=True):
            p = (a.get("predicate") or "").strip()
            if not p or p in preds:
                continue
            preds[p] = (graph.nodes[s].get("type", "") or "",
                        graph.nodes[t].get("type", "") or "")
        return cls(node_types=node_types, predicates=preds)

    @classmethod
    def preset(cls, name: str) -> "BuilderSchema":
        """Load a named preset. Unknown names raise ValueError."""
        if name not in SCHEMA_PRESETS:
            raise ValueError(
                f"unknown schema preset '{name}' "
                f"(available: {', '.join(SCHEMA_PRESETS)})")
        return cls.from_dict(SCHEMA_PRESETS[name])

    @classmethod
    def preset_names(cls) -> List[str]:
        return list(SCHEMA_PRESETS)

    @classmethod
    def preset_document(cls) -> "BuilderSchema":
        """General document-knowledge schema (aicoach 도메인 모델의 일반화)."""
        return cls.preset("document")


# 이름 있는 프리셋 레지스트리 — 스키마는 코드가 아니라 데이터
SCHEMA_PRESETS: Dict[str, Dict[str, Any]] = {
    # 문서·약관형: 조항/보장/규제 구조가 있는 문서
    "document": {
        "node_types": ["Document", "Section", "Concept", "Entity",
                       "Clause", "Coverage", "Regulation"],
        "predicates": {
            "hasSection": ("Document", "Section"),
            "hasProvision": ("Document", "Clause"),
            "hasCoverage": ("Document", "Coverage"),
            "citesRegulation": ("Clause", "Regulation"),
            "mentions": ("Section", "Entity"),
            "relatedTo": ("Concept", "Concept"),
        },
    },
    # 범용: 구조를 가정하지 않는 개념-개체 그래프
    "generic": {
        "node_types": ["Concept", "Entity", "Event", "Organization",
                       "Person", "Place"],
        "predicates": {
            "relatedTo": ("Concept", "Concept"),
            "mentions": ("Concept", "Entity"),
            "partOf": ("Entity", "Entity"),
            "locatedIn": ("Entity", "Place"),
            "involvedIn": ("Entity", "Event"),
        },
    },
}


@dataclass
class BuildReport:
    """Outcome summary of one build run."""
    namespace: str = "default"
    files_read: int = 0
    chunks_processed: int = 0
    chunks_failed: int = 0
    entities_added: int = 0
    relations_added: int = 0
    # 검수 묘비(tombstone)에 막혀 병합이 건너뛰어진 개체 수 — 0 이 아니면
    # "재빌드했는데 노드가 안 생겼다"의 원인이 여기 보인다 (조용한 skip 금지)
    entities_rejected: int = 0
    # 재분류 재지도(P-4)로 옛 id → 새 id 로 옮겨 병합된 개체 수. 0 이 아니면
    # "재빌드했는데 옛 타입 노드가 안 생겼다"가 결함이 아니라 재지도라는 뜻.
    entities_remapped: int = 0
    # 품질 게이트가 trust 를 강등한 청크 수 + 이유별 집계(toc | garbled).
    # 조용한 강등 금지 — 등급이 내려간 사실과 그 이유가 리포트에 남아야
    # "왜 이 청크가 인용에서 밀렸나"를 나중에 설명할 수 있다.
    chunks_demoted: int = 0
    demote_reasons: Dict[str, int] = field(default_factory=dict)
    # 문서 내 이미지: 캡션 청크에 붙어 검색·인용 가능해진 수 / 텍스트 앵커가 없어
    # 검색 불가인 수. 후자를 보고하지 않으면 "그림이 왜 안 나오나"를 설명할 수
    # 없다 — 캡션 없는 이미지는 OCR·VLM(L2/L3)이 있어야 검색된다.
    images_attached: int = 0
    images_unanchored: int = 0
    # 재적재에서 끊겼던 근거 링크를 되살린 수 (core.relink). 0 이 아니면 그
    # 재적재가 노드를 고아로 만들 뻔했다는 뜻이다 — 조용히 복원하면 그 사실이
    # 사라지고, 추출이 왜 불안정한지 물을 기회도 사라진다.
    links_relinked: int = 0
    errors: List[str] = field(default_factory=list)
    proposed_schema: Optional[Dict[str, Any]] = None  # auto 모드에서 LLM이 제안한 스키마
    # 스키마 큐레이터(④)가 제안을 기존 어휘에 병합한 기록 {types: {...},
    # predicates: {...}} — 조용한 개명 금지: 무엇이 어디로 합쳐졌는지 보인다
    schema_mappings: Optional[Dict[str, Any]] = None
