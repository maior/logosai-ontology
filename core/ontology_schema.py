"""
Ontology Schema Layer (TBox)

Defines the data-perspective ontology for the knowledge graph:
- Class hierarchy: node types organized under upper classes (is-a taxonomy)
- Relation definitions: each predicate declares its domain (allowed source
  types), range (allowed target types), and whether it is transitive

Policy: validation is WARN-ONLY. The graph never rejects writes — a schema
violation is logged so data quality issues surface without breaking runtime
flows. Unknown predicates and unclassified node types pass with a note so
the schema can grow behind the data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Wildcard: any node type
ANY: Tuple[str, ...] = ("*",)


@dataclass(frozen=True)
class RelationDef:
    """Declaration of a relation (predicate) in the ontology."""
    predicate: str
    domain: Tuple[str, ...]  # allowed source node types ("*" = any)
    range: Tuple[str, ...]   # allowed target node types ("*" = any)
    transitive: bool = False
    description: str = ""


# ─── Class hierarchy (node_type → parent class) ─────────────────────
# Top class is "entity"; a type absent from this map has no declared parent.

CLASS_HIERARCHY: Dict[str, str] = {
    # actors
    "agent": "actor",
    "actor": "entity",
    # semantic traits attached to actors
    "capability": "trait",
    "tag": "trait",
    "trait": "entity",
    # classification taxonomy
    "query_category": "classification",
    "domain": "classification",
    "classification": "entity",
    # learned / observed records
    "query_agent_mapping": "record",
    "execution_result": "record",
    "query": "record",
    "record": "entity",
    # processes
    "workflow": "process",
    "task": "process",
    "process": "entity",
}


# ─── Relation definitions ───────────────────────────────────────────

RELATIONS: Dict[str, RelationDef] = {
    "is_a": RelationDef(
        "is_a", ANY, ANY, transitive=True,
        description="Instance/class generalization — child is a kind of parent",
    ),
    "subcategory_of": RelationDef(
        "subcategory_of", ("query_category",), ("query_category",), transitive=True,
        description="Category taxonomy edge",
    ),
    "has_capability": RelationDef(
        "has_capability", ("agent",), ("capability",),
        description="Agent possesses a capability",
    ),
    "has_tag": RelationDef(
        "has_tag", ("agent",), ("tag",),
        description="Agent is annotated with a tag",
    ),
    "has_mapping": RelationDef(
        "has_mapping", ("agent",), ("query_agent_mapping",),
        description="Agent's learned query-success pattern",
    ),
    "belongs_to_category": RelationDef(
        "belongs_to_category", ("query_agent_mapping", "agent", "query"),
        ("query_category",),
        description="Record classified under a query category",
    ),
    # structural relations already emitted elsewhere — acknowledged loosely
    "produces": RelationDef("produces", ANY, ANY),
    "executes": RelationDef("executes", ("agent",), ANY),
    "contains_entity": RelationDef("contains_entity", ("query",), ANY),
    "involves_concept": RelationDef("involves_concept", ("query",), ANY),
    "uses_relation": RelationDef("uses_relation", ("query",), ANY),
}

# Node types that have not been classified yet — validation lets them pass
# so half-built data never blocks writes.
_UNCLASSIFIED_TYPES = {None, "", "auto_created", "unknown", "inferred"}


def get_class_ancestors(node_type: str) -> List[str]:
    """Return the upper-class chain for a node type (nearest first)."""
    ancestors: List[str] = []
    current: Optional[str] = CLASS_HIERARCHY.get(node_type)
    while current is not None and current not in ancestors:
        ancestors.append(current)
        current = CLASS_HIERARCHY.get(current)
    return ancestors


def _type_matches(node_type: Optional[str], allowed: Tuple[str, ...]) -> bool:
    if "*" in allowed or node_type in _UNCLASSIFIED_TYPES:
        return True
    if node_type in allowed:
        return True
    # a subclass satisfies a superclass constraint
    return any(ancestor in allowed for ancestor in get_class_ancestors(node_type))


def validate_relation(
    predicate: str,
    source_type: Optional[str],
    target_type: Optional[str],
) -> Tuple[bool, str]:
    """Validate a relation against the schema.

    Returns (ok, message). Unknown predicates return (True, note) —
    the warn-only policy means callers log, never reject.
    """
    relation = RELATIONS.get(predicate)
    if relation is None:
        return True, f"predicate '{predicate}' is not declared in the ontology schema"

    if not _type_matches(source_type, relation.domain):
        return False, (
            f"'{predicate}' domain violation: source type '{source_type}' "
            f"not in {relation.domain}"
        )
    if not _type_matches(target_type, relation.range):
        return False, (
            f"'{predicate}' range violation: target type '{target_type}' "
            f"not in {relation.range}"
        )
    return True, ""


# ⚠️ `RelationDef.transitive` 는 **선언일 뿐 아무도 읽지 않는다** (2026-08-21
# 실측). 전이 폐포는 `engines/knowledge_graph_clean.get_ancestors/
# get_descendants` 가 담당하는데 그쪽은 `predicate="is_a"` 를 인자 기본값으로
# 하드코딩하므로, 예컨대 `subcategory_of(transitive=True)` 선언은 런타임에
# 아무 효과가 없다.
#
# 이 사실을 소비하던 `is_transitive()` 를 여기서 지웠다 — 호출자가 0이었고
# (테스트조차 RELATIONS 를 직접 읽었다), 남겨두면 "이 플래그가 쓰인다"는
# 인상을 준다. 전이성을 실제로 쓰려면 폐포 함수가 이 선언을 조회하도록
# 배선해야 하고, 그건 검색 동작이 바뀌는 별도 결정이다.
