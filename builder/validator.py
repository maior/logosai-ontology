"""
Validator — aicoach clean_extraction + noise-filter port.

Everything the LLM returns is treated as untrusted:
- entity types / relation predicates outside the closed schema are dropped
- relations whose endpoints were not extracted as entities are dropped
  (dangling references are a classic hallucination shape)
- noise names are dropped: demonstrative stubs ("이 특약"), too-short names
"""

from typing import Any, Dict, List, Optional

from .models import BuilderSchema, Extraction

# demonstrative prefixes that mark a stub, not a real entity name
_STUB_PREFIXES = ("이 ", "그 ", "저 ", "해당 ", "위 ", "본 ", "동 ")
_MIN_NAME_LENGTH = 2


def _is_noise(name: str) -> bool:
    name = (name or "").strip()
    if len(name) < _MIN_NAME_LENGTH:
        return True
    return any(name.startswith(prefix) for prefix in _STUB_PREFIXES)


def clean_extraction(raw: Optional[Dict[str, Any]],
                     schema: BuilderSchema) -> Extraction:
    """Filter an LLM extraction down to schema-conformant, non-noise facts."""
    if not raw or not isinstance(raw, dict):
        return Extraction()

    entities: List[Dict[str, Any]] = []
    seen = set()
    for entity in raw.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "").strip()
        entity_type = entity.get("type")
        if entity_type not in schema.node_types or _is_noise(name):
            continue
        key = (name, entity_type)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"name": name, "type": entity_type,
                         "attrs": entity.get("attrs") or {}})

    extracted_names = {e["name"] for e in entities}
    relations: List[Dict[str, Any]] = []
    for relation in raw.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        subject = str(relation.get("subject") or "").strip()
        obj = str(relation.get("object") or "").strip()
        predicate = relation.get("predicate")
        if predicate not in schema.predicates:
            continue
        if subject not in extracted_names or obj not in extracted_names:
            continue  # dangling endpoint — hallucination shape
        relations.append({"subject": subject, "predicate": predicate,
                          "object": obj})

    return Extraction(entities=entities, relations=relations)
