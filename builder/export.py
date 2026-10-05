"""
Standard-format export — the graph as an OWL ontology document (Turtle).

Hand-generated Turtle (no rdflib dependency): node types become owl:Class,
predicates become owl:ObjectProperty, nodes become typed individuals with
rdfs:label and provenance. Valid input for Protégé / rdflib / GraphDB.
"""

from urllib.parse import quote
from typing import Set

BASE_IRI = "http://logosai.dev/ontology"


def _local(name: str) -> str:
    """IRI-safe local name (percent-encoded, spaces folded)."""
    return quote(str(name).strip().replace(" ", "_"), safe="")


def _literal(value: str) -> str:
    """Escape a Turtle string literal."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def to_turtle(graph, namespace: str = "default") -> str:
    lines = [
        f"@prefix : <{BASE_IRI}/{_local(namespace)}#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"<{BASE_IRI}/{_local(namespace)}> a owl:Ontology ;",
        f'    rdfs:comment "LogosAI Ontology Builder export (namespace: {_literal(namespace)})" .',
        "",
    ]

    node_types: Set[str] = set()
    predicates: Set[str] = set()
    for _, attrs in graph.nodes(data=True):
        if attrs.get("type"):
            node_types.add(attrs["type"])
    for _, _, attrs in graph.edges(data=True):
        if attrs.get("predicate"):
            predicates.add(attrs["predicate"])

    lines.append("# ─── Classes (TBox) ───")
    for node_type in sorted(node_types):
        lines.append(f":{_local(node_type)} a owl:Class .")
    lines.append("")
    lines.append("# ─── Object properties ───")
    for predicate in sorted(predicates):
        lines.append(f":{_local(predicate)} a owl:ObjectProperty .")
    lines.append("")

    lines.append("# ─── Individuals (ABox) ───")
    for node_id, attrs in graph.nodes(data=True):
        subject = f":{_local(node_id)}"
        node_type = attrs.get("type")
        parts = [f"{subject} a {':' + _local(node_type) if node_type else 'owl:NamedIndividual'}"]
        label = attrs.get("name") or str(node_id)
        parts.append(f'    rdfs:label "{_literal(label)}"')
        if attrs.get("source"):
            parts.append(f'    :source "{_literal(attrs["source"])}"')
        for geo_key in ("lat", "lng"):
            value = attrs.get(geo_key)
            if isinstance(value, (int, float)):
                parts.append(f'    :{geo_key} "{value}"^^xsd:decimal')
        lines.append(" ;\n".join(parts) + " .")
    lines.append("")

    lines.append("# ─── Relations ───")
    for source, target, attrs in graph.edges(data=True):
        predicate = attrs.get("predicate")
        if predicate:
            lines.append(f":{_local(source)} :{_local(predicate)} :{_local(target)} .")

    return "\n".join(lines) + "\n"
