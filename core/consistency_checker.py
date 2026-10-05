"""
일관성 감시 에이전트 (Consistency Linter) — 그래프 내부 모순 탐지.

②번 검수 보조. ①근거대조(evidence_checker)와 의도적으로 대비된다:
근거대조는 "이 사실이 원문에 근거하는가"라는 **판단**이라 LLM 이 필요하지만,
여기의 세 검사(이름-타입 충돌 · 별칭 충돌 · domain/range 위반)는 전부
그래프 스캔으로 **결정 가능**하다. 탐지는 계산이지 판단이 아니다 — 프로젝트
절대 원칙(결정 가능한 곳에 LLM 을 쓰지 않는다)이 이 모듈의 설계점이다.
LLM 을 쓰면 같은 그래프에 다른 결과가 나오고, 그 순간 린터가 아니라
점쟁이가 된다.

검사 목록:
- name_type_conflict (warn): 같은 name 이 다른 type 아래 존재 —
  빌드 간 스키마 drift 로 같은 개체가 두 노드로 갈라진 전형. 병합 검토 대상.
- alias_collision (warn): 별칭이 다른 노드의 이름/별칭과 겹침 —
  검색 확장(축 4)이 별칭을 확장어로 쓰므로 두 노드가 함께 끌려온다.
- range_violation (error): 스키마에 **선언된** 술어의 endpoint 타입이
  domain/range 와 불일치. 미선언 술어는 건너뛴다 — 열린 어휘 허용
  (계층 술어 is_a 등은 설계상 미선언이다).

출력 계약:
- 결과는 정렬되어 결정적이다 — 같은 그래프면 같은 순서. error 가 warn 보다
  앞선다 (UI 가 위에서부터 읽는 순서 = 심각도 순서).
- 발견은 재계산 가능한 파생물이므로 **저장하지 않는다** — 호출자(service)도
  마찬가지다. 저장하면 그래프와 어긋난 순간 어느 쪽이 진실인지 알 수 없다.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

_SEVERITY_ORDER = {"error": 0, "warn": 1}


def _norm(value: Any) -> str:
    """이름·별칭 비교용 정규화 — 공백 압축 + casefold.

    LLM 추출은 같은 개체를 공백·대소문자만 다르게 내놓는 일이 잦다
    (evidence_checker 의 인용 비교와 같은 이유). 글자가 다르면 다른 개체다.
    """
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _predicates_of(schema: Any) -> Dict[str, Tuple[str, str]]:
    """스키마 정규화 — BuilderSchema 객체든 {predicates: {...}} dict 든
    같은 모양(name → (domain, range))으로 받는다. 스키마는 데이터다."""
    if schema is None:
        return {}
    predicates = getattr(schema, "predicates", None)
    if predicates is None and isinstance(schema, dict):
        predicates = schema.get("predicates")
    return {name: tuple(dom_range)
            for name, dom_range in (predicates or {}).items()
            if dom_range and len(tuple(dom_range)) == 2}


def _finding(kind: str, severity: str, node_ids: List[str],
             detail: str, suggestion: str) -> Dict[str, Any]:
    return {"kind": kind, "severity": severity,
            "node_ids": node_ids, "detail": detail, "suggestion": suggestion}


def _name_type_conflicts(graph) -> List[Dict[str, Any]]:
    """같은 name, 다른 type — 스키마 drift 로 갈라진 중복 후보."""
    by_name: Dict[str, List[str]] = {}
    for node_id, attrs in graph.nodes(data=True):
        name = attrs.get("name")
        if name:
            by_name.setdefault(_norm(name), []).append(node_id)

    findings = []
    for node_ids in by_name.values():
        types = {graph.nodes[n].get("type", "") for n in node_ids}
        if len(node_ids) < 2 or len(types) < 2:
            continue
        node_ids = sorted(node_ids)
        display = graph.nodes[node_ids[0]].get("name", "")
        findings.append(_finding(
            "name_type_conflict", "warn", node_ids,
            f"이름 '{display}' 이(가) 서로 다른 타입 "
            f"{sorted(types)} 아래에 중복 존재합니다.",
            "같은 개체가 빌드 간 스키마 drift 로 갈라졌을 수 있습니다 — "
            "병합 검토 후 한쪽을 거절하세요."))
    return findings


def _alias_collisions(graph) -> List[Dict[str, Any]]:
    """별칭이 다른 노드의 이름/별칭과 겹침 — 확장이 두 노드를 함께 끈다."""
    name_owner: Dict[str, str] = {}          # norm(name) → node_id
    for node_id, attrs in graph.nodes(data=True):
        name = attrs.get("name")
        if name:
            name_owner.setdefault(_norm(name), node_id)

    findings = []
    seen_pairs = set()                       # 같은 (쌍, 별칭) 중복 보고 방지
    alias_owner: Dict[str, Tuple[str, str]] = {}  # norm(alias) → (node_id, 원문)

    def report(alias: str, node_a: str, node_b: str, other_kind: str) -> None:
        pair = (tuple(sorted((node_a, node_b))), _norm(alias))
        if node_a == node_b or pair in seen_pairs:
            return
        seen_pairs.add(pair)
        findings.append(_finding(
            "alias_collision", "warn", sorted((node_a, node_b)),
            f"별칭 '{alias}' 이(가) 다른 노드의 {other_kind}과(와) 겹칩니다 — "
            f"검색 확장(축 4)이 두 노드를 함께 끌어옵니다.",
            "별칭을 한 노드로 정리하거나 두 노드의 병합을 검토하세요."))

    for node_id, attrs in sorted(graph.nodes(data=True)):
        for alias in attrs.get("aliases") or []:
            if not alias:
                continue
            key = _norm(alias)
            owner = name_owner.get(key)
            if owner is not None:
                report(str(alias), node_id, owner, "이름")
            prior = alias_owner.get(key)
            if prior is not None:
                report(str(alias), node_id, prior[0], "별칭")
            else:
                alias_owner[key] = (node_id, str(alias))
    return findings


def _range_violations(graph, predicates: Dict[str, Tuple[str, str]]
                      ) -> List[Dict[str, Any]]:
    """선언된 술어만 검사한다 — 미선언은 열린 어휘로 허용."""
    findings = []
    for subj, obj, attrs in graph.edges(data=True):
        predicate = attrs.get("predicate", "")
        declared = predicates.get(predicate)
        if declared is None:
            continue
        domain, range_ = declared
        subj_type = graph.nodes[subj].get("type", "")
        obj_type = graph.nodes[obj].get("type", "")
        if subj_type == domain and obj_type == range_:
            continue
        findings.append(_finding(
            "range_violation", "error", [subj, obj],
            f"술어 '{predicate}' 은(는) ({domain} → {range_}) 로 선언됐지만 "
            f"실제 간선은 ({subj_type} → {obj_type}) 입니다.",
            "간선의 술어가 잘못됐거나 endpoint 노드의 타입이 오추출입니다 — "
            "간선 또는 노드 타입을 수정하세요."))
    return findings


def lint_graph(graph, schema: Optional[Any] = None) -> List[Dict[str, Any]]:
    """그래프 일관성 검사 — 순수 함수, 결정적, 그래프 무변경.

    schema 는 BuilderSchema 또는 {predicates: {p: (domain, range)}} dict.
    없으면 range_violation 검사는 생략된다 (검사할 선언이 없으므로).
    """
    findings = _name_type_conflicts(graph)
    findings += _alias_collisions(graph)
    findings += _range_violations(graph, _predicates_of(schema))

    for finding in findings:
        finding["node_ids"] = sorted(finding["node_ids"])
    # error 우선 → kind → node_ids → detail. 같은 그래프면 항상 같은 순서 —
    # UI diff 와 테스트가 순서에 기대도 안전하다.
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f["severity"], 9),
                                 f["kind"], tuple(f["node_ids"]), f["detail"]))
    return findings
