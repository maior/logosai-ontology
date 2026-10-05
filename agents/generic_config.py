"""generic_config — GenericGroundedSearchAgent 의 config row 순수 로직.

설계 근거: docs/generic-grounded-agent-design.md — "네임스페이스마다 에이전트를
새로 만들지 않는다. 도메인 = config row 하나." acp 로더는 agents.json 의
`parameters` 를 `AgentConfig.config` 로 넘겨 `cls(config)` 로 생성하므로
(agent_loader.py:126-138 실측), 같은 클래스의 인스턴스 여럿이 row 만 다르게
가질 수 있다 — 이 모듈이 그 row 를 해석한다.

원칙:
- **파싱은 관대하되 검증은 소리낸다** — 생성자에서 던지면 로더가 None 을 돌려
  유령 에이전트(등록됐는데 없음)가 된다(레지스트리 위생 사고의 그 결함류).
  그래서 (config, errors) 를 돌려주고, 에이전트는 handle 에서 errors 를 소리낸다.
- stdlib 만 — grounded_common 과 같은 이유로 순수(결정적 직접 실행 테스트).
"""

from typing import Any, Dict, List, Tuple

__all__ = ["parse_agent_params", "slot_query", "assemble_template", "citation_line"]

DEFAULT_TOP_K = 8
VALID_MODES = ("qa", "template")


def _markers(v: Any) -> Tuple[str, ...]:
    if isinstance(v, str):
        return (v,) if v.strip() else ()
    if isinstance(v, (list, tuple)):
        return tuple(str(m) for m in v if str(m).strip())
    return ()


def _top_k(v: Any, fallback: int) -> int:
    try:
        n = int(v)
        return n if n > 0 else fallback
    except (TypeError, ValueError):
        return fallback


def parse_agent_params(params: Any) -> Tuple[Dict[str, Any], List[str]]:
    """agents.json row 의 `parameters` → (config, errors).

    errors 가 비어 있지 않으면 이 row 는 **동작 불능**이다 — 에이전트가 handle
    에서 그대로 사용자에게 알린다. 조용히 기본값으로 덮으면 잘못된 row 가
    "그럭저럭 도는" 상태로 살아남아 원인을 못 찾는다 (retrieval_config 의
    값·키 검증과 같은 이유).
    """
    p = params if isinstance(params, dict) else {}
    errors: List[str] = []

    mode = str(p.get("mode") or "qa")
    if mode not in VALID_MODES:
        errors.append(f"mode 가 잘못됨: {mode!r} (허용: {'/'.join(VALID_MODES)})")

    cfg: Dict[str, Any] = {
        "namespace": str(p.get("namespace") or ""),
        "instruction": str(p.get("instruction") or ""),
        "source_markers": _markers(p.get("source_markers")),
        "top_k": _top_k(p.get("top_k"), DEFAULT_TOP_K),
        "mode": mode,
        "template": None,
    }

    if mode == "template":
        tpl = p.get("template")
        if not isinstance(tpl, dict):
            errors.append("template 모드인데 template 정의가 없음")
        else:
            title = str(tpl.get("title") or "")
            raw_slots = tpl.get("slots")
            slots = []
            if not isinstance(raw_slots, list) or not raw_slots:
                errors.append("template.slots 가 비었음 — 슬롯 없는 템플릿은 산출물이 없다")
            else:
                for i, s in enumerate(raw_slots):
                    if not isinstance(s, dict) or not str(s.get("title") or "").strip():
                        errors.append(f"slots[{i}] 에 title 이 없음")
                        continue
                    slots.append({
                        "slot_id": str(s.get("slot_id") or f"s{i + 1}"),
                        "title": str(s["title"]).strip(),
                        "query": str(s.get("query") or ""),
                        "instruction": str(s.get("instruction") or ""),
                        "source_markers": _markers(s.get("source_markers")),
                        "top_k": _top_k(s.get("top_k"), cfg["top_k"]),
                    })
            if not title:
                errors.append("template.title 이 없음")
            cfg["template"] = {"title": title, "slots": slots}

    return cfg, errors


def slot_query(user_query: str, slot: Dict[str, Any]) -> str:
    """슬롯의 회수 질의. slot.query 의 `{query}` 를 사용자 질의로 치환.

    query 미지정이면 `사용자질의 + 슬롯제목` — 슬롯 제목이 곧 그 절의 주제라
    회수 질의로 자연스럽다("PROJ-A 구축 주요 요구사항"). 사용자 질의를 버리지
    않는 것이 핵심: 슬롯 제목만으로 회수하면 템플릿이 질의와 무관해진다.
    """
    q = str(user_query or "").strip()
    tpl = str(slot.get("query") or "").strip()
    if tpl:
        return tpl.replace("{query}", q).strip()
    return f"{q} {slot.get('title', '')}".strip()


def citation_line(cites: List[Dict[str, Any]]) -> str:
    """슬롯 하단의 압축 근거 줄: `[1] source §sec · [2] …`.

    번호는 **슬롯 안에서만** 유효하다 — 슬롯마다 근거를 따로 회수하므로 전역
    번호로 다시 매기면 본문의 [n] 인용과 어긋난다(인용이 거짓이 되는 지점).
    """
    parts = []
    for c in cites:
        src = str(c.get("source") or "?")
        sec = str(c.get("section") or "").strip()
        parts.append(f"[{c.get('n')}] {src}" + (f" §{sec}" if sec else ""))
    return " · ".join(parts)


def assemble_template(title: str, filled: List[Dict[str, Any]], note: str = "") -> str:
    """슬롯 결과들 → 최종 마크다운 문서.

    filled: [{title, answer, citations}] (슬롯 순서 유지 — 템플릿이 정한 순서가
    곧 문서 구조다). 근거 없는 슬롯도 **자리를 지운다** — 빼면 "그 절이 원래
    없는 템플릿"으로 오독된다. note(측정 품질)는 문서 끝에 한 번.
    """
    lines = [f"# {title}", ""]
    for s in filled:
        lines.append(f"## {s.get('title', '')}")
        lines.append(str(s.get("answer") or "").strip() or "_근거를 찾지 못했습니다._")
        cl = citation_line(s.get("citations") or [])
        if cl:
            lines.append("")
            lines.append(f"> 근거: {cl}")
        lines.append("")
    if note:
        lines.append(f"_{note}_")
    return "\n".join(lines).strip() + "\n"


# ── 직접 실행 테스트 (grounded_common 과 같은 방식: pytest 아님) ─────────────
if __name__ == "__main__":
    import sys

    def test_qa_defaults():
        cfg, errs = parse_agent_params({"namespace": "ins"})
        assert errs == [], errs
        assert cfg["mode"] == "qa" and cfg["top_k"] == DEFAULT_TOP_K
        assert cfg["source_markers"] == () and cfg["template"] is None

    def test_invalid_mode_is_loud():
        _, errs = parse_agent_params({"mode": "chat"})
        assert any("mode" in e for e in errs), errs

    def test_garbage_params_safe():
        for bad in (None, "x", 3, []):
            cfg, errs = parse_agent_params(bad)
            assert cfg["mode"] == "qa" and errs == []

    def test_template_requires_slots_and_title():
        _, errs = parse_agent_params({"mode": "template", "template": {}})
        assert any("slots" in e for e in errs) and any("title" in e for e in errs), errs

    def test_template_slot_parse():
        cfg, errs = parse_agent_params({
            "mode": "template", "top_k": 6,
            "template": {"title": "보고서", "slots": [
                {"title": "개요", "source_markers": ["요청서"]},
                {"slot_id": "req", "title": "요구사항", "query": "{query} 요건", "top_k": 4},
            ]},
        })
        assert errs == [], errs
        s1, s2 = cfg["template"]["slots"]
        assert s1["slot_id"] == "s1" and s1["top_k"] == 6       # 에이전트 top_k 상속
        assert s2["slot_id"] == "req" and s2["top_k"] == 4      # 슬롯 오버라이드
        assert s1["source_markers"] == ("요청서",)

    def test_slot_without_title_is_loud_but_others_survive():
        cfg, errs = parse_agent_params({
            "mode": "template",
            "template": {"title": "t", "slots": [{"query": "x"}, {"title": "살아남"}]},
        })
        assert any("slots[0]" in e for e in errs), errs
        assert [s["title"] for s in cfg["template"]["slots"]] == ["살아남"]

    def test_slot_query_placeholder_and_default():
        assert slot_query("PROJ-A 구축", {"query": "{query} 요건", "title": "x"}) == "PROJ-A 구축 요건"
        assert slot_query("PROJ-A 구축", {"title": "주요 요구사항"}) == "PROJ-A 구축 주요 요구사항"
        assert slot_query("", {"title": "개요"}) == "개요"

    def test_citation_line():
        line = citation_line([{"n": 1, "source": "a.pdf", "section": "제3조"},
                              {"n": 2, "source": "b.txt", "section": ""}])
        assert line == "[1] a.pdf §제3조 · [2] b.txt"
        assert citation_line([]) == ""

    def test_assemble_keeps_slot_order_and_empty_slots():
        md = assemble_template("보고서", [
            {"title": "개요", "answer": "내용 [1]", "citations": [{"n": 1, "source": "a"}]},
            {"title": "빈 절", "answer": "", "citations": []},
        ], note="측정 품질: hit@5 0.98")
        assert md.index("## 개요") < md.index("## 빈 절")
        assert "_근거를 찾지 못했습니다._" in md
        assert "> 근거: [1] a" in md and md.rstrip().endswith("_측정 품질: hit@5 0.98_")

    def test_markers_accept_string_or_list():
        cfg, _ = parse_agent_params({"source_markers": "요청서"})
        assert cfg["source_markers"] == ("요청서",)
        cfg, _ = parse_agent_params({"source_markers": ["a", "", "b"]})
        assert cfg["source_markers"] == ("a", "b")

    failed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL {name}: {e}")
    print("OK" if not failed else f"FAILED {failed}")
    sys.exit(1 if failed else 0)
