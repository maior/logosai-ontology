"""
구조 단위(조항·문서) 후보 탐지 — 로드맵 4 P-1 회귀 계약.

근거: docs/clause-type-migration-plan.md. 자동 판별은 측정으로 기각됐다 —
포함 매칭은 오탐(`계약자`)을 만들고, 원문 그대로 비교는 놓침(`제24조(계약의
소멸)` vs `제24조 【계약의 소멸】` 표기차)을 만든다. 여기 계약:

- **전체-정규화 동등성만** 후보다. 부분문자열 매칭을 버리는 것이 오탐 차단의
  본체다 (실측 사례를 그대로 테스트로 고정).
- 분류는 신호이지 판정이 아니다 — evidence_matched·lifecycle 등 판단 재료를
  동봉하되 거르지 않는다 (dup_review 와 같은 규율).
- 조문 정규식 하드코딩 금지 — 라벨 사전은 청크 스토어의 section 전량이다.
"""

from dataclasses import dataclass, field
from typing import List

import networkx as nx

from ontology.core.structural_units import find_structural_candidates


@dataclass
class FakeChunk:
    chunk_id: str
    section: str = ""
    source: str = "약관.pdf"
    node_ids: List[str] = field(default_factory=list)


def _graph(nodes):
    g = nx.MultiDiGraph()
    for nid, attrs in nodes:
        g.add_node(nid, **attrs)
    return g


# ─── 신호 A: section 라벨 전체-정규화 동등성 ─────────────────────────


def test_matches_bracket_and_spacing_variants():
    """실측 놓침 해결: 노드는 `()`, 라벨은 `【】` + 공백 — normalize 동등."""
    g = _graph([
        ("InsuranceTerm:제24조(계약의 소멸)", {"name": "제24조(계약의 소멸)"}),
        ("InsuranceTerm:제1조 【목적 】", {"name": "제1조 【목적 】"}),
    ])
    chunks = [
        FakeChunk("c1", section="제24조 【계약의 소멸】"),
        FakeChunk("c2", section="제1조 【목적】"),
    ]
    result = find_structural_candidates(g, chunks)
    ids = {c["node_id"] for c in result["candidates"]}
    assert "InsuranceTerm:제24조(계약의 소멸)" in ids
    assert "InsuranceTerm:제1조 【목적 】" in ids
    by_id = {c["node_id"]: c for c in result["candidates"]}
    assert by_id["InsuranceTerm:제24조(계약의 소멸)"]["kind"] == "section"
    assert by_id["InsuranceTerm:제24조(계약의 소멸)"]["matched_sections"] == ["제24조 【계약의 소멸】"]


def test_substring_is_not_a_match():
    """실측 오탐 차단: `계약자`는 라벨들 안에 부분문자열로 흔하지만 어떤 라벨
    **전체**와도 동등하지 않다 — 포함 매칭을 버리는 것이 이 테스트의 본체."""
    g = _graph([("InsuranceTerm:계약자", {"name": "계약자"})])
    chunks = [
        FakeChunk("c1", section="제20조(계약자의 임의해지)"),
        FakeChunk("c2", section="계약자 또는 피보험자의 의무"),
    ]
    result = find_structural_candidates(g, chunks)
    assert result["candidates"] == []


def test_composite_label_segments_match_whole_segment():
    """계층 라벨(` > `)은 구획별로 사전에 든다 — 여전히 구획 **전체** 동등만."""
    g = _graph([
        ("Strategy:사업 개요", {"name": "사업 개요"}),
        ("Strategy:개요", {"name": "개요"}),  # 구획의 부분문자열 — 매칭 금지
    ])
    chunks = [FakeChunk("c1", section="1부 개요서 > 사업 개요")]
    result = find_structural_candidates(g, chunks)
    ids = {c["node_id"] for c in result["candidates"]}
    assert ids == {"Strategy:사업 개요"}


# ─── 신호 B: 자기 근거 정합 (거르지 않고 동봉) ───────────────────────


def test_evidence_matched_signal():
    g = _graph([
        ("T:제6조(지급사유)", {"name": "제6조(지급사유)"}),
        ("T:제7조(면책)", {"name": "제7조(면책)"}),
    ])
    chunks = [
        FakeChunk("c1", section="제6조(지급사유)", node_ids=["T:제6조(지급사유)"]),
        FakeChunk("c2", section="제7조(면책)", node_ids=["T:다른노드"]),
    ]
    by_id = {c["node_id"]: c
             for c in find_structural_candidates(g, chunks)["candidates"]}
    assert by_id["T:제6조(지급사유)"]["evidence_matched"] is True
    assert by_id["T:제7조(면책)"]["evidence_matched"] is False  # 거르지 않는다


# ─── 신호 C: 문서 단위 ───────────────────────────────────────────────


def test_document_kind_matches_source_title():
    g = _graph([("Strategy:제안요청서_발주기관_PROJ-A",
                 {"name": "제안요청서_발주기관_PROJ-A"})])
    chunks = [FakeChunk("c1", section="사업 개요",
                        source="제안요청서_발주기관_PROJ-A.txt")]
    result = find_structural_candidates(g, chunks)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["kind"] == "document"
    assert result["by_kind"] == {"section": 0, "document": 1}


def test_section_kind_wins_over_document():
    """양쪽 다 맞으면 section — 더 구체적인 신호가 이긴다 (결정적)."""
    g = _graph([("T:개요", {"name": "개요"})])
    chunks = [FakeChunk("c1", section="개요", source="개요.pdf")]
    result = find_structural_candidates(g, chunks)
    assert result["candidates"][0]["kind"] == "section"


# ─── 판단 재료 동봉 + 제외 규칙 ──────────────────────────────────────


def test_tombstoned_excluded():
    g = _graph([("T:제6조(지급사유)", {"name": "제6조(지급사유)"})])
    chunks = [FakeChunk("c1", section="제6조(지급사유)")]
    result = find_structural_candidates(
        g, chunks, tombstoned={"T:제6조(지급사유)"})
    assert result["candidates"] == []


def test_active_lifecycle_carries_caution():
    """active 는 개명이 차단된다(생애주기 관문) — 후보에서 빼지 않고 caution
    으로 알린다: 검수자가 '왜 승인이 거부될지'를 미리 봐야 한다."""
    g = _graph([
        ("T:제6조(지급사유)", {"name": "제6조(지급사유)", "lifecycle": "active"}),
        ("T:제7조(면책)", {"name": "제7조(면책)"}),
    ])
    chunks = [FakeChunk("c1", section="제6조(지급사유)"),
              FakeChunk("c2", section="제7조(면책)")]
    by_id = {c["node_id"]: c
             for c in find_structural_candidates(g, chunks)["candidates"]}
    assert by_id["T:제6조(지급사유)"]["caution"] is True
    assert by_id["T:제7조(면책)"]["caution"] is False


def test_materials_included():
    g = _graph([("T:제6조(지급사유)", {
        "name": "제6조(지급사유)", "definition": "지급사유 조항",
    })])
    g.add_node("T:암")
    g.add_edge("T:제6조(지급사유)", "T:암", predicate="coversDisease")
    chunks = [
        FakeChunk("c1", section="제6조(지급사유)",
                  node_ids=["T:제6조(지급사유)"], source="약관.pdf"),
        FakeChunk("c2", section="본문", node_ids=["T:제6조(지급사유)"]),
    ]
    cand = find_structural_candidates(g, chunks)["candidates"][0]
    assert cand["definition"] == "지급사유 조항"
    assert cand["type"] == "T"
    assert cand["evidence_count"] == 2           # 근거 링크 전체 (매칭 외 포함)
    assert cand["matched_chunk_ids"] == ["c1"]
    assert cand["sources"] == ["약관.pdf"]
    assert cand["out_predicates"] == {"coversDisease": 1}


def test_empty_inputs_never_throw():
    assert find_structural_candidates(_graph([]), [])["candidates"] == []
    g = _graph([("T:x", {"name": "x"})])
    result = find_structural_candidates(g, [])
    assert result["total"] == 0
    assert result["by_kind"] == {"section": 0, "document": 0}
