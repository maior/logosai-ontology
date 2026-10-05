"""관계 백필 — 온톨로지의 뼈대를 채운다 (순수 함수 계약).

**동기(실측)**: 노드 191개에 엣지 97개, 술어는 definesTerm 58 · hasParty 15 ·
coversDisease 15 · hasCondition 8 뿐이고 **`is_a` 는 0개**다. `graph_retrieval`
의 is_a 폐포 확장(`HIERARCHY_PREDICATE`)은 이 데이터에서 죽은 코드다. 그리고
커버리지·고아 회복으로 되살린 노드 123개는 **관계가 하나도 없다** — 근거 링크는
촘촘해졌지만(고아 2개) 노드끼리는 여전히 성기다.

**왜 청크 단위인가**: 같은 청크를 공유하는 노드 쌍이 652개다. 쌍마다 물으면
652콜이고 각 호출이 문맥을 잃는다. 청크마다 한 번 물으면 92콜이고 LLM 이 조문
전체를 본다 — 빌더의 `_merge` 추출과 같은 모양이다.

**새 노드를 만들지 않는다.** 이 경로는 관계만 채운다. subject/object 가 그 청크에
연결된 노드가 아니면 버린다 — 관계 백필이 노드 생성의 뒷문이 되면 커버리지
승인이 지키는 검증(원문 대조·타입 관문·묘비)을 우회한다.

**환각 차단은 인용이 한다.** 관계는 문자열이 아니라 주장이므로 `(subject,
predicate, object)` 자체를 원문에서 찾을 수 없다. 그래서 LLM 에게 `evidence_quote`
를 요구하고 그것이 청크 원문의 부분문자열인지 검증한다 — `evidence_checker` 의
`parse_check_verdict` 와 같은 규정이다.
"""
import pytest

from ontology.core.relation_backfill import (
    build_relation_prompt,
    parse_relations,
)


CHUNK = ("제3조 이 계약에서 “암”이라 함은 악성 신생물을 말하며, "
         "“갑상선암”은 C73 에 해당하는 암의 일종이다. "
         "회사는 피보험자가 암으로 진단확정된 경우 암진단비를 지급한다.")

NODES = [
    {"node_id": "D:암", "name": "암", "type": "D"},
    {"node_id": "D:갑상선암", "name": "갑상선암", "type": "D"},
    {"node_id": "T:암진단비", "name": "암진단비", "type": "T"},
    {"node_id": "P:회사", "name": "회사", "type": "P"},
]
NODE_IDS = {n["node_id"] for n in NODES}
PREDS = ["is_a", "coversDisease", "definesTerm"]


# ─── 프롬프트 ────────────────────────────────────────────────────────

class TestBuildRelationPrompt:
    def test_includes_chunk_and_nodes_and_predicates(self):
        p = build_relation_prompt(CHUNK, NODES, PREDS)
        assert "갑상선암" in p and CHUNK[:20] in p
        for nid in NODE_IDS:
            assert nid in p
        for pred in PREDS:
            assert pred in p

    def test_demands_evidence_quote(self):
        """인용을 요구하지 않으면 검증할 것이 없다 — 환각 차단의 유일한 수단."""
        p = build_relation_prompt(CHUNK, NODES, PREDS)
        assert "evidence_quote" in p

    def test_forbids_new_nodes_and_predicates(self):
        p = build_relation_prompt(CHUNK, NODES, PREDS)
        assert "목록" in p          # 후보 목록 밖을 쓰지 말라는 지시가 있어야

    def test_allows_empty_answer(self):
        """관계가 없는 청크가 대부분이다 — 억지로 만들게 하면 잡음이 쏟아진다."""
        p = build_relation_prompt(CHUNK, NODES, PREDS)
        assert "빈 배열" in p or "[]" in p

    def test_deterministic_node_order(self):
        """호출 순서가 프롬프트를 바꾸면 같은 청크가 회차마다 다른 답을 낸다."""
        assert (build_relation_prompt(CHUNK, NODES, PREDS)
                == build_relation_prompt(CHUNK, list(reversed(NODES)), PREDS))


# ─── 파서 (LLM 출력 불신) ────────────────────────────────────────────

def _raw(items):
    import json
    return json.dumps({"relations": items}, ensure_ascii=False)


class TestParseRelations:
    GOOD = {"subject": "D:갑상선암", "predicate": "is_a", "object": "D:암",
            "evidence_quote": "“갑상선암”은 C73 에 해당하는 암의 일종이다"}

    def test_valid_relation_passes(self):
        out = parse_relations(_raw([self.GOOD]), CHUNK, NODE_IDS, PREDS)
        assert len(out) == 1
        assert out[0]["subject"] == "D:갑상선암"
        assert out[0]["predicate"] == "is_a"
        assert out[0]["object"] == "D:암"

    def test_quote_not_in_chunk_is_rejected(self):
        """지어낸 근거 — 이걸 통과시키면 그래프에 환각 관계가 들어간다."""
        bad = {**self.GOOD, "evidence_quote": "갑상선암은 암이 아니다"}
        assert parse_relations(_raw([bad]), CHUNK, NODE_IDS, PREDS) == []

    def test_missing_quote_is_rejected(self):
        bad = {**self.GOOD}
        bad.pop("evidence_quote")
        assert parse_relations(_raw([bad]), CHUNK, NODE_IDS, PREDS) == []

    def test_whitespace_differing_quote_is_accepted(self):
        """LLM 은 개행·공백을 바꿔 인용한다 — 글자가 같으면 통과시킨다."""
        bad = {**self.GOOD,
               "evidence_quote": "“갑상선암”은  C73 에\n해당하는 암의 일종이다"}
        assert len(parse_relations(_raw([bad]), CHUNK, NODE_IDS, PREDS)) == 1

    def test_unknown_node_is_rejected(self):
        """새 노드를 만드는 뒷문을 막는다."""
        bad = {**self.GOOD, "object": "D:없는병"}
        assert parse_relations(_raw([bad]), CHUNK, NODE_IDS, PREDS) == []

    def test_unknown_predicate_is_rejected(self):
        """술어 어휘가 무한정 늘면 스키마가 무의미해진다."""
        bad = {**self.GOOD, "predicate": "그냥관련있음"}
        assert parse_relations(_raw([bad]), CHUNK, NODE_IDS, PREDS) == []

    def test_self_loop_is_rejected(self):
        bad = {**self.GOOD, "object": "D:갑상선암"}
        assert parse_relations(_raw([bad]), CHUNK, NODE_IDS, PREDS) == []

    def test_duplicate_triples_collapse(self):
        out = parse_relations(_raw([self.GOOD, dict(self.GOOD)]),
                             CHUNK, NODE_IDS, PREDS)
        assert len(out) == 1

    def test_same_pair_different_predicate_both_kept(self):
        """관계는 여럿일 수 있다 — 쌍으로 dedup 하면 사실이 사라진다."""
        other = {"subject": "P:회사", "predicate": "coversDisease",
                 "object": "D:암",
                 "evidence_quote": "회사는 피보험자가 암으로 진단확정된 경우"}
        out = parse_relations(_raw([self.GOOD, other]), CHUNK, NODE_IDS, PREDS)
        assert len(out) == 2

    def test_garbage_never_raises(self):
        for raw in ("", "not json", "[]", "null", '{"relations": "x"}',
                    '{"relations": [null, 3, "s"]}'):
            assert parse_relations(raw, CHUNK, NODE_IDS, PREDS) == []

    def test_prose_wrapped_json_tolerated(self):
        import json
        raw = "```json\n" + _raw([self.GOOD]) + "\n```"
        assert len(parse_relations(raw, CHUNK, NODE_IDS, PREDS)) == 1

    def test_names_in_quote_is_reported_for_triage(self):
        """검수자가 강한 근거와 약한 근거를 가려야 한다 — 두 이름이 다 인용에
        있으면 강하다. 판정에는 쓰지 않는다(대명사 지시를 버리지 않기 위해)."""
        out = parse_relations(_raw([self.GOOD]), CHUNK, NODE_IDS, PREDS)
        assert out[0]["names_in_quote"] == 2

    def test_weak_evidence_still_passes_but_marked(self):
        weak = {**self.GOOD, "evidence_quote": "제3조 이 계약에서"}
        out = parse_relations(_raw([weak]), CHUNK, NODE_IDS, PREDS)
        assert len(out) == 1 and out[0]["names_in_quote"] == 0

    def test_signature_helpers_are_reporting_not_gating(self):
        """타입 시그니처는 **거부 관문이 아니라 검수 신호**다.

        실측: 제안 36건 중 30건(83%)이 기존 시그니처를 벗어났는데 **다수는 정상
        관계**였다 — 조항(`제3조 【…】`)과 문서(`사업방법서`)가 InsuranceTerm 으로
        잡혀 있어 "조항이 용어를 정의한다"는 옳은 관계가 새 패턴으로 보인다.
        관문으로 쓰면 그 정상 관계까지 막힌다. 그래서 parse_relations 는 시그니처를
        **보지 않는다** — 이 테스트가 그 계약을 고정한다.
        """
        import networkx as nx
        from ontology.core.relation_backfill import (signature_of,
                                                     type_signatures)
        g = nx.MultiDiGraph()
        g.add_node("D:암", type="D")
        g.add_node("D:갑상선암", type="D")
        g.add_edge("P:회사", "D:암", predicate="coversDisease")
        g.add_node("P:회사", type="P")
        assert ("P", "coversDisease", "D") in type_signatures(g)
        # 새 시그니처(D→D)는 알려진 집합에 없다 …
        assert signature_of(g, "D:갑상선암", "coversDisease", "D:암") \
            == ("D", "coversDisease", "D")
        # … 그런데 파서는 그것 때문에 버리지 않는다
        raw = _raw([{"subject": "D:갑상선암", "predicate": "is_a",
                     "object": "D:암",
                     "evidence_quote": "“갑상선암”은 C73 에 해당하는 암의 일종이다"}])
        assert len(parse_relations(raw, CHUNK, NODE_IDS, PREDS)) == 1

    def test_signature_of_unknown_node_is_empty_type(self):
        import networkx as nx
        from ontology.core.relation_backfill import signature_of
        assert signature_of(nx.MultiDiGraph(), "a", "p", "b") == ("", "p", "")

    def test_type_signatures_never_raises(self):
        from ontology.core.relation_backfill import type_signatures

        class Boom:
            def edges(self, data=False):
                raise RuntimeError("down")

        assert type_signatures(Boom()) == set()

    def test_empty_predicate_list_rejects_everything(self):
        """어휘가 비면 무엇도 통과하지 못한다 — 빈 목록을 '전부 허용'으로 읽으면
        술어가 무한정 늘어난다."""
        assert parse_relations(_raw([self.GOOD]), CHUNK, NODE_IDS, []) == []
