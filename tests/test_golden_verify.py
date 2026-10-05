"""골든셋 왕복 검증 — 라벨 수를 늘리되 "인간이 확인했다"고 주장하지 않는다.

**왜 필요한가**: evidence 자로 재니 hit@5 가 1.0(16/16)으로 **포화**했다. 남은
신호는 hit@1 뿐이고 16 케이스에서 1건은 0.0625 다. 실제로 `max_terms` 6 이 2·8
보다 나쁜 비단조가 나왔는데 그건 튜닝 가능한 신호가 아니라 잡음이다. 자가 부족해
결론이 두 번 뒤집혔다(임베딩 오진 · max_terms 왕복). 세 번째를 막으려면 케이스가
더 필요하다.

**그런데 초안(draft)을 그냥 쓸 수는 없다.** LLM 이 만든 라벨이 맞는지 아무도
확인하지 않았고, `confirmed` 는 **인간**이 확정한다는 계약이 있다. 그 사이가
비어 있었다.

**왕복 검증**: 노드 N → 질의 Q(이름 금지 패러프레이즈) → 혼동 후보와 함께
"이 질의가 묻는 개체는?" → N 을 고르면 라벨이 왕복을 통과했다. 새 status
`verified` 로 `confirmed`(인간)와 구별한다.

**후보는 의미 이웃으로 뽑는다.** 무작위 후보면 과제가 너무 쉬워 통과율이
무의미해진다. 검색기의 *순위*를 쓰는 게 아니라 후보 *집합*만 임베더에서 뽑고
판정은 LLM 이 노드 목록을 보고 하므로, 검색 지표에 대해 순환이 아니다.

**한계를 숨기지 않는다**: 왕복은 "LLM 이 명확하다고 보는 질의"에 편향된다.
생성과 검증에 같은 모델을 쓰면 자기 일관성 검사에 가깝다. 후보를 혼동 가능하게
만드는 것은 완화지 해결이 아니다. 그래서 기본 평가는 여전히 `confirmed` 만
쓰고, `verified` 는 명시해야 들어간다 — 두 숫자를 나란히 본다.
"""
import pytest

from ontology.core.search_qa import (
    GoldenCase,
    GoldenSet,
    build_roundtrip_prompt,
    evaluate_cases,
    parse_roundtrip,
)


# ─── 왕복 프롬프트 (순수 함수) ───────────────────────────────────────

CANDS = [
    {"node_id": "T:계약자적립액", "name": "계약자적립액",
     "definition": "계약 소멸 시 지급 여부를 결정하는 적립 금액"},
    {"node_id": "T:보험료", "name": "보험료", "definition": "계약자가 납입하는 금액"},
    {"node_id": "T:보험금", "name": "보험금", "definition": "회사가 지급하는 금액"},
]


class TestBuildRoundtripPrompt:
    def test_includes_query_and_every_candidate(self):
        p = build_roundtrip_prompt("적립된 돈은 어떻게 되나?", CANDS)
        assert "적립된 돈은 어떻게 되나?" in p
        for c in CANDS:
            assert c["node_id"] in p

    def test_does_not_reveal_which_is_expected(self):
        """정답을 흘리면 검증이 아니라 받아쓰기다. 후보는 **id 정렬**이라
        expected 가 몇 번째인지에 정보가 없다."""
        a = build_roundtrip_prompt("q", CANDS)
        b = build_roundtrip_prompt("q", list(reversed(CANDS)))
        assert a == b          # 입력 순서가 출력에 새지 않는다
        for word in ("정답", "expected", "기대"):
            assert word not in a

    def test_allows_refusal(self):
        """모르면 모른다고 할 길이 있어야 한다 — 강제 선택은 추측을 만든다."""
        p = build_roundtrip_prompt("q", CANDS)
        assert "none" in p.lower()

    def test_empty_candidates_is_safe(self):
        assert build_roundtrip_prompt("q", []) != ""


class TestParseRoundtrip:
    IDS = {c["node_id"] for c in CANDS}

    def test_valid_pick(self):
        assert parse_roundtrip('{"node_id": "T:보험료"}', self.IDS) == "T:보험료"

    def test_pick_outside_candidates_is_rejected(self):
        """후보에 없는 id 는 지어낸 것이다 — 통과시키면 라벨이 오염된다."""
        assert parse_roundtrip('{"node_id": "T:없는것"}', self.IDS) == ""

    def test_refusal_is_empty(self):
        assert parse_roundtrip('{"node_id": "none"}', self.IDS) == ""
        assert parse_roundtrip('{"node_id": ""}', self.IDS) == ""

    def test_garbage_never_raises(self):
        for raw in ("", "not json", "[]", '{"other": 1}', "null"):
            assert parse_roundtrip(raw, self.IDS) == ""

    def test_prose_wrapped_json_is_tolerated(self):
        """LLM 은 코드펜스·설명을 붙인다 — 그것 때문에 케이스를 잃지 않는다."""
        assert parse_roundtrip('```json\n{"node_id": "T:보험금"}\n```',
                               self.IDS) == "T:보험금"

    def test_whitespace_in_id_is_normalized(self):
        assert parse_roundtrip('{"node_id": " T:보험료 "}', self.IDS) == "T:보험료"


# ─── status=verified ────────────────────────────────────────────────

class TestVerifiedStatus:
    def _set(self, tmp_path):
        gs = GoldenSet(namespace="vns", path=tmp_path / "g.jsonl")
        cid = gs.add("적립된 돈은?", "T:계약자적립액", status="draft",
                     source="generator")
        return gs, cid

    def test_verify_promotes_draft(self, tmp_path):
        gs, cid = self._set(tmp_path)
        assert gs.verify(cid) is True
        assert gs.cases()[0].status == "verified"

    def test_verify_does_not_demote_confirmed(self, tmp_path):
        """인간 판정이 기계보다 강하다 — 강등하면 사람 작업을 지운다."""
        gs, cid = self._set(tmp_path)
        gs.confirm(cid)
        assert gs.verify(cid) is False
        assert gs.cases()[0].status == "confirmed"

    def test_confirm_can_still_upgrade_verified(self, tmp_path):
        """나중 인간 판정이 이긴다 (confirm 의 기존 계약)."""
        gs, cid = self._set(tmp_path)
        gs.verify(cid)
        assert gs.confirm(cid) is True
        assert gs.cases()[0].status == "confirmed"

    def test_verify_unknown_case(self, tmp_path):
        gs, cid = self._set(tmp_path)
        assert gs.verify("nope") is False

    def test_verify_survives_replay(self, tmp_path):
        """골든셋은 추가전용 로그다 — 재생하지 못하면 재시작에 판정이 사라진다."""
        gs, cid = self._set(tmp_path)
        gs.verify(cid)
        reloaded = GoldenSet(namespace="vns", path=tmp_path / "g.jsonl")
        assert reloaded.load_from_disk()
        assert reloaded.cases()[0].status == "verified"

    def test_replay_does_not_downgrade_confirmed(self, tmp_path):
        """로그 순서가 verify → confirm 이면 최종은 confirmed 여야 한다."""
        gs, cid = self._set(tmp_path)
        gs.verify(cid)
        gs.confirm(cid)
        reloaded = GoldenSet(namespace="vns", path=tmp_path / "g.jsonl")
        reloaded.load_from_disk()
        assert reloaded.cases()[0].status == "confirmed"


# ─── evaluate_cases(statuses=...) ───────────────────────────────────

def _mixed():
    return [
        GoldenCase(case_id="c", query="q1", expected_node_id="N:1",
                   status="confirmed"),
        GoldenCase(case_id="v", query="q2", expected_node_id="N:2",
                   status="verified"),
        GoldenCase(case_id="d", query="q3", expected_node_id="N:3",
                   status="draft"),
    ]


class TestStatusSelection:
    RANK = {"ch": lambda q, k: ["N:1", "N:2", "N:3"]}

    def test_default_is_confirmed_only(self):
        """하위호환 관문: statuses 를 안 주면 오늘과 같아야 한다. 깨지면 기존
        측정치 전부가 비교 불가능해진다 — verified 가 조용히 섞이면 최악이다."""
        assert evaluate_cases(_mixed(), self.RANK, k=5)["cases"] == 1

    def test_include_drafts_still_includes_everything(self):
        assert evaluate_cases(_mixed(), self.RANK, k=5,
                              include_drafts=True)["cases"] == 3

    def test_explicit_statuses_selects(self):
        res = evaluate_cases(_mixed(), self.RANK, k=5,
                             statuses={"confirmed", "verified"})
        assert res["cases"] == 2
        assert {r["case_id"] for r in res["per_case"]} == {"c", "v"}

    def test_statuses_overrides_include_drafts(self):
        """둘 다 주면 명시적인 쪽이 이긴다 — 애매한 조합이 조용히 다른 집합을
        재는 것보다 낫다."""
        res = evaluate_cases(_mixed(), self.RANK, k=5, include_drafts=True,
                             statuses={"confirmed"})
        assert res["cases"] == 1

    def test_statuses_is_reported_back(self):
        """어느 집합을 쟀는지 응답에 없으면 두 숫자를 헷갈린다."""
        res = evaluate_cases(_mixed(), self.RANK, k=5,
                             statuses={"confirmed", "verified"})
        assert set(res["statuses"]) == {"confirmed", "verified"}

    def test_empty_statuses_falls_back_to_default(self):
        """빈 집합을 '아무것도 재지 마라'로 읽으면 조용히 0건이 된다."""
        assert evaluate_cases(_mixed(), self.RANK, k=5,
                              statuses=set())["cases"] == 1

    def test_unknown_status_yields_zero_but_says_so(self):
        res = evaluate_cases(_mixed(), self.RANK, k=5, statuses={"bogus"})
        assert res["cases"] == 0 and res["measured"] is False


class TestAcceptAdditionalAnswer:
    """정답 집합 확장 — **골든셋도 그래프 상태에 종속적이다** (실측).

    PROJ-A 커버리지 회복 후 hit@1 이 내려갔는데(0.66→0.60) 진단하니 검색 회귀가
    아니라 **라벨 노후화**였다: "질의서 분석…" 질의의 1~3위가 요구서
    조문으로 질의에 정확히 답하는데, 골든 라벨(제안서 유래 노드)이 커버리지
    회복으로 생긴 같은 개념(요구서 유래 노드)을 몰랐다. 코퍼스가 자라면 정답
    집합도 자라야 한다 — `accepted` 가 그 문서화된 용도이고, 여기는 그 쓰기 경로다.
    """

    def _set(self, tmp_path):
        gs = GoldenSet(namespace="ans", path=tmp_path / "g.jsonl")
        cid = gs.add("질의서 분석 서비스?", "Service:업무지원 AI",
                     status="confirmed", source="hand")
        return gs, cid

    def test_accept_expands_the_answer_set(self, tmp_path):
        gs, cid = self._set(tmp_path)
        assert gs.accept(cid, "Service:업무지원 AI서비스") is True
        case = gs.cases()[0]
        assert case.accepted_ids() == {"Service:업무지원 AI",
                                       "Service:업무지원 AI서비스"}

    def test_status_is_untouched(self, tmp_path):
        """정답 확장은 판정 번복이 아니다 — confirmed 가 흔들리면 안 된다."""
        gs, cid = self._set(tmp_path)
        gs.accept(cid, "Service:다른것")
        assert gs.cases()[0].status == "confirmed"

    def test_duplicate_accept_is_noop(self, tmp_path):
        gs, cid = self._set(tmp_path)
        gs.accept(cid, "Service:X")
        assert gs.accept(cid, "Service:X") is False
        assert gs.cases()[0].accepted.count("Service:X") == 1

    def test_expected_itself_is_not_duplicated(self, tmp_path):
        """이미 정답인 노드를 accepted 에 또 넣으면 집합이 거짓으로 커진다."""
        gs, cid = self._set(tmp_path)
        assert gs.accept(cid, "Service:업무지원 AI") is False

    def test_unknown_case_or_blank_node(self, tmp_path):
        gs, cid = self._set(tmp_path)
        assert gs.accept("nope", "Service:X") is False
        assert gs.accept(cid, "  ") is False

    def test_survives_replay(self, tmp_path):
        """추가전용 로그 계약 — 재시작에 정답 확장이 사라지면 지표가 되돌아간다."""
        gs, cid = self._set(tmp_path)
        gs.accept(cid, "Service:업무지원 AI서비스")
        again = GoldenSet(namespace="ans", path=tmp_path / "g.jsonl")
        assert again.load_from_disk()
        assert "Service:업무지원 AI서비스" in again.cases()[0].accepted_ids()
