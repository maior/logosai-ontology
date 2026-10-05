"""청크 품질 게이트 — 근거가 아닌 조각을 어휘 없이 구조로 판정한다.

**왜 어휘 목록이 아닌가**: aicoach 는 `참조순보험요율`·`예정사업비` 같은 블록리스트로
요약서를 걸러냈다(rag/search.py). 그 방식이 통한 것은 그쪽 코퍼스에 **상품요약서가
별도 문서로** 들어오기 때문이다. 우리 데이터로 같은 마커를 돌려보면 걸리는 것이
`제18조 【보험계약의 성립】`·`제29조 【…부활(효력회복)】` — **정당한 조항이 본문에서
그 단어를 언급**한 것이다(2026-07-27 실측). 이식하면 개선이 아니라 해악이다.
게다가 한국어 어휘 하드코딩은 이 저장소의 원칙 위반이다.

그래서 신호를 **구조**로 잡는다:
  · 점선 leader 밀도 — 목차 페이지의 보편적 형태(도메인·언어 무관)
  · 문자 구성비      — OCR 잡음("a) 은 olnt zl 오 Pal 중")의 보편적 형태

**삭제하지 않고 강등한다**: TRUST_RANK 에 summary=0 이 이미 있고
guard_attrs_by_trust 가 "요약서만 아는 사실은 유효하다 — 금지는 덮어쓰기지
기여가 아니다"라고 규정했다. 그 철학을 따른다.
"""
import pytest

from ontology.builder.chunk_quality import (
    DEMOTED_TRUST,
    assess_chunk,
    garbled_ratio,
    toc_ratio,
)


# ─── toc_ratio — 목차 판정 (점선 leader 밀도) ────────────────────────

class TestTocRatio:
    def test_toc_page_has_high_ratio(self):
        text = ("제1장 총칙 ……………………………… 3\n"
                "제2장 보험금의 지급 …………………… 12\n"
                "제3장 계약의 성립과 유지 ………………… 25\n")
        assert toc_ratio(text) > 0.05

    def test_middle_dot_leaders_also_counted(self):
        text = "총칙 ·········· 3\n지급 ·········· 12\n유지 ·········· 25\n"
        assert toc_ratio(text) > 0.05

    def test_prose_has_low_ratio(self):
        text = ("보험계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 "
                "있습니다. 다만 진단계약의 경우에는 예외가 있습니다.")
        assert toc_ratio(text) < 0.01

    def test_clause_with_a_few_dots_not_flagged(self):
        """정상 조문에도 가운뎃점은 쓰인다(제3호·제4호) — 밀도가 낮아야 통과."""
        text = ("제18조 【보험계약의 성립】 계약은 계약자의 청약과 회사의 승낙으로 "
                "이루어집니다. 회사는 제3호·제4호의 내용을 계약자에게 안내합니다.")
        assert toc_ratio(text) < 0.02

    def test_empty_and_none_safe(self):
        assert toc_ratio("") == 0.0
        assert toc_ratio(None) == 0.0


# ─── garbled_ratio — OCR 잡음 (문자 구성비) ──────────────────────────

class TestGarbledRatio:
    def test_ocr_garbage_has_high_ratio(self):
        """PROJ-A 실데이터에서 관측된 잡음 — 고립 단문자가 이어진다."""
        assert garbled_ratio("a) 은 olnt zl 오 Pal 중 e a 1 l") > 0.4

    def test_clean_korean_prose_low(self):
        text = ("보험계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 "
                "철회할 수 있습니다.")
        assert garbled_ratio(text) < 0.2

    def test_clean_english_prose_low(self):
        text = ("The policyholder may withdraw the application within 15 days "
                "of receiving the insurance policy document.")
        assert garbled_ratio(text) < 0.2

    def test_numeric_table_not_garbage(self):
        """숫자 표는 잡음이 아니다 — 정당한 데이터다.

        표본에 **한 자리 숫자 토큰**을 넣는 것이 핵심이다. 변이 검사(2026-07-27)에서
        여러 자리 숫자만 있는 표본은 '숫자 제외' 규칙을 전혀 밟지 않아 규칙을
        지워도 테스트가 통과했다 — 규칙이 실제로 작동함을 보이려면 단자리가 필요하다.
        """
        text = ("항목 1 2 3 4 5 6 7 8 9\n"
                "값 1 2 3 4 5 6 7 8 9\n"
                "비고 1 2 3 4 5 6 7 8 9")
        assert garbled_ratio(text) < 0.3

    def test_empty_and_none_safe(self):
        assert garbled_ratio("") == 0.0
        assert garbled_ratio(None) == 0.0


# ─── assess_chunk — 종합 판정 ────────────────────────────────────────

class TestAssessChunk:
    def test_prose_is_ok(self):
        v, reason, _ = assess_chunk(
            "보험계약자는 보험증권을 받은 날부터 15일 이내에 청약을 철회할 수 있습니다.")
        assert v == "ok"
        assert reason == ""

    def test_toc_is_demoted_with_reason(self):
        text = ("제1장 총칙 ……………………………… 3\n"
                "제2장 보험금의 지급 …………………… 12\n"
                "제3장 계약의 성립 ………………………… 25\n")
        v, reason, metrics = assess_chunk(text)
        assert v == DEMOTED_TRUST
        assert reason == "toc"
        assert metrics["toc_ratio"] > 0

    def test_garbled_is_demoted_with_reason(self):
        # 실제 OCR 잡음 청크 길이로 — 34자 표본은 MIN_ASSESS_LEN 보류 규칙과
        # 충돌한다(짧은 표본은 비율이 튀므로 판정하지 않는 것이 설계다).
        text = ("a) 은 olnt zl 오 Pal 중 e a 1 l m x n o p q r "
                "s t u v w z l o 은 중 오 a b c d e f g h")
        v, reason, _ = assess_chunk(text)
        assert v == DEMOTED_TRUST
        assert reason == "garbled"

    def test_legitimate_clause_mentioning_rate_words_survives(self):
        """**이 작업의 핵심 회귀 방어**: aicoach 블록리스트가 탈락시켰을 조항.
        우리 실데이터(ins_cancer_demo)에 실제로 있는 형태다."""
        text = ("제29조 【“보험료의 납입연체로 인하여 해지된 계약”의 부활(효력회복)】 "
                "계약자는 해약환급금을 받지 아니한 경우 계약이 해지된 날부터 3년 이내에 "
                "회사가 정한 절차에 따라 부활을 청약할 수 있습니다. 이 경우 회사는 "
                "예정사업비와 보험요율을 재적용하지 아니합니다.")
        v, reason, _ = assess_chunk(text)
        assert v == "ok", f"정당한 조항이 강등됨 (reason={reason})"

    def test_short_text_not_demoted_on_thin_evidence(self):
        """짧은 조각은 비율이 쉽게 튄다 — 표본이 얇으면 판정을 보류한다."""
        v, _, _ = assess_chunk("제1조 목적")
        assert v == "ok"

    def test_empty_is_ok_not_demoted(self):
        """빈 청크는 품질 판정 대상이 아니다(다른 층에서 걸러진다)."""
        assert assess_chunk("")[0] == "ok"
        assert assess_chunk(None)[0] == "ok"

    def test_metrics_always_returned(self):
        _, _, metrics = assess_chunk("아무 문장")
        assert set(metrics) == {"toc_ratio", "garbled_ratio"}

    def test_toc_beats_garbled_when_both(self):
        """두 신호가 다 걸리면 이유를 하나로 정해야 집계가 흔들리지 않는다."""
        text = ("제1장 ……………… 3 제2장 ……………… 9 제3장 ……………… 12 "
                "a b c d e f g h i j k l m n o p")
        v, reason, _ = assess_chunk(text)
        assert v == DEMOTED_TRUST
        assert reason == "toc"

    def test_demoted_trust_matches_pipeline_rank(self):
        """강등 값은 TRUST_RANK 에 이미 정의된 등급이어야 한다 — 새 어휘를
        만들면 guard_attrs_by_trust 가 그것을 모른다."""
        from ontology.builder.pipeline import TRUST_RANK
        assert DEMOTED_TRUST in TRUST_RANK
        assert TRUST_RANK[DEMOTED_TRUST] < TRUST_RANK["authoritative"]
