"""병렬 결과 리스트 핵심 추출(압축) 테스트 (2026-07-07).

하이브리드 실측(서울·제주 날씨→비교)에서 발견: _extract_core_result 가 다중
항목 리스트를 통 JSON(metadata·source_info 포함)으로 직렬화 → 2000자 truncate
예산을 잠식해 뒷 병렬 결과가 잘림. 계약(b):
  - 다중 항목 리스트: 각 항목의 핵심(answer/result/content)만 추출해
    [결과 N] 라벨로 join — metadata 는 포함하지 않는다.
  - 단일 항목 리스트: 기존과 동일(라벨 없이 핵심만).
  - 아무것도 추출 못 하면 기존 JSON fallback 유지.
  - enrich: 핵심만 넣으면 병렬 결과 2개가 2000자 예산 안에 온전히 들어간다.

직접 실행: python ontology/orchestrator/test_extract_core_parallel.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ONTOLOGY = os.path.join(_ROOT, "ontology")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ONTOLOGY)  # 내부 모듈이 'core.*' 절대 import 를 쓰므로 필요


def _mk_agent_result(city: str, answer: str) -> dict:
    """라이브 실측과 동일한 중첩 구조(success/data/result/answer + metadata)."""
    return {
        "success": True,
        "data": {
            "result": {
                "answer": answer,
                "reasoning": "Tomorrow.io API를 통해 실시간 날씨 데이터를 가져와 분석했습니다.",
                "location": city,
                "data_source": "Tomorrow.io API",
                "source_info": {
                    "api_provider": "Tomorrow.io",
                    "api_url": "https://api.tomorrow.io/v4/weather/forecast",
                    "disclaimer": "실제 기상 상황과 다를 수 있으니 외출 전 최신 정보를 확인하세요.",
                    "agent_version": "WeatherAgent v2.0",
                },
            },
            "response_type": "success",
            "message": f"{city}의 현재 날씨를 알려드립니다.",
            "metadata": {"location": city, "city": city, "district": None},
        },
    }


def main():
    fails = []

    def t(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    from ontology.orchestrator.execution_engine import ExecutionEngine

    engine = ExecutionEngine.__new__(ExecutionEngine)  # 의존성 없이 메서드만

    seoul_ans = "# seoul 현재 날씨\n현재 기온 24.59°C, 체감 24.6°C, 습도 88%" + " 상세." * 100
    jeju_ans = "# jeju 현재 날씨\n현재 기온 22.48°C, 체감 22.5°C, 습도 91%" + " 상세." * 100
    data = [_mk_agent_result("seoul", seoul_ans), _mk_agent_result("jeju", jeju_ans)]

    # ── 다중 항목: 각 answer 만, metadata 배제 ──
    out = engine._extract_core_result(data)
    t("X-1 두 병렬 결과의 answer 모두 포함",
      "24.59°C" in out and "22.48°C" in out)
    t("X-2 metadata/source_info 배제(압축)",
      "source_info" not in out and "api_provider" not in out and '"success"' not in out)
    t("X-3 [결과 N] 라벨로 구분",
      "[결과 1]" in out and "[결과 2]" in out)

    # ── 압축 효과: enrich 2000자 예산 안에 둘 다 생존 ──
    enriched = engine._enrich_query_with_input(
        "두 도시 날씨를 비교해 추천", data, "llm_search_agent")
    t("X-4 enriched 에 두 결과 핵심 모두 생존(truncate 미발동)",
      "24.59°C" in enriched and "22.48°C" in enriched)
    t("X-5 [이전 단계 결과]/[요청] 형식 유지",
      "[이전 단계 결과]" in enriched and "[요청]" in enriched)

    # ── 단일 항목: 기존 동작(라벨 없음) ──
    single = engine._extract_core_result([_mk_agent_result("seoul", "기온 24.59°C")])
    t("X-6 단일 항목 리스트는 기존과 동일(라벨 없음)",
      "24.59°C" in single and "[결과 1]" not in single)

    # ── 문자열 리스트도 라벨 join ──
    strs = engine._extract_core_result(["첫 번째 결과", "두 번째 결과"])
    t("X-7 문자열 리스트도 항목별 라벨", "[결과 1]" in strs and "두 번째 결과" in strs)

    # ── 추출 불가 항목 → fallback 유지(빈 문자열 아님) ──
    weird = engine._extract_core_result([{"zzz": 1}, {"yyy": 2}])
    t("X-8 추출 불가여도 비어있지 않음(fallback)", bool(weird.strip()))

    # ── 라이브 실경로: DataTransformer 가 stage 간 결과를 pretty-JSON '문자열'로
    #    직렬화해 전달(data_transformer.py:575) — str 로 온 JSON 도 파싱해 압축.
    import json as _json
    json_str = _json.dumps(data, ensure_ascii=False, indent=2)
    out_s = engine._extract_core_result(json_str)
    t("X-9 JSON 문자열 입력도 파싱→핵심 압축",
      "24.59°C" in out_s and "22.48°C" in out_s and "[결과 1]" in out_s)
    t("X-10 JSON 문자열 입력도 metadata 배제",
      "source_info" not in out_s and "api_provider" not in out_s)

    # ── 일반 텍스트 문자열은 기존대로 그대로 반환 ──
    plain = engine._extract_core_result("그냥 평범한 텍스트 결과입니다")
    t("X-11 비JSON 문자열은 그대로(기존 동작)", plain == "그냥 평범한 텍스트 결과입니다")

    # ── full-HTML answer(scheduler 프리미엄 뷰 등)는 핸드오프용 텍스트로 변환 ──
    # (2026-07-07 D3 실측: HTML 이 다음 stage 입력과 최종 join 을 오염)
    html_page = (
        "<style>.card{color:red}</style>\n"
        '<div class="schedule-container"><div class="event">'
        "<span>09:00</span> 팀 회의</div>"
        "<div class='event'><span>14:00</span> 프로젝트 리뷰 &amp; 정리</div></div>"
    )
    txt = engine._extract_core_result(html_page)
    t("X-12 full-HTML: 태그 제거된 텍스트", "<div" not in txt and "<style" not in txt)
    t("X-13 full-HTML: 일정 내용은 보존", "팀 회의" in txt and "프로젝트 리뷰" in txt)
    t("X-14 full-HTML: CSS 본문 제거 + 엔티티 복원", ".card" not in txt and "&" in txt)

    doc_page = "<!DOCTYPE html>\n<html><body><h1>주간 일정</h1><p>수요일 세미나</p></body></html>"
    txt2 = engine._extract_core_result(doc_page)
    t("X-15 DOCTYPE 문서도 텍스트 변환", "<html" not in txt2 and "주간 일정" in txt2 and "수요일 세미나" in txt2)

    # dict answer 값이 HTML 이어도 (재귀로 str 분기 통과) 변환
    nested_html = {"success": True, "data": {"result": {"answer": html_page}}}
    txt3 = engine._extract_core_result(nested_html)
    t("X-16 중첩 dict 의 HTML answer 도 변환", "<div" not in txt3 and "팀 회의" in txt3)

    # 인라인 태그 살짝 섞인 markdown 은 건드리지 않음 (보수적 감지)
    md = "## 제목\n일부 <b>강조</b> 텍스트"
    t("X-17 markdown+인라인 태그는 그대로", engine._extract_core_result(md) == md)

    print("RESULT:", "GREEN" if not fails else f"RED ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
