# 간단한 온톨로지 시스템 지식 그래프

```mermaid
graph TD
    query_1["오늘 서울 날씨 알려줘"]
    analysis_1<"쿼리 분석">
    agent_weather_1_0("weather 에이전트")
    result_weather_1_0["weather 결과"]
    concept_오늘{오늘}
    concept_서울{서울}
    concept_날씨{날씨}
    query_2["USD/KRW 환율을 조회하고 차트로"]
    analysis_2<"쿼리 분석">
    agent_currency_2_0("currency 에이전트")
    result_currency_2_0["currency 결과"]
    agent_visualization_2_1("visualization 에이전트")
    result_visualization_2_1["visualization 결과"]
    concept_usd/krw{USD/KRW}
    concept_환율을{환율을}
    concept_조회하고{조회하고}
    concept_차트로{차트로}
    concept_만들어줘{만들어줘}
    query_3["1+1을 계산하고 결과를 메모에 저장"]
    analysis_3<"쿼리 분석">
    agent_calculator_3_0("calculator 에이전트")
    result_calculator_3_0["calculator 결과"]
    agent_memo_3_1("memo 에이전트")
    result_memo_3_1["memo 결과"]
    concept_1+1을{1+1을}
    concept_계산하고{계산하고}
    concept_결과를{결과를}
    concept_메모에{메모에}
    concept_저장해줘{저장해줘}
    query_4["최신 AI 기술 동향을 검색해줘"]
    analysis_4<"쿼리 분석">
    agent_search_4_0("search 에이전트")
    result_search_4_0["search 결과"]
    concept_최신{최신}
    concept_ai{AI}
    concept_기술{기술}
    concept_동향을{동향을}
    concept_검색해줘{검색해줘}
    query_5["Python 웹 크롤링 방법 알려줘"]
    analysis_5<"쿼리 분석">
    agent_general_5_0("general 에이전트")
    result_general_5_0["general 결과"]
    concept_python{Python}
    concept_크롤링{크롤링}
    concept_방법{방법}
    query_1 -..-> analysis_1
    analysis_1 --> agent_weather_1_0
    agent_weather_1_0 --> result_weather_1_0
    query_1 --- concept_오늘
    query_1 --- concept_서울
    query_1 --- concept_날씨
    query_2 -..-> analysis_2
    analysis_2 --> agent_currency_2_0
    agent_currency_2_0 --> result_currency_2_0
    analysis_2 --> agent_visualization_2_1
    agent_visualization_2_1 --> result_visualization_2_1
    query_2 --- concept_usd/krw
    query_2 --- concept_환율을
    query_2 --- concept_조회하고
    query_2 --- concept_차트로
    query_2 --- concept_만들어줘
    query_3 -..-> analysis_3
    analysis_3 --> agent_calculator_3_0
    agent_calculator_3_0 --> result_calculator_3_0
    analysis_3 --> agent_memo_3_1
    agent_memo_3_1 --> result_memo_3_1
    query_3 --- concept_1+1을
    query_3 --- concept_계산하고
    query_3 --- concept_결과를
    query_3 --- concept_메모에
    query_3 --- concept_저장해줘
    query_4 -..-> analysis_4
    analysis_4 --> agent_search_4_0
    agent_search_4_0 --> result_search_4_0
    query_4 --- concept_최신
    query_4 --- concept_ai
    query_4 --- concept_기술
    query_4 --- concept_동향을
    query_4 --- concept_검색해줘
    query_5 -..-> analysis_5
    analysis_5 --> agent_general_5_0
    agent_general_5_0 --> result_general_5_0
    query_5 --- concept_python
    query_5 --- concept_크롤링
    query_5 --- concept_방법
```
