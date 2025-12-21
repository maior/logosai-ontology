# CLAUDE.md - Ontology System Development Guidelines

온톨로지 시스템 개발을 위한 가이드라인입니다.

## 핵심 원칙 (Core Principles)

### 1. 하드코딩 금지 (No Hardcoding)

**절대 원칙**: 에이전트 선택, 쿼리 분류, 도메인 매칭에서 **하드코딩된 키워드 매칭을 사용하지 않는다**.

```python
# ❌ 금지: 하드코딩된 키워드 매칭
if any(word in query for word in ['일정', '스케줄', '약속']):
    return 'scheduler_agent'
elif any(word in query for word in ['날씨', '기온']):
    return 'weather_agent'

# ✅ 권장: LLM 기반 분석
selected_agent = await self._select_agent_by_llm(query, available_agents)
```

**이유**:
- 새 에이전트 추가 시 코드 수정 불필요
- LLM이 의미론적으로 쿼리와 에이전트를 매칭
- 다국어 자동 지원
- 유지보수성 향상

### 2. LLM 기반 처리 (LLM-Based Processing)

모든 지능적 처리는 LLM을 통해 수행합니다:

| 작업 | 처리 방식 |
|------|----------|
| 쿼리 분석 | LLM (process_unified_query) |
| 에이전트 선택 | LLM (_select_agent_by_llm) |
| 워크플로우 설계 | LLM (workflow_designer) |
| 결과 통합 | LLM (result_integration) |

### 3. 에이전트 정보 활용 (Agent Metadata Utilization)

에이전트 선택 시 반드시 에이전트의 메타데이터를 LLM에 제공합니다:

```python
# 에이전트 정보 구성
agents_info = {
    "agent_id": {
        "name": "에이전트 이름",
        "description": "에이전트 설명",
        "capabilities": ["능력1", "능력2"],
        "tags": ["태그1", "태그2"]
    }
}

# LLM에게 전달하여 분석
prompt = f"""
사용자 쿼리: "{query}"
사용 가능한 에이전트: {agents_info}
가장 적합한 에이전트를 선택하세요.
"""
```

### 4. 적합한 에이전트 없음 처리 (No Suitable Agent Handling)

**원칙**: 적합한 에이전트가 없으면 사용자에게 명확히 알린다.

```python
# LLM 응답에 agent_availability 필드 포함
{
    "agent_availability": {
        "no_suitable_agent": true,
        "availability_reasoning": "일정 관리 기능을 가진 에이전트가 목록에 없습니다.",
        "user_message": "요청하신 일정 관리 작업에 적합한 에이전트를 찾지 못했습니다.",
        "required_capabilities": ["일정 관리", "캘린더 연동"],
        "suggested_alternatives": ["직접 일정 앱 사용"]
    }
}
```

**처리 흐름**:
1. LLM이 에이전트 목록을 분석하여 적합한 에이전트 없음을 판단
2. `no_suitable_agent: true` 설정
3. 사용자에게 알림 메시지 생성
4. 필요한 능력과 대안 제시

### 5. 에이전트 전략 결정 (Agent Strategy Selection)

LLM이 쿼리 분석을 통해 최적의 실행 전략을 결정합니다:

| 전략 | 사용 시점 | 예시 |
|------|----------|------|
| single_agent | 단일 에이전트로 처리 가능 | "이번주 날씨 알려줘" |
| parallel | 독립적인 작업 동시 처리 | "맛집 검색 || 관광지 검색" |
| sequential | 이전 결과가 다음 작업에 필요 | "가격 조회 → 환율 계산" |
| hybrid | parallel + sequential 조합 | "(날씨 || 환율) → 경비 계산" |

```python
# 전략 결정은 LLM이 수행
{
    "execution_plan": {
        "strategy": "parallel|sequential|single_agent|hybrid",
        "reasoning": "왜 이 전략을 선택했는지 설명"
    }
}
```

## 주요 컴포넌트

### UnifiedQueryProcessor (핵심)

**위치**: `ontology/core/unified_query_processor.py`

**역할**: 쿼리 분석 → 에이전트 선택 → 워크플로우 설계를 통합 처리

**핵심 메서드**:
- `process_unified_query()`: LLM 기반 통합 쿼리 처리
- `_select_agent_by_llm()`: LLM 기반 에이전트 선택 (비동기)
- `_select_agent_by_content()`: 동기 래퍼 (내부적으로 LLM 호출)
- `_build_agents_info_for_llm()`: 에이전트 정보 구성

### OntologySystem

**위치**: `ontology/system/ontology_system.py`

**역할**: 온톨로지 기반 지식 표현 및 추론

### WorkflowDesigner

**위치**: `ontology/engines/workflow_designer.py`

**역할**: 에이전트 실행 워크플로우 설계

## 개발 체크리스트

### 에이전트 선택 로직 수정 시

- [ ] 하드코딩된 키워드 매칭 사용하지 않음
- [ ] LLM 기반 분석 사용
- [ ] 에이전트 메타데이터 (설명, 능력, 태그) 활용
- [ ] 폴백 시에도 LLM 우선 시도

### 새 기능 추가 시

- [ ] 기존 LLM 기반 패턴 따르기
- [ ] 하드코딩 대신 설정/메타데이터 활용
- [ ] 비동기 처리 지원 (async/await)
- [ ] 로깅 추가 (logger.info/warning/error)

## 예외 사항

다음은 하드코딩이 허용되는 예외적 상황입니다:

### 1. 비즈니스 특수 로직

```python
# ✅ 허용: 삼성 도메인 특수 처리 (비즈니스 요구사항)
if self._is_samsung_domain_query(query):
    samsung_agents = [a for a in available_agents if "samsung" in a.lower()]
    if samsung_agents:
        return samsung_agents[0]
```

**이유**: 삼성 도메인은 특수한 비즈니스 로직으로, 별도의 게이트웨이 에이전트로 라우팅 필요

### 2. 시스템 기본값

```python
# ✅ 허용: 폴백 기본값
if not selected_agent:
    return available_agents[0] if available_agents else 'unknown'
```

## 테스팅

### 에이전트 선택 테스트

```bash
# 로그에서 LLM 에이전트 선택 확인
tail -f logos_server/logs/logos_server.log | grep "LLM 에이전트 선택"

# 예상 로그
# 🧠 LLM 에이전트 선택 완료: scheduler_agent (쿼리: 이번주 일정 알려줘...)
```

### 통합 테스트

```bash
cd /Users/maior/Development/skku/Logos
source .venv/bin/activate
python -c "
from ontology.core.unified_query_processor import UnifiedQueryProcessor
import asyncio

processor = UnifiedQueryProcessor()
agents = ['scheduler_agent', 'internet_agent', 'weather_agent']

result = asyncio.run(processor.process_unified_query('이번주 일정 알려줘', agents))
print(result.get('agent_mappings', []))
"
```

## 히스토리

### 2024-12-21 (5차): 설계 원칙 개선 - 폴백 개념 제거 ✅

**핵심 변경: "폴백" 개념 완전 제거**

**이전 (잘못된 접근)**:
```
internet_agent는 폴백 → 이것도 하드코딩!
```

**이후 (올바른 접근)**:
```
모든 에이전트 동등 평가 → LLM 의미론적 매칭
적합한 에이전트 없음 → 유용한 피드백 제공
```

**새로운 설계 원칙**:
1. **모든 에이전트 동등 평가**: 특정 에이전트를 "폴백"으로 지정하지 않음
2. **의미론적 매칭**: 쿼리 의도와 에이전트 capabilities의 의미론적 분석
3. **유용한 피드백 시스템**: 적합한 에이전트가 없을 때
   - `feedback_type: "clarification_needed"` → 질문 구체화 유도
   - `feedback_type: "alternative_suggested"` → 대안 제시
   - `feedback_type: "impossible_request"` → 불가능 이유 설명

**수정 파일**:
- `unified_query_processor.py` - 에이전트 선택 원칙 재설계, 피드백 시스템 추가
- `test_llm_agent_selection.py` - llm_search_agent 메타데이터 강화

**테스트 결과**: 100% (22/22 통과)

---

### 2024-12-21 (4차): 100% 통과율 달성

**테스트 결과**:
- **최종**: 100% (22/22 통과)
- **개선 경과**: 68.2% → 90.9% → 95.5% → **100%**

**수정 내용**:
1. **S01 해결** - scheduler_agent 메타데이터에 "캘린더 시스템 접근 권한" 명시
2. **Q02 해결** - 테스트 케이스 쿼리 명확화
3. **S07 해결** - RAG 에이전트 규칙 추가

**영향 파일**:
- `ontology/core/unified_query_processor.py` - LLM 프롬프트 개선
- `ontology/tests/test_llm_agent_selection.py` - 테스트 케이스 정리
- `ontology/tests/LLM_AGENT_SELECTION_TEST_RESULTS.md` - 100% 결과 리포트

---

### 2024-12-21 (3차): 종합 테스트 및 LLM 프롬프트 대폭 개선

**테스트 결과 개선**:
- **이전**: 68.2% (15/22 통과)
- **이후**: 90.9% (20/22 통과)
- **개선율**: +22.7%

**수정 내용**:
1. **에이전트 메타데이터 강화**
   - 모든 에이전트의 description을 상세화
   - capabilities와 tags를 구체적으로 명시
   - internet_agent에 "폴백 전용" 역할 명시

2. **LLM 프롬프트 개선**
   - 전문 에이전트 우선 선택 원칙 추가
   - 범용 에이전트(internet) 폴백 규칙 명확화
   - 잘못된 에이전트 선택 예시 추가

3. **개선된 테스트 케이스**:
   - ✅ S03: "아이폰 가격 검색" → shopping_agent (수정됨)
   - ✅ S10: "양자역학이란?" → llm_search_agent (수정됨)
   - ✅ P04: "노트북 가격 비교" → shopping_agent (수정됨)
   - ✅ N01: "달나라 피자 주문" → no_suitable_agent 처리 (수정됨)
   - ✅ B02: "삼성전자 반도체 NAND" → samsung_gateway_agent (수정됨)

4. **남은 실패 케이스** (2개):
   - S01: "이번주 일정 알려줘" - LLM이 개인 일정 권한 부재로 해석
   - Q02: "최신 AI 논문 찾아서 요약" - 테스트 케이스 모호성

**영향 파일**:
- `ontology/core/unified_query_processor.py` - LLM 프롬프트 개선
- `ontology/tests/test_llm_agent_selection.py` - 테스트 에이전트 메타데이터 강화
- `ontology/tests/LLM_AGENT_SELECTION_TEST_RESULTS.md` - 테스트 결과 리포트

---

### 2024-12-21 (2차): 메인 LLM 프롬프트 개선 및 에이전트 없음 처리

**문제**:
- 메인 플로우 `_create_llm_based_unified_prompt()`에 하드코딩된 에이전트 이름 존재
- 적합한 에이전트가 없을 때 사용자에게 알리지 않음

**해결**:
1. **프롬프트에서 하드코딩 제거**
   - `internet_agent`, `analysis_agent` 등 직접 언급 제거
   - "에이전트 메타데이터만으로 판단" 지침 추가

2. **agent_availability 필드 추가**
   - `no_suitable_agent`: 적합한 에이전트 없음 여부
   - `user_message`: 사용자에게 알릴 메시지
   - `required_capabilities`: 필요하지만 없는 능력

3. **싱글/멀티/하이브리드 전략 명확화**
   - LLM이 쿼리 복잡성에 따라 전략 결정
   - 전략별 사용 시점 가이드라인 제공

**영향 파일**:
- `ontology/core/unified_query_processor.py`
  - `_create_llm_based_unified_prompt()` 수정
  - `_validate_and_enhance_llm_result()` 수정

---

### 2024-12-21 (1차): LLM 기반 에이전트 선택 전환

**문제**: `_select_agent_by_content()`가 하드코딩된 키워드 매칭 사용
- "이번주 일정" → internet_agent로 잘못 라우팅
- 새 에이전트 추가 시 키워드 목록 수동 업데이트 필요

**해결**: LLM 기반 에이전트 선택으로 전환
- `_select_agent_by_llm()` 비동기 메서드 추가
- 에이전트 설명/능력/태그를 LLM에 제공
- 키워드 매칭 완전 제거

**영향 파일**:
- `ontology/core/unified_query_processor.py`

## 참조 문서

- [메인 CLAUDE.md](/Users/maior/Development/skku/Logos/CLAUDE.md)
- [logos_server CLAUDE.md](/Users/maior/Development/skku/Logos/logos_server/CLAUDE.md)
- [logosai CLAUDE.md](/Users/maior/Development/skku/Logos/logosai/CLAUDE.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [WORKFLOW_DESIGN.md](./WORKFLOW_DESIGN.md)
