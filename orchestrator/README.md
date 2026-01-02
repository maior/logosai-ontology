# Workflow Orchestrator

LLM 기반 지능형 멀티 에이전트 워크플로우 오케스트레이션 시스템입니다.

## 개요

사용자 쿼리를 분석하여 최적의 에이전트 조합과 실행 전략을 자동으로 결정하고, 실시간 진행 상황을 프론트엔드에 스트리밍합니다.

```
쿼리 → QueryPlanner(Flash-Lite) → ExecutionPlan → Validator → ExecutionEngine → Result
                                                                      ↓
                                                              ProgressStreamer → Frontend
```

## 핵심 특징

- **단일 LLM 호출 계획**: gemini-2.5-flash-lite로 빠른 계획 생성 (1-2초)
- **실시간 스트리밍**: SSE, WebSocket, 콜백 지원
- **한글/영문 메시지**: 프론트엔드 UI 친화적
- **자동 데이터 변환**: 에이전트 간 스키마 호환성 자동 처리
- **병렬/순차 실행**: 쿼리 복잡도에 따른 최적 전략 선택

## 빠른 시작

### 기본 사용법

```python
from ontology.orchestrator import WorkflowOrchestrator

# 에이전트 실행 함수 정의
async def my_agent_executor(agent_id: str, sub_query: str, context: dict):
    # 실제 에이전트 호출 로직
    result = await call_agent(agent_id, sub_query)
    return result

# 오케스트레이터 생성
orchestrator = WorkflowOrchestrator(
    agent_executor=my_agent_executor,
    enable_validation=True,
    enable_streaming=True,
)

# 쿼리 실행
result = await orchestrator.run("삼성전자 5일 종가 그래프로 그려줘")
print(result.final_output)
```

### 스트리밍 사용법 (프론트엔드 연동)

```python
# SSE 스트리밍 (FastAPI)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_query(query: str):
    return StreamingResponse(
        orchestrator.run_sse(query),
        media_type="text/event-stream"
    )
```

```python
# 이벤트 핸들링
async for event in orchestrator.run_streaming(query):
    print(f"[{event.type.value}] {event.message_ko}")
    print(f"  Progress: {event.progress_percent}%")

    if event.type == ProgressEventType.AGENT_COMPLETE:
        print(f"  {event.icon} {event.display_name} 완료")
```

### 계획만 미리보기

```python
# 실행 없이 계획만 확인
plan = await orchestrator.create_plan_only("삼성전자와 애플 실적 비교해줘")

print(f"전략: {plan.workflow_strategy}")  # "hybrid"
print(f"스테이지: {plan.get_stage_count()}")  # 2
print(f"에이전트: {plan.get_total_agents()}")  # 3
print(f"이유: {plan.reasoning}")
```

## 컴포넌트 구조

```
ontology/orchestrator/
├── __init__.py              # 패키지 exports
├── models.py                # 데이터 모델 (ProgressEvent, ExecutionPlan, etc.)
├── exceptions.py            # 예외 클래스
├── progress_streamer.py     # 실시간 스트리밍 엔진
├── agent_registry.py        # 에이전트 레지스트리
├── query_planner.py         # LLM 기반 쿼리 계획 (Flash-Lite)
├── plan_validator.py        # 실행 계획 검증
├── data_transformer.py      # 에이전트 간 데이터 변환
├── execution_engine.py      # 순차/병렬 실행 엔진
├── result_aggregator.py     # 결과 통합
├── workflow_orchestrator.py # 메인 오케스트레이터
└── test_orchestrator.py     # 통합 테스트
```

## 스트리밍 이벤트

### 이벤트 타입

| 이벤트 | 설명 | 진행률 |
|--------|------|--------|
| `planning_start` | 쿼리 분석 시작 | 5% |
| `planning_complete` | 실행 계획 생성 완료 | 10% |
| `validation_start` | 계획 검증 시작 | 12% |
| `validation_complete` | 검증 완료 | 15% |
| `workflow_start` | 워크플로우 실행 시작 | 0% |
| `stage_start` | 스테이지 시작 | 동적 계산 |
| `agent_queued` | 에이전트 대기열 추가 | - |
| `agent_start` | 에이전트 실행 시작 | - |
| `agent_complete` | 에이전트 실행 완료 | 동적 계산 |
| `stage_complete` | 스테이지 완료 | 동적 계산 |
| `transform_start` | 데이터 변환 시작 | - |
| `transform_complete` | 데이터 변환 완료 | - |
| `workflow_complete` | 전체 완료 | 100% |

### 이벤트 데이터 구조

```python
@dataclass
class ProgressEvent:
    type: ProgressEventType       # 이벤트 타입
    timestamp: datetime           # 발생 시각
    workflow_id: str              # 워크플로우 ID
    stage_id: int                 # 스테이지 번호
    agent_id: str                 # 에이전트 ID
    status: AgentStatus           # 상태 (pending/running/completed/failed)
    progress_percent: float       # 진행률 (0-100)
    message: str                  # 영문 메시지
    message_ko: str               # 한글 메시지
    display_name: str             # UI 표시명
    icon: str                     # 에이전트 아이콘 (🌐📊📈🔍 등)
    elapsed_time_ms: float        # 경과 시간
    error: str                    # 에러 메시지 (실패 시)
```

### 프론트엔드 연동 예시 (React)

```typescript
// EventSource를 사용한 SSE 연동
const eventSource = new EventSource(`/api/stream?query=${encodeURIComponent(query)}`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  // 진행률 업데이트
  setProgress(data.progress_percent);

  // 상태 메시지 표시
  setMessage(data.message_ko);

  // 에이전트 상태 업데이트
  if (data.type === 'agent_complete') {
    updateAgentStatus(data.agent_id, 'completed');
  }

  // 완료 처리
  if (data.type === 'workflow_complete') {
    setFinalResult(data.data);
    eventSource.close();
  }
};
```

## 실행 전략

### Sequential (순차)

```
Stage 1: [internet_agent] → Stage 2: [analysis_agent] → Stage 3: [visualization_agent]
```

이전 스테이지 결과가 다음 스테이지 입력으로 전달됩니다.

### Parallel (병렬)

```
Stage 1: [internet_agent(삼성)] + [internet_agent(애플)]  (동시 실행)
         ↓
Stage 2: [analysis_agent]  (결과 병합 후 분석)
```

독립적인 작업을 동시에 실행하여 시간을 단축합니다.

### Hybrid (하이브리드)

복잡한 쿼리에서 병렬과 순차를 조합합니다.

## 에이전트 레지스트리

### 기본 등록 에이전트

| ID | 이름 | 아이콘 | 입력 | 출력 |
|----|------|--------|------|------|
| `internet_agent` | 인터넷 검색 | 🌐 | query | text |
| `analysis_agent` | 데이터 분석 | 📊 | structured_data | json |
| `data_visualization_agent` | 시각화 | 📈 | json | html |
| `llm_search_agent` | LLM 검색 | 🔍 | query | text |
| `samsung_gateway_agent` | 삼성 게이트웨이 | 🏭 | query | json |
| `shopping_agent` | 쇼핑 검색 | 🛒 | query | json |
| `code_agent` | 코드 생성 | 💻 | query | text |
| `rag_search_agent` | RAG 검색 | 📚 | query | text |

### 커스텀 에이전트 등록

```python
from ontology.orchestrator import AgentRegistry, AgentRegistryEntry, AgentSchema

registry = AgentRegistry()

# 커스텀 에이전트 등록
registry.register(AgentRegistryEntry(
    agent_id="my_custom_agent",
    name="내 커스텀 에이전트",
    description="특수 분석을 수행하는 에이전트",
    capabilities=["custom_analysis", "special_processing"],
    tags=["custom", "analysis"],
    schema=AgentSchema(input_type="json", output_type="json"),
    icon="🔧",
    color="#10b981",
))
```

## 스키마 호환성

에이전트 간 데이터 전달 시 자동으로 타입 변환됩니다:

| 출력 타입 | 입력 타입 | 변환 방식 |
|-----------|-----------|-----------|
| text | query | 직접 전달 |
| text | structured_data | JSON 파싱 시도 |
| json | query | 문자열화 |
| json | structured_data | 직접 전달 |
| structured_data | query | 문자열화 |
| any | * | 항상 호환 |

## 테스트

### 통합 테스트 실행

```bash
cd /Users/maior/Development/skku/Logos
source .venv/bin/activate
python ontology/orchestrator/test_orchestrator.py
```

### 예상 출력

```
============================================================
🚀 WORKFLOW ORCHESTRATOR INTEGRATION TEST
============================================================

✅ Agent Registry: 8개 에이전트 등록
✅ Plan Creation: 3 stages, 3 agents (1535ms)
✅ Parallel Execution: hybrid 전략 감지
✅ Full Execution: 3/3 agents 성공 (1510ms)
✅ Streaming: 29개 이벤트 정상 스트리밍

============================================================
✅ ALL TESTS COMPLETED SUCCESSFULLY!
============================================================
```

### 수정 후 반드시 테스트

```bash
# 빠른 테스트 (모든 수정 후 필수)
python ontology/orchestrator/test_orchestrator.py

# 개별 컴포넌트 테스트
python -c "
from ontology.orchestrator import AgentRegistry
registry = AgentRegistry()
registry.initialize()
print(f'Agents: {len(registry)}')
"
```

## 트러블슈팅

### 스키마 호환성 에러

```
Schema incompatibility: agent_a outputs 'X' but agent_b expects 'Y'
```

**해결**: `models.py`의 `AgentSchema.is_compatible_with()` 메서드에 새 변환 규칙 추가

### LLM 계획 생성 실패

```
[QueryPlanner] Failed to parse LLM response as JSON
```

**해결**:
1. `query_planner.py`의 프롬프트 확인
2. 에이전트 레지스트리에 누락된 에이전트 추가
3. LLM 응답 로그 확인

### 스트리밍 이벤트 누락

**해결**:
1. `enable_streaming=True` 확인
2. `ProgressStreamer` 버퍼 크기 확인 (기본 100)
3. 이벤트 소비 속도 확인

## 성능 최적화

| 항목 | 현재 값 | 조정 가능 |
|------|---------|-----------|
| LLM 계획 시간 | 1-2초 | 모델 변경 |
| 에이전트 타임아웃 | 30초 | `timeout_ms` |
| 재시도 횟수 | 2회 | `max_retries` |
| 재시도 딜레이 | 1초 * 2^attempt | `RETRY_BASE_DELAY` |

## 변경 이력

### v1.0.0 (2024-12-26)

- 초기 릴리스
- gemini-2.5-flash-lite 기반 단일 LLM 호출 계획
- 실시간 스트리밍 (SSE, WebSocket, 콜백)
- 순차/병렬/하이브리드 실행 전략
- 자동 스키마 호환성 처리
- 한글/영문 메시지 지원

## 관련 문서

- [WORKFLOW_ORCHESTRATOR_DESIGN.md](./WORKFLOW_ORCHESTRATOR_DESIGN.md) - 설계 문서
- [ontology/CLAUDE.md](../CLAUDE.md) - 온톨로지 시스템 가이드라인
- [CLAUDE.md](../../CLAUDE.md) - 프로젝트 메인 가이드라인
