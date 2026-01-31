# Workflow Orchestrator Design Document

**Version**: 1.0.0
**Date**: 2025-12-26
**Author**: Claude + Human Collaboration

## 1. Overview

### 1.1 Purpose
사용자 쿼리를 분석하여 최적의 에이전트 워크플로우를 설계하고, 병렬/직렬/하이브리드 실행을 통해 결과를 생성하는 통합 오케스트레이션 시스템.

### 1.2 Design Principles
- **Single LLM Call for Planning**: Flash-Lite 1회 호출로 전체 계획 수립 (100% 정확도 검증됨)
- **Deterministic Execution**: 실행은 LLM 없이 순수 로직으로 처리
- **Data Transformation**: 에이전트 간 데이터 형식 자동 변환
- **Real-time Progress**: 사용자에게 실시간 진행상황 스트리밍

### 1.3 Key Findings (from 4-Model Comparison Test)
| Model | Accuracy | Avg Time |
|-------|----------|----------|
| **gemini-2.5-flash-lite** | **100%** | **3.63s** |
| gemini-2.5-flash | 100% | 9.39s |
| gemini-2.5-flash-lite+thinking | 81.8% | 5.75s |
| gemini-2.5-flash+thinking | 54.5% | 7.13s |

**결론**: gemini-2.5-flash-lite (non-thinking)가 최적

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Query                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    [1] Agent Registry                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • 사용 가능한 에이전트 목록 관리                                     │ │
│  │  • 각 에이전트의 입출력 스키마 정의                                   │ │
│  │  • 에이전트 capabilities 및 메타데이터                               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    [2] Query Planner (Flash-Lite)                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Input:  User Query + Agent Registry                                │ │
│  │  Output: Execution Plan (JSON)                                      │ │
│  │    • workflow_strategy: sequential | parallel | hybrid              │ │
│  │    • stages: [{stage_id, execution_type, agents[]}]                 │ │
│  │    • agents: [{agent_id, sub_query, input_from, output_to}]         │ │
│  │    • final_aggregation: {type, format}                              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    [3] Plan Validator                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • 에이전트 존재 여부 검증                                           │ │
│  │  • 의존성 그래프 검증 (순환 참조 체크)                                │ │
│  │  • 입출력 스키마 호환성 검증                                         │ │
│  │  • 첫 stage의 input_from이 null인지 검증                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    [4] Execution Engine                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  for stage in plan.stages:                                          │ │
│  │      inputs = DataTransformer.transform(stage.input_from, results)  │ │
│  │                                                                      │ │
│  │      if stage.execution_type == "parallel":                         │ │
│  │          results = await asyncio.gather(*[                          │ │
│  │              agent.execute(inputs) for agent in stage.agents        │ │
│  │          ])                                                         │ │
│  │      else:  # sequential                                            │ │
│  │          for agent in stage.agents:                                 │ │
│  │              result = await agent.execute(inputs)                   │ │
│  │              inputs = result                                        │ │
│  │                                                                      │ │
│  │      ResultValidator.validate(stage, results)                       │ │
│  │      ProgressStreamer.emit(stage, results)                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    [5] Data Transformer                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • internet_agent output → analysis_agent input                     │ │
│  │  • analysis_agent output → visualization_agent input                │ │
│  │  • 규칙 기반 변환 우선, 필요시 LLM 폴백                              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    [6] Result Aggregator                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • combine: 단순 병합 (LLM 불필요)                                   │ │
│  │  • summarize: 요약 필요 (LLM 1회)                                    │ │
│  │  • format: 사용자 출력 형식 지정                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    [7] Progress Streamer                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • WebSocket / SSE로 실시간 진행상황 전송                            │ │
│  │  • 각 stage 시작/완료 이벤트                                         │ │
│  │  • 에이전트별 상태 (pending → running → completed)                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Structures

### 3.1 Agent Registry Entry

```python
@dataclass
class AgentSchema:
    """에이전트 입출력 스키마"""
    input_type: str  # "query" | "structured_data" | "any"
    output_type: str  # "text" | "json" | "html" | "chart_data"
    output_format: Optional[Dict]  # JSON schema if applicable

@dataclass
class AgentRegistryEntry:
    """에이전트 레지스트리 엔트리"""
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    tags: List[str]
    schema: AgentSchema
    priority: int = 0  # 에이전트 선택 우선순위
```

### 3.2 Execution Plan

```python
@dataclass
class AgentTask:
    """개별 에이전트 태스크"""
    agent_id: str
    sub_query: str
    input_from: Optional[List[str]]  # ["stage_1.internet_agent"]
    output_to: Optional[List[str]]   # ["stage_2", "final"]
    expected_output: Optional[str]

@dataclass
class ExecutionStage:
    """실행 스테이지"""
    stage_id: int
    execution_type: str  # "sequential" | "parallel"
    agents: List[AgentTask]

@dataclass
class ExecutionPlan:
    """전체 실행 계획"""
    query: str
    workflow_strategy: str  # "sequential" | "parallel" | "hybrid"
    stages: List[ExecutionStage]
    final_aggregation: Dict  # {"type": "combine|summarize", "format": "..."}
    reasoning: str
```

### 3.3 Execution Result

```python
@dataclass
class AgentResult:
    """에이전트 실행 결과"""
    agent_id: str
    stage_id: int
    success: bool
    data: Any
    error: Optional[str]
    execution_time: float
    metadata: Dict

@dataclass
class StageResult:
    """스테이지 실행 결과"""
    stage_id: int
    execution_type: str
    results: List[AgentResult]
    total_time: float

@dataclass
class WorkflowResult:
    """전체 워크플로우 결과"""
    success: bool
    stages: List[StageResult]
    final_output: Any
    total_time: float
    plan: ExecutionPlan
```

---

## 4. Component Details

### 4.1 Agent Registry

**위치**: `ontology/orchestrator/agent_registry.py`

**책임**:
- 사용 가능한 에이전트 목록 관리
- 에이전트 메타데이터 및 스키마 제공
- 동적 에이전트 등록/해제

**주요 메서드**:
```python
class AgentRegistry:
    def get_all_agents() -> List[AgentRegistryEntry]
    def get_agent(agent_id: str) -> Optional[AgentRegistryEntry]
    def register_agent(entry: AgentRegistryEntry) -> None
    def get_agents_by_capability(capability: str) -> List[AgentRegistryEntry]
    def build_prompt_context() -> str  # Planner용 에이전트 목록 생성
```

### 4.2 Query Planner

**위치**: `ontology/orchestrator/query_planner.py`

**책임**:
- 사용자 쿼리 분석
- 최적 워크플로우 설계
- 에이전트 선택 및 서브쿼리 추출

**LLM 설정**:
- Model: `gemini-2.5-flash-lite`
- Thinking: OFF
- Temperature: 0.3

**프롬프트 구조**:
```
1. 에이전트 목록 (Registry에서 제공)
2. 핵심 원칙 (데이터 흐름, 도메인 구분)
3. 사용자 쿼리
4. JSON 응답 형식
```

### 4.3 Plan Validator

**위치**: `ontology/orchestrator/plan_validator.py`

**검증 항목**:
1. **에이전트 존재 검증**: 모든 agent_id가 Registry에 존재
2. **의존성 검증**: input_from 참조가 유효
3. **순환 참조 검증**: A→B→A 같은 순환 없음
4. **스키마 호환성**: 출력→입력 타입 호환

**에러 처리**:
```python
class PlanValidationError(Exception):
    code: str  # "AGENT_NOT_FOUND" | "CIRCULAR_DEPENDENCY" | ...
    message: str
    details: Dict
```

### 4.4 Execution Engine

**위치**: `ontology/orchestrator/execution_engine.py`

**실행 전략**:

**Sequential**:
```python
async def execute_sequential(stage: ExecutionStage, inputs: Any):
    current_input = inputs
    results = []
    for agent_task in stage.agents:
        result = await execute_agent(agent_task, current_input)
        results.append(result)
        current_input = result.data  # 다음 에이전트에 전달
    return results
```

**Parallel**:
```python
async def execute_parallel(stage: ExecutionStage, inputs: Any):
    tasks = [
        execute_agent(agent_task, inputs)
        for agent_task in stage.agents
    ]
    return await asyncio.gather(*tasks)
```

**Hybrid**: Stage 단위로 sequential/parallel 조합

### 4.5 Data Transformer

**위치**: `ontology/orchestrator/data_transformer.py`

**변환 규칙**:

| Source Agent | Target Agent | Transformation |
|--------------|--------------|----------------|
| internet_agent | analysis_agent | HTML/Text → Structured Data |
| analysis_agent | visualization_agent | Dict → Chart Data Format |
| analysis_agent | llm_search_agent | Dict → Context String |
| * | * | LLM Fallback |

**구현**:
```python
class DataTransformer:
    # 규칙 기반 변환 (빠름)
    def transform_internet_to_analysis(data: str) -> Dict
    def transform_analysis_to_visualization(data: Dict) -> ChartData

    # LLM 폴백 (느림, 비용)
    async def llm_transform(source: str, target: str, data: Any) -> Any
```

### 4.6 Result Aggregator

**위치**: `ontology/orchestrator/result_aggregator.py`

**집계 전략**:
- `combine`: 결과 단순 병합 (LLM 불필요)
- `summarize`: LLM으로 요약 생성
- `format`: 특정 형식으로 포맷팅

### 4.7 Progress Streamer

**위치**: `ontology/orchestrator/progress_streamer.py`

**이벤트 타입**:
```python
class ProgressEvent:
    type: str  # "stage_start" | "stage_complete" | "agent_start" | "agent_complete" | "error"
    stage_id: Optional[int]
    agent_id: Optional[str]
    status: str  # "pending" | "running" | "completed" | "failed"
    message: str
    data: Optional[Any]
    timestamp: datetime
```

---

## 5. Execution Flow Examples

### 5.1 Sequential: 주가 그래프

**Query**: "삼성전자 5일 종가 그래프로 그려줘"

```
Stage 1 (sequential):
  └─ internet_agent: "삼성전자 5일 종가 데이터"
       ↓ result: "72,500원, 71,200원, ..."

Stage 2 (sequential):
  └─ analysis_agent: (internet 결과 받음)
       ↓ result: [{date: "...", price: 72500}, ...]

Stage 3 (sequential):
  └─ visualization_agent: (analysis 결과 받음)
       ↓ result: "<svg>...</svg>"

Final: 차트 HTML 반환
```

### 5.2 Parallel: 비교 분석

**Query**: "삼성전자와 애플 실적 비교"

```
Stage 1 (parallel):
  ├─ internet_agent: "삼성전자 실적"
  └─ internet_agent: "애플 실적"
       ↓ results: [samsung_data, apple_data]

Stage 2 (sequential):
  └─ analysis_agent: (두 결과 병합하여 비교)
       ↓ result: {comparison: ...}

Final: 비교 분석 결과 반환
```

### 5.3 Hybrid: 그래프 + 설명

**Query**: "환율 그래프 보여주고 왜 그런지 알려줘"

```
Stage 1 (sequential):
  └─ internet_agent: "원달러 환율 데이터"
       ↓ result: exchange_data

Stage 2 (sequential):
  └─ analysis_agent: (데이터 가공)
       ↓ result: processed_data

Stage 3 (parallel):
  ├─ visualization_agent: (그래프 생성)
  │    ↓ result: "<svg>...</svg>"
  └─ llm_search_agent: (변동 원인 분석)
       ↓ result: "환율 변동 원인은..."

Final: 그래프 + 설명 통합
```

---

## 6. Error Handling

### 6.1 Error Types

| Error | Handling |
|-------|----------|
| Agent Not Found | PlanValidationError 발생 |
| Agent Timeout | 재시도 (max 2회) → 스킵 or 실패 |
| Agent Error | 에러 로깅 → 폴백 or 실패 |
| Transform Error | LLM 폴백 시도 → 실패 |
| Empty Result | 경고 로깅 → 다음 에이전트에 empty 전달 |

### 6.2 Retry Strategy

```python
@retry(max_attempts=2, delay=1.0, exponential_backoff=True)
async def execute_agent_with_retry(agent_task, inputs):
    return await execute_agent(agent_task, inputs)
```

---

## 7. File Structure

```
ontology/
├── orchestrator/
│   ├── __init__.py
│   ├── agent_registry.py      # Agent Registry
│   ├── query_planner.py       # Query Planner (Flash-Lite)
│   ├── plan_validator.py      # Plan Validator
│   ├── execution_engine.py    # Execution Engine
│   ├── data_transformer.py    # Data Transformer
│   ├── result_aggregator.py   # Result Aggregator
│   ├── progress_streamer.py   # Progress Streamer
│   ├── models.py              # Data classes
│   └── exceptions.py          # Custom exceptions
├── docs/
│   └── WORKFLOW_ORCHESTRATOR_DESIGN.md  # This document
└── tests/
    └── test_orchestrator.py   # Integration tests
```

---

## 8. Integration Points

### 8.1 기존 시스템 연동

- **UnifiedQueryProcessor**: Query Planner가 대체/보완
- **OntologyEnhancedMultiAgentSystem**: Execution Engine이 연동
- **logos_server**: API 엔드포인트에서 Orchestrator 호출

### 8.2 Frontend 연동

- **Progress Streamer**: WebSocket/SSE로 실시간 상태 전송
- **Result Format**: 기존 streaming UI 형식 유지

---

## 9. Performance Targets

| Metric | Target |
|--------|--------|
| Planning Time | < 4s (Flash-Lite) |
| Execution Overhead | < 100ms per stage |
| Data Transform | < 50ms (규칙) / < 2s (LLM) |
| Total Latency | Planning + Agent Execution Time + 500ms |

---

## 10. Future Enhancements

1. **동적 워크플로우 수정**: 실행 중 결과에 따라 계획 변경
2. **에이전트 자동 생성**: 적합한 에이전트 없을 시 FORGE로 생성
3. **학습 기반 최적화**: 성공 패턴 학습하여 플래닝 개선
4. **캐싱**: 동일 쿼리 결과 캐싱

---

## Changelog

- **v1.0.0** (2025-12-26): Initial design based on 4-model comparison test results
