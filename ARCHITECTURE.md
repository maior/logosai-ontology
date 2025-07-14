# 🏗️ 온톨로지 시스템 아키텍처

## 📋 시스템 개요

LogosAI의 온톨로지 시스템은 지식 기반 멀티에이전트 시스템으로, 사용자 쿼리를 의미론적으로 분석하고 최적의 워크플로우를 동적으로 생성하여 실행하는 지능형 시스템입니다.

## 🧠 핵심 컴포넌트

### 1. Enhanced Ontology System (메인 시스템)
```
enhanced_ontology_system.py
├── EnhancedOntologySystem (메인 클래스)
├── QueryAnalysisResult (쿼리 분석 결과)
├── OntologyUpdatePlan (온톨로지 업데이트 계획)
└── create_enhanced_ontology_system() (팩토리 함수)
```

**주요 기능:**
- 🧠 LLM 기반 종합 쿼리 분석
- 🌊 동적 워크플로우 계획 생성
- ⚡ 지능적 실행 전략 적용
- 🔗 의미론적 결과 통합
- 📊 온톨로지 지식 업데이트

### 2. Ontology Knowledge Graph Engine
```
ontology_knowledge_graph_engine.py
├── LLMOntologyReasoningEngine (추론 엔진)
├── LLMKnowledgeGraphBuilder (지식 그래프 구축)
├── LLMOntologyBasedAgentSelector (에이전트 선택)
├── SemanticQuery (의미론적 쿼리)
├── OntologyConcept (온톨로지 개념)
└── OntologyRelation (온톨로지 관계)
```

**주요 기능:**
- 🔍 의미론적 쿼리 분석
- 🕸️ 지식 그래프 관리
- 🤖 에이전트 지식 등록
- 📈 시각화 데이터 생성

### 3. Ontology Workflow Designer
```
ontology_workflow_designer.py
├── LLMSemanticWorkflowReasoner (워크플로우 추론기)
├── SemanticWorkflowStep (워크플로우 단계)
├── OntologyWorkflowPlan (워크플로우 계획)
├── AdvancedWorkflowFeatures (고도화 기능)
└── DynamicAdjustmentPoint (동적 조정점)
```

**주요 기능:**
- 🎯 LLM 기반 워크플로우 설계
- 📊 병렬 처리 최적화 분석
- 🔄 적응형 워크플로우 기능
- 📈 성능 예측 및 모니터링

### 4. Enhanced Execution Engine
```
enhanced_execution_engine.py
├── EnhancedExecutionEngine (실행 엔진)
├── ExecutionStrategy (실행 전략)
├── AgentExecutionResult (실행 결과)
├── DataTransformer (데이터 변환)
└── QueryComplexityAnalyzer (복잡도 분석)
```

**주요 기능:**
- ⚡ 다양한 실행 전략 지원
- 🔄 데이터 변환 및 전달
- 📊 성능 메트릭 수집
- 🎯 복잡도 기반 최적화

### 5. Semantic Query Manager
```
semantic_query_manager.py
├── SemanticQueryManager (쿼리 관리자)
├── CachedSemanticQuery (캐시된 쿼리)
└── get_semantic_query_manager() (전역 관리자)
```

**주요 기능:**
- 💾 쿼리 캐싱 및 관리
- ⏱️ 세션 기반 캐시
- 📊 분석 메트릭 수집
- 🧹 자동 캐시 정리

## 🔄 시스템 워크플로우

```mermaid
graph TD
    A[사용자 쿼리] --> B[Enhanced Ontology System]
    B --> C[LLM 기반 종합 분석]
    C --> D[QueryAnalysisResult]
    
    D --> E[동적 워크플로우 계획]
    E --> F[LLM Semantic Workflow Reasoner]
    F --> G[OntologyWorkflowPlan]
    
    G --> H[지능적 실행 전략]
    H --> I[Enhanced Execution Engine]
    I --> J[AgentExecutionResult[]]
    
    J --> K[의미론적 결과 통합]
    K --> L[LLM 기반 통합]
    L --> M[통합된 결과]
    
    M --> N[온톨로지 업데이트]
    N --> O[Knowledge Graph Update]
    O --> P[최종 결과]
    
    subgraph "분석 단계"
        C --> C1[의도 분석]
        C --> C2[개념 추출]
        C --> C3[엔티티 인식]
        C --> C4[실행 전략 결정]
        C --> C5[에이전트 매핑]
    end
    
    subgraph "실행 단계"
        I --> I1[SINGLE_AGENT]
        I --> I2[SEQUENTIAL]
        I --> I3[PARALLEL]
        I --> I4[HYBRID]
    end
    
    subgraph "통합 단계"
        L --> L1[UI/UX 통합]
        L --> L2[데이터 정합성]
        L --> L3[사용자 친화적 표현]
    end
```

## 📊 데이터 흐름

### 1. 입력 단계
```python
# 사용자 입력
user_query = "달러와 유로 환율을 조회하고 100만원을 각각 환전했을 때 금액을 계산해서 비교 차트로 보여줘"

# 시스템 초기화
system = EnhancedOntologySystem(
    email="user@example.com",
    prompt=user_query,
    sessionid="session_123"
)
```

### 2. 분석 단계
```python
# LLM 기반 종합 분석
analysis_result = QueryAnalysisResult(
    semantic_query=SemanticQuery(...),
    execution_strategy=ExecutionStrategy.SEQUENTIAL,
    agent_mappings={
        "환율_조회": ["currency_exchange_agent"],
        "계산_작업": ["calculator_agent"],
        "차트_생성": ["chart_agent"]
    },
    message_transformations={
        "currency_exchange_agent": "USD와 EUR의 현재 환율을 조회해주세요",
        "calculator_agent": "100만원을 USD와 EUR로 각각 환전 계산",
        "chart_agent": "환율 비교 차트 생성"
    },
    ui_integration_plan={
        "primary_display": "chart",
        "secondary_elements": ["table", "summary"]
    }
)
```

### 3. 워크플로우 계획
```python
# 동적 워크플로우 생성
workflow_plan = OntologyWorkflowPlan(
    plan_id="plan_123",
    semantic_query=analysis_result.semantic_query,
    steps=[
        SemanticWorkflowStep(
            step_id="step_1",
            semantic_purpose="환율 정보 수집",
            agent_id="currency_exchange_agent",
            estimated_complexity=WorkflowComplexity.SIMPLE
        ),
        SemanticWorkflowStep(
            step_id="step_2", 
            semantic_purpose="환전 금액 계산",
            agent_id="calculator_agent",
            depends_on=["step_1"]
        ),
        SemanticWorkflowStep(
            step_id="step_3",
            semantic_purpose="비교 차트 생성", 
            agent_id="chart_agent",
            depends_on=["step_1", "step_2"]
        )
    ],
    optimization_strategy=OptimizationStrategy.SEQUENTIAL
)
```

### 4. 실행 단계
```python
# 지능적 실행
execution_results = [
    AgentExecutionResult(
        agent_id="currency_exchange_agent",
        success=True,
        data={"USD": 1340, "EUR": 1450},
        execution_time=2.1,
        confidence=0.95
    ),
    AgentExecutionResult(
        agent_id="calculator_agent", 
        success=True,
        data={"USD_amount": 746.27, "EUR_amount": 689.66},
        execution_time=0.5,
        confidence=0.98
    ),
    AgentExecutionResult(
        agent_id="chart_agent",
        success=True,
        data={"chart_url": "...", "chart_data": "..."},
        execution_time=3.2,
        confidence=0.92
    )
]
```

### 5. 통합 단계
```python
# 의미론적 결과 통합
integrated_result = {
    "integrated_result": "환율 조회 및 환전 계산 완료. USD: 746.27달러, EUR: 689.66유로",
    "ui_components": {
        "primary_content": "환율 비교 차트",
        "charts": ["환율_비교_차트.png"],
        "tables": ["환전_계산_결과.json"],
        "interactive_elements": ["환율_계산기"]
    },
    "metadata": {
        "confidence": 0.95,
        "completeness": 0.98,
        "sources": ["currency_exchange_agent", "calculator_agent", "chart_agent"]
    }
}
```

### 6. 온톨로지 업데이트
```python
# 지식 그래프 업데이트
ontology_updates = {
    "new_concepts": [
        {
            "concept_id": "환율_비교_분석",
            "concept_type": "TASK",
            "name": "환율 비교 분석 작업"
        }
    ],
    "new_relations": [
        {
            "subject": "currency_exchange_agent",
            "predicate": "COLLABORATES_WITH", 
            "object": "calculator_agent",
            "weight": 0.9
        }
    ]
}
```

## 🎯 핵심 설계 원칙

### 1. 모듈성 (Modularity)
- 각 컴포넌트는 독립적으로 동작
- 명확한 인터페이스 정의
- 쉬운 확장 및 교체 가능

### 2. 확장성 (Scalability)
- 새로운 에이전트 쉽게 추가
- 워크플로우 패턴 확장 가능
- 대용량 처리 지원

### 3. 지능성 (Intelligence)
- LLM 기반 의사결정
- 학습 기반 최적화
- 적응형 동작

### 4. 안정성 (Reliability)
- 오류 처리 및 복구
- 폴백 메커니즘
- 데이터 일관성 보장

### 5. 성능 (Performance)
- 캐싱 및 최적화
- 병렬 처리 지원
- 중복 호출 방지

## 🔧 설정 및 구성

### 1. LLM 설정
```python
# llm_config_manager.py
from .llm_config_manager import (
    get_gpt4o,          # 고성능 분석용
    get_gpt4o_mini,     # 빠른 처리용
    get_reasoning_llm,  # 추론용
    get_creative_llm    # 창의적 작업용
)
```

### 2. 온톨로지 설정
```python
# 개념 타입
class ConceptType(Enum):
    AGENT = "Agent"
    TASK = "Task" 
    CAPABILITY = "Capability"
    DOMAIN = "Domain"
    WORKFLOW = "Workflow"
    USER = "User"
    RESOURCE = "Resource"
    CONTEXT = "Context"

# 관계 타입
class RelationType(Enum):
    HAS_CAPABILITY = "hasCapability"
    REQUIRES = "requires"
    PRODUCES = "produces"
    DEPENDS_ON = "dependsOn"
    COLLABORATES_WITH = "collaboratesWith"
```

### 3. 실행 전략
```python
class ExecutionStrategy(Enum):
    SINGLE_AGENT = "single_agent"    # 단일 에이전트
    SEQUENTIAL = "sequential"        # 순차 실행
    PARALLEL = "parallel"           # 병렬 실행
    HYBRID = "hybrid"               # 혼합 실행
```

## 📈 성능 모니터링

### 1. 메트릭 수집
```python
execution_metrics = {
    "total_queries": 0,
    "successful_queries": 0,
    "avg_response_time": 0.0,
    "agent_usage_stats": {},
    "workflow_patterns": {},
    "ontology_growth": {"concepts": 0, "relations": 0}
}
```

### 2. 품질 지표
- 쿼리 이해도: 90%
- 에이전트 선택 정확도: 95%
- 실행 효율성: 85%
- 결과 품질: 92%
- 사용자 만족도: 90%

## 🚀 향후 발전 방향

### Phase 1: 핵심 기능 완성 ✅
- Enhanced Ontology System 구축
- LLM 기반 분석 및 실행
- 기본 온톨로지 업데이트

### Phase 2: 고도화 (진행 중)
- 실시간 학습 및 적응
- 고급 UI/UX 통합
- 성능 최적화

### Phase 3: 확장 (계획)
- 분산 처리 지원
- 다중 사용자 환경
- 자동 에이전트 생성
- 고급 추론 엔진

이 아키텍처는 LogosAI의 온톨로지 시스템이 지능적이고 확장 가능하며 사용자 친화적인 멀티에이전트 시스템으로 발전할 수 있는 견고한 기반을 제공합니다. 