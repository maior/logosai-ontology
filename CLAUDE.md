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

#### ⚠️ QueryPlanner 프롬프트 하드코딩 금지 (CRITICAL)

**절대 금지**: `QueryPlanner`의 LLM 프롬프트에 특정 에이전트를 하드코딩하지 않는다.

```python
# ❌ 금지: 프롬프트에 특정 에이전트 하드코딩
prompt = """
## 예시: "날씨 알려줘"
{
  "agents": [{"agent_id": "internet_agent", ...}]  # ❌ 하드코딩!
}
"""

# ✅ 권장: 에이전트 메타데이터 기반 선택 유도
prompt = """
## 에이전트 선택 원칙
1. 전문 에이전트 우선: 특정 도메인 전용 에이전트가 있으면 우선 사용
   - 날씨 쿼리 + weather_agent 존재 → weather_agent 사용
   - 쇼핑 쿼리 + shopping_agent 존재 → shopping_agent 사용
2. 범용 에이전트는 전문 에이전트가 없을 때만 사용
3. 에이전트 description을 분석하여 가장 적합한 에이전트 선택
"""
```

**문제 사례** (query_planner.py):
```python
## 예시 2: "제주 날씨 어때?"
"agent_id": "internet_agent"  # ❌ weather_agent가 있어도 무시됨!
```

**결과**: `weather_agent`가 존재해도 LLM이 프롬프트 예시를 따라 `internet_agent`를 선택

**해결 원칙**:
1. 프롬프트에 특정 agent_id를 예시로 넣지 않음
2. "전문 에이전트 우선" 규칙을 프롬프트에 명시
3. 에이전트 registry의 description/capabilities를 기반으로 LLM이 판단하도록 유도

### 2. 하이브리드 에이전트 선택 (Hybrid Agent Selection) ⭐ v2.0

**Knowledge Graph + LLM** 하이브리드 방식으로 에이전트를 선택합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│              Hybrid Agent Selection v2.0 (2026-01-31)            │
├─────────────────────────────────────────────────────────────────┤
│   Phase 1: Knowledge Graph Analysis (Enhanced)                   │
│   ├─ 엔티티 추출: [삼성전자, 주가]                               │
│   ├─ 관련 개념 탐색: [기업, 금융, 실시간]                        │
│   ├─ 🆕 LLM 의미 분석: category=finance, pattern=stock_query    │
│   ├─ 🆕 시간 감쇠 적용: 7일전 패턴 ×0.9, 30일전 ×0.5            │
│   └─ 🆕 패턴 일반화: "삼성주가" → stock_query → LG주가에도 적용  │
│                           ↓                                      │
│   Phase 2: LLM Final Decision                                    │
│   ├─ Input: query + graph_insights + agent_metadata              │
│   ├─ 의미론적 분석 + 그래프 근거                                 │
│   └─ Output: internet_agent (확신도 95%)                         │
│                           ↓                                      │
│   Phase 3: Feedback Loop (v2.0 Enhanced)                         │
│   ├─ 🆕 LLM 쿼리 분석: 카테고리, 의도, 엔티티, 일반화 패턴      │
│   ├─ 🆕 EMA 성공률: 최근 결과에 높은 가중치 (α=0.3)             │
│   └─ 🆕 카테고리 노드 연결: 패턴 일반화 학습                     │
└─────────────────────────────────────────────────────────────────┘
```

#### v2.0 핵심 기능 (2026-01-31)

| 기능 | 설명 | 효과 |
|------|------|------|
| **시간 감쇠** | 30일 반감기, 1년 후 10% 가중치 | 오래된 패턴 영향 감소 |
| **LLM 의미 분석** | 하드코딩 키워드 제거, LLM 기반 | 다국어 지원, 확장성 |
| **패턴 일반화** | 쿼리 카테고리 기반 학습 | 유사 쿼리에 패턴 적용 |
| **EMA 성공률** | 지수이동평균 (α=0.3) | 최근 결과 더 중요 |

**시간 감쇠 설정**:
```python
TIME_DECAY_CONFIG = {
    "half_life_days": 30,   # 30일 후 가중치 50%
    "min_weight": 0.1,      # 최소 10% 가중치
    "max_age_days": 365     # 1년 이상 → 최소 가중치
}
```

**쿼리 의미 분석 결과 예시**:
```python
{
    "category": "finance",              # 쿼리 도메인
    "intent": "search",                 # 사용자 의도
    "entities": ["삼성전자", "주가"],    # 추출된 엔티티
    "keywords": ["알려줘", "금융"],      # 핵심 키워드
    "generalization_pattern": "stock_price_query"  # 일반화 패턴
}
```

**핵심 파일**: `ontology/core/hybrid_agent_selector.py`

**사용법**:
```python
from ontology.core.hybrid_agent_selector import get_hybrid_selector

selector = get_hybrid_selector()
agent, metadata = await selector.select_agent(
    query="삼성전자 주가 알려줘",
    available_agents=["internet_agent", "analysis_agent", ...],
    agents_info={...}
)

# 실행 후 피드백 저장 (학습)
await selector.store_feedback(query, agent, success=True)
```

**장점**:
- 과거 성공 패턴 학습 → 시간이 지날수록 정확도 향상
- LLM 호출 전 그래프 기반 후보 필터링 → 비용 절감
- 설명 가능성: 왜 이 에이전트가 선택되었는지 근거 제공

### 2-1. 에이전트 동기화 서비스 (Agent Sync Service) ⭐ NEW

**Agent Marketplace ↔ Knowledge Graph** 실시간 동기화 서비스입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Sync Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Source of Truth                                                │
│   ┌─────────────────────────────────────────────────────┐       │
│   │ ACP Server (logosai/logosai/examples/agents/)       │       │
│   │ - 61+ 에이전트 Python 파일                           │       │
│   │ - 런타임에 실행되는 실제 에이전트 코드                │       │
│   └────────────────────┬────────────────────────────────┘       │
│                        │ 스캔 & 파싱                             │
│                        ▼                                        │
│   ┌─────────────────────────────────────────────────────┐       │
│   │           AgentSyncService                           │       │
│   │  - full_sync(): 전체 동기화                          │       │
│   │  - sync_single_agent(): 단일 에이전트 동기화         │       │
│   │  - check_for_changes(): 변경사항 감지                │       │
│   └────────────────────┬────────────────────────────────┘       │
│                        │                                        │
│          ┌─────────────┼─────────────┐                         │
│          ▼             ▼             ▼                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────────┐                 │
│   │ Registry │  │ Knowledge│  │  Metadata    │                 │
│   │ (8개)    │  │  Graph   │  │  JSON        │                 │
│   └──────────┘  │ (61개)   │  └──────────────┘                 │
│                 └──────────┘                                    │
│                                                                  │
│   AgentFileWatcher (백그라운드)                                  │
│   └─ 5초마다 변경사항 체크 → 자동 동기화                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 파일**: `ontology/core/agent_sync_service.py`

**사용법**:
```python
from ontology.core.agent_sync_service import (
    get_sync_service,
    initialize_agent_sync,
    start_agent_file_watching
)

# 1. 서버 시작 시 전체 동기화
result = await initialize_agent_sync()
# {"total_agents": 61, "added": 54, "updated": 7}

# 2. 파일 감시 시작 (백그라운드)
await start_agent_file_watching()
# → 새 에이전트 추가 시 자동 감지 & 동기화

# 3. 수동으로 단일 에이전트 동기화
sync_service = get_sync_service()
await sync_service.sync_single_agent("new_agent", agent_info)
```

**동기화 대상**:
| 소스 | 경로 | 설명 |
|------|------|------|
| ACP Server | `logosai/logosai/examples/agents/*.py` | 실제 에이전트 코드 (Source of Truth) |
| Metadata JSON | `agents/config/agent_metadata.json` | 에이전트 메타데이터 |
| Agent Registry | `orchestrator/agent_registry.py` | 런타임 에이전트 등록 |
| Knowledge Graph | `engines/knowledge_graph_clean.py` | 그래프 기반 추론 |

**자동 동기화 조건**:
- **시작 시**: HybridAgentSelector 초기화 시 자동 full_sync()
- **파일 변경 시**: AgentFileWatcher가 5초마다 체크
- **새 에이전트 추가 시**: 자동 감지 → sync_single_agent()

### 2-2. GNN+RL 지능형 에이전트 선택 시스템 ⭐ NEW (2026-01-31)

**GNN (Graph Neural Network) + RL (Reinforcement Learning)** 기반 지능형 에이전트 선택 시스템입니다.

Knowledge Graph의 구조를 학습하고, 강화학습으로 최적의 에이전트 선택 정책을 학습합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              GNN+RL Intelligent Agent Selection System                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Query                                                                 │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │   Query Embedder    │  sentence-transformers (384-dim)                  │
│   │   paraphrase-       │  다국어 지원                                      │
│   │   multilingual-     │                                                   │
│   │   MiniLM-L12-v2     │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │              State Composition (512-dim)                         │       │
│   │   = Query Embedding (384) + Graph Context (64) + History (64)   │       │
│   └──────────┬──────────────────────────────────────────────────────┘       │
│              │                                                               │
│      ┌───────┴───────┐                                                      │
│      ▼               ▼                                                      │
│   ┌─────────────┐ ┌─────────────────┐                                       │
│   │ GNN Encoder │ │   RL Policy     │                                       │
│   │ GraphSAGE + │ │   PPO (Actor-   │                                       │
│   │ GAT (64-dim)│ │   Critic)       │                                       │
│   └──────┬──────┘ └────────┬────────┘                                       │
│          │                 │                                                 │
│          └────────┬────────┘                                                │
│                   ▼                                                         │
│          ┌───────────────────┐                                              │
│          │  Agent Selection  │                                              │
│          │  + Confidence     │                                              │
│          └─────────┬─────────┘                                              │
│                    │                                                         │
│                    ▼                                                         │
│          ┌───────────────────┐                                              │
│          │ Experience Buffer │  Prioritized Replay                          │
│          │ + Feedback Loop   │  + Knowledge Graph Update                    │
│          └───────────────────┘                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 시스템 구성 요소

| 컴포넌트 | 파일 | 설명 |
|----------|------|------|
| **GNN Encoder** | `ml/gnn_encoder.py` | GraphSAGE + GAT 기반 Knowledge Graph 인코딩 |
| **RL Policy** | `ml/rl_policy.py` | PPO 기반 Actor-Critic 에이전트 선택 정책 |
| **Experience Buffer** | `ml/experience_buffer.py` | 우선순위 기반 경험 리플레이 버퍼 |
| **Intelligent Selector** | `ml/intelligent_selector.py` | 통합 선택기 (GNN+RL+KG) |

#### 핵심 특징

| 특징 | 설명 |
|------|------|
| **State Composition** | Query(384) + Graph(64) + History(64) = 512-dim |
| **GNN Architecture** | 3-layer GraphSAGE + GAT, 128-dim hidden, 64-dim output |
| **RL Algorithm** | PPO with GAE (λ=0.95), clip ratio 0.2 |
| **Experience Buffer** | 100K capacity, prioritized replay (α=0.6, β=0.4) |
| **Training Mode** | 온라인/오프라인 학습 모두 지원 |

#### 사용법

```python
from ontology.ml import IntelligentAgentSelector

# 1. 선택기 생성
selector = IntelligentAgentSelector(
    query_embedding_dim=384,    # Query 임베딩 차원
    graph_embedding_dim=64,     # Graph 컨텍스트 차원
    num_agents=50,              # 최대 에이전트 수
    device='cpu'                # 'cuda' for GPU
)

# 2. 에이전트 등록
agents = ['internet_agent', 'weather_agent', 'shopping_agent', ...]
selector.rl_policy.register_agents(agents)

# 3. 에이전트 선택
agent, metadata = await selector.select_agent(
    query="삼성전자 주가 알려줘",
    available_agents=["internet_agent", "analysis_agent"],
    deterministic=False  # True for greedy selection
)

print(f"선택: {agent}")
print(f"신뢰도: {metadata['confidence']:.1%}")
print(f"가치 추정: {metadata['value_estimate']:.2f}")

# 4. 피드백 저장 (학습)
await selector.store_feedback(
    success=True,           # 성공 여부
    reward=1.0,             # 보상 (optional, 자동 계산)
    execution_result={...}  # 실행 결과 (optional)
)

# 5. 온라인 학습 활성화
selector.enable_training(True)

# 6. 오프라인 학습
await selector.train_offline(
    num_iterations=1000,
    batch_size=64,
    num_epochs=4
)

# 7. 모델 저장/로드
selector.save_models()
selector.load_models()
```

#### 합성 데이터 생성 및 초기 학습

새 시스템 시작 시 합성 데이터로 초기 학습:

```python
# 합성 데이터 생성 및 학습 (초기 부트스트랩)
result = await selector.generate_synthetic_data(
    num_samples=1000,
    train_immediately=True
)

print(f"생성된 데이터: {result['data_generated']}개")
print(f"학습 결과: {result['training']}")
```

#### HybridAgentSelector와 통합

GNN+RL 시스템은 기존 HybridAgentSelector v2.0과 통합되어 작동합니다:

```python
# IntelligentAgentSelector가 피드백 저장 시 자동으로 HybridAgentSelector에도 저장
await selector.store_feedback(success=True)
# → Knowledge Graph에도 패턴 저장
# → GNN+RL 버퍼에도 경험 저장
```

#### 모델 파일 위치

```
ontology/ml/models/
├── intelligent_selector_gnn.pt      # GNN Encoder (430KB)
├── intelligent_selector_policy.pt   # RL Policy (3.1MB)
└── intelligent_selector_buffer.pkl  # Experience Buffer (878KB)
```

#### 학습 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [Phase 1: Data Collection]                                     │
│   ├─ 실제 쿼리 처리 → Experience 저장                           │
│   ├─ 합성 데이터 생성 (SyntheticDataGenerator)                  │
│   └─ Knowledge Graph 피드백 통합                                │
│                                                                  │
│   [Phase 2: Training]                                            │
│   ├─ Prioritized Replay로 배치 샘플링                           │
│   ├─ PPO 알고리즘으로 정책 업데이트                             │
│   └─ 주기적 모델 체크포인트                                     │
│                                                                  │
│   [Phase 3: Evaluation]                                          │
│   ├─ 정책 손실, 가치 손실 모니터링                              │
│   ├─ 성공률, 평균 보상 추적                                     │
│   └─ A/B 테스트 지원 (production)                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 테스트

```bash
# GNN+RL 시스템 테스트
cd /Users/maior/Development/skku/Logos
source .venv/bin/activate

python -c "
import asyncio
from ontology.ml import IntelligentAgentSelector

async def test():
    selector = IntelligentAgentSelector(auto_load=False)
    agents = ['internet_agent', 'weather_agent', 'shopping_agent']
    selector.rl_policy.register_agents(agents)

    # 합성 데이터로 학습
    await selector.generate_synthetic_data(num_samples=100, train_immediately=True)

    # 선택 테스트
    agent, meta = await selector.select_agent('오늘 날씨 어때?', agents)
    print(f'선택: {agent} (신뢰도: {meta[\"confidence\"]:.1%})')

asyncio.run(test())
"
```

#### Production 고려사항

| 항목 | 권장 설정 |
|------|----------|
| **학습 주기** | 오프라인: 매일 새벽, 온라인: 실시간 |
| **배치 크기** | 64 (GPU), 32 (CPU) |
| **버퍼 크기** | 100K (약 1GB 메모리) |
| **모델 체크포인트** | 100 iterations마다 |
| **A/B 테스트** | 10% 트래픽으로 새 모델 테스트 |

### 3. LLM 기반 처리 (LLM-Based Processing)

모든 지능적 처리는 LLM을 통해 수행합니다:

| 작업 | 처리 방식 |
|------|----------|
| 쿼리 분석 | LLM (process_unified_query) |
| 에이전트 선택 | **하이브리드** (Knowledge Graph + LLM) |
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

### 2026-01-31: GNN+RL 지능형 에이전트 선택 시스템 추가 ✅

**업그레이드 내용**: Knowledge Graph + GNN + RL 통합 지능형 에이전트 선택 시스템

**새 모듈**: `ontology/ml/`

**추가된 컴포넌트**:

1. **GNN Encoder** (`ml/gnn_encoder.py`)
   - GraphSAGE + GAT 3-layer 아키텍처
   - Input: 14-dim 노드 특징 → Hidden: 128-dim → Output: 64-dim
   - 노드 타입별 특징 추출 (agent, query_mapping, category, domain)
   - Knowledge Graph 구조 학습

2. **RL Policy** (`ml/rl_policy.py`)
   - PPO (Proximal Policy Optimization) 알고리즘
   - Actor-Critic 네트워크 (공유 특징 추출)
   - GAE (Generalized Advantage Estimation) λ=0.95
   - State dim: 512 (query + graph + history)

3. **Experience Buffer** (`ml/experience_buffer.py`)
   - 용량: 100K 경험
   - Prioritized Replay (α=0.6, β=0.4)
   - 합성 데이터 생성기 (SyntheticDataGenerator)
   - 디스크 저장/로드 지원

4. **Intelligent Selector** (`ml/intelligent_selector.py`)
   - GNN + RL + Knowledge Graph 통합
   - 온라인/오프라인 학습 지원
   - HybridAgentSelector v2.0과 통합
   - 자동 모델 저장/로드

**핵심 차원 설정**:
```python
state_dim = 512  # query(384) + graph(64) + history(64)
query_embedding_dim = 384  # sentence-transformers
graph_embedding_dim = 64   # GNN output
hidden_dim = 256           # RL policy hidden
```

**테스트 결과**:
- ✅ 합성 데이터 생성: 100개 샘플
- ✅ 학습: policy_loss=0.2824, value_loss=0.7112
- ✅ 에이전트 선택: 3개 쿼리 성공
- ✅ 피드백 저장: Knowledge Graph + Experience Buffer

**모델 파일**:
- `ml/models/intelligent_selector_gnn.pt` (430KB)
- `ml/models/intelligent_selector_policy.pt` (3.1MB)
- `ml/models/intelligent_selector_buffer.pkl` (878KB)

**커밋**: `e0ac599` - feat(ml): Add GNN+RL intelligent agent selector module

---

### 2026-01-31: Hybrid Agent Selector v2.0 업그레이드 ✅

**업그레이드 내용**: Ontology 학습 메커니즘 전체 업그레이드

**추가된 기능**:

1. **시간 감쇠 (Time Decay)**
   - 30일 반감기 지수 감쇠 적용
   - 최소 10% 가중치 유지 (1년 이상 패턴)
   - `_calculate_time_decay()` 메서드 추가

2. **LLM 기반 의미론적 매칭**
   - `_extract_intent_keywords()` 하드코딩 제거
   - `_analyze_query_semantics()` LLM 기반 분석 추가
   - 쿼리 카테고리, 의도, 엔티티, 일반화 패턴 추출

3. **패턴 일반화 학습**
   - `generalization_pattern` 저장
   - 동일 패턴 쿼리에 학습 결과 적용
   - 카테고리 노드 연결로 패턴 그룹화

4. **EMA 성공률**
   - 지수이동평균 사용 (α=0.3)
   - 최근 결과에 높은 가중치

**수정 파일**: `ontology/core/hybrid_agent_selector.py`

**성과**:
- 하드코딩 의존성 제거
- 학습 효과 향상 (시간 감쇠로 최신 패턴 우선)
- 일반화 능력 향상 (카테고리 기반 패턴 적용)

---

### 2026-01-31: ExecutionEngine 스테이지 간 데이터 전달 수정 ✅

**문제**: 멀티 스테이지 워크플로우에서 이전 스테이지의 결과가 다음 스테이지에 전달되지 않음

**증상**:
- 쿼리: "사과를 하나 샀다. 그리고 배를 하나 샀다. 그럼 내가 과일을 몇개 가지고 있지?"
- Stage 1: 과일 개수 계산 (정상 작동 - "2개")
- Stage 2 sub_query: "계산된 총 과일 개수를 사용자에게 명확하게 전달" (추상적)
- Stage 2 응답: "계산된 과일 개수는 [계산된 과일 개수]입니다." ❌ (플레이스홀더)

**원인**:
- QueryPlanner가 추상적인 sub_query를 생성
- ExecutionEngine이 `context["input_data"]`로 이전 결과를 전달했지만
- ACP 에이전트는 `sub_query`만 처리하고 `input_data`를 무시

**해결**:
`execution_engine.py`에 두 가지 메서드 추가:
1. `_extract_core_result()`: 중첩된 데이터에서 핵심 결과(answer, result, content) 재귀 추출
2. `_enrich_query_with_input()`: 이전 단계 결과를 sub_query에 통합

**수정 후 쿼리 형식**:
```
[이전 단계 결과]
사과 1개와 배 1개를 구매했을 때 총 과일 개수는 2개입니다.

[요청]
계산된 총 과일 개수를 사용자에게 명확하게 전달

위의 이전 단계 결과를 활용하여 요청에 응답해주세요.
```

**수정 후 응답**: "사과 1개와 배 1개를 구매하셨으므로, 총 과일 개수는 2개입니다." ✅

**수정 파일**: `orchestrator/execution_engine.py`

---

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

## Workflow Orchestrator

### 개요

**위치**: `ontology/orchestrator/`

LLM 기반 지능형 멀티 에이전트 워크플로우 오케스트레이션 시스템입니다.

```
쿼리 → QueryPlanner(Flash-Lite) → ExecutionPlan → Validator → ExecutionEngine → Result
                                                                      ↓
                                                              ProgressStreamer → Frontend
```

### 필수 테스트 규칙

**⚠️ 중요**: Orchestrator 모듈 수정 후 반드시 테스트 실행

```bash
# 모든 수정 후 필수 실행
./ontology/orchestrator/run_tests.sh

# 또는 직접 실행
cd /Users/maior/Development/skku/Logos
source .venv/bin/activate
python ontology/orchestrator/test_orchestrator.py
```

### 테스트 체크리스트

수정 후 다음을 확인:

- [ ] **Agent Registry**: 8개 에이전트 등록 확인
- [ ] **Plan Creation**: 계획 생성 성공 (1-2초)
- [ ] **Validation**: 검증 통과
- [ ] **Execution**: 에이전트 실행 성공
- [ ] **Streaming**: 이벤트 스트리밍 정상

### 자주 발생하는 문제

#### 스키마 호환성 에러

```
Schema incompatibility: agent_a outputs 'X' but agent_b expects 'Y'
```

**해결**: `models.py`의 `AgentSchema.is_compatible_with()` 수정

```python
# 새 변환 규칙 추가 예시
if self.output_type == "new_type" and other.input_type == "target_type":
    return True
```

#### LLM 응답 파싱 실패

```
Failed to parse LLM response as JSON
```

**해결**:
1. `query_planner.py`의 프롬프트 확인
2. 에이전트 레지스트리 메타데이터 확인
3. LLM 응답 로그 확인

### 컴포넌트 구조

| 파일 | 역할 | 수정 시 주의사항 |
|------|------|-----------------|
| `models.py` | 데이터 모델 | 스키마 호환성 규칙 확인 |
| `query_planner.py` | LLM 계획 | 프롬프트 변경 시 테스트 필수 |
| `execution_engine.py` | 실행 엔진 | 타임아웃, 재시도 로직 확인 |
| `progress_streamer.py` | 스트리밍 | 이벤트 타입 추가 시 프론트엔드 연동 확인 |

### 문서

- [README.md](./orchestrator/README.md) - 사용 가이드
- [WORKFLOW_ORCHESTRATOR_DESIGN.md](./orchestrator/WORKFLOW_ORCHESTRATOR_DESIGN.md) - 설계 문서

---

## 참조 문서

- [메인 CLAUDE.md](/Users/maior/Development/skku/Logos/CLAUDE.md)
- [logos_server CLAUDE.md](/Users/maior/Development/skku/Logos/logos_server/CLAUDE.md)
- [logosai CLAUDE.md](/Users/maior/Development/skku/Logos/logosai/CLAUDE.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [WORKFLOW_DESIGN.md](./WORKFLOW_DESIGN.md)
