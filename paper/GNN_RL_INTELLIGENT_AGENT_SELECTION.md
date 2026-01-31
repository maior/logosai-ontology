# GNN+RL Intelligent Agent Selection System

**Graph Neural Network + Reinforcement Learning 기반 지능형 에이전트 선택 시스템**

*LogosAI Multi-Agent System을 위한 학습 기반 에이전트 선택 메커니즘*

---

## Abstract

본 연구에서는 멀티 에이전트 AI 시스템에서 사용자 쿼리에 최적의 에이전트를 선택하기 위한 **GNN(Graph Neural Network) + RL(Reinforcement Learning)** 기반 지능형 에이전트 선택 시스템을 제안한다. 기존의 키워드 매칭 또는 규칙 기반 에이전트 선택 방식의 한계를 극복하고, Knowledge Graph의 구조적 정보를 GNN으로 학습하며, PPO(Proximal Policy Optimization) 알고리즘을 통해 최적의 에이전트 선택 정책을 학습한다.

**Keywords**: Multi-Agent System, Graph Neural Network, Reinforcement Learning, Agent Selection, Knowledge Graph, PPO

---

## 1. Introduction

### 1.1 Research Background

멀티 에이전트 AI 시스템에서 사용자 쿼리에 적합한 에이전트를 선택하는 것은 시스템 성능의 핵심 요소이다. 기존 접근 방식들의 한계:

| 방식 | 문제점 |
|------|--------|
| **키워드 매칭** | 새 에이전트 추가 시 코드 수정 필요, 다국어 지원 어려움 |
| **규칙 기반** | 복잡한 쿼리 처리 불가, 유지보수 비용 증가 |
| **단순 LLM 기반** | 높은 API 비용, 일관성 부족, 학습 불가 |

### 1.2 Research Objectives

1. Knowledge Graph의 구조적 정보를 활용한 에이전트 표현 학습
2. 강화학습 기반 최적 에이전트 선택 정책 학습
3. 온라인/오프라인 학습을 통한 지속적 성능 향상
4. 하드코딩 없는 확장 가능한 에이전트 선택 시스템 구현

### 1.3 Contributions

1. **GNN 기반 Knowledge Graph Encoding**: GraphSAGE + GAT 아키텍처로 에이전트 관계 학습
2. **PPO 기반 에이전트 선택 정책**: Actor-Critic 네트워크를 통한 최적 정책 학습
3. **Prioritized Experience Replay**: 중요 경험 우선 학습으로 샘플 효율성 향상
4. **Hybrid Architecture**: Knowledge Graph + GNN + RL + LLM 통합 시스템

---

## 2. System Architecture

### 2.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 GNN+RL Intelligent Agent Selection System                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Query                                                                 │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                    Query Encoder                                 │       │
│   │   Model: sentence-transformers/paraphrase-multilingual-MiniLM   │       │
│   │   Output: 384-dimensional query embedding                        │       │
│   └──────────────────────────┬──────────────────────────────────────┘       │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                State Composition Module                          │       │
│   │                                                                  │       │
│   │   State Vector (512-dim) = [Query Emb | Graph Context | History]│       │
│   │                             384-dim    64-dim         64-dim    │       │
│   └──────────────────────────┬──────────────────────────────────────┘       │
│                              │                                               │
│              ┌───────────────┼───────────────┐                              │
│              ▼               │               ▼                              │
│   ┌──────────────────┐       │    ┌──────────────────┐                      │
│   │   GNN Encoder    │       │    │    RL Policy     │                      │
│   │                  │       │    │                  │                      │
│   │  ┌────────────┐  │       │    │  ┌────────────┐  │                      │
│   │  │ GraphSAGE  │  │       │    │  │   Actor    │  │                      │
│   │  │  Layer 1   │  │       │    │  │  Network   │  │                      │
│   │  └─────┬──────┘  │       │    │  └─────┬──────┘  │                      │
│   │        ▼         │       │    │        ▼         │                      │
│   │  ┌────────────┐  │       │    │  Action Probs    │                      │
│   │  │ GraphSAGE  │  │       │    │                  │                      │
│   │  │  Layer 2   │  │       │    │  ┌────────────┐  │                      │
│   │  └─────┬──────┘  │       │    │  │   Critic   │  │                      │
│   │        ▼         │       │    │  │  Network   │  │                      │
│   │  ┌────────────┐  │       │    │  └─────┬──────┘  │                      │
│   │  │    GAT     │  │       │    │        ▼         │                      │
│   │  │  Layer 3   │  │       │    │  Value Estimate  │                      │
│   │  └─────┬──────┘  │       │    │                  │                      │
│   │        ▼         │       │    └────────┬─────────┘                      │
│   │  Node Embeddings │       │             │                                │
│   │    (64-dim)      │       │             │                                │
│   └────────┬─────────┘       │             │                                │
│            │                 │             │                                │
│            └─────────────────┼─────────────┘                                │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                   Agent Selection                                │       │
│   │   - Selected Agent ID                                            │       │
│   │   - Confidence Score                                             │       │
│   │   - Value Estimate                                               │       │
│   └──────────────────────────┬──────────────────────────────────────┘       │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │              Experience Buffer & Feedback Loop                   │       │
│   │   - Prioritized Experience Replay (α=0.6, β=0.4)                │       │
│   │   - Knowledge Graph Update                                       │       │
│   │   - Online/Offline Training                                      │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 State Representation

시스템의 상태(State)는 세 가지 컴포넌트로 구성된다:

$$\mathbf{s} = [\mathbf{e}_q \| \mathbf{c}_g \| \mathbf{h}] \in \mathbb{R}^{512}$$

| Component | Dimension | Description |
|-----------|-----------|-------------|
| $\mathbf{e}_q$ | 384 | Query embedding (Sentence-BERT) |
| $\mathbf{c}_g$ | 64 | Graph context (GNN output) |
| $\mathbf{h}$ | 64 | History embedding (previous selections) |

**Query Embedding**:
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- 다국어 지원 (한국어, 영어, 일본어 등 50+ 언어)
- Semantic similarity 기반 쿼리 표현

---

## 3. Graph Neural Network Encoder

### 3.1 Knowledge Graph Structure

Knowledge Graph는 에이전트, 쿼리 패턴, 도메인 간의 관계를 표현한다:

```
G = (V, E)

V = {agent nodes} ∪ {query_pattern nodes} ∪ {category nodes} ∪ {capability nodes}

E = {(agent, query_pattern): successful_mapping}
    ∪ {(agent, capability): has_capability}
    ∪ {(query_pattern, category): belongs_to}
```

### 3.2 Node Feature Extraction

각 노드 $v \in V$의 특징 벡터 $\mathbf{x}_v \in \mathbb{R}^{14}$:

| Index | Feature | Description |
|-------|---------|-------------|
| 0-5 | Node type | One-hot encoding (agent, mapping, category, domain, concept, unknown) |
| 6 | Success rate | Historical success rate [0, 1] |
| 7 | Usage count | Normalized usage frequency |
| 8 | Recency | Time decay factor (365-day basis) |
| 9 | Capability count | Number of capabilities / 10 |
| 10 | Keyword count | Number of keywords / 10 |
| 11 | Confidence | Selection confidence [0, 1] |
| 12 | Is active | Binary activity flag |
| 13 | Importance | Degree centrality |

### 3.3 GNN Architecture

3-layer GNN with GraphSAGE + GAT:

**Layer 1-2: GraphSAGE**
$$\mathbf{h}_v^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \cdot \text{CONCAT}\left(\mathbf{h}_v^{(l)}, \text{AGG}\left(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)\right)$$

**Layer 3: Graph Attention Network (GAT)**
$$\mathbf{h}_v^{(3)} = \|_{k=1}^{K} \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}^k \mathbf{W}^k \mathbf{h}_u^{(2)}\right)$$

where attention coefficients:
$$\alpha_{vu} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T[\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_u]\right)\right)}{\sum_{j \in \mathcal{N}(v)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T[\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_j]\right)\right)}$$

**Architecture Parameters**:

| Parameter | Value |
|-----------|-------|
| Input dimension | 14 |
| Hidden dimension | 128 |
| Output dimension | 64 |
| Number of layers | 3 |
| GAT heads | 4 |
| Dropout | 0.1 |
| Activation | ReLU (intermediate), None (final) |

### 3.4 Graph Context Computation

쿼리와 관련된 그래프 컨텍스트 계산:

$$\mathbf{c}_g = \sum_{v \in \text{Top-}k} \text{softmax}(\text{sim}(\mathbf{e}_q, \mathbf{h}_v)) \cdot \mathbf{h}_v$$

where $\text{sim}(\cdot, \cdot)$ is cosine similarity.

---

## 4. Reinforcement Learning Policy

### 4.1 Problem Formulation

에이전트 선택을 **Contextual Bandit** 문제로 정의:

- **State** $s$: Query + Graph context + History
- **Action** $a$: Agent selection (discrete)
- **Reward** $r$: Execution success/failure

### 4.2 Actor-Critic Network

**Shared Feature Extractor**:
$$\mathbf{f} = \text{ReLU}(\text{LayerNorm}(\mathbf{W}_2 \cdot \text{ReLU}(\text{LayerNorm}(\mathbf{W}_1 \cdot \mathbf{s}))))$$

**Actor (Policy Network)**:
$$\pi(a|s) = \text{softmax}(\mathbf{W}_\pi \cdot \mathbf{f})$$

**Critic (Value Network)**:
$$V(s) = \mathbf{W}_V \cdot \mathbf{f}$$

**Network Architecture**:

```
State (512) → Linear(512, 256) → LayerNorm → ReLU → Dropout(0.1)
           → Linear(256, 256) → LayerNorm → ReLU → Dropout(0.1)

           ├→ Actor: Linear(256, num_agents) → Softmax → π(a|s)
           └→ Critic: Linear(256, 1) → V(s)
```

### 4.3 PPO Algorithm

**Objective Function**:
$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- $\epsilon = 0.2$ (clipping ratio)

**Generalized Advantage Estimation (GAE)**:
$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**Loss Functions**:

| Loss | Formula | Weight |
|------|---------|--------|
| Policy Loss | $-L^{CLIP}(\theta)$ | 1.0 |
| Value Loss | $\text{MSE}(V(s), R)$ | 0.5 |
| Entropy Bonus | $-H(\pi)$ | 0.01 |

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Discount factor (γ) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Clip ratio (ε) | 0.2 |
| Value loss coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| Max gradient norm | 0.5 |
| Batch size | 64 |
| Epochs per update | 4 |

---

## 5. Experience Buffer

### 5.1 Prioritized Experience Replay

경험의 중요도에 따라 샘플링 확률을 조정:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

where $p_i = |\delta_i| + \epsilon$ (TD-error based priority)

**Importance Sampling Weights**:
$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

### 5.2 Buffer Configuration

| Parameter | Value |
|-----------|-------|
| Capacity | 100,000 experiences |
| Priority exponent (α) | 0.6 |
| IS exponent (β) | 0.4 → 1.0 (annealing) |
| State dimension | 512 |
| Minimum priority | 1e-6 |

### 5.3 Experience Structure

```python
Experience = {
    'state': np.array[512],      # State vector
    'action': int,               # Selected agent index
    'reward': float,             # Execution reward
    'next_state': np.array[512], # Next state
    'done': bool,                # Episode termination
    'log_prob': float,           # Action log probability
    'value': float,              # Value estimate
    'priority': float            # Sampling priority
}
```

---

## 6. Training Pipeline

### 6.1 Online Learning

실시간으로 사용자 쿼리와 피드백을 학습:

```
1. Receive query → Compute state
2. Select agent (ε-greedy or sampling)
3. Execute agent → Get reward
4. Store experience → Update priorities
5. If buffer_size > batch_size:
   - Sample batch (prioritized)
   - Compute advantages (GAE)
   - Update policy (PPO)
   - Update value network
```

### 6.2 Offline Learning

축적된 경험으로 배치 학습:

```python
async def train_offline(num_iterations=1000, batch_size=64, num_epochs=4):
    for iteration in range(num_iterations):
        # Sample batch with prioritization
        batch = experience_buffer.sample(batch_size)

        # Compute returns and advantages
        returns = compute_returns(batch['rewards'], gamma=0.99)
        advantages = compute_gae(batch, lambda_=0.95)

        # PPO update
        for epoch in range(num_epochs):
            policy_loss, value_loss = ppo_update(batch, advantages, returns)

        # Checkpoint
        if iteration % 100 == 0:
            save_models()
```

### 6.3 Synthetic Data Generation

초기 학습을 위한 합성 데이터 생성:

```python
class SyntheticDataGenerator:
    def generate(self, num_samples):
        for _ in range(num_samples):
            # Random query embedding
            query_emb = np.random.randn(384)
            query_emb = query_emb / np.linalg.norm(query_emb)

            # Random graph context
            graph_ctx = np.random.randn(64) * 0.1

            # Random history
            history = np.random.randn(64) * 0.1

            # Compose state
            state = np.concatenate([query_emb, graph_ctx, history])

            # Random agent and reward
            action = np.random.randint(0, num_agents)
            reward = np.random.choice([0.0, 1.0], p=[0.3, 0.7])

            yield state, action, reward
```

---

## 7. Integration with Hybrid Agent Selector

### 7.1 Hybrid Architecture

GNN+RL 시스템은 기존 Hybrid Agent Selector v2.0과 통합:

```
                    ┌─────────────────────────────────┐
                    │      User Query                 │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │   Knowledge Graph Analysis      │
                    │   - Entity extraction           │
                    │   - Related concepts            │
                    │   - Historical patterns         │
                    └─────────────┬───────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   GNN+RL        │ │   LLM-based     │ │   Pattern       │
    │   Selection     │ │   Selection     │ │   Matching      │
    │   (Learned)     │ │   (Semantic)    │ │   (Historical)  │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ▼
                    ┌─────────────────────────────────┐
                    │   Ensemble Decision             │
                    │   - Weighted voting             │
                    │   - Confidence aggregation      │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │   Final Agent Selection         │
                    └─────────────────────────────────┘
```

### 7.2 Feedback Loop

선택 결과가 양쪽 시스템에 피드백:

```python
async def store_feedback(success: bool, reward: float = None):
    # 1. GNN+RL Experience Buffer
    experience_buffer.add(current_state, action, reward, next_state, done)

    # 2. Knowledge Graph Update
    hybrid_selector.store_feedback(query, selected_agent, success)

    # 3. Update priorities
    experience_buffer.update_priorities()
```

---

## 8. Experimental Results

### 8.1 Test Configuration

| Item | Value |
|------|-------|
| Test queries | 5 |
| Training samples | 100 (synthetic) |
| Registered agents | 5 (internet, weather, shopping, code, analysis) |
| Hardware | CPU (Apple M-series) |

### 8.2 Performance Metrics

**Training Results**:
```
Policy Loss: 0.2842
Value Loss:  0.4476
Training Time: ~50ms (100 samples)
```

**Selection Performance**:

| Query | Expected | Selected | Confidence | Latency |
|-------|----------|----------|------------|---------|
| 오늘 서울 날씨 어때? | weather_agent | analysis_agent | 37.5% | 220ms |
| 삼성전자 주가 알려줘 | internet_agent | analysis_agent | 31.4% | 45ms |
| 아이폰 16 가격 검색해줘 | shopping_agent | analysis_agent | 43.1% | 12ms |
| 파이썬으로 퀵소트 구현해줘 | code_agent | weather_agent | 26.5% | 45ms |
| 이 데이터를 분석해줘 | analysis_agent | analysis_agent | 37.2% | 11ms |

**Note**: 낮은 정확도는 합성 데이터 100개만으로 학습한 결과. 실제 운영 환경에서 더 많은 데이터로 학습 시 정확도 향상 예상.

### 8.3 System Performance

| Metric | Value |
|--------|-------|
| Initialization time | ~5 sec (model loading) |
| Selection latency | 11-220ms |
| Memory usage | ~1GB (with buffer) |
| Model file sizes | GNN: 430KB, Policy: 3.1MB, Buffer: 878KB |

---

## 9. Production Considerations

### 9.1 Deployment Configuration

| Parameter | Recommendation |
|-----------|----------------|
| Training frequency | Offline: Daily (night), Online: Real-time |
| Batch size | 64 (GPU), 32 (CPU) |
| Buffer size | 100K (1GB memory) |
| Checkpoint interval | Every 100 iterations |
| A/B testing ratio | 10% traffic for new model |

### 9.2 Monitoring Metrics

1. **Selection accuracy**: Correct agent selection rate
2. **Execution success rate**: Agent task completion rate
3. **Latency distribution**: P50, P95, P99 selection time
4. **Learning progress**: Policy loss, value loss trends
5. **Buffer health**: Size, priority distribution

### 9.3 Scaling Considerations

| Scenario | Solution |
|----------|----------|
| Large agent pool (100+) | Hierarchical selection, domain clustering |
| High throughput (1K+ QPS) | Model parallelism, batch inference |
| Multi-region deployment | Federated learning, local models |

---

## 10. Conclusion

본 연구에서 제안한 GNN+RL 기반 에이전트 선택 시스템의 주요 기여:

1. **Knowledge Graph 구조 학습**: GNN을 통해 에이전트 간 관계와 쿼리 패턴을 학습
2. **강화학습 기반 최적화**: PPO 알고리즘으로 선택 정책을 지속적으로 개선
3. **확장 가능한 아키텍처**: 새 에이전트 추가 시 재학습만으로 적응
4. **하이브리드 통합**: 기존 LLM 기반 시스템과 상호 보완적 동작

### Future Work

1. **멀티태스크 학습**: 여러 도메인에서 동시 학습
2. **메타 학습**: Few-shot adaptation for new agents
3. **분산 학습**: 대규모 배포를 위한 federated learning
4. **Explainability**: 선택 이유에 대한 설명 생성

---

## References

1. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. NeurIPS.
2. Veličković, P., et al. (2018). Graph attention networks. ICLR.
3. Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv.
4. Schaul, T., et al. (2015). Prioritized experience replay. ICLR.
5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. EMNLP.

---

## Appendix A: Implementation Details

### A.1 File Structure

```
ontology/ml/
├── __init__.py                    # Package exports
├── gnn_encoder.py                 # GNN Encoder (GraphSAGE + GAT)
├── rl_policy.py                   # RL Policy (PPO Actor-Critic)
├── experience_buffer.py           # Prioritized Experience Replay
├── intelligent_selector.py        # Integrated selector
└── models/
    ├── intelligent_selector_gnn.pt      # GNN weights (430KB)
    ├── intelligent_selector_policy.pt   # Policy weights (3.1MB)
    └── intelligent_selector_buffer.pkl  # Experience buffer (878KB)
```

### A.2 Dependencies

```
torch>=2.0.0
torch-geometric>=2.3.0
sentence-transformers>=2.2.0
networkx>=3.0
numpy>=1.24.0
loguru>=0.7.0
```

### A.3 Usage Example

```python
from ontology.ml import IntelligentAgentSelector

# Initialize
selector = IntelligentAgentSelector(
    query_embedding_dim=384,
    graph_embedding_dim=64,
    num_agents=50,
    device='cpu'
)

# Register agents
agents = ['internet_agent', 'weather_agent', 'shopping_agent']
selector.rl_policy.register_agents(agents)

# Bootstrap with synthetic data
await selector.generate_synthetic_data(num_samples=1000, train_immediately=True)

# Select agent
agent, metadata = await selector.select_agent(
    query="오늘 날씨 어때?",
    available_agents=agents,
    deterministic=False
)

print(f"Selected: {agent} (confidence: {metadata['confidence']:.1%})")

# Store feedback for learning
await selector.store_feedback(success=True, reward=1.0)

# Save models
selector.save_models()
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-31*
*Author: LogosAI Research Team*
