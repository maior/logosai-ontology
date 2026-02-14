# LogosAI Ontology System

**Knowledge-driven multi-agent orchestration with LLM-powered query analysis and intelligent agent selection.**

The Ontology System is the brain of the LogosAI platform. It analyzes user queries semantically, selects the optimal agent(s), designs execution workflows, and integrates results from multiple agents.

---

## How It Works

```
User Query
    |
    v
Query Analysis (LLM) ──> Intent, entities, complexity
    |
    v
Agent Selection (Hybrid: Knowledge Graph + LLM)
    |
    v
Workflow Design ──> single | sequential | parallel | hybrid
    |
    v
Execution Engine ──> Agent calls with data piping
    |
    v
Result Integration ──> Unified response
```

## Key Features

### LLM-Based Agent Selection
- **No hardcoded keyword matching** -- all agent selection is semantic via LLM
- Agents are evaluated equally based on their metadata (description, capabilities, tags)
- When no suitable agent exists, the system provides constructive feedback instead of a wrong answer

### Hybrid Agent Selection (v2.0)
Combines Knowledge Graph pattern learning with LLM reasoning:

| Phase | Method | Purpose |
|-------|--------|---------|
| 1 | Knowledge Graph | Entity extraction, pattern matching, time-decayed success rates |
| 2 | LLM Decision | Semantic analysis using graph insights + agent metadata |
| 3 | Feedback Loop | EMA success tracking, pattern generalization |

### Workflow Orchestration
Automatically determines the optimal execution strategy:

| Strategy | When to Use | Example |
|----------|-------------|---------|
| `single_agent` | One agent can handle it | "What's the weather?" |
| `parallel` | Independent subtasks | "Search restaurants AND tourist spots" |
| `sequential` | Results feed forward | "Get price -> Convert currency" |
| `hybrid` | Mix of both | "(Weather \|\| Exchange rate) -> Calculate expenses" |

### Agent Sync Service
Automatically synchronizes agent metadata from the ACP runtime server:
- Full sync on system startup
- File watcher detects new/changed agents every 5 seconds
- Updates Knowledge Graph, Agent Registry, and metadata in real-time

## Project Structure

```
ontology/
├── core/                          # Core processing modules
│   ├── unified_query_processor.py # LLM-based unified query processing
│   ├── hybrid_agent_selector.py   # Knowledge Graph + LLM agent selection
│   ├── agent_sync_service.py      # Agent metadata synchronization
│   ├── llm_manager.py             # LLM client management
│   ├── llm_config_loader.py       # LLM configuration loading
│   ├── context_manager.py         # Query context management
│   ├── models.py                  # Data models
│   └── interfaces.py              # Abstract interfaces
│
├── engines/                       # Processing engines
│   ├── workflow_designer.py       # Dynamic workflow generation
│   ├── execution_engine.py        # Agent execution with data piping
│   ├── knowledge_graph.py         # Knowledge graph operations
│   ├── semantic_query_manager.py  # Semantic query handling
│   └── graph/                     # Graph engine and visualization
│
├── orchestrator/                  # Workflow orchestration
│   ├── query_planner.py           # LLM-powered execution planning
│   ├── execution_engine.py        # Multi-stage execution
│   ├── workflow_orchestrator.py   # Top-level orchestration
│   ├── progress_streamer.py       # Real-time progress streaming
│   ├── agent_registry.py          # Agent registration and discovery
│   ├── result_aggregator.py       # Multi-agent result aggregation
│   └── models.py                  # Orchestration data models
│
├── system/                        # System-level modules
│   ├── ontology_system.py         # Main ontology system
│   ├── knowledge_graph_manager.py # Knowledge graph lifecycle
│   ├── reasoning_generator.py     # Reasoning and inference
│   ├── result_integration.py      # Result integration logic
│   ├── strategy_manager.py        # Execution strategy selection
│   └── metrics_manager.py         # Performance metrics
│
├── services/                      # Support services
│   ├── agent_detector.py          # Agent capability detection
│   └── visualization_response_formatter.py
│
├── processors/                    # Query processors
│   └── enhanced_ontology_query_processor.py
│
├── config/                        # Configuration
│   └── llm_config.yaml            # LLM provider settings
│
├── utils/                         # Utilities
│   └── performance_analyzer.py
│
└── examples/                      # Usage examples
    ├── basic_usage.py
    └── advanced_usage.py
```

## Quick Start

### Prerequisites
- Python 3.11+
- Google API key (for Gemini LLM) or OpenAI API key

### Installation

```bash
pip install -r requirements.txt

# Set API keys
export GOOGLE_API_KEY="your-google-api-key"
```

### Basic Usage

```python
import asyncio
from ontology.core.unified_query_processor import UnifiedQueryProcessor

async def main():
    processor = UnifiedQueryProcessor()

    available_agents = ['weather_agent', 'calculator_agent', 'search_agent']

    result = await processor.process_unified_query(
        query="What's the weather in Seoul?",
        available_agents=available_agents
    )

    print(f"Selected agent: {result['agent_mappings']}")
    print(f"Strategy: {result['execution_plan']['strategy']}")

asyncio.run(main())
```

### Hybrid Agent Selection

```python
from ontology.core.hybrid_agent_selector import get_hybrid_selector

selector = get_hybrid_selector()

# Select agent with Knowledge Graph + LLM
agent, metadata = await selector.select_agent(
    query="Show me Samsung stock price",
    available_agents=["search_agent", "finance_agent", "analysis_agent"],
    agents_info={...}
)

print(f"Selected: {agent} (confidence: {metadata['confidence']:.0%})")

# Store feedback for learning
await selector.store_feedback(query, agent, success=True)
```

### Workflow Orchestration

```python
from ontology.orchestrator import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator()

# Process a complex multi-step query
async for event in orchestrator.process(query="Convert 100 USD to KRW and EUR, then compare"):
    if event["type"] == "progress":
        print(f"Stage: {event['stage']}")
    elif event["type"] == "final_result":
        print(f"Result: {event['result']}")
```

## Configuration

### LLM Settings (`config/llm_config.yaml`)

```yaml
default_provider: gemini
providers:
  gemini:
    model: gemini-2.5-flash-lite
    temperature: 0.3
    max_tokens: 4096
  openai:
    model: gpt-4
    temperature: 0.3
```

### Agent Selection Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| Time decay half-life | 30 days | How quickly old patterns lose influence |
| Min weight | 0.1 | Minimum weight for old patterns |
| EMA alpha | 0.3 | Exponential moving average smoothing factor |

## Architecture

### Core Design Principles

1. **No Hardcoded Matching** -- All agent selection uses LLM semantic analysis
2. **Hybrid Intelligence** -- Knowledge Graph patterns + LLM reasoning
3. **Dynamic Workflows** -- Execution strategies determined at runtime
4. **Continuous Learning** -- Feedback loop improves selection over time
5. **Equal Agent Evaluation** -- No agent is a "fallback"; all are scored equally

### Integration with LogosAI

The Ontology System integrates with:
- **ACP Server** (port 8888) -- Executes selected agents
- **Django Backend** (port 8080) -- Receives user queries
- **FORGE AI** (port 8030) -- Generates new agents when needed
- **Knowledge Graph** -- Stores and retrieves learned patterns

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| LLM | Google Gemini, OpenAI GPT |
| Knowledge Graph | NetworkX, custom graph engine |
| Query Processing | Async/await, aiohttp |
| Configuration | YAML, Pydantic |

## License

MIT License
