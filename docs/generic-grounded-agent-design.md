# 도메인-제네릭 Grounded 에이전트 설계

> 목표: **네임스페이스마다 에이전트를 새로 만들지 않는다.** 새 사이트가 다른 업무
> 문서(인사규정·계약서·매뉴얼 등)를 올려도, **에이전트는 하나(제네릭)로 공통**이고
> **도메인별로는 "튜닝(config)"만** 한다. 보험 약관은 그 튜닝 인스턴스의 하나일 뿐.

작성: 2026-07-23. 근거는 4개 코드베이스(ontology·logos_api·acp_server·aicoach) 실측 분석.

---

## 1. 결론 먼저

**단일 제네릭 "온톨로지 grounded 검색 에이전트"는 실현 가능하다. 하드 블로커 없음.**

이유: 층이 이미 도메인-결합도 기준으로 깨끗이 나뉘어 있다.

| 층 | 도메인 결합도 | 근거 |
|----|------------|------|
| **ontology** (사실) | **완전 제네릭** — 모든 REST 가 `{namespace}` 파라미터. 확장어를 그래프에서 읽음("하드코딩 어휘 금지"는 코드에 박힌 원칙) | `core/graph_retrieval.py:30`, `server/router.py` 전 엔드포인트 |
| **logos_api** (오케스트레이션) | **제네릭** — 에이전트 선택이 LLM+KG 의미기반, 도메인 분기 없음 | `services/orchestrator_service.py`, `HybridAgentSelector` |
| **acp_server** (런타임) | **혼합** — 제네릭 에이전트 + 보험 에이전트 ~6개(단, 이들은 aicoach `/kg/*` 로 가는 **얇은 HTTP relay**). relay 골격은 제네릭, URL·설명·base_url 만 보험 결합 | `agents/product_matcher_agent.py` 등, `configs/agents.json` |
| **aicoach** (앱) | **여기에 보험 지식이 거의 전부 하드코딩** | 아래 §3 |

핵심 사실: 팀은 이미 **aicoach 의 하드코딩 보험 검색을 대체할 도메인-제네릭 검색 엔진
(ontology `/retrieve`)을 만들어 두었다.** 제네릭 에이전트는 진행 중 작업의 의도된 종착점이다.

---

## 2. 재사용 엔진 (이미 존재 — 그대로 씀)

- `GET /graphs/{namespace}/retrieve?query&top_k` — 그래프-조건부 하이브리드 검색.
  확장어(별칭·is_a 폐포)를 **그래프에서** 읽어 RRF 융합. 히트마다 `matched_via/channels`
  (설명가능). 하드코딩 동의어 테이블 불필요. (`service.retrieve` → `GraphConditionedRetriever(namespace=…)`)
- `GET /graphs/{namespace}/object-search` — 타입/신뢰 파셋 + BM25 부분일치.
- `GET /graphs/{namespace}/neighbors` — 노드 이웃(관계) 조회.
- `get_chunk_store/index(namespace)` — 원문 청크 + 벡터검색, 네임스페이스 격리.

→ **인용 근거(원문 청크) + 그래프 관계 + 벡터/키워드**를 한 네임스페이스 안에서 결합.
이게 aicoach 가 보험 전용으로 짠 것(KG match + ES hybrid + synonym table)의 **제네릭 대체물**.

---

## 3. 외부화 대상 — code → config (도메인 결합점 목록)

전부 aicoach 에 몰려 있다. "에이전트를 새로 만든다" 대신 이것들을 **config 로 뺀다**:

| # | 지금(하드코딩) | 위치 | → config |
|---|---------------|------|----------|
| (a) | source 스코프 ES 필터 / aicoach base_url | `rag/search.py:57`, acp relay | **namespace** (한 필드) |
| (b) | `DEFAULT_TOPICS`(청약철회·면책…) | `kg/build.py:37-48` | **topics[]** (도메인 seed 토픽) |
| (c) | 보험 프롬프트("보험 세일즈 어시스턴트") + 포맷터 | `agents/advisor.py:22`, `kg/extract.py:12`, `api/chat.py:84-124` | **prompt_template / answer_format** |
| (d) | 컴플라이언스 규칙(22패턴↔법령) | `coach/compliance_rules.py:21-58` | **rules[]** (도메인 규칙팩, 없으면 빈값) |
| — | KG 스키마(NODE_TYPES/PREDICATES + type→predicate 맵) | `kg/graph.py:19-20`, `build.py:140-145` | **네임스페이스별 스키마** (ontology 가 이미 `/graphs/{ns}/schema` 지원) |
| — | 도메인 렉시콘(synonym·benefit·boilerplate) | `kg/match.py`, `kg/concepts.py`, `rag/search.py:84` | **대부분 소멸** — `/retrieve` 가 확장어를 그래프에서 파생 |
| — | 인텐트 라우터 키워드(`_RECO_STRONG`…) | `api/chat.py:55-75` | **소멸** — 오케스트레이터의 의미기반 selector 로 대체 |

즉 (a)~(d) 4가지 + 스키마 하나만 config 로 두면 되고, 렉시콘/인텐트 라우터는 제네릭
엔진으로 갈아타며 **사라진다**.

---

## 4. 설계: 단일 제네릭 에이전트 + config 튜닝

### 4.1 GenericGroundedSearchAgent (acp_server)

기존 보험 relay 에이전트들의 **공통 골격**(`process()`/`_parse_query()`/`call_tool_http`)을
그대로 재사용하되, 대상 엔드포인트를 aicoach `/kg/*` 가 아니라 **ontology `/retrieve`
(+`/object-search`,`/neighbors`)** 로 돌리고, 동작을 **config 로** 받는다.

```
질의 → (config.namespace 로) ontology /retrieve → 근거 청크 + 그래프 관계
     → config.prompt_template 로 LLM 종합(인용 [N] 부착)
     → config.rules 적용(있으면; 규제 푸터 등)
     → 인용 있는 답
```

### 4.2 config_schema (지금 `agents.json` 에 전부 `null` → 이걸 채운다)

```json
{
  "agent_id": "grounded_search__insurance_kr",
  "metadata": { "class_name": "GenericGroundedSearchAgent" },
  "config_schema": {
    "namespace": "insurance_kr",
    "topics": ["청약철회 기간", "보험금 지급사유", "면책", "특약 종류"],
    "prompt_template": "너는 보험 약관 안내자다. 아래 근거만으로 인용과 함께 답하라 …",
    "answer_format": "markdown_citations",
    "rules": ["reco_guard:금소법_적합성", "reco_guard:부당권유"]
  },
  "description": "보험 약관 grounded 검색 (인용 기반)",
  "tags": ["보험","약관","인용","grounded"]
}
```

- **도메인 = config row 하나.** 코드 변경 0. `description`/`tags` 는 오케스트레이터
  selector 가 이 에이전트를 언제 고를지 판단하는 신호(의미기반, 하드코딩 아님).
- 인사규정 사이트 → `grounded_search__hr_policy` 로 `namespace:hr_policy`, topics/prompt 만
  바꾼 row 추가. 계약서 → `grounded_search__contracts`. 끝.

### 4.3 새 사이트 온보딩 흐름 (에이전트 튜닝만)

```
1. 관리 콘솔에서 네임스페이스 생성
2. 업무 문서 업로드 → (이미 자동) KG 추출 + 벡터DB 적재 + 원문 KB + 인덱싱
3. agents.json 에 grounded_search__<domain> config row 하나 추가
   (namespace / topics / prompt_template / rules)
4. 끝 — 오케스트레이터가 description·tags 로 이 에이전트를 선택, /retrieve 로 응답
```

코드는 `GenericGroundedSearchAgent` 하나. **"튜닝"=config row 편집.**

---

## 5. 마이그레이션 경로 (단계적, 비파괴)

1. **P-1**: `GenericGroundedSearchAgent` 신설 + `config_schema` 스펙 확정(위 4.2). ontology
   `/retrieve` 소비. 새 네임스페이스(예: 20p 약관)로 E2E 검증.
2. **P-2**: 보험 relay 6개 → 제네릭 에이전트의 config 인스턴스로 대체(aicoach `/kg/*`
   의존 축소). aicoach 는 "보험 튜닝 인스턴스"로 강등 — codebase 가 아니라 config row.
3. **P-3**: aicoach 인텐트 라우터(`api/chat.py` 키워드) 제거 → 오케스트레이터 의미기반
   selector 로 일원화.
4. **P-4**: 도메인 규칙팩(`rules[]`) 플러그인 인터페이스 — 규제 있는 도메인만.

각 단계는 기존 경로를 유지한 채 추가 → 비파괴.

## 6. 경계 준수

- **에이전트 = acp_server** (제네릭 에이전트 클래스 + config)
- **사실/검색 = ontology** (`/retrieve` 등, 네임스페이스별)
- **오케스트레이션·선택 = logos_api** (HybridAgentSelector, 의미기반)
- 온톨로지는 "물으면 근거와 함께 답하는 REST"만 제공. 에이전트 판단·대화는 얹지 않는다.

## 7. 미해결 / 리스크

- **스키마 유도 품질**: 도메인마다 KG 스키마가 자동 유도(auto)돼야 함. 품질 편차 존재 →
  네임스페이스별 스키마 오버라이드(`/graphs/{ns}/schema`)로 보정.
- **규제 도메인의 rules 팩**: 보험처럼 규제가 강한 도메인은 `rules[]` 설계가 필요(제네릭
  아님). 다만 대부분 도메인은 빈 규칙으로 충분.
- **세그먼테이션 도메인성**: 조문형(약관·법령)은 조 단위(구현됨). 조 구조 없는 문서
  (매뉴얼·계약서)는 heading/window 로 처리 — 세그먼트 모드도 사실상 config.
- **selector 신뢰도**: 현재 GNN+RL 채택률 이슈(별도 조사 중)와 무관하게 LLM 폴백으로 동작.
