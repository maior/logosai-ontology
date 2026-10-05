# GNN+RL 채택률 0% 진단 (읽기 전용 조사, 2026-08-03)

> Explore 에이전트 산출. 코드 미수정 — Phase 0 처방의 근거 문서.
> 결론 한 줄: 채택 0% 는 하나의 원인이 아니라 **직렬로 놓인 두 개의 벽** —
> ① 최근 3일간 GNN+RL 이 호출조차 안 됨(임베더 예열 게이트, bge-m3 전환 커밋
> 과 시점 일치) ② 호출되던 시절에도 confidence 는 구조적으로 1/N(≈0.013)이라
> 게이트 0.7 을 원리적으로 못 넘음 (50배 갭).

## 실측 요약

| 항목 | 값 |
|---|---|
| stats (selector_stats.json, 520KB) | total 980 · **gnn_rl_selections 0** · fallback 641 · graph_assisted 972 |
| confidence | 비영 평균 **0.0137 ≈ 1/73** (78~118 액션 softmax 에서 샘플된 확률 — 균등이면 1/78=0.0128) |
| kg_confidence | 0.6/0.8 두 값만 — 확률이 아니라 **가산 루브릭**(0.3×성공률+0.3+0.2+0.2). confidence 와 같은 자가 아님 |
| 블랙아웃 | 2026-07-31T15:33 이후 37건 전부 gnn_rl:null — `ml/config.py` mtime 15:29 (bge-m3 커밋 409167b) 직후 |
| policy.pt | ko-sroberta 차원(896) · agent_to_idx **118** · update_count **63** |
| buffer.pkl | 1048건 중 **1000건이 info.random=True 순수 잡음**(KG 패턴 폴백 발화), 실데이터 48 |
| config.json stats | 전부 0 — **보고 결함**(load 가 stats 미복원 → save 가 0으로 덮음). 실제 63 스텝 |

## 원인 순위

1. **[발화 중] 임베더 예열 게이트** — `hybrid_agent_selector.py:222-226` 이
   `_embedding_model is None` 이면 select_agent 호출 없이 fallback. 예열은
   1회성·재시도 없음·debug 로그(:177-186) — 실패하면 프로세스 수명 동안
   영구 skip 인데 get_stats 는 `gnn_rl_enabled: True` 를 계속 보고.
2. **[구조적] confidence 1/N vs 게이트 0.7** — 값은 마스킹 softmax 에서
   **샘플링된** 액션의 확률(rl_policy.py:111-112, 224-225). 0.7 을 넘으려면
   한 액션에 70% 질량 = PPO+entropy 에서 병리적 붕괴 상태. **자를 바꿔야
   한다** (deterministic max-prob / 상대 마진 / kg_confidence 와 동일 자
   정규화) — 임계만 낮추면 잡음 학습 정책의 랜덤 채택.
3. **정책이 잡음 학습** — 합성 1000건이 KG-패턴이 아니라 randn 폴백
   (`knowledge_graph=None` → 빈 그래프 → 패턴 0). 그런데 라벨은 처음부터
   있었다: **kg_checkpoint.json 의 query_agent_mapping 594건**(query_sample
   원 질의 텍스트 + selected_agent + success_rate) — Phase 1·2 의 정본 소스.
4. **피드백 파이프 95% 유실** (889→48) — `_pending_state` 전역 단일 슬롯:
   select 가 안 돌면 피드백 전량 폐기 + 동시 요청 시 보상 오귀속.
5. **[예약된 지뢰] 등록 118 > max_agents 100** — mask[idx] IndexError 재현됨
   (grounded_qa_agent=112 등 현역). 예외는 :318-319 에서 삼켜지고 그 분기만
   fallback 카운터도 안 올림 — 전량 무계측. ①을 고치면 즉시 터진다.
6. **잠복** — gnn_encoder 의 torch_geometric ImportError 시 `Data` 미정의로
   모듈 임포트 자체가 죽어 영구 OFF. ml/config 의 default_factory 상대
   임포트는 top-level 실행에서 터짐(buffer.pkl 피클 경로 `ml.…` 이 그 증거).

## Phase 0 처방 (순서 고정)

| # | 조치 |
|---|---|
| P0-1 | 관측 구멍 먼저: skip/timeout/exception/low-conf 4개 분리 지표 + 예열 실패 warning + embedding_ready 노출 |
| P0-2 | 예열 재시도 가능하게 (1회 실패 영구 skip 금지) — 호출 0% → 100% |
| P0-3 | max_agents 256 + 경계 검사 (P0-2 직후 터질 지뢰 선제거) |
| P0-4 | **게이트 재정의 (본체)**: deterministic max-prob + 상대 마진 + kg_confidence 와 동일 자 |
| P0-5 | 데이터 복구: query_agent_mapping 594건 정본 선언, 버퍼 info 에 질의 텍스트, 잡음 1000건 폐기 |
| P0-6 | save/load stats 왕복 수정 (training_steps 0 거짓 지표가 진단을 지연시켰다) |

**하지 말 것**: 임계 0.7→0.02 로 낮춰 "채택률" 만들기 — 잡음 정책의 랜덤 채택이다.

상세 코드 근거(파일:줄)는 조사 원문 참조 — 본 문서는 요약이며, 원문은 이
문서와 함께 커밋된 세션 기록(CLAUDE.md 2026-08-03 (7))에서 추적.
