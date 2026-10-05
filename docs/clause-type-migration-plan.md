# 로드맵 4: 구조 단위 타입(Clause) 도입 — 구현 계획

> 작성: 2026-08-03 (Plan 에이전트 설계 + 코드 실측). 배경: 조항·문서가 개념과
> 같은 타입으로 잡혀 관계 추출이 구조적으로 헷갈린다 — 관계 백필 정답률 67%,
> 타입 시그니처 위반 83% 의 근본 원인. 자동 판별은 측정으로 기각됨(오탐
> `계약자` · 놓침 `제24조(계약의 소멸)` 표기 차이) — 반드시 제안→검수.

## 0. 설계를 구속하는 실측 사실

| 사실 | 근거 위치 |
|---|---|
| 재분류 = id 개명 (`{type}:{name}` 이 id) | `builder/pipeline.py:643`, `server/service.py:3371` |
| active 노드는 삭제·개명 차단, 미리보기에서도 알림 | `core/lifecycle.py:59`, `server/service.py:3050-3063` |
| 참조 재지정 부품이 이미 존재 | `core/chunk_store.py:165` `relabel_node`, `core/search_qa.py:180` `relabel_node` |
| 적용 순서 계약(엣지→참조→삭제) | `server/service.py:3019-3024` merge_nodes docstring |
| `rename_type` 은 id 를 안 바꾼다 — 노드 단위 재분류엔 별도 경로 필요 | `server/service.py:2973-3008` |
| 시그니처는 관문이 아니라 신호 (83% 위반, 다수 정상) | `core/relation_backfill.py:148-176` |
| section 라벨은 segmenter 가 도메인-일반으로 만든다 | `builder/segmenter.py:52-77` → `chunk_store.py:51` |
| 표기 정규화 부품 | `core/graph_health.py:66` `normalize_name`, `core/dup_review.py:32-36` 가짜 접두사 트릭 |
| 묘비는 재빌드 부활 관문 | `builder/pipeline.py:351,371,646` `_is_tombstoned` |

원칙: **자동 판별 금지**(제안→dry_run→승인) · **타입 이름 하드코딩 금지**
(타깃 타입명은 검수자가 요청에 담는다 — PROJ-A 엔 조항이 없다) · **미리보기 = 적용**.

## P-1. 후보 탐지 — `core/structural_units.py` (신규, LLM 0콜, 결정적)

- **신호 A — section 라벨 전체-정규화 동등성**: 부분문자열이 아니라 전체 라벨의
  normalize 동등만 후보. 놓침 해결(`제24조 (계약의 소멸)`≡`제24조(계약의 소멸)`
  ≡`제24조【계약의 소멸】`) + 오탐 해결(`계약자`는 어떤 라벨 전체와도 동등하지
  않다 — 포함 매칭을 버리는 것이 오탐 차단의 본체). 라벨 사전 = 청크 스토어
  section 전량(조문 regex 하드코딩 금지 — segmenter 가 이미 도메인-일반).
- **신호 B — 자기 근거 정합**: 일치 청크가 자신의 근거인가 → `evidence_matched`
  신호 동봉 (거르지 않는다 — dup_review 의 "분류는 신호이지 판정이 아니다").
- **신호 C — 문서 단위**: 노드 이름 ≡ `chunk.source` 제목 → `kind: "document"`.
- 동봉 재료: type·definition·evidence_chunks·sources·lifecycle(active 면
  caution)·out-술어 분포·matched_sections·chunk_ids. 묘비 제외.
- API: `GET /graphs/{ns}/review/structural` + `review_queues` 에 `structural` 카드.
- 테스트: 오탐/놓침 실측 사례를 회귀 계약으로 고정 (9종).

## P-2. 개명 경로 — `core/node_rename.py` + `service.rename_node`

merge_nodes 부품 재사용 지도: 엣지 재지정 루프(`service.py:3070-3086`)는
`_repoint_edges` 헬퍼로 추출해 merge 와 공유(두 벌이면 PG 순서 계약이 갈라진다).
청크·골든셋 재지정은 기존 `relabel_node` 그대로. 적용 순서 = merge 계약
(① 새 노드 upsert ② 엣지 재지정 ③ 청크·골든셋 ④ 옛 노드 삭제).

- `plan_rename(graph, node_id, new_type, chunks, cases)` — 순수, 미리보기=적용.
  `target_exists` 면 "병합이다 — /nodes/merge 를 쓰라" 힌트.
- `POST /graphs/{ns}/nodes/rename` — dry_run 기본 True.
- **묘비를 남기지 않는다** — 재분류는 "타입이 틀렸다"지 "개체가 틀렸다"가 아니다.
  묘비면 재빌드에서 근거가 통째로 버려진다. 부활은 P-4 재지도가 푼다.
- `reindex_required: True` 고정 (compose_node_text 가 id·타입 포함).
- 테스트 11종 (plan 순수성 · 관문 변이 · dry_run==적용 · PG 순서 등).

## P-3. 검수 흐름 — `POST /review/structural/approve`

- **본문 불신**: 승인 시 탐지를 재실행해 여전히 후보인 것만 (같은 함수 재호출 —
  규칙 두 벌 금지). 아닌 것은 skipped.
- `new_type` 요청당 하나·필수 — 서버 기본값 없음(하드코딩 금지의 실행 형태).
- 미선언 타입은 `schema_decl.set_type` 에 선언 + 감사.
- 항목별 실패(active·충돌)는 전체를 막지 않되 dry_run 미리보기에 미리 나타난다.
- 보드: ReviewQueueBoard 에 카드 + 패널(판단 재료 + 타깃 타입 입력 + 미리보기→승인).

## P-4. 재빌드 부활 통제 — 묘비의 자매: 재지도(reclassify registry)

차단이 아니라 **재지도**: review_store 에 `reclassify` 이벤트 → `_reclassified:
{old_id: new_id}` (replay 복원, 체인 전이 + 사이클 가드). pipeline 의 묘비 검사
3곳과 같은 자리에서 id 를 재지도 — 재추출이 중복 대신 새 노드의 **보강**(근거
합집합)이 된다. P-4 이전의 퇴화는 안전: 부활은 cross_type 클러스터로 보드에
잡힌다(보이는 결함).

## P-5. 적용 + 측정

ins_cancer_demo 에 실제 승인(조항 노드들 + `사업방법서`) → 관계 백필 재실행:
① 시그니처 위반율 83% ↓ (같은 92 청크로 재제안) ② 관계 정답률 67% ↑ (전건
인용 대조) ③ 골든셋 node/evidence 회귀 없음 (eval_history 지문 전후) ④ `계약의
소멸` vs `제24조(계약의 소멸)` cross_type 이 정당한 구분으로 검수 종결.

## 파급 요약

| 대상 | 처리 |
|---|---|
| 골든셋 | `GoldenSet.relabel_node` + 전후 evaluate 스냅샷 |
| 근거 링크 | `ChunkStore.relabel_node` (chunk_id 불변 → 벡터 캐시 유효) |
| 색인 | `index.remove(old)` + reindex_required |
| 묘비 | 만들지 않음 (P-4 재지도) |
| 감사 | `action="reclassify"` before/after |
| PG | merge 와 같은 순서 (upsert→edge→delete) |
| 생애주기 | active 는 미리보기부터 차단 |
