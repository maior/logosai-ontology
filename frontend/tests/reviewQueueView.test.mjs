// reviewQueueView — 검수 큐 보드 뷰모델 (순수, node --test 파일 명시 실행)
import test from "node:test";
import assert from "node:assert/strict";
import {
  buildQueueCards, relabelRows, relationRows, splitClusters, acceptCandidates,
  triageBands, batchPreviewSummary, agreementBadge,
} from "../app/ontology-admin/reviewQueueView.mjs";

const QUEUES = {
  namespace: "PROJ-A",
  queues: {
    node_review: 1029,
    duplicates: { variant: 0, cross_type: 30, similar: 15 },
    golden: { verified: 65, draft: 16 },
    lifecycle: { experimental: 1596 },
    lifecycle_overdue: 2,
    structural: 4,
  },
};

test("buildQueueCards: 큐별 카드 + count>0 은 warn", () => {
  const cards = buildQueueCards(QUEUES);
  const byKey = Object.fromEntries(cards.map((c) => [c.key, c]));
  assert.equal(byKey.node_review.count, 1029);
  assert.equal(byKey.dup_variant.count, 0);
  assert.equal(byKey.dup_variant.tone, "ok");
  assert.equal(byKey.dup_cross.count, 30);
  assert.equal(byKey.dup_cross.tone, "warn");
  assert.equal(byKey.golden_draft.count, 16);
  assert.equal(byKey.overdue.count, 2);
  assert.equal(byKey.structural.count, 4);      // P-1: 구조 단위 재분류 후보
  assert.equal(byKey.structural.tone, "warn");
  assert.equal(byKey.relabel.count, null);      // 계산 전 — 지어내지 않는다
});

test("buildQueueCards: 쓰레기 입력은 0 카드 (절대 던지지 않음)", () => {
  for (const bad of [null, {}, { queues: "x" }]) {
    const cards = buildQueueCards(bad);
    assert.ok(cards.length >= 6);
    assert.ok(cards.every((c) => c.count === 0 || c.count === null));
  }
});

test("relabelRows: retrieve 1위가 아닌 케이스만, 순위 오름차순", () => {
  const rows = relabelRows({ per_case: [
    { case_id: "a", query: "q1", expected: "T:x", retrieve_rank: 1 },   // 정상 — 제외
    { case_id: "b", query: "q2", expected: ["T:y"], retrieve_rank: null }, // 미검출
    { case_id: "c", query: "q3", expected: "T:z", retrieve_rank: 3 },
  ]});
  assert.deepEqual(rows.map((r) => r.caseId), ["c", "b"]);   // rank 3 이 미검출보다 먼저
  assert.deepEqual(rows[0].expected, ["T:z"]);
  assert.equal(rows[1].rank, null);
});

test("relabelRows: 쓰레기 입력은 빈 배열", () => {
  for (const bad of [null, {}, { per_case: "x" }]) {
    assert.deepEqual(relabelRows(bad), []);
  }
});

test("buildQueueCards: relations 카드 — count null (LLM 비용, 온디맨드), relabel 바로 앞", () => {
  const cards = buildQueueCards(QUEUES);
  const byKey = Object.fromEntries(cards.map((c) => [c.key, c]));
  assert.equal(byKey.relations.count, null);   // 지어내지 않는다
  const keys = cards.map((c) => c.key);
  assert.equal(keys.indexOf("relations"), keys.indexOf("relabel") - 1);
});

test("relationRows: 정렬 — 과거기각 먼저, 다음 신규패턴, 다음 나머지 (밴드 내 안정)", () => {
  const rows = relationRows({ proposals: [
    { subject: "A", predicate: "p", object: "B", signature_seen: true },
    { subject: "C", predicate: "p", object: "D", signature_seen: false },
    { subject: "E", predicate: "p", object: "F", signature_seen: true,
      previously_rejected: { reason: "전이를 보장으로 왜곡", at: "2026-08-01",
                             chunk_id: "c1", scope: "triple" } },
    { subject: "G", predicate: "p", object: "H", signature_seen: false },
  ], rejected_filtered: 8 });
  assert.deepEqual(rows.map((r) => r.subject), ["E", "C", "G", "A"]);
  assert.deepEqual(rows[0].prevRejected,
    { reason: "전이를 보장으로 왜곡", at: "2026-08-01" });
  assert.equal(rows[1].prevRejected, null);
});

test("relationRows: quote 240자 절단 + quoteFull 무절단 (승인 재대조용)", () => {
  const long = "가".repeat(300);
  const rows = relationRows({ proposals: [
    { subject: "A", predicate: "definesTerm", object: "B",
      evidence_quote: long, chunk_id: "c9", section: "제3조",
      signature_seen: true, signature: "T→T", names_in_quote: ["A", "B"] },
  ]});
  assert.equal(rows[0].quote.length, 240);
  assert.equal(rows[0].quoteFull.length, 300);
  assert.equal(rows[0].chunkId, "c9");
  assert.equal(rows[0].section, "제3조");
  assert.deepEqual(rows[0].namesInQuote, ["A", "B"]);
});

test("relationRows: 쓰레기 입력은 빈 배열 (절대 던지지 않음)", () => {
  for (const bad of [null, {}, { proposals: "x" }, { proposals: [null, 3, "y"] }]) {
    assert.deepEqual(relationRows(bad), []);
  }
});

test("splitClusters: kind 별 분리, 모르는 kind 는 버림", () => {
  const s = splitClusters({ clusters: [
    { kind: "variant", members: [] },
    { kind: "cross_type", members: [] },
    { kind: "weird", members: [] },
  ]});
  assert.equal(s.variant.length, 1);
  assert.equal(s.cross_type.length, 1);
  assert.deepEqual(s.similar, []);
});

test("acceptCandidates: 이미 정답인 노드는 후보에서 뺀다", () => {
  const hit = { node_ids: ["T:a", "T:b", "T:c"] };
  assert.deepEqual(acceptCandidates(hit, ["T:b"]), ["T:a", "T:c"]);
  assert.deepEqual(acceptCandidates(null, ["T:b"]), []);
});

// ── triageBands (C3) ─────────────────────────────────────────────

const TRIAGE = {
  namespace: "ins",
  nodes: {
    strong_confirm: [
      { node_id: "T:암진단비", name: "암진단비", band: "strong_confirm",
        reasons: ["LLM confirm + 인용 실재"], verdict: "confirm",
        rationale: "정의 조문이 근거", evidence_quote: "암".repeat(300),
        signals: { dup_member: false }, evidence_count: 3 },
    ],
    strong_reject: [
      { node_id: "T:상선암", band: "strong_reject", reasons: ["오추출"],
        verdict: "reject", rationale: "원문에 독립 출현 없음",
        evidence_quote: "", evidence_count: 0 },
    ],
    borderline: [
      { node_id: "T:계약자", band: "borderline",
        reasons: ["structural_candidate", "llm_skipped"], verdict: null },
    ],
  },
  relations: null,
  counts: { nodes: { strong_confirm: 999 } },   // 응답 counts 는 믿지 않는다
  llm_calls: 2,
};

test("triageBands: 밴드 매핑 + quote 240 절단 + counts 재계산", () => {
  const b = triageBands(TRIAGE);
  assert.equal(b.nodes.confirm.length, 1);
  assert.equal(b.nodes.reject.length, 1);
  assert.equal(b.nodes.borderline.length, 1);
  const c = b.nodes.confirm[0];
  assert.equal(c.nodeId, "T:암진단비");
  assert.equal(c.band, "confirm");
  assert.equal(c.verdict, "confirm");
  assert.equal(c.rationale, "정의 조문이 근거");
  assert.deepEqual(c.reasons, ["LLM confirm + 인용 실재"]);
  assert.equal(c.quote.length, 240);            // 표시용 절단
  assert.equal(c.evidenceCount, 3);
  assert.equal(b.nodes.borderline[0].verdict, null);  // LLM 생략 — 예측 없음
  // counts 는 응답이 아니라 매핑된 배열에서 다시 센다
  assert.deepEqual(b.counts.nodes, { confirm: 1, reject: 1, borderline: 1 });
  assert.equal(b.relations, null);
  assert.equal(b.llmCalls, 2);
});

test("triageBands: 관계 밴드 — chunkId·quoteFull(무절단) 동봉", () => {
  const long = "약".repeat(260);
  const b = triageBands({
    nodes: {},
    relations: {
      strong_confirm: [
        { subject: "A", predicate: "definesTerm", object: "B",
          chunk_id: "c1", evidence_quote: long, band: "strong_confirm",
          reasons: ["시그니처 실재"] },
      ],
      borderline: [
        { subject: "C", predicate: "p", object: "D", chunk_id: "c2",
          evidence_quote: "짧다", reasons: ["신규 패턴"] },
      ],
    },
    llm_calls: 0,
  });
  const r = b.relations.confirm[0];
  assert.equal(r.subject, "A");
  assert.equal(r.chunkId, "c1");
  assert.equal(r.quote.length, 240);
  assert.equal(r.quoteFull.length, 260);        // 승인 재대조용 — 무절단
  assert.equal(b.relations.reject.length, 0);   // 관계엔 strong_reject 없음
  assert.equal(b.relations.borderline.length, 1);
  assert.deepEqual(b.counts.relations, { confirm: 1, reject: 0, borderline: 1 });
});

test("triageBands: 쓰레기 입력은 빈 구조 (절대 던지지 않음)", () => {
  for (const bad of [null, {}, { nodes: "x" }, { nodes: { strong_confirm: "y" } },
                     { nodes: { strong_confirm: [null, 3] } }]) {
    const b = triageBands(bad);
    assert.deepEqual(b.nodes, { confirm: [], reject: [], borderline: [] });
    assert.equal(b.relations, null);
    assert.deepEqual(b.counts.nodes, { confirm: 0, reject: 0, borderline: 0 });
    assert.equal(b.llmCalls, 0);
  }
});

// ── batchPreviewSummary (C3) ─────────────────────────────────────

test("batchPreviewSummary: would_* 는 적용, 나머지는 reason 별 집계 (첫 출현 순)", () => {
  const s = batchPreviewSummary({ dry_run: true, results: [
    { kind: "node", node_id: "a", status: "would_confirm" },
    { kind: "node", node_id: "b", status: "skipped", reason: "already_judged" },
    { kind: "node", node_id: "c", status: "would_reject" },
    { kind: "node", node_id: "d", status: "blocked", reason: "lifecycle_protected" },
    { kind: "node", node_id: "e", status: "skipped", reason: "already_judged" },
  ]});
  assert.equal(s.willApply, 2);
  assert.deepEqual(s.skipped, [
    { reason: "already_judged", n: 2 },
    { reason: "lifecycle_protected", n: 1 },
  ]);
});

test("batchPreviewSummary: reason 없는 skip 은 status 로, 쓰레기는 빈 요약", () => {
  const s = batchPreviewSummary({ results: [
    { status: "skipped" },              // reason 누락 → status 가 사유
    "not-an-object",
  ]});
  assert.deepEqual(s, { willApply: 0, skipped: [{ reason: "skipped", n: 1 }] });
  for (const bad of [null, {}, { results: "x" }]) {
    assert.deepEqual(batchPreviewSummary(bad), { willApply: 0, skipped: [] });
  }
});

// ── agreementBadge (C3) ──────────────────────────────────────────

test("agreementBadge: 표본 충분 — 일치율 라벨", () => {
  const b = agreementBadge({
    per_actor: { evidence_checker: { n: 48, agree: 44, rate: 0.9167 } },
    overall: { n: 100, agree: 50, rate: 0.5 },
  });
  assert.equal(b.n, 48);
  assert.equal(b.rate, 0.9167);
  assert.equal(b.enough, true);
  assert.equal(b.label, "사람-일치율 91.7% (n=48)");
});

test("agreementBadge: 경계 — n==20 은 충분, n==19 는 부족", () => {
  const at = (n) => agreementBadge({
    per_actor: { evidence_checker: { n, agree: n, rate: 1.0 } } });
  assert.equal(at(20).enough, true);
  assert.equal(at(20).label, "사람-일치율 100.0% (n=20)");
  assert.equal(at(19).enough, false);
  assert.equal(at(19).label, "표본 부족 (n=19) — 개별 검수 권장");
});

test("agreementBadge: rate null(n==0) · 렌즈 없음 — overall 로 폴백하지 않는다", () => {
  // n==0 → 서버가 rate=null (0.0 오보고 금지) — 여기서도 null 로 남는다
  const zero = agreementBadge({
    per_actor: { evidence_checker: { n: 0, agree: 0, rate: null } } });
  assert.equal(zero.rate, null);
  assert.equal(zero.enough, false);
  assert.equal(zero.label, "표본 부족 (n=0) — 개별 검수 권장");
  // 렌즈 항목 자체가 없으면 다른 렌즈(overall)의 표본으로 지어내지 않는다
  const missing = agreementBadge({ overall: { n: 500, rate: 0.99 } });
  assert.equal(missing.n, 0);
  assert.equal(missing.enough, false);
  // 다른 actor 지정도 같은 규칙
  const other = agreementBadge(
    { per_actor: { triage: { n: 30, rate: 0.8 } } }, "triage");
  assert.equal(other.enough, true);
});

test("agreementBadge: 쓰레기 입력은 안전한 부족 배지 (절대 던지지 않음)", () => {
  for (const bad of [null, {}, { per_actor: "x" },
                     { per_actor: { evidence_checker: "y" } },
                     { per_actor: { evidence_checker: { n: "많이", rate: "높음" } } }]) {
    const b = agreementBadge(bad);
    assert.equal(b.enough, false);
    assert.equal(b.rate, null);
    assert.ok(b.label.includes("표본 부족"));
  }
});
