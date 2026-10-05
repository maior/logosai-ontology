/**
 * goldenView 단위 테스트 (node --test, 의존성 0).
 *
 * 픽스처는 **라이브 응답을 찍어서** 만들었다 (2026-07-27, ins_cancer_demo) —
 * 필드 모양을 추측했다가 matched_via 배열에 물린 전례가 있다.
 *
 * 실행: cd frontend && node --test tests/
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  normalizeCases, allTags, filterCases, normalizeEval,
  channelsIdentical, pctLabel, rankLabel,
} from "../app/ontology-admin/goldenView.mjs";

// GET /graphs/{ns}/qa 실응답
const CASES = {
  namespace: "ins_cancer_demo",
  total: 3,
  cases: [
    { case_id: "ffc9", query: "계약자는 누구를 말하나?", expected_node_id: "ContractParty:계약자",
      status: "confirmed", source: "hand", tags: ["exact", "definition"], accepted: [] },
    { case_id: "aa01", query: "청약 철회 기간은?", expected_node_id: "InsuranceTerm:청약철회",
      status: "draft", source: "llm", tags: ["semantic"], accepted: [] },
    { case_id: "bb02", query: "보장 범위", expected_node_id: "Coverage:암진단",
      status: "confirmed", source: "hand", tags: ["coverage"], accepted: ["Coverage:암"] },
  ],
};

// POST /graphs/{ns}/qa/evaluate 실응답 (k=5). hit@{k} 키가 k 에 따라 바뀐다.
const EVAL = {
  namespace: "ins_cancer_demo", k: 5, cases: 16,
  channels: {
    semantic: { cases: 16, "hit@1": 0.3125, "hit@5": 0.875, mrr: 0.5520833333333333 },
    retrieve: { cases: 16, "hit@1": 0.3125, "hit@5": 0.875, mrr: 0.5520833333333333 },
  },
  by_tag: {
    exact: {
      semantic: { cases: 8, "hit@1": 0.25, "hit@5": 1.0, mrr: 0.5729166666666666 },
      retrieve: { cases: 8, "hit@1": 0.25, "hit@5": 1.0, mrr: 0.5729166666666666 },
    },
    coverage: {
      semantic: { cases: 5, "hit@1": 0.2, "hit@5": 0.6, mrr: 0.4 },
      retrieve: { cases: 5, "hit@1": 0.2, "hit@5": 0.6, mrr: 0.4 },
    },
  },
  per_case: [
    { case_id: "ffc9", query: "계약자는 누구를 말하나?", expected: "ContractParty:계약자",
      tags: ["exact", "definition"], semantic_rank: 4, retrieve_rank: 4 },
  ],
};

// ─── normalizeCases — 목록 + 상태 집계 ──────────────────────────────

test("normalizeCases: 케이스와 confirmed/draft 집계", () => {
  const n = normalizeCases(CASES);
  assert.equal(n.namespace, "ins_cancer_demo");
  assert.equal(n.total, 3);
  assert.equal(n.cases.length, 3);
  assert.equal(n.counts.confirmed, 2);
  assert.equal(n.counts.draft, 1);
});

test("normalizeCases: 케이스를 재구성하지 않는다 (accepted·source 보존)", () => {
  const n = normalizeCases(CASES);
  assert.equal(n.cases[2], CASES.cases[2]);          // 동일 참조
  assert.deepEqual(n.cases[2].accepted, ["Coverage:암"]);
  assert.equal(n.cases[1].source, "llm");
});

test("normalizeCases: 골든셋 없는 네임스페이스도 안전 (던지지 않음)", () => {
  for (const bad of [null, undefined, {}, { cases: "nope" }]) {
    const n = normalizeCases(bad);
    assert.deepEqual(n.cases, []);
    assert.equal(n.counts.confirmed, 0);
    assert.equal(n.counts.draft, 0);
  }
});

test("normalizeCases: total 없으면 cases 길이로 대체", () => {
  assert.equal(normalizeCases({ cases: CASES.cases }).total, 3);
});

// ─── allTags / filterCases — 필터 ───────────────────────────────────

test("allTags: 유니크 + 정렬", () => {
  assert.deepEqual(allTags(CASES.cases),
                   ["coverage", "definition", "exact", "semantic"]);
  assert.deepEqual(allTags([]), []);
  assert.deepEqual(allTags(null), []);
});

test("filterCases: status 로 걸러낸다", () => {
  assert.equal(filterCases(CASES.cases, { status: "draft" }).length, 1);
  assert.equal(filterCases(CASES.cases, { status: "confirmed" }).length, 2);
  assert.equal(filterCases(CASES.cases, { status: "all" }).length, 3);
});

test("filterCases: tag 로 걸러낸다", () => {
  assert.equal(filterCases(CASES.cases, { tag: "exact" }).length, 1);
  assert.equal(filterCases(CASES.cases, { tag: "없는태그" }).length, 0);
});

test("filterCases: 질의·정답노드 부분일치 검색 (대소문자 무시)", () => {
  assert.equal(filterCases(CASES.cases, { q: "철회" }).length, 1);
  assert.equal(filterCases(CASES.cases, { q: "coverage:" }).length, 1);   // expected_node_id
  assert.equal(filterCases(CASES.cases, { q: "" }).length, 3);
});

test("filterCases: 조건 조합 (status + tag)", () => {
  assert.equal(filterCases(CASES.cases,
    { status: "confirmed", tag: "coverage" }).length, 1);
  assert.equal(filterCases(CASES.cases,
    { status: "draft", tag: "coverage" }).length, 0);
});

test("filterCases: tags 없는 케이스도 죽지 않는다 (수동 추가는 tags 생략 가능)", () => {
  const noTags = [{ case_id: "z", query: "q", expected_node_id: "N:1", status: "confirmed" }];
  assert.equal(filterCases(noTags, { tag: "exact" }).length, 0);
  assert.equal(filterCases(noTags, { status: "confirmed" }).length, 1);
  assert.deepEqual(allTags(noTags), []);
});

test("filterCases: 입력이 배열 아니면 빈 배열", () => {
  assert.deepEqual(filterCases(null, {}), []);
  assert.deepEqual(filterCases(CASES.cases, null).length, 3);   // opts 없으면 전체
});

// ─── normalizeEval — hit@k 동적 키 ──────────────────────────────────

test("normalizeEval: 채널 지표를 hit1/hitK/mrr 로 정규화", () => {
  const n = normalizeEval(EVAL);
  assert.equal(n.k, 5);
  assert.equal(n.cases, 16);
  assert.equal(n.channels.length, 2);
  const sem = n.channels.find((c) => c.name === "semantic");
  assert.equal(sem.cases, 16);
  assert.equal(sem.hit1, 0.3125);
  assert.equal(sem.hitK, 0.875);
  assert.ok(Math.abs(sem.mrr - 0.5520833333333333) < 1e-12);
});

test("normalizeEval: hit@k 키는 k 에 따라 달라진다 (hit@10)", () => {
  // k=10 이면 서버는 'hit@10' 로 준다 — 'hit@5' 하드코딩은 조용히 0 이 된다.
  const n = normalizeEval({
    k: 10, cases: 4,
    channels: { retrieve: { cases: 4, "hit@1": 0.5, "hit@10": 0.75, mrr: 0.6 } },
  });
  assert.equal(n.channels[0].hitK, 0.75);
});

test("normalizeEval: 태그별 지표도 같은 모양으로", () => {
  const n = normalizeEval(EVAL);
  assert.equal(n.byTag.length, 2);
  const ex = n.byTag.find((t) => t.tag === "exact");
  assert.equal(ex.channels.find((c) => c.name === "semantic").hitK, 1.0);
});

test("normalizeEval: per_case 는 그대로 통과 (순위 필드 보존)", () => {
  const n = normalizeEval(EVAL);
  assert.equal(n.perCase[0], EVAL.per_case[0]);
  assert.equal(n.perCase[0].semantic_rank, 4);
  assert.equal(n.perCase[0].retrieve_rank, 4);
});

test("normalizeEval: 보고되지 않은 지표는 0 이 아니라 null (0.0% 로 거짓 표기 금지)", () => {
  // hit@k 키가 없는데 0 으로 채우면 화면에 "0.0%" 가 떠 '측정된 0점'처럼 보인다.
  const n = normalizeEval({ k: 5, channels: { retrieve: { cases: 3, "hit@1": 0.5 } } });
  assert.equal(n.channels[0].hit1, 0.5);
  assert.equal(n.channels[0].hitK, null);
  assert.equal(n.channels[0].mrr, null);
  assert.equal(pctLabel(n.channels[0].hitK), "—");
});

test("normalizeEval: 빈·불량 응답도 안전", () => {
  for (const bad of [null, undefined, {}, { channels: "x", by_tag: 3, per_case: "y" }]) {
    const n = normalizeEval(bad);
    assert.deepEqual(n.channels, []);
    assert.deepEqual(n.byTag, []);
    assert.deepEqual(n.perCase, []);
  }
});

// ─── target / skipped — 채점 위치와 라벨 없는 케이스 ────────────────

test("normalizeEval: target 과 skipped 를 노출한다 (라벨 없는 케이스는 오답이 아니다)", () => {
  const n = normalizeEval({
    k: 5, target: "chunk", cases: 2, skipped: 14,
    channels: { chunk: { cases: 2, "hit@1": 0.5, "hit@5": 1.0, mrr: 0.75 } },
  });
  assert.equal(n.target, "chunk");
  assert.equal(n.skipped, 14);
  assert.equal(n.cases, 2);
});

test("normalizeEval: target 없으면 node (옛 응답 하위호환)", () => {
  const n = normalizeEval({ k: 5, cases: 3, channels: {} });
  assert.equal(n.target, "node");
  assert.equal(n.skipped, 0);
});

test("normalizeEval: 채점 0건 + 스킵 다수 = 측정 불가 상태가 드러난다", () => {
  const n = normalizeEval({ k: 5, target: "chunk", cases: 0, skipped: 16, channels: {} });
  assert.equal(n.cases, 0);
  assert.equal(n.skipped, 16);
});

// ─── channelsIdentical — "이 측정은 무의미하다" 경고 ────────────────

test("channelsIdentical: 두 채널 지표가 같으면 true (측정 무의미 신호)", () => {
  // 라이브 실측이 정확히 이 상태였다: retrieve 의 노드 순위가 semantic 에서
  // 파생돼 소수점까지 같다 → 그래프 조건화의 효과를 이 eval 로는 볼 수 없다.
  assert.equal(channelsIdentical(normalizeEval(EVAL)), true);
});

test("channelsIdentical: 하나라도 다르면 false", () => {
  const differ = JSON.parse(JSON.stringify(EVAL));
  differ.channels.retrieve["hit@1"] = 0.5;
  assert.equal(channelsIdentical(normalizeEval(differ)), false);
});

test("channelsIdentical: 채널이 하나뿐이면 비교 대상이 없어 false", () => {
  const one = normalizeEval({ k: 5, channels: { retrieve: { cases: 1, "hit@1": 1, "hit@5": 1, mrr: 1 } } });
  assert.equal(channelsIdentical(one), false);
  assert.equal(channelsIdentical(normalizeEval(null)), false);
});

// ─── 표기 ───────────────────────────────────────────────────────────

test("pctLabel: 비율을 소수 한 자리 퍼센트로", () => {
  assert.equal(pctLabel(0.3125), "31.3%");
  assert.equal(pctLabel(1), "100.0%");
  assert.equal(pctLabel(0), "0.0%");
  assert.equal(pctLabel(null), "—");
  assert.equal(pctLabel("x"), "—");
});

test("rankLabel: 순위, 못 찾으면 실패 표시", () => {
  assert.equal(rankLabel(1), "1");
  assert.equal(rankLabel(4), "4");
  assert.equal(rankLabel(null), "✗");    // 상위 k 안에 정답이 없었다
  assert.equal(rankLabel(0), "✗");
  assert.equal(rankLabel("x"), "✗");
});

// ─── 타임라인 — 평가 이력을 궤적으로 ────────────────────────────────
import { timelineSeries, configChanges } from "../app/ontology-admin/goldenView.mjs";

const HIST = [
  { at: "2026-08-02T10:00", target: "evidence", measured: true, cases: 65,
    channels: { retrieve: { "hit@1": 0.69, "hit@5": 0.98, mrr: 0.82 } },
    config: { node_model: "a", entry_k: 3 } },
  { at: "2026-08-01T09:00", target: "evidence", measured: true, cases: 65,
    channels: { retrieve: { "hit@1": 0.66, "hit@5": 0.97, mrr: 0.78 } },
    config: { node_model: "a", entry_k: 5 } },
  { at: "2026-08-03T11:00", target: "node", measured: true, cases: 46,   // 다른 자 — 제외
    channels: { retrieve: { "hit@1": 0.5, mrr: 0.6 } }, config: {} },
  { at: "2026-08-03T12:00", target: "evidence", measured: false,          // 미측정 — 제외
    channels: { retrieve: { "hit@1": null, mrr: null } }, config: {} },
];

test("timelineSeries: target 필터 + 시간 오름차순 + 미측정 제외", () => {
  const pts = timelineSeries(HIST, { target: "evidence" });
  assert.deepEqual(pts.map((p) => p.at), ["2026-08-01T09:00", "2026-08-02T10:00"]);
  assert.equal(pts[0].hit1, 0.66);
  assert.equal(pts[1].mrr, 0.82);
  assert.equal(pts[1].hitk, 0.98);   // hit@k 는 k 가 뭐든 hit@1 아닌 것
});

test("timelineSeries: 쓰레기 입력은 빈 배열", () => {
  for (const bad of [null, [], [{}], [{ channels: 3 }]]) {
    assert.deepEqual(timelineSeries(bad), []);
  }
});

test("configChanges: 연속 지점의 지문 차이 → 바뀐 키 이름", () => {
  const pts = timelineSeries(HIST, { target: "evidence" });
  const marks = configChanges(pts);
  assert.deepEqual(marks, [{ index: 1, keys: ["entry_k"] }]);   // 5 → 3
});

test("configChanges: 변화 없으면 마커 없음, 키 추가/삭제도 변화다", () => {
  assert.deepEqual(configChanges([{ config: { a: 1 } }, { config: { a: 1 } }]), []);
  assert.deepEqual(configChanges([{ config: {} }, { config: { m: "x" } }]),
    [{ index: 1, keys: ["m"] }]);
});
