// experimentView — 실험 탭 뷰모델 (순수, node --test 파일 명시 실행)
import test from "node:test";
import assert from "node:assert/strict";
import {
  paretoView, comboTable, fingerprintGroups, latestGraphHash, currentMarker,
} from "../app/ontology-admin/experimentView.mjs";

// 레코드 팩토리 — service.run_retrieval_experiments 의 실제 스키마
const REC = (over = {}) => ({
  run_id: "exp-20260803120000",
  namespace: "ins",
  layer: "retrieval",
  axes: { channel: "graph" },
  config: { channel: "graph", k: 5, entry_k: 3, max_terms: 8,
            use_propagation: false, propagation_channel: false },
  graph: { hash: "aaa111", nodes: 191, edges: 132 },
  golden: { cases: 46, target: "evidence", statuses: ["confirmed", "verified"] },
  metrics: { "hit@1": 0.7, "hit@5": 0.9, mrr: 0.8, measured: true },
  cost: { latency_ms_p50: 100, latency_ms_p95: 140, embed_calls: 50,
          llm_calls: 0, queries: 47 },
  warnings: ["small_sample"],
  at: "2026-08-03T10:00:00",
  ...over,
});

// ─── paretoView ──────────────────────────────────────────────────────

test("paretoView: 비지배 저품질·저비용 점이 프런티어에 남는다 (가중합 mutant 사살)", () => {
  const A = REC({ metrics: { mrr: 0.8, measured: true },
                  cost: { latency_ms_p50: 100 } });
  const B = REC({ axes: { channel: "vector" },
                  metrics: { mrr: 0.5, measured: true },
                  cost: { latency_ms_p50: 10 } });   // 품질 낮고 비용 쌈 — 생존해야
  const C = REC({ metrics: { mrr: 0.7, measured: true },
                  cost: { latency_ms_p50: 120 } });  // A 에 지배됨 (품질↓ 비용↑)
  const { points, frontier } = paretoView([A, B, C]);
  assert.equal(points.length, 3);
  // 어떤 가중합 w·q − (1−w)·c 를 골라도 B 를 떨어뜨릴 수 있다 — 지배 정의만이
  // B 를 항상 살린다. 프런티어 = [B, A] (비용 오름차순).
  assert.deepEqual(frontier.map((p) => p.y), [0.5, 0.8]);
  assert.deepEqual(frontier.map((p) => p.x), [10, 100]);
  const byMrr = Object.fromEntries(points.map((p) => [p.y, p.onFrontier]));
  assert.equal(byMrr[0.8], true);
  assert.equal(byMrr[0.5], true);
  assert.equal(byMrr[0.7], false);   // 지배된 점은 프런티어 밖
});

test("paretoView: 동률 점은 서로 지배하지 않는다 — 둘 다 남는다", () => {
  const A = REC({ metrics: { mrr: 0.8, measured: true }, cost: { latency_ms_p50: 50 } });
  const B = REC({ metrics: { mrr: 0.8, measured: true }, cost: { latency_ms_p50: 50 } });
  const { frontier } = paretoView([A, B]);
  assert.equal(frontier.length, 2);
});

test("paretoView: measured=false 는 지표가 최고여도 배제 (허구 지배점 방지)", () => {
  const good = REC({ metrics: { mrr: 0.6, measured: true }, cost: { latency_ms_p50: 50 } });
  const fake = REC({ metrics: { mrr: 0.99, measured: false }, cost: { latency_ms_p50: 1 } });
  const { points, frontier, excluded } = paretoView([good, fake]);
  assert.equal(points.length, 1);
  assert.equal(excluded, 1);
  assert.equal(frontier.length, 1);
  assert.equal(frontier[0].y, 0.6);   // fake 가 있었다면 good 은 지배당했을 것
});

test("paretoView: null 비용·null 지표는 점이 되지 못한다", () => {
  const noCost = REC({ cost: { latency_ms_p50: null, embed_calls: 3 } });
  const noMetric = REC({ metrics: { mrr: null, measured: true } });
  const { points, excluded } = paretoView([noCost, noMetric]);
  assert.equal(points.length, 0);
  assert.equal(excluded, 2);
});

test("paretoView: 점이 축·runId·그래프 hash·경고를 들고 다닌다 (tooltip 재료)", () => {
  const { points } = paretoView([REC()]);
  assert.deepEqual(points[0].axes, { channel: "graph" });
  assert.equal(points[0].runId, "exp-20260803120000");
  assert.equal(points[0].graphHash, "aaa111");
  assert.deepEqual(points[0].warnings, ["small_sample"]);
  assert.equal(points[0].hit1, 0.7);
  assert.equal(points[0].cases, 46);
});

test("paretoView: 쓰레기 입력은 빈 결과 (절대 던지지 않음)", () => {
  for (const bad of [null, undefined, {}, "x", [null, 7, "y", []]]) {
    const v = paretoView(bad);
    assert.deepEqual(v.points, []);
    assert.deepEqual(v.frontier, []);
  }
});

// ─── comboTable ──────────────────────────────────────────────────────

test("comboTable: 축 열은 모든 레코드 axes 키의 합집합 (하드코딩이면 새 축이 사라진다)", () => {
  const a = REC({ axes: { channel: "vector" } });
  const b = REC({ axes: { channel: "graph", entry_k: 3 } });
  const { axisKeys } = comboTable([a, b]);
  assert.deepEqual(axisKeys, ["channel", "entry_k"]);
});

test("comboTable: 최신 먼저 + 행 필드 (hit@k 키는 k 에 따라 동적)", () => {
  const old = REC({ at: "2026-08-01T09:00:00", run_id: "exp-old" });
  const recent = REC({ at: "2026-08-03T10:00:00", run_id: "exp-new",
                       metrics: { "hit@1": 0.6, "hit@10": 0.95, mrr: 0.75, measured: true } });
  const { rows } = comboTable([old, recent]);
  assert.deepEqual(rows.map((r) => r.runId), ["exp-new", "exp-old"]);
  assert.equal(rows[0].hitKKey, "hit@10");
  assert.equal(rows[0].hitK, 0.95);
  assert.equal(rows[0].p50, 100);
  assert.equal(rows[0].embed, 50);
  assert.equal(rows[0].cases, 46);
  assert.equal(rows[0].graphHash, "aaa111");
});

test("comboTable: 쓰레기 입력은 빈 표", () => {
  for (const bad of [null, {}, "x", [null, 1]]) {
    const t2 = comboTable(bad);
    assert.deepEqual(t2.axisKeys, []);
    assert.deepEqual(t2.rows, []);
  }
});

// ─── fingerprintGroups / latestGraphHash ────────────────────────────

test("fingerprintGroups: 가장 최근 레코드의 hash 만 latest — 나머지가 stale 기준", () => {
  const recs = [
    REC({ graph: { hash: "old1" }, at: "2026-08-01T09:00:00" }),
    REC({ graph: { hash: "new2" }, at: "2026-08-03T10:00:00" }),
    REC({ graph: { hash: "old1" }, at: "2026-08-01T10:00:00" }),
  ];
  const groups = fingerprintGroups(recs);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].hash, "new2");
  assert.equal(groups[0].latest, true);
  assert.equal(groups[0].n, 1);
  assert.equal(groups[1].hash, "old1");
  assert.equal(groups[1].latest, false);
  assert.equal(groups[1].n, 2);
  assert.equal(latestGraphHash(groups), "new2");
});

test("fingerprintGroups: hash 없는 레코드는 최신의 기준이 될 수 없다", () => {
  const recs = [
    REC({ graph: {}, at: "2026-08-05T00:00:00" }),          // 지문 미상 — 가장 최근이지만
    REC({ graph: { hash: "aaa" }, at: "2026-08-03T00:00:00" }),
  ];
  const groups = fingerprintGroups(recs);
  assert.equal(latestGraphHash(groups), "aaa");
  assert.equal(groups.find((g) => g.hash === "").latest, false);
});

test("fingerprintGroups: 쓰레기 입력은 빈 배열, latestGraphHash 는 null", () => {
  for (const bad of [null, {}, "x"]) {
    assert.deepEqual(fingerprintGroups(bad), []);
  }
  assert.equal(latestGraphHash([]), null);
  assert.equal(latestGraphHash(null), null);
});

// ─── currentMarker ───────────────────────────────────────────────────

test("currentMarker: 확산 둘 다 on → graph+prop 채널의 knob 일치 레코드", () => {
  const live = { entry_k: 3, max_terms: 8,
                 use_propagation: true, propagation_channel: true };
  const match = REC({
    run_id: "exp-hit",
    config: { channel: "graph+prop", k: 5, entry_k: 3, max_terms: 8,
              use_propagation: true, propagation_channel: true },
  });
  const other = REC({ run_id: "exp-miss" });   // channel=graph — 불일치
  const m = currentMarker([other, match], live);
  assert.equal(m.liveChannel, "graph+prop");
  assert.equal(m.runId, "exp-hit");
  assert.equal(m.record, match);
});

test("currentMarker: knob 하나만 달라도 불일치 (entry_k 3 vs 5)", () => {
  const live = { entry_k: 3, use_propagation: false, propagation_channel: false };
  const rec = REC({ config: { channel: "graph", entry_k: 5,
                              use_propagation: false, propagation_channel: false } });
  const m = currentMarker([rec], live);
  assert.equal(m.liveChannel, "graph");     // 확산 off → graph
  assert.equal(m.record, null);
});

test("currentMarker: 레코드 config 에 없는 knob 은 대조하지 않는다", () => {
  // config 가 일부 키만 들고 있어도(구 레코드) 있는 키가 다 맞으면 일치
  const live = { entry_k: 3, max_terms: 2,
                 use_propagation: false, propagation_channel: false };
  const rec = REC({ config: { channel: "graph", entry_k: 3 } });
  const m = currentMarker([rec], live);
  assert.equal(m.record, rec);
});

test("currentMarker: 쓰레기 입력은 record null (절대 던지지 않음)", () => {
  for (const bad of [null, {}, "x", [null, 3]]) {
    const m = currentMarker(bad, null);
    assert.equal(m.record, null);
    assert.equal(m.liveChannel, "graph");   // 확산 미상 → 보수적으로 graph
  }
});
