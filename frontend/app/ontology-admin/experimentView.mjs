/**
 * experimentView — 실험 하네스 레코드를 화면 표현으로 바꾸는 순수 함수들.
 *
 * 레코드 = service.run_retrieval_experiments 가 만드는 한 조합의 측정
 * ({run_id, axes, config, graph:{hash,…}, golden:{cases,…}, metrics, cost,
 *   warnings, at}). 이 변환이 틀리면 파레토가 거짓말을 한다 — 렌더 없이
 * node --test 로 단위 테스트한다.
 *
 * 규율 (core/experiment.py 와 짝):
 * - 프런티어는 프론트에서 **재계산**한다 — 백엔드 pareto 는 실행 응답에만
 *   동봉되고 GET /experiments 의 entries 에는 없다.
 * - 단일 스칼라 합성 금지 — 가중치를 지어내면 저품질·저비용 점("벡터만")이
 *   부당하게 떨어진다. 지배의 정의는 백엔드와 한 벌: 품질 ≥ & 비용 ≤,
 *   둘 중 하나는 strict. 동률은 서로 지배하지 않는다.
 * - measured=false·null 지표·null 비용은 점이 되지 못한다 — 재지 않은
 *   결과가 지배점이 되면 화면이 허구 위에 선다. 배제 수는 숨기지 않는다.
 * - 던지지 않는다: 실험 기록이 아예 없는 네임스페이스가 정상 상태다.
 */

const isObj = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const arr = (v) => (Array.isArray(v) ? v : []);
// 보고되지 않은 값은 0 이 아니라 null — Number(null)===0 이라 null 을 먼저
// 걸러야 한다 (goldenView 의 pctLabel 이 잡은 실제 버그와 같은 부류).
const numOrNull = (v) => {
  if (v === null || v === undefined || v === "" || typeof v === "boolean") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

// 라이브 채널 유추와 설정 대조에 쓰는 knob 키 — retrieval_config.KEYS 와 한 벌.
const CFG_KEYS = [
  "entry_k", "entry_ratio", "min_entry_score", "max_terms",
  "use_propagation", "propagation_channel", "propagation_top",
  "propagation_weight",
];

/**
 * 파레토 산점도 뷰 — {points, frontier, excluded}.
 * points: [{x(비용), y(품질), onFrontier, warnings, axes, runId, graphHash,
 *           hit1, cases, at, record}] (record = 원본 참조 — 마커 대조용).
 * frontier: 비지배 점들, 비용 오름차순 (산점도 x축과 같은 순서).
 */
export function paretoView(records, opts) {
  const o = isObj(opts) ? opts : {};
  const quality = typeof o.quality === "string" && o.quality ? o.quality : "mrr";
  const cost = typeof o.cost === "string" && o.cost ? o.cost : "latency_ms_p50";
  const points = [];
  let excluded = 0;
  for (const r of arr(records)) {
    if (!isObj(r)) { excluded += 1; continue; }
    const metrics = isObj(r.metrics) ? r.metrics : {};
    // measured=false = "재지 못했다" — 0점도 프런티어 후보도 아니다.
    if (metrics.measured === false) { excluded += 1; continue; }
    const q = numOrNull(metrics[quality]);
    const c = numOrNull((isObj(r.cost) ? r.cost : {})[cost]);
    if (q === null || c === null) { excluded += 1; continue; }
    points.push({
      x: c,
      y: q,
      onFrontier: false,
      warnings: arr(r.warnings).map(String),
      axes: isObj(r.axes) ? r.axes : {},
      runId: String(r.run_id || ""),
      graphHash: String((isObj(r.graph) ? r.graph : {}).hash || ""),
      hit1: numOrNull(metrics["hit@1"]),
      cases: numOrNull((isObj(r.golden) ? r.golden : {}).cases),
      at: String(r.at || ""),
      record: r,
    });
  }
  // 지배 판정 — 백엔드 pareto_frontier 와 같은 정의 (두 벌이면 화면과
  // 추천이 다른 프런티어를 말한다). 자기 자신은 strict 조건에 걸리지 않는다.
  for (const p of points) {
    p.onFrontier = !points.some(
      (q2) => q2.y >= p.y && q2.x <= p.x && (q2.y > p.y || q2.x < p.x));
  }
  const frontier = points.filter((p) => p.onFrontier)
    .sort((a, b) => (a.x - b.x) || (b.y - a.y));
  return { points, frontier, excluded };
}

/**
 * 조합 표 — 축 열은 모든 레코드의 axes 키 **합집합**에서 자동 도출한다.
 * 하드코딩하면 새 축(entry_k 격자 등)이 조용히 표에서 사라진다.
 * 행은 최신 먼저 (at 내림차순 — GET entries 순서에 기대지 않는다:
 * 실행 응답의 records 는 조합 순서다).
 */
export function comboTable(records) {
  const keySet = new Set();
  const rows = [];
  for (const r of arr(records)) {
    if (!isObj(r)) continue;
    const axes = isObj(r.axes) ? r.axes : {};
    for (const key of Object.keys(axes)) keySet.add(key);
    const metrics = isObj(r.metrics) ? r.metrics : {};
    const cost = isObj(r.cost) ? r.cost : {};
    // hit@{k} 는 k 에 따라 키 이름이 바뀐다(hit@5/hit@10) — 하드코딩은 조용히 빈 칸.
    const hitKKey = Object.keys(metrics)
      .find((x) => x.startsWith("hit@") && x !== "hit@1") || null;
    rows.push({
      axes,
      hit1: numOrNull(metrics["hit@1"]),
      hitK: hitKKey ? numOrNull(metrics[hitKKey]) : null,
      hitKKey,
      mrr: numOrNull(metrics.mrr),
      measured: metrics.measured !== false,
      p50: numOrNull(cost.latency_ms_p50),
      embed: numOrNull(cost.embed_calls),
      cases: numOrNull((isObj(r.golden) ? r.golden : {}).cases),
      warnings: arr(r.warnings).map(String),
      graphHash: String((isObj(r.graph) ? r.graph : {}).hash || ""),
      at: String(r.at || ""),
      runId: String(r.run_id || ""),
    });
  }
  rows.sort((a, b) => (a.at < b.at ? 1 : a.at > b.at ? -1 : 0));
  return { axisKeys: [...keySet].sort(), rows };
}

/**
 * 그래프 지문 그룹 — hash 별 {hash, n, lastAt, latest}.
 * 최신 hash(비어있지 않은 것 중 가장 최근 레코드의 것)만 latest=true —
 * 그 외 레코드는 전부 stale ("그래프 상태가 변함 — 재실험 필요") 표시용.
 * hash 가 빈 레코드는 지문 미상이라 최신의 기준이 될 수 없다.
 */
export function fingerprintGroups(records) {
  const groups = new Map();
  for (const r of arr(records)) {
    if (!isObj(r)) continue;
    const hash = String((isObj(r.graph) ? r.graph : {}).hash || "");
    const at = String(r.at || "");
    const g = groups.get(hash) || { hash, n: 0, lastAt: "" };
    g.n += 1;
    if (at > g.lastAt) g.lastAt = at;
    groups.set(hash, g);
  }
  const out = [...groups.values()]
    .sort((a, b) => (a.lastAt < b.lastAt ? 1 : a.lastAt > b.lastAt ? -1 : 0));
  const anchor = out.findIndex((g) => g.hash !== "");
  return out.map((g, i) => ({ ...g, latest: i === anchor && anchor >= 0 }));
}

/** 그룹 목록 → stale 판정 기준 hash (없으면 null — 판정 불가는 stale 아님). */
export function latestGraphHash(groups) {
  const g = arr(groups).find((x) => isObj(x) && x.latest);
  return g ? g.hash : null;
}

/**
 * 현재 운영 설정 마커 — 산점도의 "지금 여기" 점.
 * liveConfig = GET /retrieval-config 의 effective. 채널은 확산 플래그로
 * 유추한다(둘 다 true → graph+prop, 아니면 graph) — 백엔드
 * recommend 의 _matches_live 와 같은 규칙 (두 벌이면 화면 ◆ 와 추천의
 * "current" 가 다른 점을 가리킨다). 레코드 config 에 있는 knob 만 대조한다.
 * records 는 최신 먼저라고 가정 — 첫 일치가 곧 최신 일치다.
 */
export function currentMarker(records, liveConfig) {
  const live = isObj(liveConfig) ? liveConfig : {};
  const liveChannel = (live.use_propagation && live.propagation_channel)
    ? "graph+prop" : "graph";
  for (const r of arr(records)) {
    if (!isObj(r)) continue;
    const cfg = isObj(r.config) ? r.config : {};
    if (cfg.channel !== liveChannel) continue;
    const ok = CFG_KEYS.every(
      (key) => !(key in cfg) || cfg[key] === live[key]);
    if (ok) return { liveChannel, record: r, runId: String(r.run_id || "") };
  }
  return { liveChannel, record: null, runId: "" };
}
