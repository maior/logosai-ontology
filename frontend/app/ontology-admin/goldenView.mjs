/**
 * goldenView — 골든셋(정답지) 응답을 화면 표현으로 바꾸는 순수 함수들.
 *
 * 골든셋은 "이 질의엔 이 노드가 나와야 한다"의 사람 확정 목록이고, 평가는 그걸로
 * 검색을 채점한 숫자(hit@1 · hit@k · MRR)다. 이 변환이 틀리면 **자(ruler)가 틀리는**
 * 것이라 회귀를 반대로 읽는다 — 렌더 없이 단위 테스트한다.
 *
 * 던지지 않는다: 골든셋이 아예 없는 네임스페이스(대부분)가 정상 상태다.
 */

const isObj = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const arr = (v) => (Array.isArray(v) ? v : []);
const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);

export function normalizeCases(resp) {
  const r = isObj(resp) ? resp : {};
  // cases 는 **재구성하지 않는다** — accepted·source 처럼 나중에 늘어날 필드가
  // pick 목록에서 빠져 조용히 사라지는 것을 막는다.
  const cases = arr(r.cases);
  let confirmed = 0;
  let draft = 0;
  for (const c of cases) {
    const st = isObj(c) ? String(c.status || "") : "";
    if (st === "confirmed") confirmed += 1;
    else if (st === "draft") draft += 1;
  }
  return {
    namespace: r.namespace ? String(r.namespace) : "",
    cases,
    total: Number.isFinite(Number(r.total)) ? Number(r.total) : cases.length,
    counts: { confirmed, draft },
  };
}

export function allTags(cases) {
  const seen = new Set();
  for (const c of arr(cases)) {
    for (const tg of arr(isObj(c) ? c.tags : null)) {
      const t = String(tg ?? "").trim();
      if (t) seen.add(t);
    }
  }
  return [...seen].sort();
}

export function filterCases(cases, opts) {
  const o = isObj(opts) ? opts : {};
  const status = o.status && o.status !== "all" ? String(o.status) : "";
  const tag = o.tag ? String(o.tag) : "";
  const q = o.q ? String(o.q).trim().toLowerCase() : "";
  return arr(cases).filter((c) => {
    if (!isObj(c)) return false;
    if (status && String(c.status || "") !== status) return false;
    if (tag && !arr(c.tags).map(String).includes(tag)) return false;
    if (q) {
      const hay = `${c.query || ""} ${c.expected_node_id || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

// 채널 지표 한 덩어리를 {name, cases, hit1, hitK, mrr} 로. hit@{k} 는 k 에 따라
// 키 이름이 바뀌므로(hit@5 / hit@10) k 로 조회해야 한다 — 하드코딩은 조용히 0.
function channelMetrics(name, m, k) {
  const o = isObj(m) ? m : {};
  // 보고되지 않은 지표는 **0 이 아니라 null** — 0 으로 채우면 화면에 "0.0%" 가
  // 떠서 '측정된 0점'으로 읽힌다. 없는 것과 0점은 다른 사실이다.
  const metric = (v) => (Number.isFinite(Number(v)) && v !== null && v !== "" ? Number(v) : null);
  return {
    name: String(name),
    cases: num(o.cases),
    hit1: metric(o["hit@1"]),
    hitK: metric(o[`hit@${k}`]),
    mrr: metric(o.mrr),
  };
}

export function normalizeEval(resp) {
  const r = isObj(resp) ? resp : {};
  const k = num(r.k) || 5;
  const ch = isObj(r.channels) ? r.channels : {};
  const bt = isObj(r.by_tag) ? r.by_tag : {};
  return {
    namespace: r.namespace ? String(r.namespace) : "",
    k,
    // 채점 위치. 옛 응답엔 없다 → node 로 폴백(하위호환).
    target: r.target === "chunk" ? "chunk" : "node",
    // 해당 타깃의 정답이 없어 채점에서 빠진 케이스 수. **오답이 아니다** —
    // 화면에 안 보이면 "16개 중 2개만 쟀다"는 사실이 숨는다.
    skipped: num(r.skipped),
    cases: num(r.cases),
    channels: Object.keys(ch).map((name) => channelMetrics(name, ch[name], k)),
    byTag: Object.keys(bt).sort().map((tag) => ({
      tag,
      channels: Object.keys(isObj(bt[tag]) ? bt[tag] : {})
        .map((name) => channelMetrics(name, bt[tag][name], k)),
    })),
    perCase: arr(r.per_case),   // 순위 필드 보존 — 통과만 시킨다
  };
}

export function channelsIdentical(norm) {
  const chs = arr(isObj(norm) ? norm.channels : null);
  if (chs.length < 2) return false;   // 비교 대상이 없으면 "같다"고 말할 수 없다
  const [a, ...rest] = chs;
  const same = (x, y) => Math.abs(x - y) < 1e-9;
  return rest.every((c) => same(c.hit1, a.hit1) && same(c.hitK, a.hitK)
                           && same(c.mrr, a.mrr));
}

export function pctLabel(v) {
  // Number(null) === 0 이라 null 을 먼저 걸러야 한다 — 안 그러면 측정 안 된 값이
  // "0.0%" 로 표시돼 측정된 0점과 구별되지 않는다(테스트가 잡은 실제 버그).
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

export function rankLabel(rank) {
  const n = Number(rank);
  // null = 상위 k 안에 정답이 없었다(실패). 0 이나 문자열도 실패로 본다.
  if (!Number.isFinite(n) || n < 1) return "✗";
  return String(Math.trunc(n));
}


/**
 * 평가 이력 → 타임라인 시리즈 — "그때 그 숫자"가 아니라 **궤적**을 본다.
 *
 * 이 세션에서 회복 궤적 표를 손으로 세 번 만들었다(라벨 갱신·OCR 재인제스트).
 * eval_history 가 설정 지문까지 자동 기록하므로, 남은 것은 그걸 시간축 위에
 * 펴는 일이다. target 이 다른 기록은 자가 다른 것이라 섞지 않는다(축이 거짓).
 * channel 은 retrieve 고정이 기본 — 운영 경로가 그것이다.
 */
export function timelineSeries(entries, { target = "evidence", channel = "retrieve" } = {}) {
  const rows = arr(entries)
    .filter((e) => isObj(e) && e.measured !== false && e.target === target
                   && isObj(e.channels) && isObj(e.channels[channel]))
    .map((e) => {
      const m = e.channels[channel];
      const hitK = Object.keys(m).find((x) => x.startsWith("hit@") && x !== "hit@1");
      return {
        at: String(e.at || ""),
        hit1: m["hit@1"] == null ? null : num(m["hit@1"]),
        hitk: hitK == null || m[hitK] == null ? null : num(m[hitK]),
        mrr: m.mrr == null ? null : num(m.mrr),
        cases: num(e.cases),
        config: isObj(e.config) ? e.config : {},
      };
    })
    .filter((p) => p.hit1 != null || p.mrr != null);
  rows.sort((a, b) => (a.at < b.at ? -1 : a.at > b.at ? 1 : 0));
  return rows;
}

/**
 * 연속 지점 간 설정 지문 변화 → 이벤트 마커. 어떤 키가 바뀌었는지 이름으로 —
 * "지표가 움직인 자리"와 "설정을 바꾼 자리"가 같은 그림에 있어야 인과를
 * 물을 수 있다 (knob 교훈: 검색 knob 은 그래프 상태에 종속적).
 */
export function configChanges(points) {
  const out = [];
  for (let i = 1; i < arr(points).length; i += 1) {
    const a = points[i - 1].config || {};
    const b = points[i].config || {};
    const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])]
      .filter((k) => JSON.stringify(a[k]) !== JSON.stringify(b[k]))
      .sort();
    if (keys.length) out.push({ index: i, keys });
  }
  return out;
}
