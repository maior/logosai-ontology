/**
 * retrieveView — /retrieve 응답을 화면 표현으로 바꾸는 순수 함수들.
 *
 * 컴포넌트에서 분리한 이유: 이 변환이 곧 "왜 이 결과인지"를 드러내는 규칙이라
 * 회귀가 나면 근거 사슬이 조용히 사라진다. 렌더 없이 단위 테스트한다.
 * (.mjs — package.json 에 type:module 이 없어 node --test 가 ESM 으로 읽게)
 *
 * 모든 함수는 **던지지 않는다**. 빈 네임스페이스·청크 0·필드 누락은 정상 상태이며
 * 그때 화면이 죽으면 안 된다 — 안전 기본값을 돌려준다.
 */

const isObj = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const arr = (v) => (Array.isArray(v) ? v : []);

/**
 * buildRetrieveGraph — /retrieve 응답 → 질의 중심 서브그래프 뷰모델.
 *
 * 배경(실측): 탐색 탭의 개체-개체 엣지는 성기다(고립 62.9%였다). 연결의 실체는
 * 근거 링크(노드↔청크)가 나른다(이분 그래프 고립 3.6%). /retrieve 응답에는 그
 * 질의 중심 서브그래프가 이미 통째로 들어 있다:
 *   진입 노드(점수) → 확장 노드(via: is_a↑·sameAs↔·인접 / from: 어느 노드에서)
 *   → 청크(근거 링크 node_ids + 검색 이유 matched_via)
 * 리스트로만 보여주던 것을 그래프로 편다.
 *
 * 규칙:
 *  - 진입이 확장보다 강하다(같은 노드가 둘 다에 오면 entry 로).
 *  - 근거 엣지 = (matched_via ∪ node_ids) ∩ **표시된 노드** — 표시 안 된 노드로의
 *    엣지는 허공을 가리킨다.
 *  - 노드 연결이 없는 청크는 direct(임베딩 채널 단독)로 따로 — "왜 걸렸는지"가
 *    구분되어야 한다.
 *  - maxChunks 상한은 dropped 로 보고한다 (조용한 절단 금지).
 *  - 청크는 상세(원문 text·score·channels·오프셋)를 **들고 다닌다** — 클릭해서
 *    내용을 보는 화면이 히트 목록을 다시 뒤지게 하지 않는다.
 *  - documents = 청크의 source 를 순위 첫 등장 순으로 묶은 문서 층 — 파일이
 *    그림의 끝단이다 (근거가 결국 어느 파일에서 왔는가).
 */
export function buildRetrieveGraph(resp, { maxChunks = 10 } = {}) {
  const r = isObj(resp) ? resp : {};
  const exp = isObj(r.expansion) ? r.expansion : {};

  const nodes = [];
  const seen = new Map();               // id → node (entry 우선)
  const entries = arr(exp.entry_nodes)
    .filter((e) => isObj(e) && e.node_id)
    .slice()
    .sort((a, b) => (Number(b.score) || 0) - (Number(a.score) || 0));
  for (const e of entries) {
    const id = String(e.node_id);
    if (seen.has(id)) continue;
    const node = { id, label: id.split(":").slice(1).join(":") || id,
                   kind: "entry", score: Number(e.score) || 0 };
    seen.set(id, node);
    nodes.push(node);
  }

  const expansion = [];
  for (const e of arr(exp.expanded_nodes)) {
    if (!isObj(e) || !e.node_id) continue;
    const id = String(e.node_id);
    if (!seen.has(id)) {
      const node = { id, label: String(e.name || id.split(":").slice(1).join(":") || id),
                     kind: "expanded", via: String(e.via || "") };
      seen.set(id, node);
      nodes.push(node);
    }
    // 진입으로 이미 있는 노드로의 확장 엣지는 그리지 않는다 — entry 가 이긴다.
    if (e.from && seen.get(id)?.kind === "expanded" && seen.has(String(e.from))) {
      expansion.push({ from: String(e.from), to: id, via: String(e.via || "") });
    }
  }

  const allHits = arr(r.hits).filter((h) => isObj(h) && h.chunk_id);
  const shown = allHits.slice(0, maxChunks);
  const chunks = shown.map((h, i) => {
    const source = String(h.source || "?");
    const section = String(h.section || "").trim();
    return {
      chunkId: String(h.chunk_id), rank: i + 1,
      // 파일명은 문서 층이 나른다 — 청크 라벨은 문서 안 위치(조항)가 주인공.
      label: section ? `§${section}` : source,
      source, section,
      text: String(h.text || ""),
      score: Number(h.score) || 0,
      channels: arr(h.channels).map(String).filter(Boolean),
      charStart: Number.isFinite(Number(h.char_start)) ? Number(h.char_start) : null,
      charEnd: Number.isFinite(Number(h.char_end)) ? Number(h.char_end) : null,
    };
  });

  // 문서 층 — 청크 순위의 첫 등장 순 (결정적). count 는 표시된 청크 기준.
  const documents = [];
  const docSeen = new Map();
  for (const c of chunks) {
    let d = docSeen.get(c.source);
    if (!d) { d = { source: c.source, count: 0 }; docSeen.set(c.source, d); documents.push(d); }
    d.count += 1;
  }

  const evidence = [];
  const direct = [];
  for (const h of shown) {
    const refs = new Set(
      [...arr(h.matched_via), ...arr(h.node_ids)]
        .map(String).filter((id) => seen.has(id)));
    if (refs.size === 0) {
      direct.push({ chunk: String(h.chunk_id) });
      continue;
    }
    for (const node of [...refs].sort()) {
      evidence.push({ node, chunk: String(h.chunk_id) });
    }
  }

  return {
    query: r.query ? String(r.query) : "",
    nodes, chunks, documents,
    edges: { expansion, evidence, direct },
    dropped: { chunks: Math.max(0, allHits.length - shown.length) },
  };
}

export function normalizeRetrieve(resp) {
  const r = isObj(resp) ? resp : {};
  const exp = isObj(r.expansion) ? r.expansion : {};
  const query = r.query ? String(r.query) : "";
  return {
    namespace: r.namespace ? String(r.namespace) : "",
    query,
    // 확장에 실패해도 질의는 보여야 한다 — 원 질의로 폴백.
    expandedQuery: exp.expanded_query ? String(exp.expanded_query) : query,
    terms: arr(exp.terms).filter(Boolean).map(String),
    entryNodes: arr(exp.entry_nodes),
    expandedNodes: arr(exp.expanded_nodes),
    // hits 는 **재구성하지 않는다**: matched_via·trust·node_ids 처럼 나중에 늘어날
    // 근거 필드가 pick 목록에서 빠져 조용히 사라지는 것을 막는다.
    hits: arr(r.hits),
  };
}

export function channelBadges(hit) {
  const seen = new Set();
  const out = [];
  for (const c of arr(isObj(hit) ? hit.channels : null)) {
    const key = String(c ?? "").trim();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(key);
  }
  return out;
}

export function citationLabel(hit) {
  const h = isObj(hit) ? hit : {};
  const parts = [];
  if (h.section) parts.push(String(h.section));   // 조항이 먼저 — 인용의 앵커
  if (h.source) parts.push(String(h.source));
  if (parts.length) return parts.join(" · ");
  return h.chunk_id ? String(h.chunk_id) : "";
}

export function offsetLabel(hit) {
  const h = isObj(hit) ? hit : {};
  const s = Number(h.char_start);
  const e = Number(h.char_end);
  // 무효 구간을 표시하면 "원문 어디"라는 약속이 거짓이 된다 — 차라리 비운다.
  if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || e <= s) return "";
  const fmt = (n) => n.toLocaleString("en-US");   // 로케일 고정 = 결정론적 표기
  return `${fmt(s)}–${fmt(e)}`;
}

export function viaLabel(hit) {
  // matched_via 는 **배열**이다 — 그래프 채널이 이 청크를 데려온 노드들
  // (라이브 검증 2026-07-27). 배열을 그대로 JSX 에 넣으면 React 가 구분자 없이
  // 이어붙여 "노드A노드B" 가 된다. 문자열로 바뀌어도 깨지지 않게 둘 다 받는다.
  const v = isObj(hit) ? hit.matched_via : null;
  if (typeof v === "string") return v.trim();
  return arr(v).map((x) => String(x ?? "").trim()).filter(Boolean).join(" · ");
}

export function maxScore(hits) {
  let m = 0;
  for (const h of arr(hits)) {
    const s = Number(isObj(h) ? h.score : NaN);
    if (Number.isFinite(s) && s > m) m = s;
  }
  return m;
}

export function scoreBarPct(hit, max) {
  const s = Number(isObj(hit) ? hit.score : NaN);
  const m = Number(max);
  if (!Number.isFinite(s) || !Number.isFinite(m) || m <= 0 || s <= 0) return 0;
  return Math.min(100, (s / m) * 100);
}
