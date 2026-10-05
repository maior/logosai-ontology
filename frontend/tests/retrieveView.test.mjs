/**
 * retrieveView 단위 테스트 (node --test, 의존성 0).
 *
 * 검증 대상은 "근거 사슬이 화면까지 온전히 오는가" — 확장(왜 이 질의로 찾았나),
 * 채널(어느 경로로 왔나), 인용(원문 어디인가). 이 규칙이 깨지면 검색은 돌아도
 * 신뢰 근거가 사라진다.
 *
 * 실행: cd frontend && node --test tests/
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  normalizeRetrieve, channelBadges, citationLabel,
  offsetLabel, maxScore, scoreBarPct, viaLabel,
} from "../app/ontology-admin/retrieveView.mjs";

// 서비스가 실제로 돌려주는 모양 (service.py:1218-1229 기준)
const SAMPLE = {
  namespace: "ins_cancer_demo",
  query: "청약 철회",
  expansion: {
    expanded_query: "청약 철회 청약철회권 계약취소",
    terms: ["청약철회권", "계약취소"],
    entry_nodes: [{ node_id: "Clause:청약의 철회", score: 0.41 }],
    expanded_nodes: [{ node_id: "Product:암보험", via: "is_a↑", from: "Clause:청약의 철회" }],
  },
  hits: [
    {
      chunk_id: "a1b2", text: "제19조【청약의 철회】보험계약자는 …",
      source: "dy_basic.pdf", index: 12, section: "제19조(청약의 철회)",
      char_start: 1234, char_end: 1567, node_ids: ["Clause:청약의 철회"],
      trust: "authoritative", meta: {}, score: 0.032,
      // 실측: matched_via 는 **배열**(그래프 채널이 이 청크를 데려온 노드들) —
      // 문자열로 가정했다가 라이브 검증에서 잡혔다 (2026-07-27).
      matched_via: ["InsuranceTerm:해약환급금", "ContractParty:회사"],
      channels: ["chunk", "graph"],
    },
    {
      chunk_id: "c3d4", text: "제20조【보험료의 반환】…",
      source: "dy_basic.pdf", index: 13, section: "제20조(보험료의 반환)",
      char_start: 1567, char_end: 1890, node_ids: [], trust: "", meta: {},
      score: 0.016, matched_via: [], channels: ["chunk"],
    },
  ],
};

// ─── normalizeRetrieve — 필드 누락에 죽지 않는가 ────────────────────

test("normalizeRetrieve: 정상 응답의 확장·히트를 그대로 노출", () => {
  const n = normalizeRetrieve(SAMPLE);
  assert.equal(n.namespace, "ins_cancer_demo");
  assert.equal(n.query, "청약 철회");
  assert.equal(n.expandedQuery, "청약 철회 청약철회권 계약취소");
  assert.deepEqual(n.terms, ["청약철회권", "계약취소"]);
  assert.equal(n.entryNodes.length, 1);
  assert.equal(n.expandedNodes.length, 1);
  assert.equal(n.hits.length, 2);
});

test("normalizeRetrieve: hit 을 재구성하지 않는다 — matched_via·trust·node_ids 보존", () => {
  // 근거 사슬이 조용히 사라지는 것을 막는 계약. hits 를 map/pick 으로 다시 만들면
  // 나중에 추가되는 근거 필드가 화면에 도달하지 못한다.
  const n = normalizeRetrieve(SAMPLE);
  assert.equal(n.hits[0], SAMPLE.hits[0]);              // 동일 참조(통과)
  assert.deepEqual(n.hits[0].matched_via, ["InsuranceTerm:해약환급금", "ContractParty:회사"]);
  assert.equal(n.hits[0].trust, "authoritative");
  assert.deepEqual(n.hits[0].node_ids, ["Clause:청약의 철회"]);
});

test("normalizeRetrieve: null·undefined 도 안전 기본값 (던지지 않음)", () => {
  for (const bad of [null, undefined, 0, "x", []]) {
    const n = normalizeRetrieve(bad);
    assert.equal(n.namespace, "");
    assert.deepEqual(n.terms, []);
    assert.deepEqual(n.hits, []);
    assert.deepEqual(n.entryNodes, []);
  }
});

test("normalizeRetrieve: expansion 없으면 확장질의는 원 질의로 폴백", () => {
  const n = normalizeRetrieve({ query: "암 진단", hits: [] });
  assert.equal(n.expandedQuery, "암 진단");   // 확장 못 했어도 질의는 보여야 한다
  assert.deepEqual(n.terms, []);
});

test("normalizeRetrieve: 배열 아닌 필드는 빈 배열로 (청크 0 네임스페이스)", () => {
  const n = normalizeRetrieve({
    namespace: "empty_ns", query: "q",
    expansion: { terms: "not-an-array", entry_nodes: null },
    hits: "nope",
  });
  assert.deepEqual(n.terms, []);
  assert.deepEqual(n.entryNodes, []);
  assert.deepEqual(n.hits, []);
});

test("normalizeRetrieve: terms 의 빈 값은 걸러진다", () => {
  const n = normalizeRetrieve({ expansion: { terms: ["가", "", null, "나"] } });
  assert.deepEqual(n.terms, ["가", "나"]);
});

// ─── channelBadges — 어느 경로로 왔나 (우리 차별점) ─────────────────

test("channelBadges: 두 채널 융합 히트는 순서 보존", () => {
  assert.deepEqual(channelBadges(SAMPLE.hits[0]), ["chunk", "graph"]);
  assert.deepEqual(channelBadges(SAMPLE.hits[1]), ["chunk"]);
});

test("channelBadges: 중복 제거 + 공백 제거", () => {
  assert.deepEqual(channelBadges({ channels: ["graph", "graph", " ", "chunk"] }),
                   ["graph", "chunk"]);
});

test("channelBadges: channels 없으면 빈 배열", () => {
  for (const bad of [null, undefined, {}, { channels: null }, { channels: "chunk" }]) {
    assert.deepEqual(channelBadges(bad), []);
  }
});

// ─── viaLabel — 그래프 채널이 이 청크를 데려온 노드들 ───────────────

test("viaLabel: 배열은 구분자로 이어 붙인다 (React 가 붙여쓰는 것 방지)", () => {
  assert.equal(viaLabel(SAMPLE.hits[0]),
               "InsuranceTerm:해약환급금 · ContractParty:회사");
});

test("viaLabel: 빈 배열·누락은 빈 문자열", () => {
  assert.equal(viaLabel(SAMPLE.hits[1]), "");
  assert.equal(viaLabel({}), "");
  assert.equal(viaLabel(null), "");
  assert.equal(viaLabel({ matched_via: [null, "", "  "] }), "");
});

test("viaLabel: 문자열로 오는 경우도 수용 (계약 변화 대비)", () => {
  assert.equal(viaLabel({ matched_via: "chunk+graph" }), "chunk+graph");
});

// ─── citationLabel — 조항 단위 인용 표기 ────────────────────────────

test("citationLabel: 조항 + 출처를 함께 (조항이 먼저)", () => {
  assert.equal(citationLabel(SAMPLE.hits[0]), "제19조(청약의 철회) · dy_basic.pdf");
});

test("citationLabel: 조항 없으면 출처만", () => {
  assert.equal(citationLabel({ source: "spec.pdf" }), "spec.pdf");
});

test("citationLabel: 조항·출처 모두 없으면 chunk_id 폴백", () => {
  assert.equal(citationLabel({ chunk_id: "abc123" }), "abc123");
  assert.equal(citationLabel({}), "");
  assert.equal(citationLabel(null), "");
});

// ─── offsetLabel — 출처 사슬(원문 어디인가) ─────────────────────────

test("offsetLabel: 유효 구간은 천 단위 구분", () => {
  assert.equal(offsetLabel(SAMPLE.hits[0]), "1,234–1,567");
});

test("offsetLabel: 무효 구간(끝<=시작·음수·누락)은 빈 문자열", () => {
  assert.equal(offsetLabel({ char_start: 100, char_end: 100 }), "");
  assert.equal(offsetLabel({ char_start: 200, char_end: 100 }), "");
  assert.equal(offsetLabel({ char_start: -1, char_end: 50 }), "");
  assert.equal(offsetLabel({}), "");
  assert.equal(offsetLabel(null), "");
});

// ─── 점수 막대 — 0 나눗셈·클램프 ────────────────────────────────────

test("maxScore: 최댓값, 빈 목록은 0", () => {
  assert.equal(maxScore(SAMPLE.hits), 0.032);
  assert.equal(maxScore([]), 0);
  assert.equal(maxScore(null), 0);
  assert.equal(maxScore([{ score: "x" }, { score: null }]), 0);
});

test("scoreBarPct: 최댓값 대비 비율", () => {
  assert.equal(scoreBarPct(SAMPLE.hits[0], 0.032), 100);
  assert.equal(scoreBarPct(SAMPLE.hits[1], 0.032), 50);
});

test("scoreBarPct: max 0·음수·비정상은 0 (0 나눗셈 차단)", () => {
  assert.equal(scoreBarPct({ score: 1 }, 0), 0);
  assert.equal(scoreBarPct({ score: -1 }, 5), 0);
  assert.equal(scoreBarPct({}, 5), 0);
  assert.equal(scoreBarPct(null, 5), 0);
});

test("scoreBarPct: 최댓값 초과는 100 으로 클램프", () => {
  assert.equal(scoreBarPct({ score: 10 }, 5), 100);
});

// ─── buildRetrieveGraph — 질의 중심 서브그래프 뷰모델 ────────────────
//
// 배경(실측): 탐색 탭의 개체-개체 엣지는 성기다(고립 62.9%였다). 연결의 실체는
// 근거 링크(노드↔청크)가 나른다(이분 그래프 고립 3.6%). /retrieve 응답에는 그
// 질의 중심 서브그래프가 이미 들어 있다 — 진입(점수)·확장(via·from)·청크(근거).
// 사용자가 "청약 철회 기간으로 검색했는데 연결을 보고 싶다"고 한 그 요구다.

import { buildRetrieveGraph } from "../app/ontology-admin/retrieveView.mjs";

const GRAPH_SAMPLE = {
  query: "청약 철회 기간",
  expansion: {
    entry_nodes: [
      { node_id: "T:청약철회", score: 0.7 },
      { node_id: "T:청약", score: 0.9 },
    ],
    expanded_nodes: [
      { node_id: "T:철회기간", via: "definesTerm", from: "T:청약철회", name: "철회기간" },
      { node_id: "T:떠돌이", via: "propagation" },           // from 없음 (확산)
    ],
  },
  hits: [
    { chunk_id: "c1", source: "약관.pdf", section: "제19조",
      node_ids: ["T:청약철회", "T:무관노드"], matched_via: ["T:청약"], score: 0.03,
      text: "청약을 한 날부터 15일 이내에 철회할 수 있습니다.",
      channels: ["chunk", "graph"], char_start: 120, char_end: 180 },
    { chunk_id: "c2", source: "요약서.pdf", section: "",
      node_ids: [], matched_via: [], score: 0.01 },
  ],
};

test("buildRetrieveGraph: 진입 노드는 점수 내림차순, kind=entry", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE);
  const entries = g.nodes.filter((n) => n.kind === "entry");
  assert.deepEqual(entries.map((n) => n.id), ["T:청약", "T:청약철회"]);
});

test("buildRetrieveGraph: 확장 노드는 from→node 엣지에 via 라벨", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE);
  assert.deepEqual(g.edges.expansion,
    [{ from: "T:청약철회", to: "T:철회기간", via: "definesTerm" }]);
});

test("buildRetrieveGraph: from 없는 확장(확산)도 노드로는 남는다", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE);
  const n = g.nodes.find((x) => x.id === "T:떠돌이");
  assert.ok(n && n.kind === "expanded" && n.via === "propagation");
});

test("buildRetrieveGraph: 근거 엣지 = (matched_via ∪ node_ids) ∩ 표시된 노드", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE);
  const c1 = g.edges.evidence.filter((e) => e.chunk === "c1").map((e) => e.node).sort();
  // T:무관노드 는 표시 노드가 아니므로 엣지 없음
  assert.deepEqual(c1, ["T:청약", "T:청약철회"]);
});

test("buildRetrieveGraph: 노드 연결 없는 청크는 direct(임베딩 채널)로 표시", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE);
  assert.deepEqual(g.edges.direct, [{ chunk: "c2" }]);
});

test("buildRetrieveGraph: 청크 라벨은 조항(§) 우선·없으면 source, 순위 유지", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE);
  assert.equal(g.chunks[0].label, "§제19조");     // 파일명은 문서 층이 나른다
  assert.equal(g.chunks[1].label, "요약서.pdf");  // 조항 없으면 source 로 폴백
  assert.deepEqual(g.chunks.map((c) => c.rank), [1, 2]);
});

test("buildRetrieveGraph: 청크는 상세(원문·score·채널·오프셋)를 들고 다닌다", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE);
  const c1 = g.chunks[0];
  assert.equal(c1.source, "약관.pdf");
  assert.equal(c1.section, "제19조");
  assert.ok(c1.text.includes("15일 이내"));
  assert.equal(c1.score, 0.03);
  assert.deepEqual(c1.channels, ["chunk", "graph"]);
  assert.equal(c1.charStart, 120);
  assert.equal(c1.charEnd, 180);
  // 없는 필드는 안전 기본값 — 화면이 죽지 않는다
  const c2 = g.chunks[1];
  assert.equal(c2.text, "");
  assert.deepEqual(c2.channels, []);
  assert.equal(c2.charStart, null);
});

test("buildRetrieveGraph: documents = source 를 첫 등장 순으로 묶은 문서 층", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE);
  assert.deepEqual(g.documents, [
    { source: "약관.pdf", count: 1 },
    { source: "요약서.pdf", count: 1 },
  ]);
});

test("buildRetrieveGraph: 같은 문서의 청크 여럿이면 count 로 합산", () => {
  const multi = JSON.parse(JSON.stringify(GRAPH_SAMPLE));
  multi.hits.push({ chunk_id: "c3", source: "약관.pdf", section: "제20조",
                    node_ids: [], matched_via: [] });
  const g = buildRetrieveGraph(multi);
  assert.deepEqual(g.documents,
    [{ source: "약관.pdf", count: 2 }, { source: "요약서.pdf", count: 1 }]);
});

test("buildRetrieveGraph: documents 는 표시된 청크만 센다 (dropped 제외)", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE, { maxChunks: 1 });
  assert.deepEqual(g.documents, [{ source: "약관.pdf", count: 1 }]);
});

test("buildRetrieveGraph: maxChunks 상한은 dropped 로 보고 (조용한 절단 금지)", () => {
  const g = buildRetrieveGraph(GRAPH_SAMPLE, { maxChunks: 1 });
  assert.equal(g.chunks.length, 1);
  assert.equal(g.dropped.chunks, 1);
});

test("buildRetrieveGraph: 진입과 확장에 같은 노드가 오면 entry 가 이긴다", () => {
  const dup = JSON.parse(JSON.stringify(GRAPH_SAMPLE));
  dup.expansion.expanded_nodes.push({ node_id: "T:청약", via: "is_a↑", from: "T:청약철회" });
  const g = buildRetrieveGraph(dup);
  const nodes = g.nodes.filter((n) => n.id === "T:청약");
  assert.equal(nodes.length, 1);
  assert.equal(nodes[0].kind, "entry");
});

test("buildRetrieveGraph: 쓰레기 입력은 빈 모델 (절대 던지지 않음)", () => {
  for (const bad of [null, undefined, {}, { hits: "x" }, { expansion: 3 }]) {
    const g = buildRetrieveGraph(bad);
    assert.deepEqual(g.nodes, []);
    assert.deepEqual(g.chunks, []);
  }
});

test("buildRetrieveGraph: 결정적 — 같은 입력이면 같은 모델", () => {
  assert.deepEqual(buildRetrieveGraph(GRAPH_SAMPLE), buildRetrieveGraph(GRAPH_SAMPLE));
});
