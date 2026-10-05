/**
 * reviewQueueView — 검수 큐 보드의 순수 뷰모델.
 *
 * 보드의 질문: **"오늘 뭘 검수해야 하나."** 병목이 사람 검수로 넘어온 뒤
 * (재라벨·중복·draft) 큐가 JSON 응답 속에 흩어져 있어 아무 화면도 이 질문에
 * 답하지 못했다. 여기 함수들이 그 큐들을 카드·행으로 바꾼다 — 렌더 없이
 * 단위 테스트한다 (retrieveView·coverageView 와 같은 이유).
 *
 * 모든 함수는 던지지 않는다.
 */

const isObj = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const arr = (v) => (Array.isArray(v) ? v : []);
const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);

/**
 * /review/queues 응답 → 카드 목록. count>0 인 실행형 큐는 tone:"warn" —
 * 카드의 색이 곧 "여기 일이 있다"는 신호다. 정보형(골든 verified 등)은 카드로
 * 만들지 않는다 — 큐가 아니라 상태이기 때문.
 */
export function buildQueueCards(resp) {
  const q = isObj(resp) && isObj(resp.queues) ? resp.queues : {};
  const dup = isObj(q.duplicates) ? q.duplicates : {};
  const golden = isObj(q.golden) ? q.golden : {};
  const cards = [
    { key: "node_review", label: "노드 검수", count: num(q.node_review),
      desc: "LLM 추출 노드의 confirm / reject" },
    { key: "dup_variant", label: "중복 · 표기 변형", count: num(dup.variant),
      desc: "병합 제안 동봉 (기계 판정 가능)" },
    { key: "dup_cross", label: "중복 · 타입 상이", count: num(dup.cross_type),
      desc: "같은 이름, 다른 타입 — 사람 판단" },
    { key: "dup_similar", label: "중복 · 유사", count: num(dup.similar),
      desc: "포함 관계 등 — C73 주의 동봉" },
    { key: "golden_draft", label: "골든셋 draft", count: num(golden.draft),
      desc: "왕복 검증·확정 대기 (골든 탭)" },
    { key: "structural", label: "구조 단위 재분류", count: num(q.structural),
      desc: "조항·문서 후보 — 타입 재분류 (P-1 탐지)" },
    { key: "overdue", label: "기한 초과 deprecated", count: num(q.lifecycle_overdue),
      desc: "삭제 기한이 지난 노드" },
    // count null = 재라벨 카드와 같은 온디맨드 계약 — 제안 생성이 청크당
    // LLM 1콜이라 비싸다. 없는 수를 지어내지 않는다.
    { key: "relations", label: "관계 제안", count: null,
      desc: "관계 백필 제안 — 청크당 LLM 1콜, 계산 버튼" },
    { key: "relabel", label: "재라벨 후보", count: null,
      desc: "평가 실행 후 1위 아닌 케이스 — 계산 버튼" },
  ];
  for (const c of cards) c.tone = c.count > 0 ? "warn" : "ok";
  return cards;
}

/**
 * 평가 응답(per_case) → 재라벨 후보 행. retrieve 1위가 아닌 케이스만 —
 * 1위인 케이스는 라벨이 옳다는 뜻이라 검수 대상이 아니다. 순위 오름차순
 * (rank 2 가 rank 미검출보다 먼저 — 고치기 쉬운 것부터).
 */
export function relabelRows(evalResp) {
  const pc = isObj(evalResp) ? arr(evalResp.per_case) : [];
  return pc
    .filter((c) => isObj(c) && c.retrieve_rank !== 1)
    .map((c) => ({
      caseId: String(c.case_id || ""),
      query: String(c.query || ""),
      expected: (Array.isArray(c.expected) ? c.expected : [c.expected])
        .filter(Boolean).map(String),
      // Number(null)===0 함정 — null(미검출)을 0위로 만들면 정렬이 거짓이 된다.
      rank: c.retrieve_rank != null && Number.isFinite(Number(c.retrieve_rank))
        ? Number(c.retrieve_rank) : null,
    }))
    .sort((a, b) => (a.rank ?? 1e9) - (b.rank ?? 1e9));
}

/**
 * POST /review/relations 응답 → 관계 제안 행. 정렬은 판단이 급한 것부터:
 * ① 과거 기각 재출현(prevRejected — 번복 여부의 경고를 먼저 봐야 한다)
 * ② 신규 시그니처(signatureSeen=false) ③ 나머지. 같은 밴드 안은 응답
 * 순서 유지 (Array.sort 는 안정적).
 *
 * quote 는 표시용 240자 절단 — 승인 API 는 인용을 청크 원문과 **재대조**
 * 하므로(quote_not_found 관문) 무절단 quoteFull 을 따로 들고 다닌다.
 * 절단본을 보내도 부분문자열이라 통과는 하지만, 서로게이트 절단 등
 * 경계 사고를 API 에 흘리지 않는다.
 */
export function relationRows(proposeResp) {
  const props = isObj(proposeResp) ? arr(proposeResp.proposals) : [];
  const rows = props.filter(isObj).map((p) => {
    const pr = isObj(p.previously_rejected) ? p.previously_rejected : null;
    const quoteFull = String(p.evidence_quote || "");
    return {
      subject: String(p.subject || ""),
      predicate: String(p.predicate || ""),
      object: String(p.object || ""),
      quote: quoteFull.slice(0, 240),
      quoteFull,
      section: String(p.section || ""),
      chunkId: String(p.chunk_id || ""),
      signatureSeen: !!p.signature_seen,
      signature: String(p.signature || ""),
      namesInQuote: arr(p.names_in_quote).map(String),
      prevRejected: pr
        ? { reason: String(pr.reason || ""), at: String(pr.at || "") }
        : null,
    };
  });
  const band = (r) => (r.prevRejected ? 0 : r.signatureSeen ? 2 : 1);
  return rows.sort((a, b) => band(a) - band(b));
}

/** /review/duplicates 응답 → kind 별 분리 (없는 kind 는 빈 배열). */
export function splitClusters(dupResp) {
  const out = { variant: [], cross_type: [], similar: [] };
  const clusters = isObj(dupResp) ? arr(dupResp.clusters) : [];
  for (const c of clusters) {
    if (isObj(c) && out[c.kind]) out[c.kind].push(c);
  }
  return out;
}

/**
 * 1위 청크의 노드 중 "정답으로 추가" 후보 — 이미 정답인 노드는 버튼이
 * 무의미하므로 뺀다 (accept API 도 거부한다 — 미리 걸러 왕복을 아낀다).
 */
export function acceptCandidates(hit, expected) {
  const exp = new Set(arr(expected).map(String));
  return arr(isObj(hit) ? hit.node_ids : null).map(String)
    .filter((n) => !exp.has(n));
}

// ── 트리아지 (C3) ──────────────────────────────────────────────────

// 서버 밴드 키 → 렌더 키. 밴드 이름은 서버 계약(review_triage)의 것이고,
// 화면은 짧은 키로 다룬다 — 모르는 밴드는 조용히 버리지 않고 여기 없으면
// 버려진다는 사실이 이 표 한 곳에 보인다.
const BAND_KEYS = {
  strong_confirm: "confirm",
  strong_reject: "reject",
  borderline: "borderline",
};

const emptyBands = () => ({ confirm: [], reject: [], borderline: [] });

/**
 * POST /review/triage 응답 → 렌더용 밴드 구조.
 *
 * quote 는 relationRows 와 같은 이유로 240자 절단(표시용)이고, 관계 항목만
 * quoteFull 을 따로 든다 — 관계 승인 API 는 인용을 원문과 재대조하므로
 * 무절단본이 필요하다. 노드 판정(judge-batch kind:node)은 인용을 보내지
 * 않으므로 노드 항목에는 quoteFull 이 없다.
 *
 * counts 는 응답의 counts 를 믿지 않고 매핑된 배열에서 다시 센다 —
 * 화면의 "N건 일괄 승인" 버튼 숫자와 실제 행 수가 갈라지면 안 된다.
 * 쓰레기 입력 → 빈 구조 (안 던짐).
 */
export function triageBands(triageResp) {
  const resp = isObj(triageResp) ? triageResp : {};

  const nodeItem = (it, band) => ({
    nodeId: String(it.node_id || ""),
    name: String(it.name || it.node_id || ""),
    band,
    reasons: arr(it.reasons).map(String),
    verdict: it.verdict == null ? null : String(it.verdict),
    rationale: String(it.rationale || ""),
    quote: String(it.evidence_quote || "").slice(0, 240),
    evidenceCount: num(it.evidence_count),
  });
  const relItem = (it, band) => {
    const quoteFull = String(it.evidence_quote || "");
    return {
      subject: String(it.subject || ""),
      predicate: String(it.predicate || ""),
      object: String(it.object || ""),
      chunkId: String(it.chunk_id || ""),
      band,
      reasons: arr(it.reasons).map(String),
      quote: quoteFull.slice(0, 240),
      quoteFull,
    };
  };

  const mapBands = (src, mapItem) => {
    const out = emptyBands();
    if (!isObj(src)) return out;
    for (const [serverKey, key] of Object.entries(BAND_KEYS)) {
      out[key] = arr(src[serverKey]).filter(isObj)
        .map((it) => mapItem(it, key));
    }
    return out;
  };

  const nodes = mapBands(resp.nodes, nodeItem);
  const relations = isObj(resp.relations)
    ? mapBands(resp.relations, relItem) : null;

  const countOf = (bands) => ({
    confirm: bands.confirm.length,
    reject: bands.reject.length,
    borderline: bands.borderline.length,
  });
  const counts = { nodes: countOf(nodes) };
  if (relations) counts.relations = countOf(relations);

  return { nodes, relations, counts, llmCalls: num(resp.llm_calls) };
}

/**
 * judge-batch dry-run 응답 → confirm 문구용 요약.
 * would_* 가 적용 예정이고, 나머지(skipped/blocked)는 reason 별로 센다 —
 * 첫 출현 순서 유지 (어떤 사유가 왜 나왔는지 응답 순서대로 읽힌다).
 */
export function batchPreviewSummary(judgeResp) {
  const results = isObj(judgeResp) ? arr(judgeResp.results) : [];
  let willApply = 0;
  const byReason = new Map();
  for (const r of results) {
    if (!isObj(r)) continue;
    const status = String(r.status || "");
    if (status.startsWith("would_")) { willApply += 1; continue; }
    const reason = String(r.reason || status || "unknown");
    byReason.set(reason, (byReason.get(reason) || 0) + 1);
  }
  return {
    willApply,
    skipped: [...byReason].map(([reason, n]) => ({ reason, n })),
  };
}

/**
 * GET /review/recommendation-quality 응답 → 일괄 승인의 안전핀 배지.
 *
 * **렌즈(actor)별로만 읽는다 — overall 로 폴백하지 않는다.** 게이트가 묻는
 * 것은 "이 렌즈의 추천을 믿어도 되는가"이고, 다른 렌즈의 표본으로 그 답을
 * 지어내면 자 없는 자동화가 된다 (설계 §5: 표본 n<20 이면 일괄 버튼 비활성).
 * n==0 이면 서버가 rate=null 을 주고(0.0 오보고 금지 규율), 여기서도
 * rate 는 null 로 남는다.
 */
export function agreementBadge(qualityResp, actor = "evidence_checker") {
  const perActor = isObj(qualityResp) && isObj(qualityResp.per_actor)
    ? qualityResp.per_actor : {};
  const src = isObj(perActor[actor]) ? perActor[actor] : {};
  const n = num(src.n);
  const rate = src.rate != null && Number.isFinite(Number(src.rate))
    ? Number(src.rate) : null;
  const enough = n >= 20 && rate !== null;   // 측정 없는 게이트 통과는 없다
  const label = enough
    ? `사람-일치율 ${(rate * 100).toFixed(1)}% (n=${n})`
    : `표본 부족 (n=${n}) — 개별 검수 권장`;
  return { rate, n, enough, label };
}
