"use client";

// 검수 큐 보드 — "오늘 뭘 검수해야 하나"의 한 화면.
//
// 병목이 사람 검수로 넘어온 뒤(재라벨·중복·draft) 큐가 JSON 응답 속에 흩어져
// 있어 아무 화면도 이 질문에 답하지 못했다. 골격은 커버리지 지도에서 검증한
// Overview → Zoom → Detail:
//   ① 카드 — 큐별 카운트 (GET /review/queues, 결정적·LLM 0콜)
//   ② 패널 — 카드를 클릭한 큐의 항목 + 판단 재료 (근거 수·정의·출처·순위)
//   ③ 행동 — 판정 버튼이 기존 API 로 바로 이어진다: confirm/reject(노드),
//     merge(variant, dry-run 미리보기 후 적용), accept(재라벨).
//
// 재라벨 큐만 카운트가 null 로 시작한다 — 평가 실행(케이스당 회수)이 필요해
// 비싸므로 버튼으로 온디맨드 계산한다. 없는 수를 지어내지 않는다.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  buildQueueCards, relabelRows, relationRows, splitClusters, acceptCandidates,
  triageBands, batchPreviewSummary, agreementBadge,
} from "./reviewQueueView.mjs";

const INK = "var(--ink)";
const INK_SOFT = "var(--ink-2)";
const WARN = "#f2946a";
const LINE = "rgba(255,255,255,.08)";
const ACTOR = "admin-console";
// 트리아지 렌즈 actor — recommend 이벤트와 일치율 자(per_actor)가 같은 이름을
// 봐야 게이트가 "이 렌즈"의 표본을 잰다.
const TRIAGE_ACTOR = "evidence_checker";

// 관계 제안 행의 안정 키 — 트리플만으로는 부족하다 (같은 트리플이 다른
// 청크에서 재제안될 수 있고 그 구분이 묘비 스코프의 본체다).
const relKey = (r) => `${r.subject}|${r.predicate}|${r.object}|${r.chunkId}`;

function Card({ card, active, onClick }) {
  const warn = card.tone === "warn";
  return (
    <button onClick={onClick}
      style={{
        flex: "1 1 130px", minWidth: 130, textAlign: "left", cursor: "pointer",
        padding: "10px 12px", borderRadius: 10,
        border: active ? "2px solid #2a78d6" : `1px solid ${LINE}`,
        background: active ? "rgba(42,120,214,.14)" : "var(--surface)",
      }}>
      <div style={{ fontSize: 22, fontWeight: 800, fontVariantNumeric: "tabular-nums",
                    color: card.count === null ? INK_SOFT : warn ? WARN : "#4cc9a4" }}>
        {card.count === null ? "?" : card.count}
      </div>
      <div style={{ color: INK, fontSize: 12.5, fontWeight: 600 }}>{card.label}</div>
      <div style={{ color: INK_SOFT, fontSize: 11 }}>{card.desc}</div>
    </button>
  );
}

/* 중복 클러스터 한 줄 — 판단 재료(정의·근거 수·출처·생애주기)를 나란히. */
function ClusterRow({ cluster, onMerge, busy }) {
  const s = cluster.suggested;
  return (
    <div style={{ borderBottom: `1px solid ${LINE}`, padding: "8px 4px" }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        {(cluster.members || []).map((m) => (
          <span key={m.node_id} className="mono" title={m.definition || m.node_id}
                style={{ fontSize: 11.5, padding: "2px 8px", borderRadius: 10,
                         background: "rgba(42,120,214,.12)", color: INK,
                         border: `1px solid ${LINE}` }}>
            {m.node_id} <span style={{ color: INK_SOFT }}>근거{m.evidence_chunks ?? 0}</span>
          </span>
        ))}
        {s && onMerge && (
          <button className="oa-rt-more" disabled={busy}
                  onClick={() => onMerge(s)}
                  title={`승자 ${s.winner} ← ${s.losers?.join(", ")}`}>
            병합 (승자: {String(s.winner).split(":").slice(1).join(":")})
          </button>
        )}
      </div>
      {(cluster.members || []).some((m) => m.definition) && (
        <div style={{ marginTop: 4, color: INK_SOFT, fontSize: 11.5 }}>
          {(cluster.members || []).map((m) => m.definition
            ? <div key={m.node_id}>· <b style={{ color: INK }}>{m.name}</b>: {String(m.definition).slice(0, 110)}</div>
            : null)}
        </div>
      )}
    </div>
  );
}

/* 트리아지 밴드 그룹 (C3) — 헤더 색이 곧 추천 방향. 행의 판정 버튼은
   기존 judgeNode 경로 그대로다 (판정 쓰기 두 벌 금지 — 일괄도 개별도
   결국 confirm/reject API 하나로 모인다). */
function TriageBand({ title, color, rows, note, action, done, busy, onJudge }) {
  return (
    <div style={{ marginTop: 10, border: `1px solid ${LINE}`, borderRadius: 8 }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap",
                    padding: "6px 10px", borderBottom: `1px solid ${LINE}` }}>
        <b style={{ color, fontSize: 12.5 }}>{title} · {rows.length}건</b>
        {note && <span style={{ color: INK_SOFT, fontSize: 11.5 }}>{note}</span>}
        {action}
      </div>
      {rows.length === 0
        ? <p className="hint" style={{ padding: "6px 10px", margin: 0 }}>이 밴드는 비어 있다.</p>
        : rows.map((r) => (
          <div key={r.nodeId}
               style={{ borderBottom: `1px solid ${LINE}`, padding: "7px 10px",
                        display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <span className="mono" style={{ color: INK, fontSize: 12 }}>{r.nodeId}</span>
            {r.verdict && (
              <span style={{ fontSize: 11, padding: "1px 7px", borderRadius: 9,
                             border: `1px solid ${LINE}`, color }}>
                추천: {r.verdict}
              </span>
            )}
            {r.reasons.map((reason, i) => (
              <span key={i}
                    style={{ fontSize: 10.5, padding: "1px 6px", borderRadius: 8,
                             border: `1px solid ${LINE}`, color: INK_SOFT }}>
                {reason}
              </span>
            ))}
            {r.rationale && (
              <span style={{ color: INK_SOFT, fontSize: 11.5 }}
                    title={r.quote || undefined}>
                {String(r.rationale).slice(0, 90)}
              </span>
            )}
            <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
              {done[r.nodeId]
                ? <span style={{ color: INK, fontSize: 12 }}>{done[r.nodeId]}</span>
                : <>
                    <button className="oa-rt-more" disabled={busy}
                            onClick={() => onJudge(r.nodeId, "confirm")}>확정</button>
                    <button className="oa-rt-more" style={{ color: WARN }} disabled={busy}
                            onClick={() => onJudge(r.nodeId, "reject")}>거절</button>
                  </>}
            </span>
          </div>
        ))}
    </div>
  );
}

export default function ReviewQueueBoard({ api, namespace }) {
  const [queues, setQueues] = useState(null);
  const [open, setOpen] = useState(null);          // 카드 key
  const [busyMsg, setBusyMsg] = useState("");
  const [err, setErr] = useState("");
  // 패널 데이터 (열 때 로드)
  const [nodes, setNodes] = useState(null);        // 노드 검수 항목
  const [dups, setDups] = useState(null);          // 중복 클러스터
  const [relabel, setRelabel] = useState(null);    // 재라벨 행
  const [structural, setStructural] = useState(null); // 구조 단위 후보 (P-1)
  const [structType, setStructType] = useState("");   // 재분류 타깃 타입 (P-3)
  const [relations, setRelations] = useState(null);   // 관계 제안 행 (온디맨드 — LLM)
  const [relFiltered, setRelFiltered] = useState(0);  // 묘비에 걸러진 재출현 수
  const [relReject, setRelReject] = useState({});     // relKey → {reason, evidenceOnly}
  const [evidence, setEvidence] = useState({});    // caseId → 1위 히트
  const [done, setDone] = useState({});            // 항목 id → 판정 결과 문구
  const [rejectReason, setRejectReason] = useState("");
  const [triage, setTriage] = useState(null);      // triageBands 결과 (C3)
  const [quality, setQuality] = useState(null);    // agreementBadge — 일괄 승인 안전핀

  const base = `${api}/graphs/${encodeURIComponent(namespace)}`;

  const loadQueues = useCallback(() => {
    fetch(`${base}/review/queues`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setQueues)
      .catch((e) => setErr(String(e?.message || e)));
  }, [base]);

  useEffect(() => {
    if (!namespace) return;
    setQueues(null); setOpen(null); setNodes(null); setDups(null);
    setRelabel(null); setRelations(null); setRelFiltered(0); setRelReject({});
    setEvidence({}); setDone({}); setErr("");
    setTriage(null); setQuality(null);
    loadQueues();
  }, [namespace, loadQueues]);

  const cards = useMemo(() => buildQueueCards(queues || {}), [queues]);

  const openPanel = useCallback(async (key) => {
    setOpen((k) => (k === key ? null : key));
    setErr("");
    try {
      if (key === "node_review" && nodes === null) {
        const r = await fetch(`${base}/review?limit=30`).then((x) => x.json());
        setNodes(r.items || []);
      }
      if ((key === "dup_variant" || key === "dup_cross" || key === "dup_similar") && dups === null) {
        const r = await fetch(`${base}/review/duplicates`).then((x) => x.json());
        setDups(splitClusters(r));
      }
      if (key === "structural" && structural === null) {
        const r = await fetch(`${base}/review/structural`).then((x) => x.json());
        setStructural(r.candidates || []);
      }
    } catch (e) { setErr(String(e?.message || e)); }
  }, [base, nodes, dups, structural]);

  // 재라벨 후보 계산 — 평가 1회 (LLM 0콜, 케이스당 회수라 수십 초 걸릴 수 있음)
  const computeRelabel = useCallback(async () => {
    setBusyMsg("평가 실행 중 (케이스당 회수 — 수십 초 걸릴 수 있다)…");
    try {
      const r = await fetch(`${base}/qa/evaluate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ k: 5, target: "evidence",
                               statuses: ["confirmed", "verified"] }),
      }).then((x) => x.json());
      setRelabel(relabelRows(r));
      setOpen("relabel");
    } catch (e) { setErr(String(e?.message || e)); }
    setBusyMsg("");
  }, [base]);

  // 관계 제안 생성 — 청크당 LLM 1콜이라 비싸다 (카드 count 도 null 온디맨드).
  // 판정 후에도 재생성하지 않고 로컬 done 마킹만 한다.
  const computeRelations = useCallback(async () => {
    setBusyMsg("관계 제안 생성 중 (청크당 LLM 1콜 — 수십 초 걸릴 수 있다)…");
    setErr("");
    try {
      const r = await fetch(`${base}/review/relations`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      }).then((x) => x.json());
      if (r && r.error) throw new Error(r.detail || r.error);
      setRelations(relationRows(r));
      setRelFiltered(Number(r?.rejected_filtered) || 0);
      setOpen("relations");
    } catch (e) { setErr(String(e?.message || e)); }
    setBusyMsg("");
  }, [base]);

  // 관계 승인 — dry-run 미리보기 = 적용 계약 (merge 와 동일). 과거 기각 행은
  // override_rejected 를 명시해야만 묘비를 걷는다 — confirm 문구로 번복을 알린다.
  const approveRelation = useCallback(async (row) => {
    const key = relKey(row);
    setBusyMsg("승인 미리보기…");
    setErr("");
    try {
      const rel = { subject: row.subject, predicate: row.predicate,
                    object: row.object, chunk_id: row.chunkId,
                    // 서버가 인용을 청크 원문과 재대조한다 — 무절단 원문을 보낸다
                    evidence_quote: row.quoteFull,
                    ...(row.prevRejected ? { override_rejected: true } : {}) };
      const body = { relations: [rel], actor: ACTOR };
      const prev = await fetch(`${base}/review/relations/approve`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, dry_run: true }),
      }).then((x) => x.json());
      if (!(prev.added_total > 0)) {
        const why = ((prev.skipped || [])[0] || {}).reason || prev.error || "불가";
        setDone((d) => ({ ...d, [key]: `⚠ ${why}` }));
        setBusyMsg(""); return;
      }
      const msg = `관계 추가 미리보기:\n${row.subject} —${row.predicate}→ ${row.object}\n`
        + (row.prevRejected
            ? `\n⚠ 과거 기각(${row.prevRejected.reason})을 번복합니다.\n` : "")
        + `\n엣지 ${prev.added_total}건 추가 — 적용할까?`;
      if (!window.confirm(msg)) { setBusyMsg(""); return; }
      const r = await fetch(`${base}/review/relations/approve`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, dry_run: false }),
      }).then((x) => x.json());
      setDone((d) => ({ ...d, [key]: r.added_total > 0 ? "✓ 추가됨" : "⚠ 실패" }));
      if (r.added_total > 0) loadQueues();
    } catch (e) { setErr(String(e?.message || e)); }
    setBusyMsg("");
  }, [base, loadQueues]);

  // 관계 기각 — reason 필수 (이유 없는 기각은 감사가 아니다, 서버도 거부).
  // "이 인용만" = scope evidence: 다른 인용의 재제안은 표시하며 통과시킨다.
  const rejectRelation = useCallback(async (row) => {
    const key = relKey(row);
    const inp = relReject[key] || {};
    const reason = String(inp.reason || "").trim();
    if (!reason) { setErr("기각 사유를 입력하라 — 관계판 묘비에 기록된다."); return; }
    setErr("");
    try {
      const body = { relations: [{ subject: row.subject, predicate: row.predicate,
        object: row.object, reason, chunk_id: row.chunkId,
        scope: inp.evidenceOnly ? "evidence" : "triple" }], actor: ACTOR };
      const r = await fetch(`${base}/review/relations/reject`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((x) => x.json());
      setDone((d) => ({ ...d, [key]: r.rejected_total > 0 ? "✕ 기각됨" : "⚠ 실패" }));
      if (r.rejected_total > 0) loadQueues();
    } catch (e) { setErr(String(e?.message || e)); }
  }, [base, relReject, loadQueues]);

  const showEvidence = useCallback(async (row) => {
    if (evidence[row.caseId]) return;
    try {
      const r = await fetch(`${base}/retrieve?query=${encodeURIComponent(row.query)}&top_k=1`)
        .then((x) => x.json());
      setEvidence((e) => ({ ...e, [row.caseId]: (r.hits || [])[0] || { none: true } }));
    } catch { /* 실패 시 버튼 다시 누르면 재시도 */ }
  }, [base, evidence]);

  const judgeNode = useCallback(async (nodeId, verdict) => {
    const body = { node_id: nodeId, actor: ACTOR,
                   ...(verdict === "reject" ? { reason: rejectReason || "검수 큐 보드에서 거절" } : {}) };
    try {
      const r = await fetch(`${base}/review/${verdict}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setDone((d) => ({ ...d, [nodeId]: r.ok ? (verdict === "confirm" ? "✓ 확정" : "✕ 거절") : "⚠ 실패" }));
      if (r.ok) loadQueues();
    } catch { setDone((d) => ({ ...d, [nodeId]: "⚠ 실패" })); }
  }, [base, rejectReason, loadQueues]);

  // 사전판정 (C3) — 트리아지 + 일치율 자를 함께 로드. 방금 만든 추천은
  // 아직 판정 전(pending)이라 자의 n 에 안 들어간다 — 병렬 로드 무해.
  const runTriage = useCallback(async () => {
    setBusyMsg("사전판정 실행 중 (노드당 LLM ≤1콜 × limit 30)…");
    setErr("");
    try {
      const [t, q] = await Promise.all([
        fetch(`${base}/review/triage`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ limit: 30, include_relations: false,
                                 actor: TRIAGE_ACTOR }),
        }).then((x) => x.json()),
        fetch(`${base}/review/recommendation-quality`).then((x) => x.json()),
      ]);
      if (t && t.error) throw new Error(t.detail || t.error);
      setTriage(triageBands(t));
      setQuality(agreementBadge(q, TRIAGE_ACTOR));
    } catch (e) { setErr(String(e?.message || e)); }
    setBusyMsg("");
  }, [base]);

  // 일괄 판정 (C3) — dry-run 미리보기 == 적용 계약 (merge 와 동일 흐름).
  // 서버가 본문을 불신하고(최신 추천 재대조) skip 사유를 돌려주므로,
  // 미리보기 confirm 문구에 그 사유들을 그대로 보여준다. 적용에서
  // 실제 판정된 행만 done 마킹 — skip 행은 개별 버튼이 살아 있어야 한다.
  const bulkJudge = useCallback(async (rows, verdict) => {
    const items = rows.filter((r) => !done[r.nodeId])
      .map((r) => ({ kind: "node", node_id: r.nodeId, verdict }));
    if (!items.length) return;
    const label = verdict === "confirm" ? "일괄 승인" : "일괄 거절";
    setBusyMsg(`${label} 미리보기…`);
    setErr("");
    try {
      const body = { items, actor: ACTOR };
      const prev = await fetch(`${base}/review/judge-batch`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, dry_run: true }),
      }).then((x) => x.json());
      if (prev && prev.error) throw new Error(prev.detail || prev.error);
      const sum = batchPreviewSummary(prev);
      const skipLines = sum.skipped.length
        ? `\nskip: ${sum.skipped.map((s) => `${s.reason} ${s.n}건`).join(" · ")}`
        : "";
      if (sum.willApply === 0) {
        setErr(`${label} 적용 가능 0건${skipLines || " — 전부 skip"}`);
        setBusyMsg(""); return;
      }
      if (!window.confirm(
        `${label} 미리보기:\n적용 ${sum.willApply}건${skipLines}\n\n적용할까?`)) {
        setBusyMsg(""); return;
      }
      const r = await fetch(`${base}/review/judge-batch`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, dry_run: false }),
      }).then((x) => x.json());
      if (r && r.error) throw new Error(r.detail || r.error);
      setDone((d) => {
        const nd = { ...d };
        for (const res of (r.results || [])) {
          if (res.status === "confirmed") nd[res.node_id] = "✓ 확정";
          else if (res.status === "rejected") nd[res.node_id] = "✕ 거절";
        }
        return nd;
      });
      if (r.applied > 0) loadQueues();
    } catch (e) { setErr(String(e?.message || e)); }
    setBusyMsg("");
  }, [base, done, loadQueues]);

  const mergeCluster = useCallback(async (suggested) => {
    setBusyMsg("병합 미리보기…");
    try {
      const body = { winner: suggested.winner, losers: suggested.losers, actor: ACTOR };
      const prev = await fetch(`${base}/nodes/merge`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, dry_run: true }),
      }).then((x) => x.json());
      const okMsg = `병합 미리보기: 근거 ${prev.evidence_moved ?? "?"}건 이동, `
        + `${(suggested.losers || []).length}개 노드 흡수 → 적용할까?`;
      if (!window.confirm(okMsg)) { setBusyMsg(""); return; }
      const r = await fetch(`${base}/nodes/merge`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, dry_run: false }),
      });
      setDone((d) => ({ ...d, [suggested.winner]: r.ok ? "✓ 병합됨" : "⚠ 실패" }));
      if (r.ok) { setDups(null); loadQueues(); }
    } catch (e) { setErr(String(e?.message || e)); }
    setBusyMsg("");
  }, [base, loadQueues]);

  // 구조 단위 재분류 (P-3) — 본문 불신은 서버가 한다 (탐지 재실행)
  const approveStructural = useCallback(async (nodeId) => {
    const t = structType.trim();
    if (!t) { setErr("타깃 타입을 먼저 입력하라 (예: Clause)"); return; }
    setBusyMsg("재분류 미리보기…");
    setErr("");
    try {
      const body = { items: [nodeId], new_type: t, actor: ACTOR };
      const prev = await fetch(`${base}/review/structural/approve`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, dry_run: true }),
      }).then((x) => x.json());
      const p = (prev.results || [])[0] || {};
      if (p.status !== "would_rename") {
        setDone((d) => ({ ...d, [nodeId]: `⚠ ${p.detail || p.reason || p.status || "불가"}` }));
        setBusyMsg(""); return;
      }
      const msg = `재분류 미리보기:\n${nodeId}\n→ ${p.new_id}\n\n`
        + `엣지 ${p.edges} · 청크 ${p.chunks} · 골든셋 ${p.golden} 재지정.\n`
        + `적용할까? (적용 후 /reindex 필요)`;
      if (!window.confirm(msg)) { setBusyMsg(""); return; }
      const r = await fetch(`${base}/review/structural/approve`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, dry_run: false }),
      }).then((x) => x.json());
      const a = (r.results || [])[0] || {};
      setDone((d) => ({ ...d, [nodeId]: a.status === "renamed" ? `✓ → ${a.new_id}` : "⚠ 실패" }));
      if (a.status === "renamed") {
        const rr = await fetch(`${base}/review/structural`).then((x) => x.json());
        setStructural(rr.candidates || []);
        loadQueues();
      }
    } catch (e) { setErr(String(e?.message || e)); }
    setBusyMsg("");
  }, [base, structType, loadQueues]);

  const acceptAnswer = useCallback(async (caseId, nodeId) => {
    try {
      const r = await fetch(`${base}/qa/cases/${encodeURIComponent(caseId)}/accept`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: nodeId }),
      });
      setDone((d) => ({ ...d, [`${caseId}:${nodeId}`]: r.ok ? "✓ 정답 추가" : "⚠ 실패" }));
    } catch { setDone((d) => ({ ...d, [`${caseId}:${nodeId}`]: "⚠ 실패" })); }
  }, [base]);

  const dupPanel = (kind, canMerge) => {
    const list = dups ? dups[kind] : null;
    if (list === null) return <p className="hint">불러오는 중…</p>;
    if (!list.length) return <p className="hint">이 큐는 비어 있다.</p>;
    return (
      <>
        {kind === "similar" && (
          <p style={{ color: WARN, fontSize: 12 }}>
            ⚠ 포함 관계("갑상선암" ⊂ "중증 갑상선암")는 중복이 아닐 수 있다 —
            문서가 구별한 것을 지우지 마라 (C73 교훈).
          </p>
        )}
        {list.map((c, i) => (
          <ClusterRow key={i} cluster={c} busy={!!busyMsg}
                      onMerge={canMerge ? mergeCluster : null} />
        ))}
      </>
    );
  };

  return (
    <section className="card">
      <h2>검수 큐 — 오늘 뭘 검수해야 하나</h2>
      <p style={{ marginTop: 2, color: INK_SOFT, fontSize: 13 }}>
        카드를 클릭하면 그 큐의 항목과 판단 재료가 열리고, 판정 버튼이 바로
        API 로 이어진다. 검수가 끝나는 만큼 골든셋 지표가 회복되는 것을
        골든 탭의 평가 이력에서 확인할 수 있다.
      </p>
      {err && <p className="hint" style={{ marginTop: 8 }}>⚠ {err}</p>}
      {busyMsg && <p style={{ color: INK_SOFT, marginTop: 8, fontSize: 12.5 }}>{busyMsg}</p>}

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
        {cards.map((c) => (
          <Card key={c.key} card={c} active={open === c.key}
                onClick={() => (c.key === "relabel" && relabel === null)
                  ? computeRelabel()
                  : (c.key === "relations" && relations === null)
                    ? computeRelations() : openPanel(c.key)} />
        ))}
      </div>

      {/* ── 패널 ── */}
      {open === "node_review" && (
        <div style={{ marginTop: 12 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6,
                        flexWrap: "wrap" }}>
            <button className="oa-rt-more" disabled={!!busyMsg} onClick={runTriage}>
              사전판정 실행
            </button>
            <span style={{ color: INK_SOFT, fontSize: 11.5 }}>
              렌즈 신호(근거대조·중복·일관성·구조단위) → 밴드. 추천이지 판정이
              아니다 — 판정은 이 화면의 버튼(사람)만 만든다.
            </span>
            {quality && (
              <span style={{ fontSize: 11.5, padding: "1px 8px", borderRadius: 9,
                             border: `1px solid ${LINE}`,
                             color: quality.enough ? "var(--green)" : WARN }}>
                이 렌즈의 {quality.label}
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
            <span style={{ color: INK_SOFT, fontSize: 12.5 }}>거절 사유(선택):</span>
            <input value={rejectReason} onChange={(e) => setRejectReason(e.target.value)}
                   placeholder="예: 오추출 — 원문에 없는 개체"
                   style={{ flex: 1, padding: "4px 8px", borderRadius: 6,
                            border: `1px solid ${LINE}`, background: "var(--surface)",
                            color: INK, fontSize: 12.5 }} />
          </div>
          {triage !== null ? (
            <>
              <TriageBand title="강한 확정 추천" color="var(--green)"
                rows={triage.nodes.confirm} done={done} busy={!!busyMsg}
                onJudge={judgeNode}
                action={triage.nodes.confirm.length > 0 && (
                  <button className="oa-rt-more" style={{ marginLeft: "auto" }}
                          disabled={!quality?.enough || !!busyMsg}
                          title={quality?.enough
                            ? "dry-run 미리보기 후 적용 — 서버가 최신 추천을 재대조한다"
                            : quality?.label || "일치율 자 미로드"}
                          onClick={() => bulkJudge(triage.nodes.confirm, "confirm")}>
                    추천 {triage.nodes.confirm.length}건 일괄 승인
                  </button>
                )} />
              <TriageBand title="강한 거절 추천" color="var(--red)"
                rows={triage.nodes.reject} done={done} busy={!!busyMsg}
                onJudge={judgeNode}
                action={triage.nodes.reject.length > 0 && (
                  <button className="oa-rt-more" style={{ marginLeft: "auto" }}
                          disabled={!quality?.enough || !!busyMsg}
                          title={quality?.enough
                            ? "dry-run 미리보기 후 적용 — 서버가 최신 추천을 재대조한다"
                            : quality?.label || "일치율 자 미로드"}
                          onClick={() => bulkJudge(triage.nodes.reject, "reject")}>
                    추천 {triage.nodes.reject.length}건 일괄 거절
                  </button>
                )} />
              <TriageBand title="경계 — 사람 판단" color={WARN}
                rows={triage.nodes.borderline} done={done} busy={!!busyMsg}
                onJudge={judgeNode}
                note="결정적 신호가 강등했거나 예측이 약하다 — 일괄 없음, 한 건씩" />
              <p className="hint" style={{ marginTop: 6 }}>
                LLM {triage.llmCalls}콜 사용. 일괄 버튼은 사람-일치율 표본
                n≥20 에서만 열린다 — 자 없이 자동화 없음.
              </p>
            </>
          ) : (
            <>
              {nodes === null ? <p className="hint">불러오는 중…</p>
                : nodes.length === 0 ? <p className="hint">검수 대기 노드가 없다.</p>
                : nodes.map((n) => (
                  <div key={n.node_id} style={{ borderBottom: `1px solid ${LINE}`,
                                                padding: "7px 4px", display: "flex",
                                                gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    <span className="mono" style={{ color: INK, fontSize: 12 }}>{n.node_id}</span>
                    <span style={{ color: INK_SOFT, fontSize: 11.5 }}>
                      근거 {n.evidence_count ?? 0} · {n.trust || "?"} ·
                      {String(n.definition || "").slice(0, 60) || " (정의 없음)"}
                    </span>
                    {n.recommendation && (
                      <span style={{ color: WARN, fontSize: 11.5 }}>추천: {n.recommendation}</span>
                    )}
                    <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                      {done[n.node_id]
                        ? <span style={{ color: INK, fontSize: 12 }}>{done[n.node_id]}</span>
                        : <>
                            <button className="oa-rt-more" onClick={() => judgeNode(n.node_id, "confirm")}>확정</button>
                            <button className="oa-rt-more" style={{ color: WARN }}
                                    onClick={() => judgeNode(n.node_id, "reject")}>거절</button>
                          </>}
                    </span>
                  </div>
                ))}
              <p className="hint" style={{ marginTop: 6 }}>상위 30건 표시 — 판정하면 큐에서 빠진다.</p>
            </>
          )}
        </div>
      )}

      {open === "dup_variant" && <div style={{ marginTop: 12 }}>{dupPanel("variant", true)}</div>}
      {open === "dup_cross" && <div style={{ marginTop: 12 }}>{dupPanel("cross_type", false)}</div>}
      {open === "dup_similar" && <div style={{ marginTop: 12 }}>{dupPanel("similar", false)}</div>}

      {open === "relabel" && relabel !== null && (
        <div style={{ marginTop: 12 }}>
          <p style={{ color: INK_SOFT, fontSize: 12.5 }}>
            retrieve 1위가 아닌 케이스 {relabel.length}건. "1위 근거 보기"로 실제
            1위 청크를 확인하고, 그 청크가 질의에 답하면 그 노드를 정답으로
            추가하라(라벨 노후화 처방 — 판정은 사람이 한다).
          </p>
          {relabel.map((row) => {
            const hit = evidence[row.caseId];
            return (
              <div key={row.caseId} style={{ borderBottom: `1px solid ${LINE}`, padding: "8px 4px" }}>
                <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                  <span style={{ color: INK, fontSize: 12.5, fontWeight: 600 }}>{row.query}</span>
                  <span style={{ color: row.rank ? INK_SOFT : WARN, fontSize: 11.5 }}>
                    {row.rank ? `현재 ${row.rank}위` : "미검출"}</span>
                  <span className="mono" style={{ color: INK_SOFT, fontSize: 11 }}>
                    기대: {row.expected.join(", ")}</span>
                  {!hit && <button className="oa-rt-more" style={{ marginLeft: "auto" }}
                                   onClick={() => showEvidence(row)}>1위 근거 보기</button>}
                </div>
                {hit && !hit.none && (
                  <div style={{ marginTop: 6, padding: "8px 10px", borderRadius: 6,
                                background: "var(--surface)", border: `1px solid ${LINE}` }}>
                    <div style={{ color: INK_SOFT, fontSize: 11.5 }}>
                      1위: {hit.source} {hit.section ? `§${hit.section}` : ""}</div>
                    <p style={{ color: INK, fontSize: 12.5, margin: "4px 0", lineHeight: 1.55 }}>
                      {String(hit.text || "").slice(0, 240)}…</p>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {acceptCandidates(hit, row.expected).map((n) => (
                        done[`${row.caseId}:${n}`]
                          ? <span key={n} style={{ color: INK, fontSize: 11.5 }}>{n} {done[`${row.caseId}:${n}`]}</span>
                          : <button key={n} className="oa-rt-more" title={n}
                                    onClick={() => acceptAnswer(row.caseId, n)}>
                              {n} ← 정답으로 추가
                            </button>
                      ))}
                    </div>
                  </div>
                )}
                {hit && hit.none && <p className="hint">이 질의는 히트가 없다.</p>}
              </div>
            );
          })}
        </div>
      )}

      {open === "structural" && (
        <div style={{ marginTop: 12 }}>
          <p style={{ color: INK_SOFT, fontSize: 12.5 }}>
            이름이 section 라벨·문서 제목과 <b>전체-정규화 동등</b>인 노드 —
            개념이 아니라 구조 단위(조항·문서)일 가능성. 판정은 사람이 한다:
            정의 절 헤딩과 동명인 <b>개념</b> 노드(예: 계약자)도 섞인다.
            재분류 = id 개명 (미리보기 → 적용, 적용 후 /reindex 필요).
          </p>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
            <span style={{ color: INK_SOFT, fontSize: 12.5 }}>타깃 타입:</span>
            <input value={structType} onChange={(e) => setStructType(e.target.value)}
                   placeholder="예: Clause (검수자가 정한다 — 서버 기본값 없음)"
                   style={{ flex: 1, maxWidth: 340, padding: "4px 8px", borderRadius: 6,
                            border: `1px solid ${LINE}`, background: "var(--surface)",
                            color: INK, fontSize: 12.5 }} />
          </div>
          {structural === null ? <p className="hint">불러오는 중…</p>
            : structural.length === 0 ? <p className="hint">이 큐는 비어 있다.</p>
            : structural.map((c) => (
              <div key={c.node_id} style={{ borderBottom: `1px solid ${LINE}`,
                                            padding: "7px 4px", display: "flex",
                                            gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <span className="mono" style={{ color: INK, fontSize: 12 }}>{c.node_id}</span>
                <span style={{ fontSize: 11, padding: "1px 7px", borderRadius: 9,
                               border: `1px solid ${LINE}`,
                               color: c.kind === "section" ? "#8ab8f0" : "#c9a86a" }}>
                  {c.kind === "section" ? "조항" : "문서"}
                </span>
                {c.caution && <span style={{ color: WARN, fontSize: 11 }}>active — 개명 차단</span>}
                <span style={{ color: INK_SOFT, fontSize: 11.5 }}>
                  근거 {c.evidence_count ?? 0}
                  {c.evidence_matched ? " · 라벨=자기근거" : ""}
                  {" · "}일치 라벨: {(c.matched_sections || []).slice(0, 2).join(" | ")}
                </span>
                <span style={{ marginLeft: "auto" }}>
                  {done[c.node_id]
                    ? <span style={{ color: INK, fontSize: 12 }}>{done[c.node_id]}</span>
                    : <button className="oa-rt-more" disabled={!!busyMsg || c.caution}
                              title={c.caution ? "active — 먼저 deprecated 로 내려라" : c.node_id}
                              onClick={() => approveStructural(c.node_id)}>
                        재분류
                      </button>}
                </span>
              </div>
            ))}
        </div>
      )}

      {open === "relations" && relations !== null && (
        <div style={{ marginTop: 12 }}>
          <p style={{ color: INK_SOFT, fontSize: 12.5 }}>
            관계 제안 {relations.length}건 — 인용의 <b>실재</b>는 서버가 승인 시
            재검증하지만, 인용이 관계를 <b>지지하는지</b>는 사람이 판정한다
            (인용 실재 ≠ 옳은 관계 — 정답률 67% 실측). 기각 사유는 관계판
            묘비에 남아 같은 오제안의 재출현을 거른다.
            {relFiltered > 0 && (
              <span style={{ color: WARN }}>
                {" "}묘비에 걸러진 재출현 {relFiltered}건.
              </span>
            )}
          </p>
          {relations.length === 0
            ? <p className="hint">제안이 없다.</p>
            : relations.map((row) => {
              const key = relKey(row);
              const inp = relReject[key] || {};
              return (
                <div key={key} style={{ borderBottom: `1px solid ${LINE}`, padding: "8px 4px" }}>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <span className="mono" style={{ color: INK, fontSize: 12 }}>
                      {row.subject} —{row.predicate}→ {row.object}
                    </span>
                    {row.section && (
                      <span style={{ color: INK_SOFT, fontSize: 11 }}>§{row.section}</span>
                    )}
                    {!row.signatureSeen && (
                      <span title={row.signature || "기존 시그니처에 없는 타입 조합"}
                            style={{ color: WARN, fontSize: 11, padding: "1px 7px",
                                     borderRadius: 9, border: `1px solid ${LINE}` }}>
                        신규 패턴
                      </span>
                    )}
                    {row.prevRejected && (
                      <span style={{ color: WARN, fontSize: 11 }}
                            title={row.prevRejected.at}>
                        ⚠ 과거 기각: {row.prevRejected.reason}
                      </span>
                    )}
                    <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                      {done[key]
                        ? <span style={{ color: INK, fontSize: 12 }}>{done[key]}</span>
                        : <>
                            <button className="oa-rt-more" disabled={!!busyMsg}
                                    onClick={() => approveRelation(row)}>승인</button>
                            <button className="oa-rt-more" style={{ color: WARN }}
                                    disabled={!!busyMsg}
                                    onClick={() => rejectRelation(row)}>기각</button>
                          </>}
                    </span>
                  </div>
                  {row.quote && (
                    <div style={{ marginTop: 6, padding: "8px 10px", borderRadius: 6,
                                  background: "var(--surface)", border: `1px solid ${LINE}` }}>
                      <p style={{ color: INK, fontSize: 12.5, margin: 0, lineHeight: 1.55 }}>
                        {row.quote}{row.quoteFull.length > 240 ? "…" : ""}
                      </p>
                    </div>
                  )}
                  {!done[key] && (
                    <div style={{ display: "flex", gap: 8, alignItems: "center",
                                  marginTop: 6, flexWrap: "wrap" }}>
                      <input value={inp.reason || ""}
                             onChange={(e) => setRelReject((m) =>
                               ({ ...m, [key]: { ...m[key], reason: e.target.value } }))}
                             placeholder="기각 사유 (필수 — 관계판 묘비에 기록된다)"
                             style={{ flex: 1, minWidth: 220, padding: "4px 8px",
                                      borderRadius: 6, border: `1px solid ${LINE}`,
                                      background: "var(--surface)", color: INK,
                                      fontSize: 12 }} />
                      <label style={{ color: INK_SOFT, fontSize: 11.5, display: "flex",
                                      gap: 4, alignItems: "center", cursor: "pointer" }}>
                        <input type="checkbox" checked={!!inp.evidenceOnly}
                               onChange={(e) => setRelReject((m) =>
                                 ({ ...m, [key]: { ...m[key], evidenceOnly: e.target.checked } }))} />
                        이 인용만 (다른 인용의 재제안은 표시)
                      </label>
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      )}

      {open === "golden_draft" && (
        <p className="hint" style={{ marginTop: 12 }}>
          draft 케이스의 왕복 검증·확정은 <b>골든셋 탭</b>에서 한다 — 이 카드는 카운트만 담당.
        </p>
      )}
      {open === "overdue" && (
        <p className="hint" style={{ marginTop: 12 }}>
          기한 초과 deprecated 노드 목록은 <b>/health</b> 의 lifecycle_overdue 에 있다 —
          삭제 전 대체 노드 지정 여부를 확인하라.
        </p>
      )}
    </section>
  );
}
