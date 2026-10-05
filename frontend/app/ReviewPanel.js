"use client";

// 검수(Review) 탭 — LLM 추출 노드의 확정/거절 큐 + 근거대조 에이전트 사전판정.
// 백엔드: GET /graphs/{ns}/review → POST review/precheck → POST review/confirm|reject
//         → GET review/history (감사 이력) · GET node/chunks (근거 원문)
import { useEffect, useRef, useState } from "react";
import { useT } from "./i18n";

const VERDICT_COLORS = {
  confirm: "#059669",      // 초록 — 확정 추천
  reject: "#dc2626",       // 빨강 — 거절 추천
  unsure: "#d97706",       // 호박 — 불확실
  no_evidence: "#6b7280",  // 회색 — 근거 없음
};
const VERDICTS = Object.keys(VERDICT_COLORS);
const TRUST_COLORS = { authoritative: "#059669", unknown: "#98a2b3", summary: "#d97706" };
const TRUST_LEVELS = ["authoritative", "unknown", "summary"];
const ACTION_COLORS = { confirm: "#059669", reject: "#dc2626", recommend: "#4f46e5" };

const pill = (bg) => ({
  display: "inline-block", padding: "2px 9px", borderRadius: 999,
  background: bg || "#98a2b3", color: "#fff", fontSize: "0.68rem",
  fontWeight: 700, letterSpacing: "0.3px",
});
const outlinePill = {
  display: "inline-block", padding: "1px 8px", borderRadius: 999,
  border: "1px solid var(--border)", color: "var(--muted)",
  fontSize: "0.68rem", fontWeight: 700, letterSpacing: "0.3px",
};
const smallBtn = { padding: "4px 10px", fontSize: "0.74rem" };
const quoteStyle = {
  margin: "6px 0 0", padding: "5px 10px",
  borderLeft: "3px solid var(--border)", color: "var(--muted)",
  fontSize: "0.78rem", fontStyle: "italic",
};

// ─── 일관성/커버리지 검사 (아래 두 섹션 전용) ────────────────────
const SEVERITY_COLORS = { error: "#dc2626", warn: "#d97706" };
const CONSISTENCY_KINDS = ["name_type_conflict", "alias_collision", "range_violation"];
const codeChip = {
  display: "inline-block", padding: "1px 7px", borderRadius: 6,
  border: "1px solid var(--border)", color: "var(--muted)",
  fontSize: "0.7rem", fontFamily: "monospace",
};
// FastAPI 는 detail 을 문자열 또는 검증 오류 배열로 준다 — 있는 그대로 노출
const detailMsg = async (res) => {
  try {
    const d = (await res.json()).detail;
    if (typeof d === "string" && d) return d;
    if (d) return JSON.stringify(d);
  } catch { /* 본문이 JSON 이 아니면 statusText 폴백 */ }
  return res.statusText;
};

export default function ReviewPanel({ api, namespace }) {
  const { t } = useT();
  const base = `${api}/api/v1/ontology/graphs/${namespace}`;
  const [items, setItems] = useState(null);        // null = 아직 미로드
  const [loading, setLoading] = useState(false);
  const [trustFilter, setTrustFilter] = useState("");
  const [precheckBusy, setPrecheckBusy] = useState(false);
  const [actionBusy, setActionBusy] = useState(null);  // 판정 진행 중 node_id
  const [rejecting, setRejecting] = useState(null);    // 거절 사유 입력이 열린 node_id
  const [rejectReason, setRejectReason] = useState("");
  const [expanded, setExpanded] = useState({});        // node_id → bool (근거 펼침)
  const [chunksMap, setChunksMap] = useState({});      // node_id → {status, chunks, error}
  const [history, setHistory] = useState([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const noticeRef = useRef(null);
  // 일관성 검사 (결정적 · 무료)
  const [consistOpen, setConsistOpen] = useState(false);
  const [consistBusy, setConsistBusy] = useState(false);
  const [consistData, setConsistData] = useState(null);  // null = 아직 미실행
  const [consistError, setConsistError] = useState("");
  // 커버리지 검사 (청크당 LLM 1콜)
  const [coverOpen, setCoverOpen] = useState(false);
  const [coverBusy, setCoverBusy] = useState(false);
  const [coverData, setCoverData] = useState(null);
  const [coverError, setCoverError] = useState("");
  const [coverLimit, setCoverLimit] = useState(10);

  const flash = (msg) => {
    setNotice(msg);
    clearTimeout(noticeRef.current);
    noticeRef.current = setTimeout(() => setNotice(""), 4000);
  };
  useEffect(() => () => clearTimeout(noticeRef.current), []);

  const loadQueue = async (trust = trustFilter) => {
    if (!namespace) return;
    setLoading(true); setError("");
    try {
      const q = trust ? `&trust=${encodeURIComponent(trust)}` : "";
      const res = await fetch(`${base}/review?limit=100${q}`);
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      setItems((await res.json()).items || []);
    } catch (e) { setError(t("review.err.load", { e: e.message || e })); }
    setLoading(false);
  };

  const loadHistory = async () => {
    if (!namespace) return;
    try {
      const res = await fetch(`${base}/review/history?limit=30`);
      if (!res.ok) return;  // 이력은 보조 정보 — 실패해도 큐는 유효
      const data = await res.json();
      setHistory(data.history || data.entries || []);
    } catch { /* 이력 실패는 조용히 무시 */ }
  };

  useEffect(() => {
    setItems(null); setExpanded({}); setChunksMap({});
    setRejecting(null); setNotice(""); setError("");
    loadQueue(); loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namespace, trustFilter]);

  // 네임스페이스가 바뀌면 검사 결과는 무효 — 리셋 (열림 상태는 유지)
  useEffect(() => {
    setConsistData(null); setConsistError("");
    setCoverData(null); setCoverError("");
  }, [namespace]);

  const runConsistency = async () => {
    setConsistBusy(true); setConsistError("");
    try {
      const res = await fetch(`${base}/review/consistency`);
      if (!res.ok) throw new Error(await detailMsg(res));
      setConsistData(await res.json());
    } catch (e) {
      setConsistError(t("review.consistency.err", { e: e.message || e }));
    }
    setConsistBusy(false);
  };

  const runCoverage = async () => {
    const limit = Math.min(30, Math.max(1, Number(coverLimit) || 10));
    setCoverBusy(true); setCoverError("");
    try {
      const res = await fetch(`${base}/review/coverage`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      setCoverData(await res.json());
    } catch (e) {
      setCoverError(t("review.coverage.err", { e: e.message || e }));
    }
    setCoverBusy(false);
  };

  const runPrecheck = async () => {
    setPrecheckBusy(true); setError("");
    try {
      const body = { limit: 20 };
      if (trustFilter) body.trust = trustFilter;
      const res = await fetch(`${base}/review/precheck`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      flash(t("review.precheckDone", { n: data.checked ?? 0 }));
      await loadQueue();
      await loadHistory();
    } catch (e) { setError(t("review.err.precheck", { e: e.message || e })); }
    setPrecheckBusy(false);
  };

  // 확정/거절 — 낙관적 제거, 실패 시 복원 + 오류를 그대로 노출
  const act = async (item, kind, reason = "") => {
    const prev = items;
    setActionBusy(item.node_id);
    setItems((p) => p.filter((x) => x.node_id !== item.node_id));
    setRejecting(null); setRejectReason("");
    try {
      const body = { node_id: item.node_id, actor: "builder_ui" };
      if (kind === "reject") body.reason = reason;
      const res = await fetch(`${base}/review/${kind}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      flash(t(kind === "confirm" ? "review.done.confirm" : "review.done.reject",
              { name: item.name }));
      loadHistory();
    } catch (e) {
      setItems(prev);  // 복원 — 실패를 감추지 않는다
      setError(t("review.err.action", { e: e.message || e }));
    }
    setActionBusy(null);
  };

  // 근거 청크 lazy-load (펼칠 때 1회)
  const toggleExpand = (nodeId) => {
    const open = !expanded[nodeId];
    setExpanded((p) => ({ ...p, [nodeId]: open }));
    if (open && !chunksMap[nodeId]) {
      setChunksMap((p) => ({ ...p, [nodeId]: { status: "loading" } }));
      fetch(`${base}/node/chunks?node_id=${encodeURIComponent(nodeId)}`)
        .then(async (res) => {
          if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
          const d = await res.json();
          setChunksMap((p) => ({ ...p, [nodeId]: { status: "ok", chunks: d.chunks || [] } }));
        })
        .catch((e) => setChunksMap((p) => ({
          ...p, [nodeId]: { status: "error", error: e.message || String(e) } })));
    }
  };

  if (!namespace) return <div className="empty-state">{t("review.noNs")}</div>;

  const itemCard = (item) => {
    const rec = item.recommendation || null;
    const verdictLabel = rec
      ? (VERDICTS.includes(rec.verdict) ? t(`review.badge.${rec.verdict}`) : rec.verdict)
      : t("review.badge.none");
    const chunks = chunksMap[item.node_id];
    return (
      <section className="card" key={item.node_id}>
        {/* 헤더 — 클릭하면 근거 펼침 */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
                      cursor: "pointer" }}
             onClick={() => toggleExpand(item.node_id)}>
          <b style={{ fontSize: "0.94rem" }}>{item.name}</b>
          {item.type && <span style={pill("#7c3aed")}>{item.type}</span>}
          <span style={pill(TRUST_COLORS[item.trust] || "#98a2b3")}>
            {item.trust || "unknown"}</span>
          <span className="hint-inline">{item.source}</span>
          <span className="hint-inline">{t("review.evidence", { n: item.evidence_count })}</span>
          <span style={{ marginLeft: "auto" }}>
            {rec
              ? <span style={pill(VERDICT_COLORS[rec.verdict])}>{verdictLabel}</span>
              : <span style={outlinePill}>{verdictLabel}</span>}
          </span>
        </div>
        {item.definition && (
          <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: 6 }}>
            {item.definition}
          </div>
        )}
        {/* 사전판정 근거 (rationale + 인용) */}
        {rec?.rationale && (
          <div style={{ fontSize: "0.8rem", marginTop: 8 }}>{rec.rationale}</div>
        )}
        {rec?.evidence_quote && (
          <blockquote style={quoteStyle}>“{rec.evidence_quote}”</blockquote>
        )}
        {/* 근거 청크 (lazy) */}
        {expanded[item.node_id] && (
          <div style={{ marginTop: 10, padding: "8px 12px",
                        border: "1px solid var(--border)", borderRadius: 8 }}>
            <div style={{ fontSize: "0.74rem", fontWeight: 700, color: "var(--faint)",
                          letterSpacing: "0.4px", marginBottom: 6 }}>
              {t("review.evidenceTitle")}
            </div>
            {(!chunks || chunks.status === "loading") && (
              <div style={{ display: "flex", alignItems: "center", gap: 8,
                            color: "var(--muted)", fontSize: "0.8rem" }}>
                <span className="spinner" />{t("review.evidenceLoading")}
              </div>
            )}
            {chunks?.status === "error" && (
              <div className="error">{t("review.err.evidence", { e: chunks.error })}</div>
            )}
            {chunks?.status === "ok" && (chunks.chunks.length === 0
              ? <div style={{ color: "var(--muted)", fontSize: "0.8rem" }}>
                  {t("review.evidenceEmpty")}</div>
              : chunks.chunks.map((ch, i) => (
                  <div key={ch.chunk_id || i} style={{ marginBottom: 8 }}>
                    <blockquote style={{ ...quoteStyle, margin: 0 }}>{ch.text}</blockquote>
                    <div className="hint-inline" style={{ marginTop: 2 }}>
                      {(ch.source || "").split("/").pop()}
                      {ch.trust ? ` · ${ch.trust}` : ""}
                    </div>
                  </div>
                )))}
          </div>
        )}
        {/* 판정 액션 */}
        <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center",
                      flexWrap: "wrap" }}>
          {rejecting === item.node_id ? (
            <>
              <input type="text" value={rejectReason} autoFocus
                     placeholder={t("review.rejectReason.ph")}
                     style={{ padding: "5px 8px", fontSize: "0.8rem", flex: 1, maxWidth: 340 }}
                     onChange={(e) => setRejectReason(e.target.value)}
                     onKeyDown={(e) => e.key === "Enter" && act(item, "reject", rejectReason)} />
              <button style={smallBtn} disabled={actionBusy === item.node_id}
                      onClick={() => act(item, "reject", rejectReason)}>
                {t("review.rejectSubmit")}
              </button>
              <button className="ghost" style={smallBtn}
                      onClick={() => { setRejecting(null); setRejectReason(""); }}>
                {t("review.rejectCancel")}
              </button>
            </>
          ) : (
            <>
              <button style={smallBtn} disabled={!!actionBusy}
                      onClick={() => act(item, "confirm")}>{t("review.confirm")}</button>
              <button className="ghost" style={smallBtn} disabled={!!actionBusy}
                      onClick={() => { setRejecting(item.node_id); setRejectReason(""); }}>
                {t("review.reject")}
              </button>
            </>
          )}
        </div>
      </section>
    );
  };

  return (
    <>
      {error && <div className="error card soft" style={{ marginBottom: 14 }}>{error}</div>}
      {notice && (
        <div className="card soft" style={{ marginBottom: 14, color: "#059669",
                                            fontSize: "0.84rem" }}>{notice}</div>
      )}

      {/* 툴바 — 새로고침 · 사전판정 · 대기 건수 */}
      <section className="card soft" style={{ display: "flex", alignItems: "center",
                                              gap: 10, flexWrap: "wrap" }}>
        <button className="ghost" style={smallBtn} disabled={loading || precheckBusy}
                onClick={() => { loadQueue(); loadHistory(); }}>
          {t("review.refresh")}
        </button>
        <button style={smallBtn} disabled={precheckBusy || loading || !items?.length}
                onClick={runPrecheck}>
          {precheckBusy && <span className="spinner" />}
          {precheckBusy ? t("review.precheckRunning") : t("review.precheck")}
        </button>
        {items && (
          <span className="hint-inline" style={{ marginLeft: "auto" }}>
            {t("review.total", { n: items.length })}
          </span>
        )}
      </section>

      {/* 신뢰 등급 필터 */}
      <div className="filter-bar">
        <span className="filter-label">{t("gate.trust")}</span>
        <span className={`filter-chip ${trustFilter === "" ? "on" : ""}`}
              onClick={() => setTrustFilter("")}>
          {t("review.filter.all")}
        </span>
        {TRUST_LEVELS.map((tr) => (
          <span key={tr} className={`filter-chip ${trustFilter === tr ? "on" : ""}`}
                onClick={() => setTrustFilter(trustFilter === tr ? "" : tr)}>
            <span className="dot" style={{ background: TRUST_COLORS[tr] }} />{tr}
          </span>
        ))}
      </div>

      {/* 큐 */}
      {(items === null || loading) ? (
        <section className="card" style={{ display: "flex", alignItems: "center", gap: 10,
                                           color: "var(--muted)" }}>
          <span className="spinner" />{t("review.loading")}
        </section>
      ) : items.length === 0 ? (
        <div className="empty-state">{t("review.empty")}</div>
      ) : (
        items.map(itemCard)
      )}

      {/* 일관성 검사 (접이식 · 결정적 · 무료) */}
      <section className="card" style={{ marginTop: 4 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                      cursor: "pointer" }}
             onClick={() => setConsistOpen(!consistOpen)}>
          <b style={{ fontSize: "0.88rem" }}>{t("review.consistency.title")}</b>
          <span className="hint-inline">
            {consistOpen ? t("review.history.hide") : t("review.history.show")}
          </span>
        </div>
        {consistOpen && (
          <div style={{ marginTop: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <button style={smallBtn} disabled={consistBusy} onClick={runConsistency}>
                {consistBusy && <span className="spinner" />}
                {consistBusy ? t("review.consistency.running") : t("review.consistency.run")}
              </button>
              <span className="hint-inline">{t("review.consistency.hint")}</span>
            </div>
            {consistError && (
              <div className="error" style={{ marginTop: 8 }}>{consistError}</div>
            )}
            {consistData && !consistError && (
              (consistData.findings || []).length === 0 ? (
                <div style={{ color: "#059669", fontSize: "0.84rem", marginTop: 10 }}>
                  {t("review.consistency.empty")}
                </div>
              ) : (
                <>
                  <div className="hint-inline" style={{ display: "block", marginTop: 10 }}>
                    {t("review.consistency.summary", { n: consistData.findings.length })}
                    {" — "}
                    {CONSISTENCY_KINDS.filter((k) => consistData.counts?.[k])
                      .map((k) => `${t(`review.consistency.kind.${k}`)} ${consistData.counts[k]}`)
                      .join(" · ")}
                  </div>
                  {consistData.findings.map((f, i) => (
                    <div key={i} style={{ marginTop: 10, paddingTop: 10,
                                          borderTop: "1px solid var(--border)" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8,
                                    flexWrap: "wrap" }}>
                        <span style={pill(SEVERITY_COLORS[f.severity] || "#98a2b3")}>
                          {f.severity}</span>
                        <span style={outlinePill}>
                          {CONSISTENCY_KINDS.includes(f.kind)
                            ? t(`review.consistency.kind.${f.kind}`) : f.kind}
                        </span>
                      </div>
                      <div style={{ fontSize: "0.8rem", marginTop: 6 }}>{f.detail}</div>
                      {(f.node_ids || []).length > 0 && (
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap",
                                      marginTop: 6 }}>
                          {f.node_ids.map((id) => (
                            <span key={id} style={codeChip}>{id}</span>
                          ))}
                        </div>
                      )}
                      {f.suggestion && (
                        <div style={{ color: "var(--muted)", fontSize: "0.78rem",
                                      marginTop: 6 }}>
                          {f.suggestion}
                        </div>
                      )}
                    </div>
                  ))}
                </>
              )
            )}
          </div>
        )}
      </section>

      {/* 커버리지 검사 (접이식 · 청크당 LLM 1콜) */}
      <section className="card" style={{ marginTop: 4 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                      cursor: "pointer" }}
             onClick={() => setCoverOpen(!coverOpen)}>
          <b style={{ fontSize: "0.88rem" }}>{t("review.coverage.title")}</b>
          <span className="hint-inline">
            {coverOpen ? t("review.history.hide") : t("review.history.show")}
          </span>
        </div>
        {coverOpen && (
          <div style={{ marginTop: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <label className="hint-inline" htmlFor="coverage-limit">
                {t("review.coverage.limit")}
              </label>
              <input id="coverage-limit" type="number" min={1} max={30} value={coverLimit}
                     disabled={coverBusy}
                     style={{ width: 70, padding: "4px 8px", fontSize: "0.8rem" }}
                     onChange={(e) => setCoverLimit(e.target.value)} />
              <button style={smallBtn} disabled={coverBusy} onClick={runCoverage}>
                {coverBusy && <span className="spinner" />}
                {coverBusy ? t("review.coverage.running") : t("review.coverage.run")}
              </button>
              <span className="hint-inline">{t("review.coverage.cost")}</span>
            </div>
            {coverError && (
              <div className="error" style={{ marginTop: 8 }}>{coverError}</div>
            )}
            {coverData && !coverError && !coverBusy && (
              <>
                <div className="hint-inline" style={{ display: "block", marginTop: 10 }}>
                  {t("review.coverage.result", {
                    n: coverData.chunks_checked ?? 0,
                    m: (coverData.gaps || []).length,
                  })}
                </div>
                {(coverData.gaps || []).length === 0 ? (
                  <div style={{ color: "#059669", fontSize: "0.84rem", marginTop: 8 }}>
                    {t("review.coverage.empty")}
                  </div>
                ) : (
                  <>
                    {coverData.gaps.map((g, i) => (
                      <div key={i} style={{ marginTop: 10, paddingTop: 10,
                                            borderTop: "1px solid var(--border)" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8,
                                      flexWrap: "wrap" }}>
                          <b style={{ fontSize: "0.88rem" }}>{g.name}</b>
                          {g.type && <span style={pill("#7c3aed")}>{g.type}</span>}
                          <span className="hint-inline">
                            {(g.source || "").split("/").pop()}
                          </span>
                        </div>
                        {g.quote && <blockquote style={quoteStyle}>“{g.quote}”</blockquote>}
                      </div>
                    ))}
                    <div style={{ color: "var(--muted)", fontSize: "0.78rem",
                                  marginTop: 10 }}>
                      {t("review.coverage.addHint")}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        )}
      </section>

      {/* 감사 이력 (접이식) */}
      <section className="card" style={{ marginTop: 4 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                      cursor: "pointer" }}
             onClick={() => setHistoryOpen(!historyOpen)}>
          <b style={{ fontSize: "0.88rem" }}>{t("review.history")}</b>
          <span className="hint-inline">
            {historyOpen ? t("review.history.hide") : t("review.history.show")}
          </span>
        </div>
        {historyOpen && (history.length === 0
          ? <div style={{ color: "var(--muted)", fontSize: "0.8rem", marginTop: 8 }}>
              {t("review.history.empty")}</div>
          : <table>
              <thead><tr>
                <th>{t("review.th.action")}</th><th>{t("review.th.node")}</th>
                <th>{t("review.th.actor")}</th><th>{t("review.th.reason")}</th>
                <th>{t("review.th.at")}</th>
              </tr></thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i}>
                    <td><span style={pill(ACTION_COLORS[h.action] || "#98a2b3")}>{h.action}</span></td>
                    <td className="mono" style={{ fontSize: "0.74rem" }}>{h.node_id}</td>
                    <td style={{ fontSize: "0.78rem" }}>{h.actor || "—"}</td>
                    <td style={{ color: "var(--muted)", fontSize: "0.78rem" }}>
                      {h.reason || h.after?.rationale || ""}
                    </td>
                    <td className="mono" style={{ fontSize: "0.72rem" }}>
                      {(h.at || "").replace("T", " ").slice(0, 19)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>)}
      </section>

      <p className="hint">{t("review.hint")}</p>
    </>
  );
}
