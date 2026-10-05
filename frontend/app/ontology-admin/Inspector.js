"use client";

// 우측 인스펙터 (3-pane 탐색기) — 선택 노드의 속성 편집 · 관계(pivot) · 근거 청크 ·
// 이력 · 거절(묘비). 상세는 GET /node?id= (백엔드 계약의 비대칭 — PATCH/chunks/history 는
// ?node_id=). PATCH 는 변경된 키만 전송(value null = 프로퍼티 삭제).
// 우리 고유 1급 요소: trust 색 pill · 근거 청크(provenance) · 거절=묘비.
// 데이터 값(타입명·술어·trust·프로퍼티키)은 번역하지 않는다 — i18n 규약.

import { useCallback, useEffect, useRef, useState } from "react";
import { useT } from "../i18n";

const TRUST_COLORS = {
  authoritative: "#059669", unknown: "#98a2b3", summary: "#d97706", unset: "#cbd5e1",
};
const TRUST_VALUES = ["authoritative", "unknown", "summary", "unset"];
const INTERNAL_KEYS = new Set([
  "type", "name", "definition", "aliases", "trust", "source", "created_at", "last_updated",
]);
const ACTION_COLORS = {
  confirm: "#059669", reject: "#dc2626", recommend: "#4f46e5", coverage_gap: "#d97706",
  create: "#0e7490", edit: "#6d28d9", edge_added: "#0e7490", edge_removed: "#9f1239",
};
const ACTOR = "admin-console";

const detailMsg = async (res) => {
  try {
    const d = (await res.json()).detail;
    if (typeof d === "string" && d) return d;
    if (d) return JSON.stringify(d);
  } catch { /* 본문이 JSON 이 아니면 statusText 폴백 */ }
  return res.statusText;
};
const asText = (v) => (typeof v === "string" ? v : JSON.stringify(v));
const fmtTime = (iso, lang) => {
  try {
    return new Date(iso).toLocaleString(lang === "ko" ? "ko-KR" : "en-US",
      { dateStyle: "short", timeStyle: "short" });
  } catch { return iso || ""; }
};

export function TrustPill({ trust }) {
  const v = trust || "unset";
  return (
    <span className="oa-exp-trust" style={{ background: TRUST_COLORS[v] || "#98a2b3" }}>{v}</span>
  );
}

export default function Inspector({
  base, nodeId, readOnly, types, targetOptions,
  crumbs, onPivot, onCrumb, onClose, onChanged, onEdited, flash, setError,
}) {
  const { t, lang } = useT();

  const [sel, setSel] = useState(null);     // GET /node 상세
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState(null);   // 기본 필드
  const [custom, setCustom] = useState([]); // 커스텀 프로퍼티 행
  const [newProp, setNewProp] = useState({ k: "", v: "" });
  const [saveBusy, setSaveBusy] = useState(false);
  const [edgeForm, setEdgeForm] = useState({ predicate: "", target: "" });
  const [edgeBusy, setEdgeBusy] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const [rejectBusy, setRejectBusy] = useState(false);

  // 접이식 근거·이력 (열 때 lazy load)
  const [provOpen, setProvOpen] = useState(false);
  const [chunks, setChunks] = useState(null);
  const [provBusy, setProvBusy] = useState(false);
  const [histOpen, setHistOpen] = useState(false);
  const [history, setHistory] = useState(null);
  const [histBusy, setHistBusy] = useState(false);

  const loadDetail = useCallback(async () => {
    if (!nodeId) return;
    setBusy(true);
    setRejecting(false); setRejectReason("");
    setEdgeForm({ predicate: "", target: "" }); setNewProp({ k: "", v: "" });
    setProvOpen(false); setChunks(null); setHistOpen(false); setHistory(null);
    try {
      const res = await fetch(`${base}/node?id=${encodeURIComponent(nodeId)}`);
      if (!res.ok) throw new Error(await detailMsg(res));
      const d = await res.json();
      const attrs = d.attrs || {};
      setSel({ ...d, node_id: d.node_id || d.id || nodeId, attrs });
      setForm({
        name: attrs.name || "",
        type: attrs.type || "",
        trust: attrs.trust || "unset",
        definition: attrs.definition || "",
        aliases: Array.isArray(attrs.aliases) ? attrs.aliases.join(", ") : "",
      });
      setCustom(Object.entries(attrs)
        .filter(([k]) => !INTERNAL_KEYS.has(k))
        .map(([k, v]) => ({ key: k, value: asText(v), orig: v, removed: false, added: false })));
    } catch (e) { setError(t("admin.nodes.err.detail", { e: e.message || e })); }
    setBusy(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, nodeId]);

  useEffect(() => { loadDetail(); }, [loadDetail]);

  // 변경된 키만 계산 → PATCH body 의 updates
  const buildUpdates = () => {
    if (!sel || !form) return {};
    const a = sel.attrs; const u = {};
    if (form.name !== (a.name || "")) u.name = form.name;
    if (form.type !== (a.type || "")) u.type = form.type;
    if (form.trust !== (a.trust || "unset")) u.trust = form.trust === "unset" ? null : form.trust;
    if (form.definition !== (a.definition || "")) u.definition = form.definition === "" ? null : form.definition;
    const aliases = form.aliases.split(",").map((s) => s.trim()).filter(Boolean);
    const origAliases = Array.isArray(a.aliases) ? a.aliases : [];
    if (JSON.stringify(aliases) !== JSON.stringify(origAliases)) u.aliases = aliases;
    for (const c of custom) {
      const key = c.key.trim();
      if (!key) continue;
      if (c.added) u[key] = c.value;
      else if (c.removed) u[key] = null;           // null = 프로퍼티 삭제
      else if (c.value !== asText(c.orig)) u[key] = c.value;
    }
    return u;
  };

  const save = async () => {
    const updates = buildUpdates();
    if (!Object.keys(updates).length || !sel) return;
    setSaveBusy(true); setError("");
    try {
      const res = await fetch(`${base}/node?node_id=${encodeURIComponent(sel.node_id)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates, actor: ACTOR }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.nodes.edit.saved", { name: form.name || sel.node_id }));
      await loadDetail();
      onEdited && onEdited();
    } catch (e) { setError(t("admin.nodes.edit.err", { e: e.message || e })); }
    setSaveBusy(false);
  };

  const addEdge = async () => {
    const predicate = edgeForm.predicate.trim();
    const target = edgeForm.target.trim();
    if (!predicate || !target || !sel) return;
    setEdgeBusy(true); setError("");
    try {
      const res = await fetch(`${base}/edges`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: sel.node_id, predicate, target, actor: ACTOR }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.nodes.edges.added"));
      setEdgeForm({ predicate: "", target: "" });
      await loadDetail();
      onEdited && onEdited();
    } catch (e) { setError(t("admin.nodes.edges.err", { e: e.message || e })); }
    setEdgeBusy(false);
  };

  const removeEdge = async (source, predicate, target) => {
    setEdgeBusy(true); setError("");
    try {
      const p = new URLSearchParams({ source, predicate, target });
      const res = await fetch(`${base}/edges?${p.toString()}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.nodes.edges.removed"));
      await loadDetail();
      onEdited && onEdited();
    } catch (e) { setError(t("admin.nodes.edges.err", { e: e.message || e })); }
    setEdgeBusy(false);
  };

  const doReject = async () => {
    if (!sel) return;
    setRejectBusy(true); setError("");
    try {
      const res = await fetch(`${base}/review/reject`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: sel.node_id, reason: rejectReason, actor: ACTOR }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.nodes.reject.done", { name: form?.name || sel.node_id }));
      onClose();
      onChanged && onChanged();
    } catch (e) { setError(t("admin.nodes.reject.err", { e: e.message || e })); }
    setRejectBusy(false);
  };

  const toggleProv = async () => {
    const next = !provOpen; setProvOpen(next);
    if (next && chunks === null && sel) {
      setProvBusy(true);
      try {
        const res = await fetch(`${base}/node/chunks?node_id=${encodeURIComponent(sel.node_id)}`);
        if (!res.ok) throw new Error(await detailMsg(res));
        setChunks((await res.json()).chunks || []);
      } catch (e) { setError(t("review.err.evidence", { e: e.message || e })); setChunks([]); }
      setProvBusy(false);
    }
  };

  const toggleHist = async () => {
    const next = !histOpen; setHistOpen(next);
    if (next && history === null && sel) {
      setHistBusy(true);
      try {
        const res = await fetch(`${base}/review/history?node_id=${encodeURIComponent(sel.node_id)}&limit=50`);
        if (!res.ok) throw new Error(await detailMsg(res));
        setHistory((await res.json()).history || []);
      } catch (e) { setError(t("admin.err.detail", { e: e.message || e })); setHistory([]); }
      setHistBusy(false);
    }
  };

  const dirty = sel ? Object.keys(buildUpdates()).length > 0 : false;

  if (busy && !sel) {
    return (
      <aside className="oa-exp-inspector">
        <div className="empty-state" style={{ marginTop: 0 }}><span className="spinner" />{t("admin.loading")}</div>
      </aside>
    );
  }
  if (!sel || !form) return <aside className="oa-exp-inspector" />;

  const attrs = sel.attrs || {};
  const outEdges = sel.out_edges || [];
  const inEdges = sel.in_edges || [];

  return (
    <aside className="oa-exp-inspector">
      {/* 브레드크럼 (pivot 히스토리) */}
      {crumbs && crumbs.length > 1 && (
        <div className="oa-exp-crumbs">
          {crumbs.map((c, i) => (
            <span key={`${c.id}_${i}`}>
              {i > 0 && <span className="oa-exp-crumb-sep">›</span>}
              <button className={`oa-exp-crumb ${i === crumbs.length - 1 ? "cur" : ""}`}
                      onClick={() => onCrumb(i)}>{c.name || c.id}</button>
            </span>
          ))}
        </div>
      )}

      {/* 헤더 */}
      <div className="oa-exp-insp-head">
        <div style={{ minWidth: 0 }}>
          <div className="oa-exp-insp-title">
            <span className="oa-exp-insp-type">{attrs.type || "—"}</span>
            <b>{attrs.name || sel.node_id}</b>
          </div>
          <div className="oa-exp-insp-id mono">{sel.node_id}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <TrustPill trust={form.trust === "unset" ? attrs.trust : form.trust} />
          <button className="oa-exp-x" onClick={onClose} title={t("admin.nodes.edit.close")}>✕</button>
        </div>
      </div>

      {readOnly && (
        <div className="admin-notice" style={{
          marginTop: 0, marginBottom: 12,
          background: "#fffaeb", borderColor: "#fedf89", color: "#b54708",
        }}>{t("admin.nodes.readonly")}</div>
      )}

      {/* 속성 (prominent) */}
      <h4 className="oa-exp-h4">{t("detail.attrs")}</h4>
      <div className="oa-exp-field">
        <label>{t("admin.nodes.edit.name")}</label>
        <input type="text" disabled={readOnly} value={form.name}
               onChange={(e) => setForm({ ...form, name: e.target.value })} />
      </div>
      <div className="row" style={{ marginTop: 10 }}>
        <div>
          <label>{t("admin.nodes.edit.type")}</label>
          <input type="text" list="oa-exp-type-list" disabled={readOnly} value={form.type}
                 onChange={(e) => setForm({ ...form, type: e.target.value })} />
        </div>
        <div>
          <label>{t("admin.nodes.edit.trust")}</label>
          <select disabled={readOnly} value={form.trust}
                  onChange={(e) => setForm({ ...form, trust: e.target.value })}>
            {TRUST_VALUES.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
      </div>
      <div className="oa-exp-field" style={{ marginTop: 10 }}>
        <label>{t("admin.nodes.edit.definition")}</label>
        <textarea rows={3} disabled={readOnly} value={form.definition}
                  onChange={(e) => setForm({ ...form, definition: e.target.value })} />
      </div>
      <div className="oa-exp-field" style={{ marginTop: 10 }}>
        <label>{t("admin.nodes.edit.aliases")}</label>
        <input type="text" disabled={readOnly} value={form.aliases}
               onChange={(e) => setForm({ ...form, aliases: e.target.value })} />
      </div>

      {/* 커스텀 프로퍼티 */}
      <h4 className="oa-exp-h4" style={{ marginTop: 16 }}>{t("admin.nodes.edit.custom")}</h4>
      {custom.filter((c) => !c.removed).length === 0 && (
        <p className="hint" style={{ marginTop: 4 }}>{t("admin.nodes.edit.custom.empty")}</p>
      )}
      {custom.map((c, i) => !c.removed && (
        <div key={`${c.key}_${i}`} className="prop-row">
          <input className="k-input" type="text" value={c.key} disabled={!c.added || readOnly}
                 onChange={(e) => setCustom(custom.map((x, j) => j === i ? { ...x, key: e.target.value } : x))} />
          <input type="text" value={c.value} disabled={readOnly}
                 onChange={(e) => setCustom(custom.map((x, j) => j === i ? { ...x, value: e.target.value } : x))} />
          {!readOnly && (
            <button className="tree-toggle" title="✕"
                    onClick={() => setCustom(c.added
                      ? custom.filter((_, j) => j !== i)
                      : custom.map((x, j) => j === i ? { ...x, removed: true } : x))}>✕</button>
          )}
        </div>
      ))}
      {!readOnly && (
        <div className="prop-row">
          <input className="k-input" type="text" value={newProp.k}
                 placeholder={t("admin.nodes.edit.keyPh")}
                 onChange={(e) => setNewProp({ ...newProp, k: e.target.value })} />
          <input type="text" value={newProp.v}
                 placeholder={t("admin.nodes.edit.valuePh")}
                 onChange={(e) => setNewProp({ ...newProp, v: e.target.value })} />
          <button className="ghost" style={{ padding: "6px 12px", fontSize: "0.76rem" }}
                  disabled={!newProp.k.trim()}
                  onClick={() => {
                    setCustom([...custom, { key: newProp.k.trim(), value: newProp.v, orig: undefined, removed: false, added: true }]);
                    setNewProp({ k: "", v: "" });
                  }}>{t("admin.nodes.edit.addProp")}</button>
        </div>
      )}

      {/* 관계 — 상대 노드명 클릭 = pivot */}
      <h4 className="oa-exp-h4" style={{ marginTop: 16 }}>{t("admin.nodes.edges.title")}</h4>
      <div className="oa-exp-rel-label">{t("admin.nodes.edges.out")}</div>
      {outEdges.length === 0 && <p className="hint" style={{ marginTop: 2 }}>{t("admin.nodes.edges.none")}</p>}
      {outEdges.map((e, i) => (
        <div key={`o${i}`} className="oa-exp-rel">
          <span className="pred">—{e.predicate}→</span>
          <button className="oa-exp-pivot"
                  onClick={() => onPivot({ id: e.target, name: e.target_name, type: e.target_type })}>
            {e.target_name || e.target}</button>
          {!readOnly && (
            <button className="tree-toggle" disabled={edgeBusy} title="✕"
                    onClick={() => removeEdge(sel.node_id, e.predicate, e.target)}>✕</button>
          )}
        </div>
      ))}
      <div className="oa-exp-rel-label" style={{ marginTop: 8 }}>{t("admin.nodes.edges.in")}</div>
      {inEdges.length === 0 && <p className="hint" style={{ marginTop: 2 }}>{t("admin.nodes.edges.none")}</p>}
      {inEdges.map((e, i) => {
        const other = e.source ?? e.target;                     // in_edge 상대노드 (계약 방어)
        const otherName = e.source_name ?? e.target_name;
        const otherType = e.source_type ?? e.target_type;
        return (
          <div key={`i${i}`} className="oa-exp-rel">
            <button className="oa-exp-pivot"
                    onClick={() => onPivot({ id: other, name: otherName, type: otherType })}>
              {otherName || other}</button>
            <span className="pred">—{e.predicate}→</span>
            {!readOnly && (
              <button className="tree-toggle" disabled={edgeBusy} title="✕"
                      onClick={() => removeEdge(other, e.predicate, sel.node_id)}>✕</button>
            )}
          </div>
        );
      })}
      {!readOnly && (
        <div className="row" style={{ marginTop: 8, alignItems: "flex-end" }}>
          <div>
            <input type="text" value={edgeForm.predicate} placeholder={t("admin.nodes.edges.predPh")}
                   onChange={(e) => setEdgeForm({ ...edgeForm, predicate: e.target.value })} />
          </div>
          <div>
            <input type="text" list="oa-exp-target-list" value={edgeForm.target}
                   placeholder={t("admin.nodes.edges.targetPh")}
                   onChange={(e) => setEdgeForm({ ...edgeForm, target: e.target.value })} />
          </div>
          <div style={{ flex: 0, minWidth: 0 }}>
            <button className="ghost" onClick={addEdge}
                    disabled={edgeBusy || !edgeForm.predicate.trim() || !edgeForm.target.trim()}>
              {edgeBusy && <span className="spinner" />}{t("admin.nodes.edges.add")}
            </button>
          </div>
        </div>
      )}

      {/* 근거 청크 (provenance) — 우리 고유 기능, 인용 스타일 */}
      <button className="oa-exp-fold" onClick={toggleProv}>
        <span>{provOpen ? "▾" : "▸"}</span> {t("review.evidenceTitle")}
        {chunks !== null && <span className="oa-exp-fold-cnt">{chunks.length}</span>}
      </button>
      {provOpen && (
        <div className="oa-exp-fold-body">
          {provBusy && <p className="hint"><span className="spinner" />{t("review.evidenceLoading")}</p>}
          {!provBusy && chunks && chunks.length === 0 && (
            <p className="hint">{t("review.evidenceEmpty")}</p>
          )}
          {!provBusy && (chunks || []).map((c) => (
            <div key={c.chunk_id} className="oa-exp-prov">
              {c.section && <div className="oa-exp-prov-cite">{c.section}</div>}
              <div className="oa-exp-prov-text">{c.text}</div>
              <div className="oa-exp-prov-src mono">
                {c.source || "—"} · [{c.char_start}–{c.char_end}]</div>
            </div>
          ))}
        </div>
      )}

      {/* 이력 */}
      <button className="oa-exp-fold" onClick={toggleHist}>
        <span>{histOpen ? "▾" : "▸"}</span> {t("review.history")}
        {history !== null && <span className="oa-exp-fold-cnt">{history.length}</span>}
      </button>
      {histOpen && (
        <div className="oa-exp-fold-body">
          {histBusy && <p className="hint"><span className="spinner" />{t("admin.loading")}</p>}
          {!histBusy && history && history.length === 0 && (
            <p className="hint">{t("review.history.empty")}</p>
          )}
          {!histBusy && (history || []).map((h, i) => (
            <div key={i} className="oa-exp-hist">
              <span className="oa-exp-action" style={{ background: ACTION_COLORS[h.action] || "#98a2b3" }}>
                {h.action}</span>
              <span className="oa-exp-hist-detail">
                {h.reason || h.after?.rationale || h.after?.name || "—"}</span>
              <span className="oa-exp-hist-meta mono">{h.actor || "—"} · {fmtTime(h.at, lang)}</span>
            </div>
          ))}
        </div>
      )}

      {/* 액션 — 저장 · 거절(묘비) */}
      {!readOnly && (
        <div style={{ display: "flex", gap: 8, marginTop: 18, flexWrap: "wrap" }}>
          <button onClick={save} disabled={saveBusy || !dirty}>
            {saveBusy && <span className="spinner" />}
            {saveBusy ? t("admin.nodes.edit.saving") : t("admin.nodes.edit.save")}
          </button>
          {!rejecting ? (
            <button className="btn-danger" onClick={() => setRejecting(true)}>
              {t("admin.nodes.reject.btn")}</button>
          ) : (
            <>
              <input type="text" value={rejectReason} style={{ width: 180, flex: "0 1 auto" }}
                     placeholder={t("admin.nodes.reject.reasonPh")}
                     onChange={(e) => setRejectReason(e.target.value)} />
              <button className="btn-danger" onClick={doReject} disabled={rejectBusy}>
                {rejectBusy && <span className="spinner" />}{t("admin.nodes.reject.submit")}
              </button>
              <button className="ghost" onClick={() => setRejecting(false)}>
                {t("admin.nodes.reject.cancel")}</button>
            </>
          )}
        </div>
      )}

      <datalist id="oa-exp-type-list">
        {(types || []).map((ty) => <option key={ty} value={ty} />)}
      </datalist>
      <datalist id="oa-exp-target-list">
        {(targetOptions || []).map((n) => <option key={n.node_id} value={n.node_id}>{n.name}</option>)}
      </datalist>
    </aside>
  );
}
