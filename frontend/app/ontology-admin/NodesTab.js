"use client";

// 노드 탭 — 검색·필터·페이지네이션 목록 + 행 클릭 편집 패널.
// 편집: PATCH 는 변경된 키만 보낸다 (value null = 프로퍼티 삭제).
// 거절은 삭제가 아니라 묘비(review/reject) — 재인제스트에서 부활하지 않는다.
// protected 네임스페이스는 읽기 전용 (백엔드 403 이 최종 방어선, 여기는 UX).

import { useCallback, useEffect, useRef, useState } from "react";
import { useT } from "../i18n";

const LIMIT = 50;
const TRUST_COLORS = {
  authoritative: "#059669", unknown: "#98a2b3", summary: "#d97706",
  unset: "#cbd5e1",
};
const TRUST_VALUES = ["authoritative", "unknown", "summary", "unset"];
// 내부 키 — 기본 필드 폼에서 다루므로 커스텀 프로퍼티 목록에서 제외
const INTERNAL_KEYS = new Set([
  "type", "name", "definition", "aliases", "trust", "source",
  "created_at", "last_updated",
]);
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

function TrustPill({ trust }) {
  const v = trust || "unset";
  return (
    <span style={{
      display: "inline-block", padding: "2px 9px", borderRadius: 999,
      background: TRUST_COLORS[v] || "#98a2b3", color: "#fff",
      fontSize: "0.68rem", fontWeight: 700, letterSpacing: "0.3px",
    }}>{v}</span>
  );
}

export default function NodesTab({ api, namespace, protected: readOnly, onChanged }) {
  const { t } = useT();
  const base = `${api}/graphs/${namespace}`;

  // ── 목록 ──
  const [qInput, setQInput] = useState("");
  const [q, setQ] = useState("");
  const [typeF, setTypeF] = useState("");
  const [trustF, setTrustF] = useState("");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState(null);
  const [listBusy, setListBusy] = useState(false);
  const [types, setTypes] = useState([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const noticeRef = useRef(null);

  const flash = (msg) => {
    setNotice(msg);
    clearTimeout(noticeRef.current);
    noticeRef.current = setTimeout(() => setNotice(""), 5000);
  };
  useEffect(() => () => clearTimeout(noticeRef.current), []);

  // ── 편집 패널 ──
  const [sel, setSel] = useState(null);        // GET /node 상세
  const [selBusy, setSelBusy] = useState(false);
  const [form, setForm] = useState(null);      // 기본 필드
  const [custom, setCustom] = useState([]);    // 커스텀 프로퍼티 행
  const [newProp, setNewProp] = useState({ k: "", v: "" });
  const [saveBusy, setSaveBusy] = useState(false);
  const [edgeForm, setEdgeForm] = useState({ predicate: "", target: "" });
  const [edgeBusy, setEdgeBusy] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const [rejectBusy, setRejectBusy] = useState(false);

  // ── 새 노드 폼 ──
  const [creating, setCreating] = useState(false);
  const [cForm, setCForm] = useState({ node_type: "", name: "", definition: "", aliases: "" });
  const [createBusy, setCreateBusy] = useState(false);

  // 타입 셀렉트는 스키마의 classes 로 채운다 (하드코딩 어휘 금지)
  useEffect(() => {
    let alive = true;
    fetch(`${base}/schema`)
      .then(async (r) => {
        if (!r.ok || !alive) return;
        const s = await r.json();
        if (alive) setTypes((s.classes || []).map((c) => c.type));
      })
      .catch(() => { /* 타입 목록은 보조 — 실패해도 검색은 동작 */ });
    return () => { alive = false; };
  }, [base]);

  const loadList = useCallback(async () => {
    setListBusy(true); setError("");
    try {
      const p = new URLSearchParams({ offset: String(offset), limit: String(LIMIT) });
      if (q) p.set("q", q);
      if (typeF) p.set("node_type", typeF);
      if (trustF) p.set("trust", trustF);
      const res = await fetch(`${base}/nodes?${p.toString()}`);
      if (!res.ok) throw new Error(await detailMsg(res));
      setData(await res.json());
    } catch (e) { setError(t("admin.nodes.err.load", { e: e.message || e })); }
    setListBusy(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, q, typeF, trustF, offset]);

  useEffect(() => { loadList(); }, [loadList]);

  const openNode = async (nodeId) => {
    setSelBusy(true); setError(""); setCreating(false);
    setRejecting(false); setRejectReason("");
    setEdgeForm({ predicate: "", target: "" }); setNewProp({ k: "", v: "" });
    try {
      // 주의: 상세 GET 은 ?id=, 편집 PATCH 는 ?node_id= — 백엔드 계약의 비대칭 (확정)
      const res = await fetch(`${base}/node?id=${encodeURIComponent(nodeId)}`);
      if (!res.ok) throw new Error(await detailMsg(res));
      const d = await res.json();
      const attrs = d.attrs || {};
      setSel({ ...d, node_id: d.node_id || nodeId, attrs });
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
    setSelBusy(false);
  };

  const closePanel = () => { setSel(null); setForm(null); setCustom([]); };

  // 변경된 키만 계산 — 그대로 PATCH body 의 updates 가 된다
  const buildUpdates = () => {
    if (!sel || !form) return {};
    const a = sel.attrs; const u = {};
    if (form.name !== (a.name || "")) u.name = form.name;
    if (form.type !== (a.type || "")) u.type = form.type;
    if (form.trust !== (a.trust || "unset")) {
      u.trust = form.trust === "unset" ? null : form.trust;
    }
    if (form.definition !== (a.definition || "")) {
      u.definition = form.definition === "" ? null : form.definition;
    }
    const aliases = form.aliases.split(",").map((s) => s.trim()).filter(Boolean);
    const origAliases = Array.isArray(a.aliases) ? a.aliases : [];
    if (JSON.stringify(aliases) !== JSON.stringify(origAliases)) u.aliases = aliases;
    for (const c of custom) {
      const key = c.key.trim();
      if (!key) continue;
      if (c.added) u[key] = c.value;
      else if (c.removed) u[key] = null;             // null = 프로퍼티 삭제
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
      await openNode(sel.node_id);
      await loadList();
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
      await openNode(sel.node_id);
      await loadList();
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
      await openNode(sel.node_id);
      await loadList();
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
      closePanel(); setRejecting(false); setRejectReason("");
      await loadList();
      if (onChanged) onChanged();
    } catch (e) { setError(t("admin.nodes.reject.err", { e: e.message || e })); }
    setRejectBusy(false);
  };

  const createNode = async () => {
    const node_type = cForm.node_type.trim();
    const name = cForm.name.trim();
    if (!node_type || !name) return;
    setCreateBusy(true); setError("");
    try {
      const body = { node_type, name, actor: ACTOR };
      if (cForm.definition.trim()) body.definition = cForm.definition.trim();
      const aliases = cForm.aliases.split(",").map((s) => s.trim()).filter(Boolean);
      if (aliases.length) body.aliases = aliases;
      const res = await fetch(`${base}/nodes`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.nodes.create.done", { name }));
      setCreating(false);
      setCForm({ node_type: "", name: "", definition: "", aliases: "" });
      await loadList();
      if (onChanged) onChanged();
    } catch (e) { setError(t("admin.nodes.create.err", { e: e.message || e })); }
    setCreateBusy(false);
  };

  const items = data?.items || [];
  const total = data?.total || 0;
  const from = total ? offset + 1 : 0;
  const to = Math.min(offset + LIMIT, total);
  const dirty = sel ? Object.keys(buildUpdates()).length > 0 : false;
  const inSource = (e) => e.source ?? e.target;  // in_edge 상대 노드 (계약 방어)

  return (
    <>
      {readOnly && (
        <div className="admin-notice" style={{
          marginTop: 18, marginBottom: 0,
          background: "#fffaeb", borderColor: "#fedf89", color: "#b54708",
        }}>{t("admin.nodes.readonly")}</div>
      )}
      {notice && <div className="admin-notice" style={{ marginTop: 14, marginBottom: 0 }}>{notice}</div>}
      {error && <div className="error" style={{ marginTop: 14 }}>{error}</div>}

      <section className="card">
        <div className="admin-head">
          <h2>{t("admin.nodes.title")}
            <span className="badge" style={{ marginLeft: 8 }}>{total.toLocaleString()}</span></h2>
          {!readOnly && (
            <button className="ghost" onClick={() => { setCreating(!creating); closePanel(); }}>
              {t("admin.nodes.new")}</button>
          )}
        </div>

        {/* 필터바 — 검색은 Enter 로 발동 */}
        <div className="row" style={{ marginTop: 12, alignItems: "flex-end" }}>
          <div style={{ flex: 2 }}>
            <label>{t("admin.nodes.search.label")}</label>
            <input type="text" value={qInput} placeholder={t("admin.nodes.search.ph")}
                   onChange={(e) => setQInput(e.target.value)}
                   onKeyDown={(e) => {
                     if (e.key === "Enter") { setOffset(0); setQ(qInput.trim()); }
                   }} />
          </div>
          <div>
            <label>{t("admin.nodes.filter.type")}</label>
            <select value={typeF}
                    onChange={(e) => { setOffset(0); setTypeF(e.target.value); }}>
              <option value="">{t("admin.nodes.filter.allTypes")}</option>
              {types.map((ty) => <option key={ty} value={ty}>{ty}</option>)}
            </select>
          </div>
          <div>
            <label>{t("admin.nodes.filter.trust")}</label>
            <select value={trustF}
                    onChange={(e) => { setOffset(0); setTrustF(e.target.value); }}>
              <option value="">{t("admin.nodes.filter.allTrust")}</option>
              {TRUST_VALUES.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div style={{ flex: 0, minWidth: 0 }}>
            <button className="ghost" onClick={() => {
              setQInput(""); setQ(""); setTypeF(""); setTrustF(""); setOffset(0);
            }}>{t("admin.nodes.reset")}</button>
          </div>
        </div>

        {/* 새 노드 폼 */}
        {creating && !readOnly && (
          <div className="card soft" style={{ marginTop: 14 }}>
            <h2 style={{ fontSize: "0.95rem" }}>{t("admin.nodes.create.title")}</h2>
            <div className="row" style={{ marginTop: 10 }}>
              <div>
                <label>{t("admin.nodes.create.type")}</label>
                <input type="text" list="admin-type-list" value={cForm.node_type}
                       onChange={(e) => setCForm({ ...cForm, node_type: e.target.value })} />
              </div>
              <div>
                <label>{t("admin.nodes.create.name")}</label>
                <input type="text" value={cForm.name}
                       onChange={(e) => setCForm({ ...cForm, name: e.target.value })} />
              </div>
              <div>
                <label>{t("admin.nodes.create.aliases")}</label>
                <input type="text" value={cForm.aliases}
                       onChange={(e) => setCForm({ ...cForm, aliases: e.target.value })} />
              </div>
            </div>
            <div style={{ marginTop: 10 }}>
              <label>{t("admin.nodes.create.definition")}</label>
              <textarea rows={2} value={cForm.definition}
                        onChange={(e) => setCForm({ ...cForm, definition: e.target.value })} />
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button onClick={createNode}
                      disabled={createBusy || !cForm.node_type.trim() || !cForm.name.trim()}>
                {createBusy && <span className="spinner" />}{t("admin.nodes.create.submit")}
              </button>
              <button className="ghost" onClick={() => setCreating(false)}>
                {t("admin.nodes.create.cancel")}</button>
            </div>
          </div>
        )}

        {/* 목록 */}
        {listBusy && !data
          ? <div className="empty-state"><span className="spinner" />{t("admin.loading")}</div>
          : items.length === 0
            ? <p className="hint" style={{ marginTop: 12 }}>{t("admin.nodes.empty")}</p>
            : <table>
                <thead><tr>
                  <th>{t("admin.nodes.th.name")}</th>
                  <th>{t("admin.nodes.th.id")}</th>
                  <th>{t("admin.nodes.th.type")}</th>
                  <th style={{ width: 110 }}>{t("admin.nodes.th.trust")}</th>
                  <th style={{ textAlign: "right", width: 80 }}>{t("admin.nodes.th.degree")}</th>
                </tr></thead>
                <tbody>
                  {items.map((n) => (
                    <tr key={n.node_id} onClick={() => openNode(n.node_id)}
                        style={{ cursor: "pointer",
                                 ...(sel?.node_id === n.node_id
                                   ? { background: "var(--accent-soft)" } : {}) }}>
                      <td><b>{n.name || n.node_id}</b></td>
                      <td className="mono" style={{ fontSize: "0.72rem" }}>{n.node_id}</td>
                      <td>{n.type || "—"}</td>
                      <td><TrustPill trust={n.trust} /></td>
                      <td className="mono" style={{ textAlign: "right" }}>
                        {(n.out_degree || 0) + (n.in_degree || 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>}

        {/* 페이지네이션 */}
        {total > 0 && (
          <div style={{ display: "flex", gap: 8, alignItems: "center",
                        justifyContent: "flex-end", marginTop: 12 }}>
            <span className="hint" style={{ marginTop: 0 }}>
              {t("admin.nodes.page", { from, to, total: total.toLocaleString() })}</span>
            <button className="ghost" style={{ padding: "4px 12px" }}
                    disabled={listBusy || offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - LIMIT))}>‹</button>
            <button className="ghost" style={{ padding: "4px 12px" }}
                    disabled={listBusy || offset + LIMIT >= total}
                    onClick={() => setOffset(offset + LIMIT)}>›</button>
          </div>
        )}
      </section>

      {/* ── 편집 패널 ── */}
      {sel && form && (
        <section className="card" style={{ borderColor: "var(--accent)" }}>
          <div className="admin-head">
            <div>
              <h2>{t("admin.nodes.edit.title")}</h2>
              <p className="sub" style={{ fontFamily: "ui-monospace, monospace",
                                          fontSize: "0.74rem" }}>{sel.node_id}</p>
            </div>
            <button className="ghost" onClick={closePanel}>{t("admin.nodes.edit.close")}</button>
          </div>
          {selBusy && <div className="empty-state"><span className="spinner" />{t("admin.loading")}</div>}

          <div className="row" style={{ marginTop: 14 }}>
            <div>
              <label>{t("admin.nodes.edit.name")}</label>
              <input type="text" disabled={readOnly} value={form.name}
                     onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </div>
            <div>
              <label>{t("admin.nodes.edit.type")}</label>
              <input type="text" list="admin-type-list" disabled={readOnly} value={form.type}
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
          <div style={{ marginTop: 12 }}>
            <label>{t("admin.nodes.edit.definition")}</label>
            <textarea rows={3} disabled={readOnly} value={form.definition}
                      onChange={(e) => setForm({ ...form, definition: e.target.value })} />
          </div>
          <div style={{ marginTop: 12 }}>
            <label>{t("admin.nodes.edit.aliases")}</label>
            <input type="text" disabled={readOnly} value={form.aliases}
                   onChange={(e) => setForm({ ...form, aliases: e.target.value })} />
          </div>

          {/* 커스텀 프로퍼티 — 내부 키 제외, ✕ = null 전송으로 삭제 */}
          <h2 style={{ fontSize: "0.95rem", marginTop: 18 }}>{t("admin.nodes.edit.custom")}</h2>
          {custom.filter((c) => !c.removed).length === 0 && (
            <p className="hint" style={{ marginTop: 6 }}>{t("admin.nodes.edit.custom.empty")}</p>
          )}
          {custom.map((c, i) => !c.removed && (
            <div key={`${c.key}_${i}`} className="prop-row">
              <input className="k-input" type="text" value={c.key} disabled={!c.added || readOnly}
                     onChange={(e) => setCustom(custom.map((x, j) =>
                       j === i ? { ...x, key: e.target.value } : x))} />
              <input type="text" value={c.value} disabled={readOnly}
                     onChange={(e) => setCustom(custom.map((x, j) =>
                       j === i ? { ...x, value: e.target.value } : x))} />
              {!readOnly && (
                <button className="tree-toggle" title="✕"
                        onClick={() => setCustom(c.added
                          ? custom.filter((_, j) => j !== i)
                          : custom.map((x, j) => j === i ? { ...x, removed: true } : x))}>
                  ✕</button>
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
                        setCustom([...custom, { key: newProp.k.trim(), value: newProp.v,
                                                orig: undefined, removed: false, added: true }]);
                        setNewProp({ k: "", v: "" });
                      }}>{t("admin.nodes.edit.addProp")}</button>
            </div>
          )}

          {/* 관계 — out / in, ✕ = DELETE /edges */}
          <h2 style={{ fontSize: "0.95rem", marginTop: 18 }}>{t("admin.nodes.edges.title")}</h2>
          <div className="admin-grid" style={{ marginTop: 8 }}>
            <div>
              <label>{t("admin.nodes.edges.out")}</label>
              {(sel.out_edges || []).length === 0 &&
                <p className="hint" style={{ marginTop: 4 }}>{t("admin.nodes.edges.none")}</p>}
              {(sel.out_edges || []).map((e, i) => (
                <div key={i} className="edge-row" style={{ cursor: "default" }}>
                  <span className="pred">—{e.predicate}→</span>
                  <span>{e.target_name || e.target}</span>
                  {!readOnly && (
                    <button className="tree-toggle" disabled={edgeBusy}
                            onClick={() => removeEdge(sel.node_id, e.predicate, e.target)}>
                      ✕</button>
                  )}
                </div>
              ))}
            </div>
            <div>
              <label>{t("admin.nodes.edges.in")}</label>
              {(sel.in_edges || []).length === 0 &&
                <p className="hint" style={{ marginTop: 4 }}>{t("admin.nodes.edges.none")}</p>}
              {(sel.in_edges || []).map((e, i) => (
                <div key={i} className="edge-row" style={{ cursor: "default" }}>
                  <span>{e.source_name || e.target_name || inSource(e)}</span>
                  <span className="pred">—{e.predicate}→</span>
                  {!readOnly && (
                    <button className="tree-toggle" disabled={edgeBusy}
                            onClick={() => removeEdge(inSource(e), e.predicate, sel.node_id)}>
                      ✕</button>
                  )}
                </div>
              ))}
            </div>
          </div>
          {!readOnly && (
            <div className="row" style={{ marginTop: 10, alignItems: "flex-end" }}>
              <div>
                <input type="text" value={edgeForm.predicate}
                       placeholder={t("admin.nodes.edges.predPh")}
                       onChange={(e) => setEdgeForm({ ...edgeForm, predicate: e.target.value })} />
              </div>
              <div>
                <input type="text" list="admin-node-ids" value={edgeForm.target}
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

          {/* 저장 · 거절(묘비) */}
          {!readOnly && (
            <div style={{ display: "flex", gap: 8, marginTop: 20, flexWrap: "wrap" }}>
              <button onClick={save} disabled={saveBusy || !dirty}>
                {saveBusy && <span className="spinner" />}
                {saveBusy ? t("admin.nodes.edit.saving") : t("admin.nodes.edit.save")}
              </button>
              {!rejecting ? (
                <button className="btn-danger" onClick={() => setRejecting(true)}>
                  {t("admin.nodes.reject.btn")}</button>
              ) : (
                <>
                  <input type="text" value={rejectReason} style={{ width: 260, flex: "0 1 auto" }}
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
        </section>
      )}

      {/* datalist — 타입은 스키마에서, node_id 는 현재 목록 페이지에서 */}
      <datalist id="admin-type-list">
        {types.map((ty) => <option key={ty} value={ty} />)}
      </datalist>
      <datalist id="admin-node-ids">
        {items.map((n) => <option key={n.node_id} value={n.node_id}>{n.name}</option>)}
      </datalist>
    </>
  );
}
