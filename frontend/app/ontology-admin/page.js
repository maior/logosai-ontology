"use client";

// 관리 콘솔 (/ontology-admin) — 온톨로지 운영자의 단일 창구.
// 백엔드: GET /admin/overview → 네임스페이스 상세는 탭 구조 —
//   탐색(ExploreTab, 3-pane) · 개요(stats) · 검수·감사 · 관리
//   (구 스키마/노드 탭은 ExploreTab 으로 통합 — SchemaTab/NodesTab 는 미사용 orphan)
// 원칙: 이 페이지는 REST 만 소비한다 (서비스 역할 경계 — 9275 는 관리 콘솔).
//       protected(default) 편집·삭제는 백엔드 403 이 최종 방어선이고, 여기의
//       읽기 전용 UI·이름 재입력 확인은 UX 안전장치다.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { hierarchy, pack, arc, pie } from "d3";
import { useT } from "../i18n";
import ExploreTab from "./ExploreTab";
import IngestTab from "./IngestTab";
import SystemPanel from "./SystemPanel";
import ManageTab from "./ManageTab";
import RetrieveTab from "./RetrieveTab";
import CoverageTab from "./CoverageTab";
import ReviewQueueBoard from "./ReviewQueueBoard";
import GoldenTab from "./GoldenTab";
import ExperimentTab from "./ExperimentTab";
import { CLUSTER_PALETTE } from "./ClusterView";

const API = process.env.NEXT_PUBLIC_ONTOLOGY_API || "http://localhost:9274";
const BASE = `${API}/api/v1/ontology`;

// 데이터 값(trust/action)은 번역하지 않는다 — i18n.js 규약과 동일
const TRUST_COLORS = {
  authoritative: "#059669", unknown: "#98a2b3", summary: "#d97706",
  unset: "#cbd5e1",
};
const TRUST_ORDER = ["authoritative", "unknown", "summary", "unset"];
const ACTION_COLORS = {
  confirm: "#059669", reject: "#dc2626", recommend: "#4f46e5",
  coverage_gap: "#d97706", create: "#0e7490", edit: "#6d28d9",
  edge_added: "#0e7490", edge_removed: "#9f1239",
};

const fmtBytes = (n) => {
  if (!n) return "0 B";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
};
const fmtNum = (n) => (n == null ? "—" : Number(n).toLocaleString());
const fmtTime = (iso, lang) => {
  try {
    return new Date(iso).toLocaleString(lang === "ko" ? "ko-KR" : "en-US",
      { dateStyle: "short", timeStyle: "medium" });
  } catch { return iso || ""; }
};
const detailMsg = async (res) => {
  try {
    const d = (await res.json()).detail;
    if (typeof d === "string" && d) return d;
    if (d) return JSON.stringify(d);
  } catch { /* 본문이 JSON 이 아니면 statusText 폴백 */ }
  return res.statusText;
};

// ─── 작은 시각 요소 ─────────────────────────────────────────────────

// 키·값 한 줄 — 시스템 카드용(SystemPanel/ManageTab 와 같은 마크업)
function KV({ k, v, mono = true }) {
  return (
    <div className="oa-sys-kv">
      <span className="oa-sys-kv-k">{k}</span>
      <span className={`oa-sys-kv-v ${mono ? "mono" : ""}`}>{v}</span>
    </div>
  );
}

function TrustBar({ trust, height = 8 }) {
  const total = Object.values(trust || {}).reduce((a, b) => a + b, 0);
  if (!total) return <div className="trust-bar" style={{ height }} />;
  return (
    <div className="trust-bar" style={{ height }}>
      {TRUST_ORDER.filter((k) => trust[k]).map((k) => (
        <span key={k} className="trust-seg" title={`${k}: ${trust[k]}`}
              style={{ width: `${(trust[k] / total) * 100}%`,
                       background: TRUST_COLORS[k] }} />
      ))}
    </div>
  );
}

function ReviewProgress({ review }) {
  const { t } = useT();
  const judged = (review?.confirmed || 0) + (review?.rejected || 0);
  const total = judged + (review?.pending || 0);
  const pct = total ? Math.round((judged / total) * 100) : 0;
  return (
    <div title={t("admin.reviewProgress", { done: judged, total })}>
      <div className="progress"><span style={{ width: `${pct}%` }} /></div>
      <span className="progress-label">
        {total ? `${pct}%` : "—"}
        {review?.pending > 0 &&
          <span className="pending-cnt"> · {review.pending}</span>}
      </span>
    </div>
  );
}

// 검색 인덱스 상태(관측) + 재색인 — 관리 콘솔 관측용(원시 ES 노출 아님).
function IndexPanel({ base, namespace, t, flash }) {
  const [st, setSt] = useState(null);
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      const r = await fetch(`${base}/graphs/${namespace}/index-status`);
      if (r.ok) setSt(await r.json());
    } catch { /* 관측용 — 조용히 실패 */ }
  }, [base, namespace]);
  useEffect(() => { load(); }, [load]);
  const reindex = async () => {
    setBusy(true);
    try {
      const r = await fetch(`${base}/graphs/${namespace}/reindex`, { method: "POST" });
      if (r.ok) { flash?.(t("admin.index.reindexed")); await load(); }
    } catch { /* noop */ }
    setBusy(false);
  };
  if (!st) return null;
  return (
    <section className="card" style={{ marginTop: 18 }}>
      <h2>{t("admin.index.title")}</h2>
      <div className="oa-index-rows">
        <div><span>{t("admin.index.es")}</span>
          <b className={st.es_available ? "oa-index-ok" : "oa-index-warn"}>
            {st.es_available ? t("admin.index.up") : t("admin.index.down")}</b></div>
        <div><span>{t("admin.index.nodeDocs")}</span>
          <b>{st.node_docs ?? "—"} / {st.nodes.toLocaleString()}</b>
          {st.node_in_sync === false && <span className="oa-index-warn"> · {t("admin.index.drift")}</span>}
          {st.node_in_sync === true && <span className="oa-index-ok"> · {t("admin.index.synced")}</span>}</div>
        <div><span>{t("admin.index.vectors")}</span>
          <b className={st.vector_search ? "oa-index-ok" : "oa-index-warn"}>
            {st.vector_search ? t("admin.index.on") : t("admin.index.off")}</b>
          <span> · {st.chunks.toLocaleString()} {t("admin.index.chunks")}</span></div>
      </div>
      <button className="ghost" style={{ marginTop: 12 }} disabled={busy} onClick={reindex}>
        {busy && <span className="spinner" />}{t("admin.index.reindex")}</button>
      <p className="hint" style={{ marginTop: 8 }}>{t("admin.index.hint")}</p>
    </section>
  );
}

// 네임스페이스 타입 분포 도넛 — stats.node_types {type: count} (d3 v7)
function NsTypeDonut({ nodeTypes, t }) {
  const SZ = 180, R = 74, IR = 44;
  const arcs = useMemo(() => {
    const items = Object.entries(nodeTypes || {})
      .map(([type, count]) => ({ type, count }))
      .filter((d) => d.count > 0)
      .sort((a, b) => b.count - a.count);
    if (!items.length) return [];
    const gen = pie().value((d) => d.count).sort(null);
    const a = arc().innerRadius(IR).outerRadius(R);
    return gen(items).map((p, i) => ({
      d: a(p), item: p.data, color: CLUSTER_PALETTE[i % CLUSTER_PALETTE.length],
    }));
  }, [nodeTypes]);
  const total = useMemo(
    () => Object.values(nodeTypes || {}).reduce((a, b) => a + b, 0), [nodeTypes]);
  if (!arcs.length) return <p className="hint">{t("admin.sys.noData")}</p>;
  return (
    <div className="oa-sys-donut-wrap">
      <svg viewBox={`0 0 ${SZ} ${SZ}`} role="img" aria-label={t("admin.nsstore.typeDist")}
           style={{ width: 156, height: 156, flexShrink: 0 }}>
        <g transform={`translate(${SZ / 2},${SZ / 2})`}>
          {arcs.map((a) => (
            <path key={a.item.type} d={a.d} fill={a.color} fillOpacity={0.82}
                  stroke="var(--surface)" strokeWidth={1.5}>
              <title>{`${a.item.type}: ${a.item.count.toLocaleString()}`}</title>
            </path>
          ))}
          <text textAnchor="middle" y={-2} fontSize={15} fontWeight={800} fill="var(--ink)">
            {total.toLocaleString()}</text>
          <text textAnchor="middle" y={14} fontSize={9} fill="var(--faint)">
            {t("admin.stat.nodes")}</text>
        </g>
      </svg>
      <div className="legend oa-sys-donut-legend">
        {arcs.slice(0, 8).map((a) => (
          <span key={a.item.type} className="legend-item" title={a.item.type}>
            <span className="dot" style={{ background: a.color }} />
            <span className="oa-sys-idxname">{a.item.type}</span>
            <b>{a.item.count.toLocaleString()}</b>
          </span>
        ))}
      </div>
    </div>
  );
}

// 네임스페이스 스토리지·상세 — 전역 지표(/admin/system)를 이 ns 로 클라이언트
// 필터 + index-status. 새 백엔드 엔드포인트 없이 개요를 이 ns 로 좁힌다.
// 각 원천은 독립적으로 degrade 한다(엔드포인트 하나가 죽어도 나머지는 그대로).
function NamespaceStorageDetail({ base, namespace, entry, stats, t }) {
  const [system, setSystem] = useState(null);
  const [ix, setIx] = useState(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch(`${base}/admin/system`);
      if (r.ok) setSystem(await r.json());
    } catch { /* degrade */ }
    try {
      const r = await fetch(`${base}/graphs/${namespace}/index-status`);
      if (r.ok) setIx(await r.json());
    } catch { /* degrade */ }
  }, [base, namespace]);
  useEffect(() => { load(); }, [load]);

  const es = system?.es, pg = system?.pg, vector = system?.vector;
  const esIdxName = `ontology-obj-${namespace.toLowerCase()}`;
  const esIdx = (es?.indices || []).find(
    (i) => i.index === esIdxName || (ix?.es_index && i.index === ix.es_index));
  // npy 캐시 네임스페이스는 파일명 유래(예: chunks_AI-Coach) — chunks_ 벗겨 매칭
  const cache = (vector?.caches || []).find(
    (c) => c.namespace === namespace
        || String(c.namespace).replace(/^chunks_/, "") === namespace);
  const pgNode = (pg?.nodes_by_namespace || []).find((n) => n.namespace === namespace);

  const nodes = stats?.nodes ?? entry?.nodes;
  const edges = stats?.edges ?? entry?.edges;
  const chunks = ix?.chunks ?? stats?.chunks ?? entry?.chunks;

  return (
    <section className="card" style={{ marginTop: 18 }}>
      <h2>{t("admin.nsstore.title")}</h2>
      <p className="hint" style={{ marginTop: 2 }}>{t("admin.nsstore.hint")}</p>
      <div className="oa-nsstore">
        {/* 그래프 + 타입 분포 도넛 */}
        <div className="oa-sys-card">
          <div className="oa-sys-card-h"><span>{t("admin.nsstore.graph")}</span></div>
          <NsTypeDonut nodeTypes={stats?.node_types} t={t} />
          <div style={{ marginTop: 8 }}>
            <KV k={t("admin.stat.nodes")} v={fmtNum(nodes)} />
            <KV k={t("admin.stat.edges")} v={fmtNum(edges)} />
            <KV k={t("admin.stat.chunks")} v={fmtNum(chunks)} />
          </div>
        </div>
        {/* ES 인덱스 */}
        <div className="oa-sys-card">
          <div className="oa-sys-card-h"><span>Elasticsearch</span>
            {ix && <span className={`oa-sys-badge ${ix.es_available ? "ok" : "down"}`}>
              {ix.es_available ? t("admin.sys.live") : t("admin.sys.down")}</span>}</div>
          <KV k={t("admin.nsstore.index")} v={esIdx?.index || esIdxName} />
          <KV k={t("admin.sys.totalDocs")} v={
            <>{fmtNum(esIdx?.docs ?? ix?.node_docs)} / {fmtNum(ix?.nodes ?? nodes)}
              {ix?.node_in_sync === false && <span className="oa-index-warn"> ⚠</span>}
              {ix?.node_in_sync === true && <span className="oa-index-ok"> ✓</span>}</>} />
          <KV k={t("admin.sys.storeSize")} v={esIdx ? fmtBytes(esIdx.size_bytes) : "—"} />
          <KV k={t("admin.sys.model")} v={<span className="oa-sys-model">{es?.embedding_model || "—"}</span>} mono={false} />
          <KV k={t("admin.sys.dim")} v={fmtNum(es?.dim)} />
        </div>
        {/* 벡터 캐시 + PG */}
        <div className="oa-sys-card">
          <div className="oa-sys-card-h"><span>{t("admin.nsstore.vectorPg")}</span>
            {ix && <span className={`oa-sys-badge ${ix.vector_search ? "ok" : "down"}`}>
              {ix.vector_search ? t("admin.index.on") : t("admin.index.off")}</span>}</div>
          <KV k={t("admin.sys.cacheSize")} v={cache ? fmtBytes(cache.size_bytes) : "—"} />
          <KV k={t("admin.sys.vectors")} v={fmtNum(cache?.vectors)} />
          <KV k={t("admin.sys.chunkCfg")} v={vector ? `${vector.chunk_size} / ${vector.overlap}` : "—"} />
          <KV k={t("admin.nsstore.pgNodes")} v={pg?.available ? fmtNum(pgNode?.count ?? 0) : "—"} />
        </div>
      </div>
    </section>
  );
}

const pill = (bg) => ({
  display: "inline-block", padding: "2px 9px", borderRadius: 999,
  background: bg || "#98a2b3", color: "#fff", fontSize: "0.68rem",
  fontWeight: 700, letterSpacing: "0.3px",
});

// 네임스페이스 상태 색 — 대시보드 버블·행 공용
const nsHealth = (e) => {
  if (e.protected) return "#4f46e5";                 // 보호(agent-routing)
  if (e.empty || !e.nodes) return "#98a2b3";         // 빈 프로젝트
  if ((e.review?.pending || 0) / Math.max(1, e.nodes) > 0.5) return "#b54708"; // 검수 대기 많음
  return "#027a48";                                  // 정상
};

// 대시보드 버블 — D3 circle pack (크기=노드수, 색=상태), 클릭 진입
function NamespaceBubbles({ entries, onEnter, t }) {
  const W = 1000, H = 300;
  const leaves = useMemo(() => {
    const items = (entries || []).filter((e) => !("error" in e));
    if (!items.length) return [];
    // 면적을 √노드로 — 9~수천 노드의 큰 편차를 압축해 작은 네임스페이스도
    // 보이고 클릭 가능하게(면적∝노드면 작은 것이 안 보임)
    const root = hierarchy({ children: items })
      .sum((d) => Math.sqrt(Math.max(1, d.nodes || 0)));
    pack().size([W, H]).padding(6)(root);
    return root.leaves();
  }, [entries]);
  if (!leaves.length) return null;
  return (
    <section className="card">
      <h2>{t("admin.bubbles.title")}</h2>
      <p className="hint" style={{ marginTop: 2, marginBottom: 8 }}>{t("admin.bubbles.hint")}</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="oa-bubbles" role="img"
           aria-label={t("admin.bubbles.title")}>
        {leaves.map((lf, i) => {
          const e = lf.data;
          const fill = CLUSTER_PALETTE[i % CLUSTER_PALETTE.length];  // 네임스페이스별 구분색
          const ring = nsHealth(e);                                  // 테두리 = 상태
          const lbl = lf.r > 22;
          return (
            <g key={e.namespace} className="oa-bubble" onClick={() => onEnter(e.namespace)}>
              <title>{`${e.namespace} · ${(e.nodes || 0).toLocaleString()} 노드`}</title>
              <circle cx={lf.x} cy={lf.y} r={lf.r} fill={fill} fillOpacity={0.42}
                      stroke={ring} strokeWidth={1.4} strokeOpacity={0.9} />
              {lbl && (
                <text x={lf.x} y={lf.y - 2} textAnchor="middle" className="oa-bubble-name"
                      fontSize={lf.r > 50 ? 13 : 11}>{e.protected ? "🔒 " : ""}{e.namespace}</text>
              )}
              {lbl && (
                <text x={lf.x} y={lf.y + 14} textAnchor="middle" className="oa-bubble-count"
                      fontSize={10}>{(e.nodes || 0).toLocaleString()}</text>
              )}
            </g>
          );
        })}
      </svg>
    </section>
  );
}

const DOMAINS = ["generic", "heritage", "legal", "medical"];

// ─── LLM 설정 뷰 (전역) — GET/PUT /admin/llm · POST /admin/llm/test ──
function LlmSettings({ t, flash }) {
  const [cfg, setCfg] = useState(null);          // 서버 저장본
  const [form, setForm] = useState(null);        // {provider, model, base_url}
  const [busy, setBusy] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testRes, setTestRes] = useState(null);  // {ok, sample?, error?}
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    setErr("");
    try {
      const res = await fetch(`${BASE}/admin/llm`);
      if (!res.ok) throw new Error(await detailMsg(res));
      const d = await res.json();
      setCfg(d);
      setForm({ provider: d.provider, model: d.model || "", base_url: d.base_url || "" });
    } catch (e) { setErr(t("admin.set.err.load", { e: e.message || e })); }
  }, [t]);
  useEffect(() => { load(); }, [load]);

  const save = async () => {
    if (!form) return;
    setBusy(true); setErr(""); setTestRes(null);
    try {
      const res = await fetch(`${BASE}/admin/llm`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      const d = await res.json();
      setCfg(d);
      setForm({ provider: d.provider, model: d.model || "", base_url: d.base_url || "" });
      flash(t("admin.set.saved"));
    } catch (e) { setErr(t("admin.set.err.save", { e: e.message || e })); }
    setBusy(false);
  };

  const test = async () => {
    setTesting(true); setTestRes(null); setErr("");
    try {
      const res = await fetch(`${BASE}/admin/llm/test`, { method: "POST" });
      setTestRes(await res.json());   // 항상 200 — {ok, sample?, error?}
    } catch (e) { setErr(t("admin.set.err.test", { e: e.message || e })); }
    setTesting(false);
  };

  return (
    <>
      <div className="oa-context">
        <div>
          <h1 className="oa-h1">{t("admin.set.title")}</h1>
          <p className="oa-sub">{t("admin.set.subtitle")}</p>
        </div>
      </div>
      {err && <div className="error" style={{ marginTop: 14 }}>{err}</div>}
      {!cfg ? (
        <div className="empty-state"><span className="spinner" />{t("admin.loading")}</div>
      ) : (
        <section className="card" style={{ marginTop: 16 }}>
          {cfg.injected && <div className="oa-set-injected">⚑ {t("admin.set.injected")}</div>}
          <div className="oa-set-form">
            <div>
              <label>{t("admin.set.provider")}</label>
              <select value={form.provider}
                      onChange={(e) => setForm({ ...form, provider: e.target.value })}>
                {(cfg.providers || []).map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <label>{t("admin.set.model")}</label>
              <input type="text" value={form.model} placeholder={cfg.default_model || ""}
                     onChange={(e) => setForm({ ...form, model: e.target.value })} />
            </div>
            {form.provider === "openai_compatible" && (
              <div>
                <label>{t("admin.set.baseUrl")}</label>
                <input type="text" value={form.base_url} placeholder="http://localhost:8000/v1"
                       onChange={(e) => setForm({ ...form, base_url: e.target.value })} />
              </div>
            )}
            <div>
              <label>{t("admin.set.key")}</label>
              <div className="oa-set-key">
                <span className="oa-set-envvar">{cfg.env_var}</span>
                <span>·</span>
                {cfg.key_present ? (
                  <span className="oa-set-present">
                    {t("admin.set.keyDetected")} <span className="oa-set-hint">{cfg.key_hint}</span>
                  </span>
                ) : (
                  <span className="oa-set-absent">{t("admin.set.keyMissing")}</span>
                )}
              </div>
              <p className="oa-set-keynote">{t("admin.set.keyNote")}</p>
            </div>
            <div className="oa-set-actions">
              <button onClick={save} disabled={busy}>
                {busy && <span className="spinner" />}{busy ? t("admin.set.saving") : t("admin.set.save")}
              </button>
              <button className="ghost" onClick={test} disabled={testing}>
                {testing && <span className="spinner" />}{testing ? t("admin.set.testing") : t("admin.set.test")}
              </button>
            </div>
            {testRes && (
              <div className={`oa-set-testres ${testRes.ok ? "ok" : "fail"}`}>
                {testRes.ok
                  ? t("admin.set.testOk", { sample: testRes.sample || "" })
                  : t("admin.set.testFail", { error: testRes.error || "" })}
              </div>
            )}
          </div>
        </section>
      )}
    </>
  );
}

// ─── 페이지 ─────────────────────────────────────────────────────────

export default function OntologyAdmin() {
  const { t, lang, setLang } = useT();

  const [overview, setOverview] = useState(null);   // null = 로딩 전
  const [loading, setLoading] = useState(false);
  const [namespace, setNamespace] = useState("");   // "" = 전체(대시보드)
  const [section, setSection] = useState("dashboard"); // dashboard|overview|ingest|explore|review|manage|settings
  const [stats, setStats] = useState(null);
  const [tombstones, setTombstones] = useState([]);
  const [history, setHistory] = useState([]);
  const [actionFilter, setActionFilter] = useState("");
  const [detailBusy, setDetailBusy] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [creatingProject, setCreatingProject] = useState(false);
  const [projForm, setProjForm] = useState({ name: "", description: "", domain: "generic" });
  const [projBusy, setProjBusy] = useState(false);
  // 네임스페이스 콤보(상단) — 열림 · 검색 · 바깥클릭/ESC 닫기
  const [nsOpen, setNsOpen] = useState(false);
  const [nsSearch, setNsSearch] = useState("");
  const noticeRef = useRef(null);
  const nselRef = useRef(null);

  const flash = (msg) => {
    setNotice(msg);
    clearTimeout(noticeRef.current);
    noticeRef.current = setTimeout(() => setNotice(""), 5000);
  };
  useEffect(() => () => clearTimeout(noticeRef.current), []);

  const loadOverview = useCallback(async () => {
    setLoading(true); setError("");
    try {
      const res = await fetch(`${BASE}/admin/overview`);
      if (!res.ok) throw new Error(await detailMsg(res));
      setOverview(await res.json());
    } catch (e) { setError(t("admin.err.load", { e: e.message || e })); }
    setLoading(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { loadOverview(); }, [loadOverview]);

  // 상세 데이터만 다시 읽는다 — 탭 상태는 건드리지 않는다
  const loadDetail = useCallback(async (ns) => {
    if (!ns) return;
    setDetailBusy(true); setError("");
    try {
      const [s, tomb, hist] = await Promise.all([
        fetch(`${BASE}/graphs/${ns}/stats`),
        fetch(`${BASE}/graphs/${ns}/tombstones`),
        fetch(`${BASE}/graphs/${ns}/review/history?limit=200`),
      ]);
      if (!s.ok) throw new Error(await detailMsg(s));
      setStats(await s.json());
      if (tomb.ok) setTombstones((await tomb.json()).tombstones || []);
      if (hist.ok) setHistory((await hist.json()).history || []);
    } catch (e) { setError(t("admin.err.detail", { e: e.message || e })); }
    setDetailBusy(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 네임스페이스 선택(=explore 진입) · ns="" → 전체 대시보드
  const openDetail = useCallback(async (ns) => {
    setNamespace(ns); setConfirmText(""); setActionFilter("");
    setSection(ns ? "explore" : "dashboard");
    setStats(null); setTombstones([]); setHistory([]);
    if (!ns) return;
    await loadDetail(ns);
  }, [loadDetail]);

  // 레일 섹션 이동 — 개요/탐색/검수/관리는 네임스페이스 선택 시에만
  const goSection = useCallback((sec) => {
    if (sec === "dashboard") { openDetail(""); return; }
    if (sec === "settings") { setSection("settings"); return; }
    if (!namespace) return;
    setSection(sec);
  }, [openDetail, namespace]);

  // 탭 컴포넌트가 노드 수를 바꿨을 때 (생성·거절) — 부모 수치 동기화
  const handleChildChange = useCallback(() => {
    if (namespace) loadDetail(namespace);
    loadOverview();
  }, [namespace, loadDetail, loadOverview]);

  // 수집 완료 — 대상 네임스페이스(새 프로젝트일 수 있다)의 개요로 착지
  const ingestDone = useCallback(async (ns) => {
    await loadOverview();
    setNamespace(ns);
    await loadDetail(ns);
    setSection("overview");
  }, [loadOverview, loadDetail]);

  // 감사 뷰는 항상 신선해야 한다 — 노드 탭에서 편집한 직후 열어도
  // 최초 로드 스냅샷("이력 없음")이 아니라 방금의 edit 이벤트가 보이게,
  // 검수·감사 탭 진입 시마다 재조회한다.
  useEffect(() => {
    if (section === "review" && namespace) loadDetail(namespace);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [section]);

  // 콤보 바깥 클릭 / ESC 로 닫기
  useEffect(() => {
    if (!nsOpen) return;
    const onDown = (e) => { if (nselRef.current && !nselRef.current.contains(e.target)) setNsOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setNsOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [nsOpen]);

  const deleteNamespace = async () => {
    if (!stats || confirmText !== namespace) return;
    setDeleteBusy(true); setError("");
    try {
      const res = await fetch(`${BASE}/graphs/${namespace}`,
                              { method: "DELETE" });
      if (!res.ok) throw new Error(await detailMsg(res));
      const body = await res.json();
      flash(t("admin.danger.deleted",
              { ns: body.namespace, n: (body.deleted_files || []).length }));
      setNamespace(""); setSection("dashboard"); setStats(null);
      await loadOverview();
    } catch (e) { setError(t("admin.danger.err", { e: e.message || e })); }
    setDeleteBusy(false);
  };

  // 새 프로젝트 생성 (빈 온톨로지) — POST /admin/projects
  const createProject = async () => {
    const name = projForm.name.trim();
    if (!name) return;
    setProjBusy(true); setError("");
    try {
      const body = { name };
      if (projForm.description.trim()) body.description = projForm.description.trim();
      if (projForm.domain) body.domain = projForm.domain;
      const res = await fetch(`${BASE}/admin/projects`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.status === 409) { flash(t("admin.proj.err.dupe")); setProjBusy(false); return; }
      if (res.status === 400) { flash(t("admin.proj.err.format")); setProjBusy(false); return; }
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.proj.done", { name }));
      setCreatingProject(false);
      setProjForm({ name: "", description: "", domain: "generic" });
      await loadOverview();
      await openDetail(name);   // 갓 만든 빈 프로젝트를 연다 (온보딩 노출)
    } catch (e) { setError(t("admin.proj.err", { e: e.message || e })); }
    setProjBusy(false);
  };

  const entries = overview?.namespaces || [];
  const totals = overview?.totals;
  const selEntry = entries.find((e) => e.namespace === namespace) || null;
  const actions = [...new Set(history.map((h) => h.action))];
  const shownHistory = actionFilter
    ? history.filter((h) => h.action === actionFilter) : history;

  // 헬스 신호: 로드 실패=red · 빈 프로젝트=muted · 미검수 노드 있음=warn · 그 외=ok
  const nsHealth = (e) =>
    "error" in e ? "err" : (e.empty ? "muted" : (e.review?.pending > 0 ? "warn" : "ok"));

  // 콤보 목록 필터 (클라이언트) + 섹션 레일 정의
  const nsQuery = nsSearch.trim().toLowerCase();
  const nsFiltered = nsQuery
    ? entries.filter((e) => e.namespace.toLowerCase().includes(nsQuery)) : entries;
  const NAV_SECTIONS = [
    { key: "overview", icon: "◎", label: "admin.nav.overviewSec", tip: "admin.nav.overviewTip" },
    { key: "ingest",   icon: "⬆", label: "admin.nav.ingest",      tip: "admin.nav.ingestTip" },
    { key: "explore",  icon: "⬡", label: "admin.nav.explore",     tip: "admin.nav.exploreTip" },
    // 커버리지 지도 — "비어 있는 구간이 어느 절인가" (회복 루프의 타깃 선정)
    { key: "coverage", icon: "▦", label: "admin.nav.coverage",    tip: "admin.nav.coverageTip" },
    // 검색 Explain — 그래프-조건부 검색의 근거 사슬을 펼쳐 보는 탭(우리 차별점 가시화)
    { key: "retrieve", icon: "🔍", label: "admin.nav.retrieve",    tip: "admin.nav.retrieveTip" },
    // 골든셋 — 검색을 채점하는 자(ruler). 상수 튜닝의 전제.
    { key: "golden",   icon: "◎", label: "admin.nav.golden",      tip: "admin.nav.goldenTip" },
    // 실험 — 조합(채널×knob) 품질×비용 파레토 (실험 하네스 Tier 0)
    { key: "experiment", icon: "⚗", label: "admin.nav.experiment", tip: "admin.nav.experimentTip" },
    { key: "review",   icon: "✓", label: "admin.nav.review",      tip: "admin.nav.reviewTip" },
    { key: "manage",   icon: "⚙", label: "admin.nav.manage",      tip: "admin.nav.manageTip" },
  ];
  const SECTION_LABEL = {
    dashboard: "admin.nav.dashboard", overview: "admin.nav.overviewSec",
    ingest: "admin.nav.ingest",
    explore: "admin.nav.explore", coverage: "admin.nav.coverage",
    retrieve: "admin.nav.retrieve",
    golden: "admin.nav.golden", experiment: "admin.nav.experiment",
    review: "admin.nav.review",
    manage: "admin.nav.manage", settings: "admin.nav.settings",
  };

  return (
    <div className="oa">
      {/* ─── 상단 커맨드 바 ─── */}
      <header className="oa-topbar">
        <div className="oa-brand">
          <span className="oa-logo">◆</span>
          <span className="oa-wordmark">ONTOLOGY&nbsp;ADMIN</span>
          <span className="oa-env">:9274</span>
        </div>

        {/* 네임스페이스 콤보 (컨텍스트 선택) */}
        <div className="oa-nsel" ref={nselRef}>
          <button className={`oa-nsel-btn ${nsOpen ? "on" : ""}`}
                  aria-label={t("admin.nsel.label")}
                  onClick={() => setNsOpen((v) => !v)}>
            {namespace ? (
              <>
                <span className={`oa-health ${selEntry ? nsHealth(selEntry) : "ok"}`} />
                <span className="oa-nsel-cur">{selEntry?.protected && "🔒 "}{namespace}</span>
              </>
            ) : (
              <span className="oa-nsel-cur oa-nsel-all">▦ {t("admin.nsel.all")}</span>
            )}
            <span className="oa-nsel-caret">▾</span>
          </button>
          {nsOpen && (
            <div className="oa-nsel-panel">
              <input className="oa-nsel-search" type="text" autoFocus
                     placeholder={t("admin.nsel.searchPh")} value={nsSearch}
                     onChange={(e) => setNsSearch(e.target.value)} />
              <div className="oa-nsel-list">
                <button className={`oa-nsel-item ${!namespace ? "on" : ""}`}
                        onClick={() => { openDetail(""); setNsOpen(false); setNsSearch(""); }}>
                  <span className="oa-nsel-item-icon">▦</span>
                  <span className="oa-nsel-item-name">{t("admin.nsel.dashboardAll")}</span>
                </button>
                {nsFiltered.map((e) => (
                  <button key={e.namespace}
                          className={`oa-nsel-item ${namespace === e.namespace ? "on" : ""}`}
                          onClick={() => { openDetail(e.namespace); setNsOpen(false); setNsSearch(""); }}>
                    <span className={`oa-health ${nsHealth(e)}`}
                          title={"error" in e ? t("admin.nsError") : ""} />
                    <span className="oa-nsel-item-name">{e.protected && "🔒 "}{e.namespace}</span>
                    <span className="oa-nsel-item-meta">
                      {"error" in e ? "—" : `${e.nodes.toLocaleString()}N`}</span>
                  </button>
                ))}
                {nsFiltered.length === 0 && (
                  <div className="oa-nsel-empty">
                    {overview === null ? t("admin.loading") : t("admin.nsel.empty")}</div>
                )}
              </div>
              <button className="oa-nsel-new"
                      onClick={() => { openDetail(""); setCreatingProject(true); setNsOpen(false); setNsSearch(""); }}>
                {t("admin.proj.new")}
              </button>
            </div>
          )}
        </div>

        {/* 현재 섹션 (작은 브레드크럼) */}
        <nav className="oa-crumbs">
          <span className="oa-crumb cur">{t(SECTION_LABEL[section] || "admin.nav.dashboard")}</span>
        </nav>

        <div className="oa-topbar-right">
          <div className="oa-lang">
            {["en", "ko"].map((l) => (
              <button key={l} className={`oa-lang-btn ${lang === l ? "on" : ""}`}
                      onClick={() => setLang(l)}>{l === "en" ? "EN" : "한국어"}</button>
            ))}
          </div>
          <a href="/" className="oa-topbar-link">{t("admin.backToBuilder")}</a>
          <span className="oa-status" title="backend :9274 · frontend :9275">
            <span className="oa-dot ok" />:9274
            <span className="oa-dot ok" style={{ marginLeft: 8 }} />:9275
          </span>
        </div>
      </header>

      <div className="oa-body">
        {/* ─── 섹션 아이콘 레일 ─── */}
        <nav className="oa-nav">
          <button className={`oa-nav-item ${section === "dashboard" ? "on" : ""}`}
                  title={t("admin.nav.dashboardTip")} onClick={() => openDetail("")}>
            <span className="oa-nav-icon">▦</span>
            <span className="oa-nav-label">{t("admin.nav.dashboard")}</span>
          </button>
          <div className="oa-nav-div" />
          {NAV_SECTIONS.map((s) => {
            const dim = !namespace;
            return (
              <button key={s.key}
                      className={`oa-nav-item ${section === s.key ? "on" : ""} ${dim ? "dim" : ""}`}
                      title={dim ? t("admin.nav.needNamespace") : t(s.tip)}
                      onClick={() => goSection(s.key)}>
                <span className="oa-nav-icon">{s.icon}</span>
                <span className="oa-nav-label">{t(s.label)}</span>
              </button>
            );
          })}
          <div className="oa-nav-spacer" />
          <button className={`oa-nav-item ${section === "settings" ? "on" : ""}`}
                  title={t("admin.nav.settingsTip")} onClick={() => setSection("settings")}>
            <span className="oa-nav-icon">🔧</span>
            <span className="oa-nav-label">{t("admin.nav.settings")}</span>
          </button>
        </nav>

        {/* ─── Main ─── */}
        <main className="oa-main">
          {error && <div className="error" style={{ marginBottom: 14 }}>{error}</div>}
          {notice && <div className="admin-notice">{notice}</div>}

          {/* ── 설정 뷰 (전역 LLM) ── */}
          {section === "settings" && <LlmSettings t={t} flash={flash} />}

          {/* ── 대시보드 (전체) ── */}
          {section === "dashboard" && (
            <>
              <div className="oa-context">
                <div>
                  <h1 className="oa-h1">{t("admin.title")}</h1>
                  <p className="oa-sub">{t("admin.subtitle")}</p>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={() => setCreatingProject((v) => !v)}>
                    {t("admin.proj.new")}
                  </button>
                  <button className="ghost" onClick={loadOverview} disabled={loading}>
                    {loading && <span className="spinner" />}{t("admin.refresh")}
                  </button>
                </div>
              </div>

              {creatingProject && (
                <div className="card soft" style={{ marginTop: 14 }}>
                  <h2 style={{ fontSize: "0.9rem" }}>{t("admin.proj.title")}</h2>
                  <div className="row" style={{ marginTop: 10 }}>
                    <div>
                      <label>{t("admin.proj.name")}</label>
                      <input type="text" value={projForm.name} placeholder={t("admin.proj.namePh")}
                             onChange={(e) => setProjForm({ ...projForm, name: e.target.value })}
                             onKeyDown={(e) => { if (e.key === "Enter") createProject(); }} />
                    </div>
                    <div>
                      <label>{t("admin.proj.desc")}</label>
                      <input type="text" value={projForm.description}
                             onChange={(e) => setProjForm({ ...projForm, description: e.target.value })} />
                    </div>
                    <div>
                      <label>{t("admin.proj.domain")}</label>
                      <select value={projForm.domain}
                              onChange={(e) => setProjForm({ ...projForm, domain: e.target.value })}>
                        {DOMAINS.map((d) => <option key={d} value={d}>{d}</option>)}
                      </select>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                    <button onClick={createProject} disabled={projBusy || !projForm.name.trim()}>
                      {projBusy && <span className="spinner" />}
                      {projBusy ? t("admin.proj.creating") : t("admin.proj.create")}
                    </button>
                    <button className="ghost" onClick={() => setCreatingProject(false)}>
                      {t("admin.proj.cancel")}</button>
                  </div>
                </div>
              )}

              {overview === null && !error && (
                <div className="empty-state"><span className="spinner" />{t("admin.loading")}</div>
              )}

              {totals && (
                <div className="oa-metrics">
                  <div className="oa-metric"><span className="oa-metric-v">{totals.namespaces}</span>
                    <span className="oa-metric-k">{t("admin.stat.namespaces")}</span></div>
                  <div className="oa-metric"><span className="oa-metric-v">{totals.nodes.toLocaleString()}</span>
                    <span className="oa-metric-k">{t("admin.stat.nodes")}</span></div>
                  <div className="oa-metric"><span className="oa-metric-v">{totals.edges.toLocaleString()}</span>
                    <span className="oa-metric-k">{t("admin.stat.edges")}</span></div>
                  <div className="oa-metric"><span className="oa-metric-v">{totals.chunks.toLocaleString()}</span>
                    <span className="oa-metric-k">{t("admin.stat.chunks")}</span></div>
                  <div className="oa-metric">
                    <span className="oa-metric-v" style={totals.pending_reviews ? { color: "var(--amber)" } : {}}>
                      {totals.pending_reviews.toLocaleString()}</span>
                    <span className="oa-metric-k">{t("admin.stat.pendingReviews")}</span></div>
                </div>
              )}

            {overview && entries.length > 0 && (
              <NamespaceBubbles entries={entries} onEnter={openDetail} t={t} />
            )}

            {overview && <SystemPanel t={t} />}

            {/* 전체(전역) 관리 콘솔 — 모든 네임스페이스(노드/관계/청크·인덱스 동기·
                디스크) + 행별 액션 + 작업 이력. 버블 지도로 진입, 상단 콤보로 선택.
                namespace prop 없음 → ManageTab 이 전 네임스페이스 모드로 동작. */}
            {overview && entries.length > 0 && (
              <div style={{ marginTop: 18 }}>
                <ManageTab
                  base={BASE}
                  entries={entries}
                  t={t}
                  flash={flash}
                  onChanged={handleChildChange}
                  onNamespaceDeleted={(ns) => {
                    if (ns === namespace) {
                      setNamespace(""); setSection("dashboard"); setStats(null);
                    }
                  }}
                />
              </div>
            )}
          </>
        )}

        {/* ── 네임스페이스 상세 (섹션별) ── */}
        {section !== "dashboard" && section !== "settings" && namespace && (
          <>
            <div className="oa-context">
              <div>
                <h1 className="oa-h1">
                  {namespace}
                  {stats?.protected &&
                    <span className="oa-tag lock">🔒 {t("admin.protected")}</span>}
                  {selEntry?.empty &&
                    <span className="oa-tag" style={{ background: "#232a3c", color: "var(--ink-2)" }}>
                      {t("admin.proj.emptyBadge")}</span>}
                </h1>
                <p className="oa-sub">
                  {selEntry?.domain &&
                    <span className="badge" style={{ marginRight: 8 }}>{selEntry.domain}</span>}
                  {selEntry?.description || t("admin.detail.subtitle")}
                </p>
              </div>
              <button className="ghost" onClick={() => loadDetail(namespace)}
                      disabled={detailBusy}>
                {detailBusy && <span className="spinner" />}{t("admin.refresh")}
              </button>
            </div>

            {detailBusy && !stats && (
              <div className="empty-state"><span className="spinner" />{t("admin.loading")}</div>
            )}

            {stats && (
              <>
                {/* 지표 스트립 — 모든 탭에서 상시 노출 (컨텍스트) */}
                <div className="oa-metrics">
                  <div className="oa-metric"><span className="oa-metric-v">{stats.nodes.toLocaleString()}</span>
                    <span className="oa-metric-k">{t("admin.stat.nodes")}</span></div>
                  <div className="oa-metric"><span className="oa-metric-v">{stats.edges.toLocaleString()}</span>
                    <span className="oa-metric-k">{t("admin.stat.edges")}</span></div>
                  <div className="oa-metric"><span className="oa-metric-v">{stats.chunks.toLocaleString()}</span>
                    <span className="oa-metric-k">{t("admin.stat.chunks")}</span></div>
                  <div className="oa-metric"><span className="oa-metric-v">{stats.golden_cases}</span>
                    <span className="oa-metric-k">{t("admin.stat.golden")}</span></div>
                  <div className="oa-metric">
                    <span className="oa-metric-v" style={stats.review.pending ? { color: "var(--amber)" } : {}}>
                      {stats.review.pending.toLocaleString()}</span>
                    <span className="oa-metric-k">{t("admin.review.pending")}</span></div>
                </div>

                {/* ── 빈 프로젝트 온보딩 (노드 0, 개요에서만) ── */}
                {stats.nodes === 0 && section === "overview" && (
                  <div className="oa-onb">
                    <div className="oa-onb-title">{t("admin.onb.title")}</div>
                    <div className="oa-onb-desc">{t("admin.onb.desc")}</div>
                    <div className="oa-onb-ctas">
                      <button className="oa-onb-cta" onClick={() => setSection("ingest")}>
                        {t("admin.onb.ingest")}</button>
                      <button className="oa-onb-cta ghost" onClick={() => setSection("explore")}>
                        {t("admin.onb.addNode")}</button>
                    </div>
                  </div>
                )}

                {/* ── 개요 섹션 ── */}
                {section === "overview" && (
                  <>
                    <div className="admin-grid" style={{ marginTop: 16 }}>
                      <section className="card" style={{ marginTop: 18 }}>
                        <h2>{t("admin.trust.title")}</h2>
                        <p className="hint" style={{ marginTop: 2 }}>{t("admin.trust.hint")}</p>
                        <div style={{ marginTop: 12 }}><TrustBar trust={stats.trust} height={12} /></div>
                        <div className="legend">
                          {TRUST_ORDER.filter((k) => stats.trust[k]).map((k) => (
                            <span key={k} className="legend-item">
                              <span className="dot" style={{ background: TRUST_COLORS[k] }} />
                              {k} <b>{stats.trust[k]}</b>
                            </span>
                          ))}
                          {!Object.keys(stats.trust).length &&
                            <span className="hint">{t("admin.trust.empty")}</span>}
                        </div>
                      </section>

                      <section className="card" style={{ marginTop: 18 }}>
                        <h2>{t("admin.review.title")}</h2>
                        <div className="review-counts">
                          <div><span className="rc-num" style={{ color: "var(--green)" }}>
                            {stats.review.confirmed}</span>
                            <span className="rc-label">{t("admin.review.confirmed")}</span></div>
                          <div><span className="rc-num" style={{ color: "var(--red)" }}>
                            {stats.review.rejected}</span>
                            <span className="rc-label">{t("admin.review.rejected")}</span></div>
                          <div><span className="rc-num" style={{ color: "var(--amber)" }}>
                            {stats.review.pending}</span>
                            <span className="rc-label">{t("admin.review.pending")}</span></div>
                          <div style={{ flex: 1, minWidth: 160, alignSelf: "center" }}>
                            <ReviewProgress review={stats.review} />
                          </div>
                        </div>
                        <p className="hint">{t("admin.review.hint")}</p>
                      </section>

                      <IndexPanel base={BASE} namespace={namespace} t={t} flash={flash} />
                    </div>
                    {/* 이 네임스페이스의 스토리지·상세 (ES 인덱스·벡터캐시·PG·타입 분포) */}
                    <NamespaceStorageDetail base={BASE} namespace={namespace}
                                            entry={selEntry} stats={stats} t={t} />
                  </>
                )}

                {/* ── 탐색 섹션 (3-pane 탐색기) ── */}
                {/* ── 수집 섹션 (업로드 → 감식 → 확인 → 수집) ── */}
                {section === "ingest" && (
                  <IngestTab t={t} namespace={namespace}
                             namespaces={entries.map((e) => e.namespace)}
                             protectedNamespaces={entries.filter((e) => e.protected).map((e) => e.namespace)}
                             onComplete={ingestDone} flash={flash} />
                )}

                {section === "explore" && (
                  <ExploreTab key={namespace} api={BASE} namespace={namespace}
                              protected={!!stats.protected}
                              onChanged={handleChildChange} />
                )}

                {/* ── 커버리지 지도 (미연결 구간 = 회복 타깃) ── */}
                {section === "coverage" && (
                  <CoverageTab t={t} api={BASE} namespace={namespace} />
                )}

                {/* ── 검색 Explain 섹션 (근거 사슬: 확장 → 채널 → 인용) ── */}
                {section === "retrieve" && (
                  <RetrieveTab t={t} api={BASE} namespace={namespace} />
                )}

                {/* ── 골든셋 섹션 (정답지 관리 + 채점) ── */}
                {section === "golden" && (
                  <GoldenTab t={t} api={BASE} namespace={namespace} />
                )}

                {/* ── 실험 섹션 (조합 × 골든셋 → 품질×비용 파레토) ── */}
                {section === "experiment" && (
                  <ExperimentTab t={t} api={BASE} namespace={namespace} />
                )}

                {/* ── 검수·감사 섹션 ── */}
                {section === "review" && (
                  <>
                    {/* 검수 큐 보드 — "오늘 뭘 검수해야 하나" (묘비·감사보다 먼저) */}
                    <ReviewQueueBoard api={BASE} namespace={namespace} />
                    <section className="card">
                      <h2>{t("admin.tombstones.title")}
                        <span className="badge" style={{ marginLeft: 8 }}>{tombstones.length}</span></h2>
                      <p className="hint" style={{ marginTop: 2 }}>{t("admin.tombstones.hint")}</p>
                      {tombstones.length === 0
                        ? <p className="hint" style={{ marginTop: 10 }}>{t("admin.tombstones.empty")}</p>
                        : <table>
                            <thead><tr>
                              <th>{t("admin.th.node")}</th><th>{t("admin.th.reason")}</th>
                              <th>{t("admin.th.actor")}</th><th>{t("admin.th.at")}</th>
                            </tr></thead>
                            <tbody>
                              {tombstones.map((s) => (
                                <tr key={s.node_id}>
                                  <td className="mono">{s.node_id}</td>
                                  <td>{s.reason || "—"}</td>
                                  <td>{s.actor || "—"}</td>
                                  <td className="mono" style={{ fontSize: "0.72rem" }}>
                                    {fmtTime(s.at, lang)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>}
                    </section>

                    <section className="card">
                      <h2>{t("admin.audit.title")}</h2>
                      <p className="hint" style={{ marginTop: 2 }}>{t("admin.audit.hint")}</p>
                      {actions.length > 1 && (
                        <div className="filter-bar">
                          <span className={`filter-chip ${!actionFilter ? "on" : ""}`}
                                onClick={() => setActionFilter("")}>
                            {t("admin.audit.all")}
                            <span className="cnt">{history.length}</span>
                          </span>
                          {actions.map((a) => (
                            <span key={a} className={`filter-chip ${actionFilter === a ? "on" : ""}`}
                                  onClick={() => setActionFilter(a)}>
                              <span className="dot" style={{ background: ACTION_COLORS[a] || "#98a2b3" }} />
                              {a}
                              <span className="cnt">{history.filter((h) => h.action === a).length}</span>
                            </span>
                          ))}
                        </div>
                      )}
                      {shownHistory.length === 0
                        ? <p className="hint" style={{ marginTop: 10 }}>{t("admin.audit.empty")}</p>
                        : <table>
                            <thead><tr>
                              <th style={{ width: 110 }}>{t("admin.th.action")}</th>
                              <th>{t("admin.th.node")}</th>
                              <th>{t("admin.th.detail")}</th>
                              <th style={{ width: 120 }}>{t("admin.th.actor")}</th>
                              <th style={{ width: 150 }}>{t("admin.th.at")}</th>
                            </tr></thead>
                            <tbody>
                              {shownHistory.slice(0, 60).map((h, i) => (
                                <tr key={i}>
                                  <td><span style={pill(ACTION_COLORS[h.action])}>{h.action}</span></td>
                                  <td className="mono" style={{ fontSize: "0.74rem" }}>{h.node_id}</td>
                                  <td style={{ color: "var(--muted)", fontSize: "0.78rem" }}>
                                    {h.reason || h.after?.rationale || h.after?.name || "—"}</td>
                                  <td style={{ fontSize: "0.78rem" }}>{h.actor || "—"}</td>
                                  <td className="mono" style={{ fontSize: "0.72rem" }}>
                                    {fmtTime(h.at, lang)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>}
                      {shownHistory.length > 60 &&
                        <p className="hint">{t("admin.audit.truncated", { n: shownHistory.length })}</p>}
                    </section>
                  </>
                )}

                {/* ── 관리 섹션 (운영 콘솔) ── */}
                {section === "manage" && (
                  <ManageTab
                    base={BASE}
                    entries={entries}
                    namespace={namespace}
                    t={t}
                    flash={flash}
                    onChanged={handleChildChange}
                    onNamespaceDeleted={(ns) => {
                      if (ns === namespace) {
                        setNamespace(""); setSection("dashboard"); setStats(null);
                      }
                    }}
                  />
                )}
              </>
            )}
          </>
        )}
        </main>
      </div>
    </div>
  );
}
