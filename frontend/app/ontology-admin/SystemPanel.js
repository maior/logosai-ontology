"use client";

// 시스템 개요 패널 (대시보드 전용) — GET /admin/system 한 콜로 ES/PG/VectorDB/
// 청크/파일의 실물 지표를 렌더한다. 각 백엔드는 독립적으로 degrade 하므로
// ({available:false}) 한 섹션이 죽어도 나머지는 그대로 보인다.
// 원칙: REST 만 소비 · 다크 관리 테마(.oa-* / var(--*)) 상속 · 외부 CDN 0.
//       D3 는 scale/shape 계산에만 쓰고 마크업은 React SVG 로 그린다
//       (NamespaceBubbles 와 같은 패턴 — DOM 을 d3 가 건드리지 않는다).

import { useCallback, useEffect, useMemo, useState } from "react";
import { scaleLinear, scalePoint, line, area, curveMonotoneX, arc, pie, max } from "d3";
import { CLUSTER_PALETTE } from "./ClusterView";

const API = process.env.NEXT_PUBLIC_ONTOLOGY_API || "http://localhost:9274";
const BASE = `${API}/api/v1/ontology`;

const fmtBytes = (n) => {
  if (!n) return "0 B";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};
const fmtNum = (n) => (n == null ? "—" : Number(n).toLocaleString());

// ─── (b) 성장 추이 라인차트 — pg.growth [{date,count}] ──────────────
function GrowthChart({ growth, t }) {
  const W = 520, H = 190, PAD = { t: 14, r: 16, b: 30, l: 48 };
  const data = growth || [];
  const geom = useMemo(() => {
    if (data.length === 0) return null;
    const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
    const x = scalePoint().domain(data.map((d) => d.date)).range([0, iw]).padding(0.1);
    const yMax = max(data, (d) => d.count) || 1;
    const y = scaleLinear().domain([0, yMax]).nice().range([ih, 0]);
    const lg = line().x((d) => x(d.date)).y((d) => y(d.count)).curve(curveMonotoneX);
    const ag = area().x((d) => x(d.date)).y0(ih).y1((d) => y(d.count)).curve(curveMonotoneX);
    const ticks = y.ticks(4);
    return { iw, ih, x, y, path: lg(data), areaPath: ag(data), ticks, yMax };
  }, [data]);

  if (!geom) return <p className="hint">{t("admin.sys.noGrowth")}</p>;
  const single = data.length === 1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="oa-sys-chart" role="img"
         aria-label={t("admin.sys.growth")} style={{ width: "100%", height: "auto" }}>
      <g transform={`translate(${PAD.l},${PAD.t})`}>
        {geom.ticks.map((tk) => (
          <g key={tk} transform={`translate(0,${geom.y(tk)})`}>
            <line x1={0} x2={geom.iw} stroke="var(--oa-line)" strokeWidth={1} />
            <text x={-8} y={4} textAnchor="end" fontSize={9} fill="var(--faint)">
              {tk.toLocaleString()}</text>
          </g>
        ))}
        {!single && <path d={geom.areaPath} fill="var(--accent)" fillOpacity={0.14} />}
        {!single && <path d={geom.path} fill="none" stroke="var(--accent)" strokeWidth={2} />}
        {data.map((d) => (
          <g key={d.date}>
            {single && (
              <rect x={geom.x(d.date) - 18} y={geom.y(d.count)} width={36}
                    height={geom.ih - geom.y(d.count)} fill="var(--accent)" fillOpacity={0.55}
                    rx={2} />
            )}
            <circle cx={geom.x(d.date)} cy={geom.y(d.count)} r={3.2}
                    fill="var(--accent)" stroke="var(--surface)" strokeWidth={1.5} />
            <title>{`${d.date}: ${d.count.toLocaleString()}`}</title>
            <text x={geom.x(d.date)} y={geom.ih + 16} textAnchor="middle"
                  fontSize={9} fill="var(--muted)">{d.date.slice(5)}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

// ─── (c) ES 인덱스 용량 도넛 — indices [{index,size_bytes,docs}] ─────
function IndexDonut({ indices, totalBytes, t }) {
  const SZ = 190, R = 78, IR = 46;
  const arcs = useMemo(() => {
    const items = (indices || []).filter((i) => i.size_bytes > 0);
    if (!items.length) return [];
    const gen = pie().value((d) => d.size_bytes).sort(null);
    const a = arc().innerRadius(IR).outerRadius(R);
    return gen(items).map((p, i) => ({
      d: a(p), item: p.data, color: CLUSTER_PALETTE[i % CLUSTER_PALETTE.length],
    }));
  }, [indices]);
  if (!arcs.length) return <p className="hint">{t("admin.sys.noData")}</p>;
  return (
    <div className="oa-sys-donut-wrap">
      <svg viewBox={`0 0 ${SZ} ${SZ}`} role="img" aria-label={t("admin.sys.esBreakdown")}
           style={{ width: 170, height: 170, flexShrink: 0 }}>
        <g transform={`translate(${SZ / 2},${SZ / 2})`}>
          {arcs.map((a) => (
            <path key={a.item.index} d={a.d} fill={a.color} fillOpacity={0.82}
                  stroke="var(--surface)" strokeWidth={1.5}>
              <title>{`${a.item.index}: ${a.item.size} · ${a.item.docs.toLocaleString()} docs`}</title>
            </path>
          ))}
          <text textAnchor="middle" y={-2} fontSize={15} fontWeight={800} fill="var(--ink)">
            {fmtBytes(totalBytes)}</text>
          <text textAnchor="middle" y={14} fontSize={9} fill="var(--faint)">
            {t("admin.sys.total")}</text>
        </g>
      </svg>
      <div className="legend oa-sys-donut-legend">
        {arcs.slice(0, 8).map((a) => (
          <span key={a.item.index} className="legend-item" title={a.item.index}>
            <span className="dot" style={{ background: a.color }} />
            <span className="oa-sys-idxname">{a.item.index.replace(/^ontology-obj-/, "")}</span>
            <b>{a.item.size}</b>
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── (a) 용량 배분 바 — 물리 디스크를 네임스페이스 비중으로 분할 ─────
// 클라우드 스토리지 콘솔 스타일: 전체 용량을 하나의 세그먼트 바로 나눠
// "누가 얼마를 차지하나"를 한눈에. 버블(노드 수)·ES 도넛(ES 크기)과 다른
// 축(물리 디스크)이라 중복이 아니다. 슬라이버는 범례가 값·%로 보완한다.
function CapacityAllocation({ entries, t }) {
  const { segs, total } = useMemo(() => {
    const items = (entries || [])
      .filter((e) => !("error" in e) && (e.disk_bytes || 0) > 0)
      .sort((a, b) => (b.disk_bytes || 0) - (a.disk_bytes || 0));
    const total = items.reduce((a, e) => a + (e.disk_bytes || 0), 0) || 1;
    const segs = items.map((e, i) => ({
      ns: e.namespace,
      bytes: e.disk_bytes || 0,
      pct: ((e.disk_bytes || 0) / total) * 100,
      color: CLUSTER_PALETTE[i % CLUSTER_PALETTE.length],
      protected: e.protected,
    }));
    return { segs, total };
  }, [entries]);
  if (!segs.length) return <p className="hint">{t("admin.sys.noData")}</p>;
  return (
    <div className="oa-sys-alloc">
      <div className="oa-sys-alloc-total">
        <b>{fmtBytes(total)}</b><span>{t("admin.sys.total")}</span>
      </div>
      <div className="oa-sys-alloc-bar" role="img" aria-label={t("admin.sys.capacity")}>
        {segs.map((s) => (
          <div key={s.ns} className="oa-sys-alloc-seg"
               style={{ width: `${s.pct}%`, background: s.color }}
               title={`${s.ns}: ${fmtBytes(s.bytes)} · ${s.pct.toFixed(1)}%`} />
        ))}
      </div>
      <div className="oa-sys-alloc-legend">
        {segs.map((s) => (
          <span key={s.ns} className="oa-sys-alloc-item" title={s.ns}>
            <span className="dot" style={{ background: s.color }} />
            <span className="oa-sys-alloc-ns">{s.protected && "🔒 "}{s.ns}</span>
            <b className="oa-sys-alloc-sz">{fmtBytes(s.bytes)}</b>
            <span className="oa-sys-alloc-pct">{s.pct.toFixed(1)}%</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// 작은 상태 배지 (available / down)
const Badge = ({ ok, t }) => (
  <span className={`oa-sys-badge ${ok ? "ok" : "down"}`}>
    {ok ? t("admin.sys.live") : t("admin.sys.down")}</span>
);
const KV = ({ k, v, mono = true }) => (
  <div className="oa-sys-kv">
    <span className="oa-sys-kv-k">{k}</span>
    <span className={`oa-sys-kv-v ${mono ? "mono" : ""}`}>{v}</span>
  </div>
);

export default function SystemPanel({ t }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setBusy(true); setErr("");
    try {
      const res = await fetch(`${BASE}/admin/system`);
      if (!res.ok) throw new Error(res.statusText);
      setData(await res.json());
    } catch (e) { setErr(e.message || String(e)); }
    setBusy(false);
  }, []);
  useEffect(() => { load(); }, [load]);

  if (err) return (
    <section className="card"><h2>{t("admin.sys.title")}</h2>
      <div className="error" style={{ marginTop: 10 }}>{err}</div></section>
  );
  if (!data) return (
    <section className="card"><h2>{t("admin.sys.title")}</h2>
      <div className="empty-state"><span className="spinner" />{t("admin.loading")}</div></section>
  );

  const { es, pg, vector, files, namespaces } = data;
  const nsEntries = namespaces?.namespaces || [];

  return (
    <section className="card oa-sys">
      <div className="admin-head" style={{ display: "flex", justifyContent: "space-between",
        alignItems: "center" }}>
        <h2 style={{ border: 0, padding: 0, margin: 0 }}>{t("admin.sys.title")}</h2>
        <button className="ghost" onClick={load} disabled={busy}
                style={{ padding: "4px 12px", fontSize: "0.76rem" }}>
          {busy && <span className="spinner" />}{t("admin.refresh")}</button>
      </div>
      <p className="hint" style={{ marginTop: 2 }}>{t("admin.sys.hint")}</p>

      {/* ── 백엔드 지표 4-패널 ── */}
      <div className="oa-sys-grid">
        {/* Elasticsearch */}
        <div className="oa-sys-card">
          <div className="oa-sys-card-h">
            <span>Elasticsearch</span><Badge ok={es.available} t={t} /></div>
          {es.available ? (
            <>
              <KV k={t("admin.sys.cluster")} v={
                <span className={`oa-sys-cluster ${es.cluster_status}`}>
                  ● {es.cluster_status}</span>} mono={false} />
              <KV k={t("admin.sys.totalDocs")} v={fmtNum(es.total_docs)} />
              <KV k={t("admin.sys.indices")} v={fmtNum(es.indices?.length)} />
              <KV k={t("admin.sys.storeSize")} v={fmtBytes(es.total_size_bytes)} />
              <KV k={t("admin.sys.shards")} v={
                `${es.active_shards ?? "—"} / ${es.unassigned_shards ?? 0} unassigned`} />
              <KV k={t("admin.sys.model")} v={<span className="oa-sys-model">{es.embedding_model}</span>} />
              <KV k={t("admin.sys.dim")} v={`${es.dim} · ${es.index_options}`} />
            </>
          ) : <p className="hint">{es.error || t("admin.sys.down")}</p>}
        </div>

        {/* PostgreSQL */}
        <div className="oa-sys-card">
          <div className="oa-sys-card-h">
            <span>PostgreSQL</span><Badge ok={pg.available} t={t} /></div>
          {pg.available ? (
            <>
              <KV k={t("admin.sys.schema")} v={pg.schema} />
              <KV k={t("admin.stat.nodes")} v={fmtNum(pg.total_nodes)} />
              <KV k={t("admin.stat.edges")} v={fmtNum(pg.total_edges)} />
              <KV k={t("admin.sys.nodeTbl")} v={fmtBytes(pg.table_sizes?.node)} />
              <KV k={t("admin.sys.edgeTbl")} v={fmtBytes(pg.table_sizes?.edge)} />
              <KV k={t("admin.sys.growthDays")} v={fmtNum(pg.growth?.length)} />
            </>
          ) : <p className="hint">{pg.error || t("admin.sys.down")}</p>}
        </div>

        {/* VectorDB */}
        <div className="oa-sys-card">
          <div className="oa-sys-card-h">
            <span>VectorDB (chunks)</span><Badge ok={vector.available} t={t} /></div>
          {vector.available ? (
            <>
              {/* 노드·청크가 다른 모델을 쓸 수 있다 — 그 조합이 지표를 크게
                  움직이므로(융합 MRR 0.8333 ↔ 0.8732 ↔ 0.8167) 갈렸을 때는
                  두 줄로 보여준다. 통일이면 한 줄로 둔다(화면을 늘리지 않는다). */}
              {vector.embedding && !vector.embedding.unified ? (
                <>
                  <KV k={`${t("admin.sys.model")} · node`}
                      v={<span className="oa-sys-model">{vector.embedding.node_model}</span>} />
                  <KV k={`${t("admin.sys.model")} · chunk`}
                      v={<span className="oa-sys-model">{vector.embedding.chunk_model}</span>} />
                </>
              ) : (
                <KV k={t("admin.sys.model")} v={<span className="oa-sys-model">{vector.embedding_model}</span>} />
              )}
              <KV k={t("admin.sys.dim")} v={fmtNum(vector.dim)} />
              <KV k={t("admin.sys.chunkCfg")} v={`${vector.chunk_size} / ${vector.overlap}`} />
              {/* 검색 knob — 지표를 움직이는 값들. 화면에 없으면 "이 점수가 어떤
                  설정에서 나왔나"를 알 수 없다 (이 저장소에서 두 번 데였다). */}
              {vector.embedding?.retrieval && (
                <KV k={t("admin.sys.retrievalCfg")}
                    v={`entry_k ${vector.embedding.retrieval.entry_k}`
                       + ` · terms ${vector.embedding.retrieval.max_terms}`
                       + ` · ratio ${vector.embedding.retrieval.entry_ratio}`
                       + (vector.embedding.retrieval.propagation_channel
                          ? ` · prop w${vector.embedding.retrieval.propagation_weight}`
                          : "")} />
              )}
              <KV k={t("admin.sys.npyCaches")} v={fmtNum(vector.caches?.length)} />
              <KV k={t("admin.sys.cacheSize")} v={fmtBytes(vector.total_size_bytes)} />
              <KV k={t("admin.sys.vectors")} v={fmtNum(
                (vector.caches || []).reduce((a, c) => a + (c.vectors || 0), 0))} />
            </>
          ) : <p className="hint">{vector.error || t("admin.sys.down")}</p>}
        </div>

        {/* Files / chunks */}
        <div className="oa-sys-card">
          <div className="oa-sys-card-h">
            <span>{t("admin.sys.filesChunks")}</span><Badge ok={files.available} t={t} /></div>
          <KV k={t("admin.sys.datasets")} v={fmtNum(files.datasets)} />
          <KV k={t("admin.sys.files")} v={fmtNum(files.total_files)} />
          <KV k={t("admin.stat.chunks")} v={fmtNum(namespaces?.totals?.chunks)} />
          <KV k={t("admin.stat.namespaces")} v={fmtNum(namespaces?.totals?.namespaces)} />
          <KV k={t("admin.sys.diskTotal")} v={fmtBytes(namespaces?.totals?.disk_bytes)} />
          {vector.tiers && (
            <div className="oa-sys-tiers">
              {Object.entries(vector.tiers).map(([k, v]) => (
                <span key={k} className="oa-sys-tier" title={v}>{k}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── D3 차트 3종 ── */}
      <div className="oa-sys-charts">
        <div className="oa-sys-chart-box">
          <div className="oa-sys-chart-t">{t("admin.sys.growth")}</div>
          <p className="hint" style={{ marginTop: 0 }}>{t("admin.sys.growthHint")}</p>
          {pg.available ? <GrowthChart growth={pg.growth} t={t} />
            : <p className="hint">{t("admin.sys.down")}</p>}
        </div>
        <div className="oa-sys-chart-box">
          <div className="oa-sys-chart-t">{t("admin.sys.esBreakdown")}</div>
          <p className="hint" style={{ marginTop: 0 }}>{t("admin.sys.esBreakdownHint")}</p>
          {es.available ? <IndexDonut indices={es.indices} totalBytes={es.total_size_bytes} t={t} />
            : <p className="hint">{t("admin.sys.down")}</p>}
        </div>
        <div className="oa-sys-chart-box wide">
          <div className="oa-sys-chart-t">{t("admin.sys.capacity")}</div>
          <p className="hint" style={{ marginTop: 0 }}>{t("admin.sys.capacityHint")}</p>
          <CapacityAllocation entries={nsEntries} t={t} />
        </div>
      </div>
    </section>
  );
}
