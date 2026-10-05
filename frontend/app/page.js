"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import DataMap from "./DataMap";
import GoogleMap, { HAS_GOOGLE_KEY } from "./GoogleMap";
import GraphExplorer, { typeColor } from "./GraphExplorer";
import Graph3D from "./Graph3D";
import IngestGate from "./IngestGate";
import ReviewPanel from "./ReviewPanel";
import { useT } from "./i18n";

const API = process.env.NEXT_PUBLIC_ONTOLOGY_API || "http://localhost:9274";

const MENU = ["build", "graph", "graph3d", "map", "search", "dataset", "review"];

// 선택 노드가 렌더된 그래프(limit) 밖일 때, 상세 응답으로 이웃 서브그래프 구성.
// 검색 결과처럼 전체 그래프 랭킹에서 온 노드도 항상 중심에 표기·활성화된다.
function buildEgoGraph(nodeId, detail) {
  const center = {
    id: nodeId,
    name: detail.attrs?.name || nodeId,
    type: detail.attrs?.type || "",
    source: detail.attrs?.source,
  };
  const nodeMap = new Map([[nodeId, center]]);
  const addNeighbor = (e) => {
    if (!nodeMap.has(e.target)) {
      nodeMap.set(e.target, {
        id: e.target, name: e.target_name || e.target, type: e.target_type || "",
      });
    }
  };
  const links = [];
  (detail.out_edges || []).forEach((e) => {
    addNeighbor(e); links.push({ source: nodeId, target: e.target, predicate: e.predicate });
  });
  (detail.in_edges || []).forEach((e) => {
    addNeighbor(e); links.push({ source: e.target, target: nodeId, predicate: e.predicate });
  });
  return { nodes: [...nodeMap.values()], links };
}

export default function Home() {
  const { t, lang, setLang } = useT();

  // 셸 상태
  const [view, setView] = useState("build");
  const [namespace, setNamespace] = useState("");
  const [namespaces, setNamespaces] = useState([]);
  // 빌드 상태
  const [buildMode, setBuildMode] = useState("gate");  // gate(감식 게이트) | simple(구 빌드)
  const [files, setFiles] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [dataset, setDataset] = useState(null);
  const [schemaMode, setSchemaMode] = useState("auto");
  const [presets, setPresets] = useState({});
  const [customSchema, setCustomSchema] = useState(
    '{\n  "node_types": ["Concept"],\n  "predicates": {"relatedTo": ["Concept", "Concept"]}\n}');
  const [chunkSize, setChunkSize] = useState(800);
  const [overlap, setOverlap] = useState(120);
  const [segmentMode, setSegmentMode] = useState("auto");
  const [llmModel, setLlmModel] = useState("");
  const [llmProvider, setLlmProvider] = useState("google");
  const [llmBaseUrl, setLlmBaseUrl] = useState("");
  const [rebuild, setRebuild] = useState(false);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  // 조회 상태
  const [graph, setGraph] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [focusGraph, setFocusGraph] = useState(null);  // ego-graph (선택 노드가 렌더 집합 밖일 때)
  const graphDataRef = useRef(null);                    // selectNode 안에서 최신 graphData 참조 (stale 방지)
  const [mapData, setMapData] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hiddenTypes, setHiddenTypes] = useState([]);
  const [showAllTypes, setShowAllTypes] = useState(false);
  const [hiddenCats, setHiddenCats] = useState([]);
  const [mapFocus, setMapFocus] = useState(null);
  const [nodeDetail, setNodeDetail] = useState(null);
  const [rollup, setRollup] = useState(null);
  const [graphFull, setGraphFull] = useState(false);
  const graphApi = useRef(null);
  const graph3dApi = useRef(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState(null);
  const [searchBusy, setSearchBusy] = useState(false);
  const [dsFormats, setDsFormats] = useState(["qa", "triples", "surface"]);
  const [dsPreds, setDsPreds] = useState([]);
  const [dsResult, setDsResult] = useState(null);
  const [dsBusy, setDsBusy] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef(null);

  const loadNamespaces = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/ontology/namespaces`).then((r) => r.json());
      setNamespaces(res.namespaces || []);
    } catch { /* server down — ignore */ }
  }, []);

  const loadGraph = useCallback(async (ns) => {
    try {
      setError("");
      const [summary, data, map] = await Promise.all([
        fetch(`${API}/api/v1/ontology/graphs/${ns}?limit=50`).then((r) => r.json()),
        fetch(`${API}/api/v1/ontology/graphs/${ns}/data?limit=200`).then((r) => r.json()),
        fetch(`${API}/api/v1/ontology/graphs/${ns}/map`).then((r) => r.json()),
      ]);
      setGraph(summary); setGraphData(data); setMapData(map);
    } catch (e) { setError(t("err.graph", { e })); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { graphDataRef.current = graphData; }, [graphData]);

  const selectNode = useCallback(async (nodeId, nsOverride) => {
    setSelectedNode(nodeId);
    if (!nodeId) { setNodeDetail(null); setRollup(null); setFocusGraph(null); return; }
    try {
      const ns = nsOverride || namespace;
      const res = await fetch(
        `${API}/api/v1/ontology/graphs/${ns}/node?id=${encodeURIComponent(nodeId)}`);
      const detail = res.ok ? await res.json() : null;
      setNodeDetail(detail);
      setRollup(null);
      // 선택 노드가 현재 렌더된 그래프에 없으면(예: 검색 결과가 limit 밖) 이웃 서브그래프로 전환
      const inCurrent = (graphDataRef.current?.nodes || []).some((n) => n.id === nodeId);
      setFocusGraph(!inCurrent && detail ? buildEgoGraph(nodeId, detail) : null);
      if (detail && (detail.in_edges.some((e) => e.predicate === "classifiedAs")
                     || detail.out_edges.some((e) => e.predicate === "is_a")
                     || detail.in_edges.some((e) => e.predicate === "is_a"))) {
        const rr = await fetch(
          `${API}/api/v1/ontology/graphs/${ns}/rollup?class_id=${encodeURIComponent(nodeId)}`);
        if (rr.ok) {
          const roll = await rr.json();
          if (roll.instances_total > 0 || roll.descendants.length > 0) setRollup(roll);
        }
      }
    } catch { setNodeDetail(null); }
  }, [namespace]);

  useEffect(() => {
    fetch(`${API}/api/v1/ontology/schemas`)
      .then((r) => r.json()).then((d) => setPresets(d.presets || {})).catch(() => {});
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const ns = params.get("ns");
      const v = params.get("view");
      // 언어 판정은 i18n(LanguageProvider)과 동일하게: localStorage 우선 → navigator → en
      const savedLang = localStorage.getItem("ontology_lang");
      const isKo = (savedLang === "ko" || savedLang === "en")
        ? savedLang === "ko"
        : (navigator.language || "").startsWith("ko");
      const preferred = isKo ? "heritage_kr" : "heritage_us";
      // 네임스페이스 목록을 받은 뒤 기본값 결정 (URL 우선 → 언어 기본 → 목록 첫째)
      fetch(`${API}/api/v1/ontology/namespaces`)
        .then((r) => r.json())
        .then((d) => {
          const list = d.namespaces || [];
          setNamespaces(list);
          if (ns) return;  // URL이 지정하면 아래에서 처리
          const names = list.map((n) => n.namespace);
          const pick = names.includes(preferred) ? preferred : (names[0] || "");
          if (pick) { setNamespace(pick); loadGraph(pick); }
        })
        .catch(() => {});
      if (ns) { setNamespace(ns); loadGraph(ns); setView(v || "graph"); }
      else if (v) setView(v);
      const node = params.get("node");
      if (node && ns) setTimeout(() => selectNode(node, ns), 400);
      const focus = params.get("focus");
      if (focus && ns) {
        const [fid, flat, flng] = focus.split(",");
        setMapFocus({ id: fid, lat: Number(flat), lng: Number(flng) });
        setView("map");
      }
    }
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const switchNamespace = (ns) => {
    setNamespace(ns); setSelectedNode(null); setNodeDetail(null); setFocusGraph(null);
    setSearchResults(null); setHiddenTypes([]); setHiddenCats([]); setMapFocus(null);
    loadGraph(ns);
  };

  const upload = async () => {
    if (!files.length) return;
    setBusy(true); setError("");
    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const res = await fetch(`${API}/api/v1/ontology/datasets`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      setDataset(await res.json());
      setJob(null);
    } catch (e) { setError(t("err.upload", { e: e.message || e })); }
    setBusy(false);
  };

  const build = async () => {
    setBusy(true); setError(""); setJob(null);
    try {
      const payload = {
        dataset_id: dataset.dataset_id, namespace,
        schema_mode: schemaMode, chunk_size: Number(chunkSize),
        overlap: Number(overlap), segment_mode: segmentMode,
        llm_model: llmModel.trim() || null, rebuild,
        llm_provider: llmProvider, llm_base_url: llmBaseUrl.trim() || null,
      };
      if (schemaMode === "custom") payload.custom_schema = JSON.parse(customSchema);
      const res = await fetch(`${API}/api/v1/ontology/build`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const { job_id } = await res.json();
      pollRef.current = setInterval(async () => {
        const j = await fetch(`${API}/api/v1/ontology/jobs/${job_id}`).then((r) => r.json());
        setJob(j);
        if (j.status === "completed" || j.status === "failed") {
          clearInterval(pollRef.current);
          setBusy(false);
          if (j.status === "completed") {
            await loadGraph(namespace);
            await loadNamespaces();
            setView("graph");
          }
        }
      }, 1500);
    } catch (e) { setError(t("err.build", { e: e.message || e })); setBusy(false); }
  };

  const search = async () => {
    if (!searchQuery.trim() || searchBusy) return;
    setError(""); setSearchBusy(true);
    try {
      const res = await fetch(`${API}/api/v1/ontology/graphs/${namespace}/search`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, top_k: 8 }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      setSearchResults((await res.json()).results);
    } catch (e) { setError(t("err.search", { e: e.message || e })); }
    finally { setSearchBusy(false); }
  };

  const extractDataset = async () => {
    setDsBusy(true); setError("");
    try {
      const res = await fetch(`${API}/api/v1/ontology/graphs/${namespace}/dataset`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ formats: dsFormats,
                               predicates: dsPreds.length ? dsPreds : null }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      setDsResult(await res.json());
    } catch (e) { setError(t("err.extract", { e: e.message || e })); }
    setDsBusy(false);
  };

  const activeGraph = focusGraph || graphData;  // ego-graph 활성 시 그쪽을 렌더
  const colors = typeColor((activeGraph?.nodes || []).map((n) => n.type));

  // ── 2D/3D 그래프 뷰 공유 조각 ──
  const focusBanner = focusGraph && (
    <div className="filter-bar" style={{ alignItems: "center" }}>
      <span className="filter-chip on" style={{ cursor: "default" }}>
        {t("graph.focus", { name: nodeDetail?.attrs?.name || selectedNode })}
      </span>
      <button className="ghost" style={{ padding: "4px 12px", fontSize: "0.76rem" }}
              onClick={() => { setFocusGraph(null); selectNode(null); }}>
        {t("graph.focus.back")}
      </button>
    </div>
  );

  const statRow = graph && (
    <div className="stat-row">
      <div className="stat-card"><div className="v">{graph.nodes}</div><div className="k">{t("graph.stat.nodes")}</div></div>
      <div className="stat-card"><div className="v">{graph.edges}</div><div className="k">{t("graph.stat.edges")}</div></div>
      <div className="stat-card"><div className="v">{Object.keys(graph.node_types).length}</div><div className="k">{t("graph.stat.types")}</div></div>
      <div className="stat-card"><div className="v">{mapData?.geo?.length ?? 0}</div><div className="k">{t("graph.stat.geo")}</div></div>
    </div>
  );

  const typeFilter = graph && (() => {
    const sorted = Object.entries(graph.node_types).sort((a, b) => b[1] - a[1]);
    const shown = showAllTypes ? sorted : sorted.slice(0, 18);
    const hiddenCount = sorted.length - shown.length;
    return (
      <div className="filter-bar">
        <span className="filter-label">{t("graph.filter.types", { n: sorted.length })}</span>
        {shown.map(([ty, c]) => {
          const on = !hiddenTypes.includes(ty);
          return (
            <span key={ty} className={`filter-chip ${on ? "on" : ""}`}
                  onClick={() => setHiddenTypes(on
                    ? [...hiddenTypes, ty] : hiddenTypes.filter((x) => x !== ty))}>
              <span className="dot" style={{ background: colors[ty] || "#98a2b3" }} />
              {ty} <span className="cnt">{c}</span>
            </span>
          );
        })}
        {sorted.length > 18 && (
          <span className="filter-chip" onClick={() => setShowAllTypes(!showAllTypes)}
                style={{ color: "var(--accent)", fontWeight: 700 }}>
            {showAllTypes ? t("graph.filter.less") : t("graph.filter.more", { n: hiddenCount })}
          </span>
        )}
      </div>
    );
  })();

  const detailPanel = nodeDetail && (
    <aside className="detail-panel">
      <div className="detail-type"
           style={{ background: colors[nodeDetail.attrs.type] || "#8a8371" }}>
        {nodeDetail.attrs.type}
      </div>
      <h3>{nodeDetail.attrs.name || nodeDetail.id}</h3>
      {nodeDetail.attrs.definition && (
        <p className="detail-def">{nodeDetail.attrs.definition}</p>
      )}
      {Array.isArray(nodeDetail.attrs.aliases) && nodeDetail.attrs.aliases.length > 0 && (
        <div style={{ marginBottom: 6 }}>
          {nodeDetail.attrs.aliases.map((a) => (
            <span key={a} className="alias-chip">= {a}</span>))}
        </div>
      )}
      {(nodeDetail.attrs.lat != null && nodeDetail.attrs.lng != null) && (
        <div className="detail-actions">
          <button onClick={() => {
            setMapFocus({ id: nodeDetail.id,
                          lat: Number(nodeDetail.attrs.lat),
                          lng: Number(nodeDetail.attrs.lng) });
            setView("map");
          }}>{t("detail.toMap")}</button>
        </div>
      )}
      {rollup && (
        <>
          <h4>{t("detail.rollup")}</h4>
          {rollup.ancestors.length > 0 && (
            <div style={{ fontSize: "0.78rem", marginBottom: 6 }}>
              {t("detail.rollup.parents")} {rollup.ancestors.map((a) => (
                <span key={a} className="alias-chip" style={{ cursor: "pointer" }}
                      onClick={() => selectNode(a)}>{a.split(":").pop()}</span>))}
            </div>
          )}
          <div style={{ fontSize: "0.82rem", margin: "4px 0" }}>
            {t("detail.rollup.summary", { d: rollup.descendants.length, t: rollup.instances_total })}
            {" "}<span className="hint-inline">{t("detail.rollup.direct", { n: rollup.instances_direct })}</span>
          </div>
          {rollup.instances.slice(0, 7).map((i) => (
            <div key={i.id} className="edge-row" onClick={() => selectNode(i.id)}>
              {i.name} <span className="edge-type">{i.type}</span>
            </div>))}
          {rollup.instances_total > 7 && (
            <div className="hint">{t("detail.rollup.more", { n: rollup.instances_total - 7 })}</div>)}
        </>
      )}
      <h4>{t("detail.attrs")}</h4>
      {Object.entries(nodeDetail.attrs)
        .filter(([k]) => !["name", "type", "definition", "aliases"].includes(k))
        .map(([k, v]) => (
          <div className="kv" key={k}>
            <span className="k">{k}</span>
            <span className="v">{String(v)}</span>
          </div>
        ))}
      {nodeDetail.out_edges.length > 0 && (
        <>
          <h4>{t("detail.outEdges")}</h4>
          {nodeDetail.out_edges.map((e, i) => (
            <div key={i} className="edge-row" onClick={() => selectNode(e.target)}>
              <span className="pred">{e.predicate}</span> → {e.target_name}
              <span className="edge-type">{e.target_type}</span>
            </div>))}
        </>
      )}
      {nodeDetail.in_edges.length > 0 && (
        <>
          <h4>{t("detail.inEdges")}</h4>
          {nodeDetail.in_edges.map((e, i) => (
            <div key={i} className="edge-row" onClick={() => selectNode(e.target)}>
              {e.target_name} <span className="pred">{e.predicate}</span> →
              <span className="edge-type">{e.target_type}</span>
            </div>))}
        </>
      )}
      <button className="ghost" style={{ marginTop: 14, padding: "6px 14px" }}
              onClick={() => selectNode(null)}>{t("detail.close")}</button>
    </aside>
  );

  return (
    <div className="shell">
      {/* ─── Sidebar ─── */}
      <aside className="sidebar">
        <div className="side-brand">
          <div className="step-label">{t("app.brand")}</div>
          <div className="side-title">{t("app.title")}</div>
        </div>

        <div className="side-section">
          <label>{t("nav.namespace")}</label>
          <select value={namespace} onChange={(e) => switchNamespace(e.target.value)}>
            {!namespaces.some((n) => n.namespace === namespace) && (
              <option value={namespace}>{namespace}</option>
            )}
            {namespaces.map((n) => (
              <option key={n.namespace} value={n.namespace}>
                {n.namespace}{n.nodes != null ? ` (${n.nodes})` : ""}
              </option>
            ))}
          </select>
          <div className="side-hint">{t("nav.sampleHint")}</div>
        </div>

        <nav className="side-menu">
          {MENU.map((key) => (
            <button key={key}
                    className={`menu-item ${view === key ? "active" : ""}`}
                    onClick={() => { setView(key); if (key !== "build" && !graph) loadGraph(namespace); }}>
              <span className="menu-label">{t(`menu.${key}`)}</span>
              <span className="menu-desc">{t(`menu.${key}.desc`)}</span>
            </button>
          ))}
        </nav>

        <div className="side-footer">
          <div className="side-section" style={{ padding: 0, marginBottom: 12 }}>
            <label>{t("nav.language")}</label>
            <div className="lang-toggle">
              {["en", "ko"].map((l) => (
                <button key={l} className={`lang-btn ${lang === l ? "on" : ""}`}
                        onClick={() => setLang(l)}>
                  {l === "en" ? "EN" : "한국어"}
                </button>
              ))}
            </div>
          </div>
          <a href={`${API}/api/v1/ontology/graphs/${namespace}/export?format=turtle`}
             className="side-link">{t("side.exportOwl")}</a>
          <div style={{ marginTop: 6 }}>
            <a href="/ontology-admin" className="side-link">{t("side.admin")}</a>
          </div>
          <div className="side-meta">backend :9274 · frontend :9275</div>
        </div>
      </aside>

      {/* ─── Main ─── */}
      <main className="content">
        {error && <div className="error card soft" style={{ marginBottom: 18 }}>{error}</div>}

        {/* Build */}
        {view === "build" && (
          <>
            <h1>{t("build.title")}</h1>
            <p className="sub" dangerouslySetInnerHTML={{
              __html: t("build.subtitle", { ns: `<b>${namespace}</b>` }) }} />

            <div className="filter-bar">
              {["gate", "simple"].map((m) => (
                <span key={m} className={`filter-chip ${buildMode === m ? "on" : ""}`}
                      onClick={() => setBuildMode(m)}>
                  <span className="dot" style={{ background: "#4f46e5" }} />
                  {t(`build.mode.${m}`)}
                </span>
              ))}
            </div>

            {buildMode === "gate" && (
              <IngestGate api={API}
                          onComplete={async (ns) => {
                            setNamespace(ns);
                            await loadGraph(ns);
                            await loadNamespaces();
                            setView("graph");
                          }} />
            )}

            {buildMode === "simple" && (
            <>
            <section className="card">
              <h2>{t("build.step1")} <span className="hint-inline">{t("build.formats")}</span></h2>
              <div className={`dropzone ${dragActive ? "active" : ""}`}
                   onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                   onDragLeave={() => setDragActive(false)}
                   onDrop={(e) => { e.preventDefault(); setDragActive(false); setFiles([...e.dataTransfer.files]); }}
                   onClick={() => document.getElementById("file-input").click()}>
                {files.length
                  ? t("build.dropzone.files", { n: files.length, names: files.map((f) => f.name).join(", ") })
                  : t("build.dropzone")}
              </div>
              <input id="file-input" type="file" multiple style={{ display: "none" }}
                     onChange={(e) => setFiles([...e.target.files])} />
              <div style={{ marginTop: 14 }}>
                <button onClick={upload} disabled={!files.length || busy}>{t("build.upload")}</button>
                {dataset && <span className="status-completed" style={{ marginLeft: 12, fontSize: "0.82rem" }}>
                  {t("build.uploaded", { id: dataset.dataset_id, n: dataset.files.length })}</span>}
              </div>
            </section>

            {dataset && (
              <section className="card">
                <h2>{t("build.step2")}</h2>
                <div className="row">
                  <div>
                    <label>{t("build.schema")}</label>
                    <select value={schemaMode} onChange={(e) => setSchemaMode(e.target.value)}>
                      <option value="auto">{t("build.schema.auto")}</option>
                      {Object.keys(presets).map((name) => (
                        <option key={name} value={name}>{t("build.schema.preset", { name })}</option>
                      ))}
                      <option value="custom">{t("build.schema.custom")}</option>
                    </select>
                  </div>
                  <div>
                    <label>{t("build.provider")}</label>
                    <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)}>
                      <option value="google">{t("build.provider.google")}</option>
                      <option value="openai">{t("build.provider.openai")}</option>
                      <option value="anthropic">{t("build.provider.anthropic")}</option>
                      <option value="openai_compatible">{t("build.provider.compat")}</option>
                    </select>
                  </div>
                  <div>
                    <label>{t("build.model")}</label>
                    <input type="text" value={llmModel}
                           placeholder={{ google: "gemini-3.5-flash", openai: "gpt-4o-mini",
                                          anthropic: "claude-haiku-4-5", openai_compatible: "required" }[llmProvider]}
                           onChange={(e) => setLlmModel(e.target.value)} />
                  </div>
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <div>
                    <label>{t("build.segment")}</label>
                    <select value={segmentMode} onChange={(e) => setSegmentMode(e.target.value)}>
                      <option value="auto">{t("build.segment.auto")}</option>
                      <option value="heading">{t("build.segment.heading")}</option>
                      <option value="window">{t("build.segment.window")}</option>
                    </select>
                  </div>
                  <div><label>{t("build.chunkSize")}</label>
                    <input type="number" value={chunkSize} onChange={(e) => setChunkSize(e.target.value)} /></div>
                  <div><label>{t("build.overlap")}</label>
                    <input type="number" value={overlap} onChange={(e) => setOverlap(e.target.value)} /></div>
                  <div style={{ display: "flex", alignItems: "flex-end", paddingBottom: 6 }}>
                    <label style={{ fontWeight: 400, marginBottom: 0 }}>
                      <input type="checkbox" checked={rebuild}
                             onChange={(e) => setRebuild(e.target.checked)}
                             style={{ width: "auto", marginRight: 6 }} />
                      {t("build.rebuild")}
                    </label>
                  </div>
                </div>
                {llmProvider === "openai_compatible" && (
                  <div style={{ marginTop: 12 }}>
                    <label>{t("build.baseUrl")}</label>
                    <input type="text" value={llmBaseUrl} placeholder="http://localhost:11434/v1"
                           onChange={(e) => setLlmBaseUrl(e.target.value)} />
                  </div>
                )}
                {schemaMode === "custom" && (
                  <div style={{ marginTop: 12 }}>
                    <label>{t("build.customSchema")}</label>
                    <textarea rows={5} value={customSchema} onChange={(e) => setCustomSchema(e.target.value)} />
                  </div>
                )}
                <div style={{ marginTop: 16 }}>
                  <button onClick={build} disabled={busy}>
                    {busy ? t("build.running") : t("build.run")}
                  </button>
                </div>
                {job && (
                  <div style={{ marginTop: 14, fontSize: "0.86rem" }}>
                    {job.status === "running" && <span className="spinner" />}
                    {t("build.status")} <span className={`status-${job.status}`}>{job.status}</span>
                    {job.status === "running" && job.progress?.stage && (
                      <span style={{ marginLeft: 10, color: "var(--muted)" }}>
                        {job.progress.stage} {job.progress.current || ""}{job.progress.total ? `/${job.progress.total}` : ""}
                      </span>)}
                    {job.report && (
                      <span style={{ marginLeft: 10 }}>
                        {t("build.report", { e: job.report.entities_added, r: job.report.relations_added, f: job.report.chunks_failed })}
                      </span>)}
                    {job.report?.proposed_schema && (
                      <div className="hint">{t("build.proposedSchema", { types: job.report.proposed_schema.node_types.join(" · ") })}</div>)}
                    {job.error && <div className="error">{job.error}</div>}
                  </div>
                )}
              </section>
            )}
            </>
            )}
          </>
        )}

        {/* Graph (2D) */}
        {view === "graph" && (
          <>
            <h1>{t("graph.title", { ns: namespace })}</h1>
            <p className="sub">{t("graph.subtitle")}</p>
            {focusBanner}
            {statRow}
            {typeFilter}
            <div className={`graph-shell ${graphFull ? "fullscreen" : ""}`}>
              <GraphExplorer ref={graphApi} data={activeGraph} selectedId={selectedNode}
                             onSelect={selectNode} hiddenTypes={hiddenTypes}
                             emptyText={t("graph.empty")} />
              <div className="graph-controls">
                <button title={t("graph.ctrl.zoomIn")} onClick={() => graphApi.current?.zoomIn()}>＋</button>
                <button title={t("graph.ctrl.zoomOut")} onClick={() => graphApi.current?.zoomOut()}>－</button>
                <button title={t("graph.ctrl.fit")} onClick={() => graphApi.current?.fit()}>⛶</button>
                <button title={graphFull ? t("graph.ctrl.exitFull") : t("graph.ctrl.full")}
                        onClick={() => setGraphFull(!graphFull)}>{graphFull ? "✕" : "⤢"}</button>
              </div>
              {detailPanel}
            </div>
            <p className="hint">{t("graph.hint")}</p>
          </>
        )}

        {/* Graph (3D) */}
        {view === "graph3d" && (
          <>
            <h1>{t("graph3d.title", { ns: namespace })}</h1>
            <p className="sub">{t("graph3d.subtitle")}</p>
            {focusBanner}
            {statRow}
            {typeFilter}
            <div className={`graph-shell dark ${graphFull ? "fullscreen" : ""}`}>
              <Graph3D ref={graph3dApi} data={activeGraph} selectedId={selectedNode}
                       onSelect={selectNode} hiddenTypes={hiddenTypes}
                       emptyText={t("graph.empty")} />
              <div className="graph-controls">
                <button title={t("graph.ctrl.zoomIn")} onClick={() => graph3dApi.current?.zoomIn()}>＋</button>
                <button title={t("graph.ctrl.zoomOut")} onClick={() => graph3dApi.current?.zoomOut()}>－</button>
                <button title={t("graph.ctrl.fit")} onClick={() => graph3dApi.current?.fit()}>⛶</button>
                <button title={graphFull ? t("graph.ctrl.exitFull") : t("graph.ctrl.full")}
                        onClick={() => setGraphFull(!graphFull)}>{graphFull ? "✕" : "⤢"}</button>
              </div>
              {detailPanel}
            </div>
            <p className="hint">{t("graph3d.hint")}</p>
          </>
        )}

        {/* Map */}
        {view === "map" && (() => {
          const OTHER = t("map.uncategorized");
          return (
          <>
            <h1>{t("map.title", { ns: namespace })}</h1>
            <p className="sub">{t("map.subtitle")}</p>
            {mapData?.geo?.length > 0 ? (
              <>
                <div className="filter-bar">
                  <span className="filter-label">{t("map.category")}</span>
                  {[...new Set(mapData.geo.map((p) => p.category || OTHER))].map((cat) => {
                    const on = !hiddenCats.includes(cat);
                    return (
                      <span key={cat} className={`filter-chip ${on ? "on" : ""}`}
                            onClick={() => setHiddenCats(on
                              ? [...hiddenCats, cat] : hiddenCats.filter((x) => x !== cat))}>
                        <span className="dot" style={{ background: "#4f46e5" }} />{cat}
                        <span className="cnt">{mapData.geo.filter((p) => (p.category || OTHER) === cat).length}</span>
                      </span>
                    );
                  })}
                </div>
                {HAS_GOOGLE_KEY
                  ? <GoogleMap
                      geo={mapData.geo.filter((p) => !hiddenCats.includes(p.category || OTHER))}
                      focusPoint={mapFocus} lang={lang}
                      onOpenNode={(id) => { setView("graph"); selectNode(id); }} />
                  : <DataMap geo={mapData.geo.filter((p) => !hiddenCats.includes(p.category || OTHER))} />}
                <p className="hint">{t("map.hint")}</p>
              </>
            ) : <div className="empty-state">{t("map.empty")}</div>}
            {mapData && Object.keys(mapData.categories).length > 0 && (
              <section className="card">
                <h2>{t("map.group")}</h2>
                <table>
                  <thead><tr><th>{t("map.th.category")}</th><th>{t("map.th.count")}</th><th>{t("map.th.items")}</th></tr></thead>
                  <tbody>
                    {Object.entries(mapData.categories).map(([cat, info]) => (
                      <tr key={cat}><td><b>{cat}</b></td><td className="mono">{info.count}</td>
                        <td style={{ color: "var(--muted)" }}>{info.items.slice(0, 8).join(", ")}{info.count > 8 ? " …" : ""}</td></tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}
          </>
          );
        })()}

        {/* Search */}
        {view === "search" && (
          <>
            <h1>{t("search.title", { ns: namespace })}</h1>
            <p className="sub">{t("search.subtitle")}</p>
            <section className="card">
              <div className="row">
                <div style={{ flex: 3 }}>
                  <input type="text" placeholder={t("search.placeholder")}
                         value={searchQuery} disabled={searchBusy}
                         onChange={(e) => setSearchQuery(e.target.value)}
                         onKeyDown={(e) => e.key === "Enter" && search()} />
                </div>
                <div style={{ flex: 0 }}>
                  <button onClick={search} disabled={searchBusy || !searchQuery.trim()}>
                    {searchBusy && <span className="spinner" />}
                    {searchBusy ? t("search.running") : t("search.run")}
                  </button>
                </div>
              </div>
              {searchBusy && (
                <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 8,
                              color: "var(--muted)", fontSize: "0.86rem" }}>
                  <span className="spinner" />{t("search.loading")}
                </div>
              )}
              {!searchBusy && searchResults && (
                <table>
                  <thead><tr><th>{t("search.th.node")}</th><th>{t("search.th.type")}</th><th>{t("search.th.score")}</th><th></th></tr></thead>
                  <tbody>
                    {searchResults.length === 0 && (
                      <tr><td colSpan={4} style={{ color: "var(--muted)" }}>{t("search.empty")}</td></tr>)}
                    {searchResults.map((r) => (
                      <tr key={r.node_id}>
                        <td className="mono">{r.node_id}</td>
                        <td>{r.node_type}</td>
                        <td className="mono">{r.score}</td>
                        <td><button className="ghost" style={{ padding: "3px 10px", fontSize: "0.74rem" }}
                                    onClick={() => { setView("graph"); selectNode(r.node_id); }}>
                          {t("search.viewGraph")}</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <p className="hint">{t("search.hint")}</p>
            </section>
          </>
        )}

        {/* Training data */}
        {view === "dataset" && (
          <>
            <h1>{t("ds.title", { ns: namespace })}</h1>
            <p className="sub">{t("ds.subtitle")}</p>
            <section className="card">
              <div className="filter-bar">
                <span className="filter-label">{t("ds.format")}</span>
                {[["qa", t("ds.fmt.qa")], ["triples", t("ds.fmt.triples")],
                  ["surface", t("ds.fmt.surface")]].map(([f, label]) => {
                  const on = dsFormats.includes(f);
                  return (
                    <span key={f} className={`filter-chip ${on ? "on" : ""}`}
                          onClick={() => setDsFormats(on
                            ? dsFormats.filter((x) => x !== f) : [...dsFormats, f])}>
                      <span className="dot" style={{ background: "#4f46e5" }} />{label}
                    </span>
                  );
                })}
              </div>
              {graph && Object.keys(graph.predicates || {}).length > 0 && (
                <div className="filter-bar">
                  <span className="filter-label">{t("ds.relations")}</span>
                  {Object.keys(graph.predicates).map((p) => {
                    const on = dsPreds.includes(p);
                    return (
                      <span key={p} className={`filter-chip ${on ? "on" : ""}`}
                            onClick={() => setDsPreds(on
                              ? dsPreds.filter((x) => x !== p) : [...dsPreds, p])}>
                        <span className="dot" style={{ background: "#059669" }} />{p}
                      </span>
                    );
                  })}
                </div>
              )}
              <div style={{ marginTop: 14, display: "flex", gap: 10 }}>
                <button onClick={extractDataset} disabled={dsBusy || !dsFormats.length}>
                  {dsBusy ? t("ds.extracting") : t("ds.preview")}
                </button>
                <a href={`${API}/api/v1/ontology/graphs/${namespace}/dataset.jsonl?formats=${dsFormats.join(",")}${dsPreds.length ? `&predicates=${dsPreds.join(",")}` : ""}`}>
                  <button className="ghost">{t("ds.download")}</button>
                </a>
              </div>
            </section>

            {dsResult && (
              <>
                <div className="stat-row">
                  <div className="stat-card"><div className="v">{dsResult.total}</div><div className="k">{t("ds.stat.total")}</div></div>
                  <div className="stat-card"><div className="v">{dsResult.counts.qa ?? 0}</div><div className="k">{t("ds.stat.qa")}</div></div>
                  <div className="stat-card"><div className="v">{dsResult.counts.triple ?? 0}</div><div className="k">{t("ds.stat.triple")}</div></div>
                  <div className="stat-card"><div className="v">{dsResult.counts.surface ?? 0}</div><div className="k">{t("ds.stat.surface")}</div></div>
                </div>
                <section className="card">
                  <h2>{t("ds.previewTitle")} <span className="hint-inline">{t("ds.previewNote")}</span></h2>
                  <table>
                    <thead><tr><th style={{width:70}}>{t("ds.th.format")}</th><th>{t("ds.th.content")}</th><th style={{width:"26%"}}>{t("ds.th.source")}</th></tr></thead>
                    <tbody>
                      {dsResult.rows.slice(0, 30).map((r, i) => (
                        <tr key={i}>
                          <td><span className="badge">{r.format}</span></td>
                          <td>
                            {r.format === "triple" && (
                              <span className="mono" style={{fontSize:"0.78rem"}}>
                                {r.subject} —{r.predicate}→ {r.object}</span>)}
                            {r.format === "qa" && (
                              <><b>Q.</b> {r.instruction}<br /><b>A.</b> {r.output}</>)}
                            {r.format === "surface" && (
                              <span>&quot;{r.input}&quot; → &quot;{r.output}&quot;</span>)}
                          </td>
                          <td className="mono" style={{fontSize:"0.68rem", color:"var(--faint)"}}>
                            {(r.source || "").split("/").pop()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </section>
              </>
            )}
          </>
        )}

        {/* Review */}
        {view === "review" && (
          <>
            <h1>{t("review.title", { ns: namespace })}</h1>
            <p className="sub">{t("review.subtitle")}</p>
            <ReviewPanel api={API} namespace={namespace} />
          </>
        )}
      </main>
    </div>
  );
}
