"use client";

// 3-pane 탐색기 (Palantir Foundry Object Explorer 영감) — 스키마 내비 · 객체/엣지 목록 ·
// 인스펙터. 스키마·노드·엣지·저장된탐색을 REST 로만 소비 (서비스 역할 경계 — 9275 관리 콘솔).
//   좌: 스키마 내비 (클래스 · 술어 · is_a 계층 · 저장된 탐색)
//   센터: 객체 목록(기본) / 엣지 목록(술어 클릭) + type-ahead 글로벌 검색
//   우: 인스펙터 (Inspector.js) — 속성 편집 · pivot · 근거 청크 · 이력 · 거절
// 필터는 하나의 상태 객체 → 저장된 탐색의 filter 로 그대로 직렬화/복원.
// 데이터 값(타입명·술어·trust)은 번역하지 않는다 — i18n 규약.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useT } from "../i18n";
import Inspector, { TrustPill } from "./Inspector";
import Graph3D from "../Graph3D";
import ClusterView from "./ClusterView";

const LIMIT = 50;
const TOP_PROPS = 6;
const ACTOR = "admin-console";
const EMPTY_FILTER = {
  mode: "objects", node_type: "", property: "", trust: "", kind: "",
  predicate: "", source_type: "", target_type: "", q: "",
  // 구조화 검색(mode:"query") — 프로퍼티 값 + 관계 제약
  prop_key: "", prop_value: "", prop_op: "eq",
  rel_predicate: "", rel_target: "", rel_target_type: "", rel_direction: "out",
};

const detailMsg = async (res) => {
  try {
    const d = (await res.json()).detail;
    if (typeof d === "string" && d) return d;
    if (d) return JSON.stringify(d);
  } catch { /* 본문이 JSON 이 아니면 statusText 폴백 */ }
  return res.statusText;
};

// ─── 스키마 편집 위젯 (좌 pane 내비) — readOnly 아닐 때만 마운트 ───
// 클래스: 개명(POST rename-type) + 설명/deprecated 선언(PUT type)
function SchemaTypeEditor({ base, cls, t, setError, onRenamed, onSaved, onCancel }) {
  const decl = cls.declared || {};
  const [nm, setNm] = useState("");
  const [desc, setDesc] = useState(decl.description || "");
  const [dep, setDep] = useState(!!decl.deprecated);
  const [busy, setBusy] = useState(false);

  const doRename = async () => {
    const nn = nm.trim();
    if (!nn || nn === cls.type) return;
    setBusy(true); setError("");
    try {
      const res = await fetch(`${base}/schema/rename-type`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old: cls.type, new: nn }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      const d = await res.json();
      onRenamed(cls.type, nn, d.renamed || 0);
    } catch (e) { setError(t("admin.sch.err", { e: e.message || e })); setBusy(false); }
  };

  const doSave = async () => {
    setBusy(true); setError("");
    try {
      const res = await fetch(`${base}/schema/type`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: cls.type, description: desc.trim(), deprecated: dep }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      onSaved();
    } catch (e) { setError(t("admin.sch.err", { e: e.message || e })); setBusy(false); }
  };

  return (
    <div className="oa-sch-form">
      <label>{t("admin.sch.rename")}</label>
      <div className="oa-sch-row">
        <input type="text" value={nm} placeholder={t("admin.sch.renamePh")} list="oa-exp-type-list"
               onChange={(e) => setNm(e.target.value)}
               onKeyDown={(e) => { if (e.key === "Enter") doRename(); }} />
        <button disabled={busy || !nm.trim()} onClick={doRename}>{t("admin.sch.rename")}</button>
      </div>
      <p className="hint" style={{ marginTop: 3 }}>{t("admin.sch.renameMerge")}</p>
      <label style={{ marginTop: 10 }}>{t("admin.sch.descPh")}</label>
      <input type="text" value={desc} onChange={(e) => setDesc(e.target.value)} />
      <label className="oa-sch-check">
        <input type="checkbox" checked={dep} onChange={(e) => setDep(e.target.checked)} />
        {t("admin.sch.deprecatedLabel")}
      </label>
      <div className="oa-sch-actions">
        <button disabled={busy} onClick={doSave}>
          {busy && <span className="spinner" />}{t("admin.sch.saveType")}</button>
        <button className="ghost" onClick={onCancel}>{t("admin.sch.cancel")}</button>
      </div>
    </div>
  );
}

// 술어: domain·range·설명 선언(PUT predicate)
function SchemaPredEditor({ base, pred, t, setError, onSaved, onCancel }) {
  const decl = pred.declared || {};
  const [domain, setDomain] = useState(decl.domain || "");
  const [range, setRange] = useState(decl.range || "");
  const [desc, setDesc] = useState(decl.description || "");
  const [busy, setBusy] = useState(false);

  const doSave = async () => {
    setBusy(true); setError("");
    try {
      const res = await fetch(`${base}/schema/predicate`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          predicate: pred.predicate,
          domain: domain.trim(), range: range.trim(), description: desc.trim(),
        }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      onSaved();
    } catch (e) { setError(t("admin.sch.err", { e: e.message || e })); setBusy(false); }
  };

  return (
    <div className="oa-sch-form">
      <div className="oa-sch-row" style={{ alignItems: "flex-end" }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <label>domain</label>
          <input type="text" value={domain} placeholder={t("admin.sch.domainPh")} list="oa-exp-type-list"
                 onChange={(e) => setDomain(e.target.value)} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <label>range</label>
          <input type="text" value={range} placeholder={t("admin.sch.rangePh")} list="oa-exp-type-list"
                 onChange={(e) => setRange(e.target.value)} />
        </div>
      </div>
      <label style={{ marginTop: 10 }}>{t("admin.sch.descPh")}</label>
      <input type="text" value={desc} onChange={(e) => setDesc(e.target.value)} />
      <div className="oa-sch-actions">
        <button disabled={busy} onClick={doSave}>
          {busy && <span className="spinner" />}{t("admin.sch.savePred")}</button>
        <button className="ghost" onClick={onCancel}>{t("admin.sch.cancel")}</button>
      </div>
    </div>
  );
}

export default function ExploreTab({ api, namespace, protected: readOnly, onChanged }) {
  const { t } = useT();
  const base = `${api}/graphs/${namespace}`;

  const [schema, setSchema] = useState(null);
  const [schemaErr, setSchemaErr] = useState("");
  const [views, setViews] = useState([]);

  const [filter, setFilter] = useState(EMPTY_FILTER);
  const [offset, setOffset] = useState(0);
  const [nodes, setNodes] = useState(null);       // 객체 모드
  const [edges, setEdges] = useState(null);       // 엣지 모드
  const [facets, setFacets] = useState(null);     // 검색 파셋 {type,trust} (ES 검색 시)
  const [searchSource, setSearchSource] = useState(null); // "es" | "fallback"
  const [advOpen, setAdvOpen] = useState(false);  // 고급 검색(구조화) 패널
  const [qForm, setQForm] = useState({ prop_key: "", prop_op: "eq", prop_value: "",
    rel_predicate: "", rel_direction: "out", rel_target_type: "", rel_target: "" });
  const [extractOpen, setExtractOpen] = useState(false); // 추출 패널
  const [extractFmts, setExtractFmts] = useState(["triples", "qa"]);
  const [extractRes, setExtractRes] = useState(null);
  const [extractBusy, setExtractBusy] = useState(false);
  const [listBusy, setListBusy] = useState(false);
  const [viewMode, setViewMode] = useState("cluster");  // cluster(2D 구조) | graph(3D) | list — 진입은 구조 우선
  // 3D는 앵커→이웃 확장 방식으로 누적한다 (전체 로드 금지 — 대용량 대응).
  // id→node 누적 맵 + 링크 누적. 노드를 클릭할 때마다 그 노드 이웃을 더 얹는다.
  const [graphNodes, setGraphNodes] = useState({});   // id -> {id,name,type,trust,degree}
  const [graphLinks, setGraphLinks] = useState([]);   // {source,target,predicate}
  const [graphBusy, setGraphBusy] = useState(false);
  const graph3dRef = useRef(null);

  const [sel, setSel] = useState(null);           // 선택 노드 id
  const [crumbs, setCrumbs] = useState([]);       // pivot 히스토리

  // 좌 pane 접기 상태
  const [secOpen, setSecOpen] = useState({ classes: true, predicates: true, hierarchy: false, saved: true });
  const [openClass, setOpenClass] = useState("");     // 프로퍼티 칩 펼친 클래스
  const [openPred, setOpenPred] = useState("");       // 시그니처 펼친 술어
  const [editClass, setEditClass] = useState("");     // 스키마 편집 폼 펼친 클래스
  const [editPred, setEditPred] = useState("");       // 스키마 편집 폼 펼친 술어
  const [treeCollapsed, setTreeCollapsed] = useState({});

  // 검색 type-ahead
  const [qInput, setQInput] = useState("");
  const [suggestNodes, setSuggestNodes] = useState([]);
  const [showSuggest, setShowSuggest] = useState(false);
  const suggestTimer = useRef(null);

  // 저장/생성
  const [savingView, setSavingView] = useState(false);
  const [viewName, setViewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [cForm, setCForm] = useState({ node_type: "", name: "", definition: "", aliases: "" });
  const [createBusy, setCreateBusy] = useState(false);

  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const noticeRef = useRef(null);
  const flash = (msg) => {
    setNotice(msg);
    clearTimeout(noticeRef.current);
    noticeRef.current = setTimeout(() => setNotice(""), 5000);
  };
  useEffect(() => () => { clearTimeout(noticeRef.current); clearTimeout(suggestTimer.current); }, []);

  // ── 스키마 · 뷰 로드 (네임스페이스별 1회) ──
  useEffect(() => {
    let alive = true;
    setSchema(null); setSchemaErr(""); setFilter(EMPTY_FILTER); setOffset(0);
    setSel(null); setCrumbs([]); setQInput("");
    setViewMode("cluster"); setGraphNodes({}); setGraphLinks([]);
    (async () => {
      try {
        const res = await fetch(`${base}/schema`);
        if (!res.ok) throw new Error(await detailMsg(res));
        if (alive) setSchema(await res.json());
      } catch (e) { if (alive) setSchemaErr(t("admin.schema.err", { e: e.message || e })); }
      try {
        const vr = await fetch(`${base}/views`);
        if (vr.ok && alive) setViews((await vr.json()).views || []);
      } catch { /* 뷰는 보조 — 실패해도 탐색은 동작 */ }
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base]);

  // 스키마 편집 후 재로드 (편집 자체는 성공 — 재로드 실패는 조용히)
  const reloadSchema = useCallback(async () => {
    try {
      const res = await fetch(`${base}/schema`);
      if (res.ok) setSchema(await res.json());
    } catch { /* noop */ }
  }, [base]);

  // ── 센터 목록 로드 (필터/offset 변동 시) ──
  const loadCenter = useCallback(async () => {
    setListBusy(true); setError("");
    try {
      if (filter.mode === "edges") {
        const p = new URLSearchParams();
        if (filter.predicate) p.set("predicate", filter.predicate);
        if (filter.source_type) p.set("source_type", filter.source_type);
        if (filter.target_type) p.set("target_type", filter.target_type);
        const res = await fetch(`${base}/edges?${p.toString()}`);
        if (!res.ok) throw new Error(await detailMsg(res));
        setEdges(await res.json());
        setFacets(null); setSearchSource(null);
      } else if (filter.mode === "query") {
        // 구조화 검색 — 프로퍼티 값 + 관계 제약 (/query, PG jsonb+EXISTS)
        const p = new URLSearchParams({ offset: String(offset), limit: String(LIMIT) });
        if (filter.node_type) p.set("type", filter.node_type);
        if (filter.trust) p.set("trust", filter.trust);
        if (filter.prop_key) {
          p.set("prop_key", filter.prop_key); p.set("prop_op", filter.prop_op || "eq");
          if (filter.prop_value) p.set("prop_value", filter.prop_value);
        }
        if (filter.rel_predicate) p.set("rel_predicate", filter.rel_predicate);
        if (filter.rel_target) p.set("rel_target", filter.rel_target);
        if (filter.rel_target_type) p.set("rel_target_type", filter.rel_target_type);
        if (filter.rel_direction) p.set("rel_direction", filter.rel_direction);
        const res = await fetch(`${base}/query?${p.toString()}`);
        if (!res.ok) throw new Error(await detailMsg(res));
        setNodes(await res.json()); setFacets(null); setSearchSource(null);
      } else if (filter.q && !filter.property) {
        // 검색: ES object-search(BM25 관련도 + 타입/신뢰 파셋). ES 없으면
        // 백엔드가 PG substring 으로 폴백(source 로 구분, 파셋 없음).
        const p = new URLSearchParams({ q: filter.q, top_k: String(LIMIT),
                                        offset: String(offset) });
        if (filter.node_type) p.set("type", filter.node_type);
        if (filter.trust) p.set("trust", filter.trust);
        const res = await fetch(`${base}/object-search?${p.toString()}`);
        if (!res.ok) throw new Error(await detailMsg(res));
        const r = await res.json();
        setNodes({ items: r.items, total: r.total, offset: r.offset,
                   limit: r.limit, capped: false });
        setFacets(r.facets); setSearchSource(r.source);
      } else {
        const p = new URLSearchParams({ offset: String(offset), limit: String(LIMIT) });
        if (filter.node_type) p.set("node_type", filter.node_type);
        if (filter.trust) p.set("trust", filter.trust);
        if (filter.property) p.set("property", filter.property);
        if (filter.kind) p.set("kind", filter.kind);
        const res = await fetch(`${base}/nodes?${p.toString()}`);
        if (!res.ok) throw new Error(await detailMsg(res));
        setNodes(await res.json());
        setFacets(null); setSearchSource(null);
      }
    } catch (e) { setError(t("admin.nodes.err.load", { e: e.message || e })); }
    setListBusy(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, filter, offset]);

  useEffect(() => { loadCenter(); }, [loadCenter]);

  // 필터 변경 헬퍼 — offset 리셋
  const applyFilter = (patch) => { setFilter({ ...EMPTY_FILTER, ...patch }); setOffset(0); };
  const patchFilter = (patch) => { setFilter((f) => ({ ...f, ...patch })); setOffset(0); };
  const clearField = (k) => {
    if (k === "signature") patchFilter({ source_type: "", target_type: "" });
    else if (k === "predicate") applyFilter({ mode: "objects" });   // 술어 해제 = 객체 모드 복귀
    else if (k === "structured") applyFilter({ mode: "objects" });  // 구조화 검색 해제
    else patchFilter({ [k]: "" });
  };

  // ── 노드 열기 / pivot / 브레드크럼 ──
  const openNode = (node, pivot = false) => {
    const id = node.node_id || node.id;
    if (pivot) setCrumbs((c) => [...c, { id, name: node.name, type: node.type }]);
    else setCrumbs([{ id, name: node.name, type: node.type }]);
    setSel(id);
    setShowSuggest(false);
  };
  const crumbTo = (i) => { setCrumbs((c) => c.slice(0, i + 1)); setSel(crumbs[i].id); };
  const closeInspector = () => { setSel(null); setCrumbs([]); };

  // ── 그래프 뷰 — 선택 노드의 ego 를 기존 그래프에 "병합"(리셋 아님) ──
  // 개요는 그대로 두고, 목록에서 고른 미표시 노드만 등장시킨다. 다른 노드는
  // 사라지지 않고 뷰에서 흐려질 뿐. 비용 O(그 노드 차수).
  const mergeEgo = useCallback(async (anchorId) => {
    if (!anchorId) return;
    setGraphBusy(true); setError("");
    try {
      const res = await fetch(
        `${base}/neighbors?node_id=${encodeURIComponent(anchorId)}&limit=80`);
      if (!res.ok) throw new Error(await detailMsg(res));
      const d = await res.json();
      if (d.error) { setGraphBusy(false); return; }
      setGraphNodes((prev) => {
        const next = { ...prev };
        for (const n of d.nodes || []) if (!next[n.id]) next[n.id] = n;
        return next;
      });
      setGraphLinks((prev) => {
        const key = (l) => `${l.source} ${l.predicate} ${l.target}`;
        const seen = new Set(prev.map(key));
        const add = (d.links || [])
          .map((l) => ({ source: l.source, target: l.target, predicate: l.predicate || "" }))
          .filter((l) => !seen.has(key(l)));
        return add.length ? [...prev, ...add] : prev;
      });
      if (d.truncated) {
        flash(t("admin.exp.graph.truncated",
                { n: d.total_neighbors, limit: d.limit }));
      }
    } catch (e) { setError(t("admin.exp.graph.err", { e: e.message || e })); }
    setGraphBusy(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base]);

  // 진입 시 구조 그래프 자동 로드 — 빈 3D 방지. 상위 노드+엣지의 연결 구조를
  // 한 번에(전체 로드가 아니라 bounded overview). 큰 네임스페이스는 상위 일부만
  // 보이고, 사용자가 노드를 클릭하면 그 이웃을 이어서 확장(expandNode).
  const loadStructure = useCallback(async () => {
    setGraphBusy(true); setError("");
    try {
      const res = await fetch(`${base}/data?limit=120`);
      if (!res.ok) throw new Error(await detailMsg(res));
      const d = await res.json();
      const nodes = {};
      for (const n of d.nodes || []) {
        nodes[n.id] = { id: n.id, name: n.name || n.id, type: n.type || "",
                        trust: n.trust || "unset", degree: 0 };
      }
      setGraphNodes(nodes);
      setGraphLinks((d.links || []).map((l) => ({
        source: l.source, target: l.target, predicate: l.predicate || "" })));
      // Graph3D 가 데이터 갱신 후 자동 zoomToFit 한다(내장). 여기서 재호출하면
      // 레이아웃이 덜 퍼진 상태를 다시 프레이밍해 오히려 어긋난다.
    } catch (e) { setError(t("admin.exp.graph.err", { e: e.message || e })); }
    setGraphBusy(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base]);

  // 네임스페이스 진입 시 구조 자동 로드 (base 변동 = 새 네임스페이스)
  useEffect(() => { loadStructure(); }, [loadStructure]);

  // 선택(sel)이 바뀌면: 이미 그래프에 있으면 그대로 두고(뷰가 흐림/강조 처리),
  // 없으면(목록·검색에서 고른 미표시 노드) ego 를 병합해 등장시킨다.
  // 개요를 리셋하지 않으므로 다른 노드는 사라지지 않는다.
  useEffect(() => {
    if (!sel) return;
    if (viewMode !== "cluster" && viewMode !== "graph") return;
    if (graphNodes[sel]) return;
    mergeEgo(sel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel, viewMode]);

  // 리셋 = 선택 해제 + 구조 개요로 되돌림
  const resetGraph = () => { setSel(null); setCrumbs([]); loadStructure(); };
  const onGraphSelect = (id) => {
    if (!id) { closeInspector(); return; }   // 빈 공간 클릭 = 선택 해제(개요 유지·이동 없음)
    const n = graphNodes[id] || { id, name: id, type: "" };
    openNode({ node_id: n.id, name: n.name, type: n.type });  // sel 변경 → 확장 트리거
  };
  // 선택은 하이라이트일 뿐 — 레이아웃 재계산을 트리거하면 안 된다. graphNodes/
  // graphLinks 가 실제로 바뀔 때만 새 참조를 만들어(useMemo), setSel 로 인한
  // 리렌더가 자식(ClusterView/Graph3D)의 레이아웃 effect 를 무효화하지 않게 한다.
  const graphNodeList = useMemo(() => Object.values(graphNodes), [graphNodes]);
  const graph3dData = useMemo(() => ({ nodes: graphNodeList, links: graphLinks }),
                              [graphNodeList, graphLinks]);

  // 그래프 컨트롤 refs — 2D 는 d3 zoom 을 ClusterView 가 소유하므로 imperative
  // handle 로 노출받고, 전체화면은 각 뷰의 래퍼 div 를 Fullscreen API 로 토글.
  const clusterRef = useRef(null);
  const graph2dWrapRef = useRef(null);
  const graph3dWrapRef = useRef(null);
  const toggleFullscreen = (el) => {
    if (!el) return;
    if (document.fullscreenElement) document.exitFullscreen?.();
    else el.requestFullscreen?.();
  };

  // ── type-ahead 검색 ──
  const onSearchInput = (v) => {
    setQInput(v); setShowSuggest(!!v.trim());
    clearTimeout(suggestTimer.current);
    if (!v.trim()) { setSuggestNodes([]); return; }
    suggestTimer.current = setTimeout(async () => {
      try {
        // 엔터 검색과 동일한 경로(object-search) 로 통일 — 자동완성과 결과가
        // 어긋나지 않게. (이전엔 /nodes substring 이라 엔터 결과와 달랐다.)
        const res = await fetch(
          `${base}/object-search?q=${encodeURIComponent(v.trim())}&top_k=8`);
        if (res.ok) setSuggestNodes((await res.json()).items || []);
      } catch { /* 제안은 보조 */ }
    }, 220);
  };
  const runSearch = () => { applyFilter({ mode: "objects", q: qInput.trim() }); setShowSuggest(false); };

  // 구조화 검색 실행 — qForm 을 filter(mode:query)로 적용
  const runQuery = () => {
    const f = { mode: "query", rel_direction: qForm.rel_direction };
    if (qForm.prop_key.trim()) {
      f.prop_key = qForm.prop_key.trim(); f.prop_op = qForm.prop_op;
      if (qForm.prop_op !== "exists") f.prop_value = qForm.prop_value.trim();
    }
    if (qForm.rel_predicate) f.rel_predicate = qForm.rel_predicate;
    if (qForm.rel_target.trim()) f.rel_target = qForm.rel_target.trim();
    if (qForm.rel_target_type) f.rel_target_type = qForm.rel_target_type;
    const hasProp = !!f.prop_key, hasRel = !!(f.rel_predicate || f.rel_target || f.rel_target_type);
    if (!hasProp && !hasRel) return;   // 빈 검색 방지
    applyFilter(f); setAdvOpen(false);
  };

  // ── 추출(3+): 현재 검색 코호트를 학습데이터(dataset)로 ──
  const COHORT_KEYS = ["node_type", "trust", "prop_key", "prop_value", "prop_op",
    "rel_predicate", "rel_target", "rel_target_type", "rel_direction"];
  const cohortFromFilter = () => {
    const c = {};
    if (filter.node_type) c.node_type = filter.node_type;
    if (filter.trust) c.trust = filter.trust;
    if (filter.mode === "query") {
      ["prop_key", "prop_value", "prop_op", "rel_predicate", "rel_target",
       "rel_target_type", "rel_direction"].forEach((k) => { if (filter[k]) c[k] = filter[k]; });
    } else if (filter.property) {
      c.prop_key = filter.property; c.prop_op = "exists";  // 프로퍼티 드릴다운 = 있음
    }
    return Object.keys(c).length ? c : null;
  };
  const runExtract = async () => {
    if (!extractFmts.length || extractBusy) return;
    setExtractBusy(true); setError(""); setExtractRes(null);
    try {
      const res = await fetch(`${base}/dataset`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ formats: extractFmts, cohort: cohortFromFilter() }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      setExtractRes(await res.json());
    } catch (e) { setError(t("admin.exp.extract.err", { e: e.message || e })); }
    setExtractBusy(false);
  };
  const extractDownloadUrl = () => {
    const p = new URLSearchParams({ formats: extractFmts.join(",") });
    const c = cohortFromFilter();
    if (c) {
      const map = { node_type: "c_type", trust: "c_trust", prop_key: "c_prop_key",
        prop_value: "c_prop_value", prop_op: "c_prop_op", rel_predicate: "c_rel_predicate",
        rel_target: "c_rel_target", rel_target_type: "c_rel_target_type",
        rel_direction: "c_rel_direction" };
      Object.entries(c).forEach(([k, v]) => p.set(map[k], v));
    }
    return `${base}/dataset.jsonl?${p.toString()}`;
  };

  // ── 뷰 저장/복원/삭제 ──
  const saveView = async () => {
    const name = viewName.trim();
    if (!name) return;
    setError("");
    try {
      const res = await fetch(`${base}/views`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, filter }),
      });
      if (res.status === 409) { flash(t("admin.exp.saved.dupe")); return; }
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.exp.saved.done"));
      setSavingView(false); setViewName("");
      const vr = await fetch(`${base}/views`);
      if (vr.ok) setViews((await vr.json()).views || []);
    } catch (e) { setError(t("admin.exp.saved.err", { e: e.message || e })); }
  };
  const restoreView = (v) => {
    applyFilter({ ...EMPTY_FILTER, ...(v.filter || {}) });
    setQInput((v.filter && v.filter.q) || "");
    closeInspector();
  };
  const deleteView = async (id) => {
    setError("");
    try {
      const res = await fetch(`${base}/views/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.exp.saved.deleted"));
      setViews((vs) => vs.filter((v) => v.id !== id));
    } catch (e) { setError(t("admin.exp.saved.err", { e: e.message || e })); }
  };

  // ── 새 노드 생성 ──
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
      setCreating(false); setCForm({ node_type: "", name: "", definition: "", aliases: "" });
      await loadCenter();
      onChanged && onChanged();
    } catch (e) { setError(t("admin.nodes.create.err", { e: e.message || e })); }
    setCreateBusy(false);
  };

  // 인스펙터 콜백
  const onRejected = () => { closeInspector(); loadCenter(); onChanged && onChanged(); };

  if (schemaErr) return <div className="error" style={{ marginTop: 18 }}>{schemaErr}</div>;
  if (!schema) {
    return <div className="empty-state" style={{ marginTop: 18 }}>
      <span className="spinner" />{t("admin.loading")}</div>;
  }

  const classes = schema.classes || [];
  const predicates = schema.predicates || [];
  const hierarchy = schema.hierarchy || [];
  const types = classes.map((c) => c.type);
  const totalObjects = classes.reduce((a, c) => a + (c.count || 0), 0);

  // 검색 제안 (클래스/술어는 클라이언트 필터)
  const qLower = qInput.trim().toLowerCase();
  const suggestClasses = qLower ? classes.filter((c) => c.type.toLowerCase().includes(qLower)).slice(0, 5) : [];
  const suggestPreds = qLower ? predicates.filter((p) => p.predicate.toLowerCase().includes(qLower)).slice(0, 5) : [];

  const nItems = nodes?.items || [];
  const nTotal = nodes?.total || 0;
  const nCapped = !!nodes?.capped;   // 백엔드가 total 을 상한에서 잘랐는지 → "N+" 표기
  const from = nTotal ? offset + 1 : 0;
  const to = Math.min(offset + LIMIT, nTotal);
  const eItems = edges?.edges || [];

  const sec = (key) => setSecOpen((s) => ({ ...s, [key]: !s[key] }));

  // 활성 필터 칩
  const chips = [];
  if (filter.node_type) chips.push(["node_type", t("admin.exp.chip.type", { v: filter.node_type })]);
  if (filter.property) chips.push(["property", t("admin.exp.chip.property", { v: filter.property })]);
  if (filter.trust) chips.push(["trust", t("admin.exp.chip.trust", { v: filter.trust })]);
  if (filter.q) chips.push(["q", t("admin.exp.chip.q", { v: filter.q })]);
  if (filter.predicate) chips.push(["predicate", t("admin.exp.chip.predicate", { v: filter.predicate })]);
  if (filter.source_type || filter.target_type) {
    chips.push(["signature", t("admin.exp.chip.signature", { v: `${filter.source_type || "*"} → ${filter.target_type || "*"}` })]);
  }
  if (filter.mode === "query") {
    const parts = [];
    if (filter.prop_key) {
      const opsym = { eq: "=", contains: "⊃", exists: "있음" }[filter.prop_op] || "=";
      parts.push(`${filter.prop_key} ${opsym} ${filter.prop_op === "exists" ? "" : filter.prop_value}`.trim());
    }
    if (filter.rel_predicate || filter.rel_target || filter.rel_target_type) {
      const arrow = filter.rel_direction === "out" ? "→" : "←";
      parts.push(`${arrow}${filter.rel_predicate || "*"}${arrow}${filter.rel_target || filter.rel_target_type || "*"}`);
    }
    if (parts.length) chips.push(["structured", t("admin.exp.chip.structured", { v: parts.join("  ·  ") })]);
  }

  const renderTree = (node, depth) => {
    const kids = node.children || [];
    const collapsed = !!treeCollapsed[node.id];
    return (
      <div key={node.id}>
        <div className="oa-exp-tree-row" style={{ paddingLeft: depth * 14 }}>
          {kids.length > 0 ? (
            <button className="tree-toggle"
                    onClick={() => setTreeCollapsed((c) => ({ ...c, [node.id]: !c[node.id] }))}>
              {collapsed ? "▸" : "▾"}</button>
          ) : <span style={{ width: 18, display: "inline-block" }} />}
          <button className="oa-exp-tree-name"
                  onClick={() => openNode({ id: node.id, name: node.name, type: "HeritageClass" })}>
            {node.name || node.id}
            {kids.length > 0 && <span className="oa-exp-item-count">{kids.length}</span>}
          </button>
        </div>
        {!collapsed && kids.map((c) => renderTree(c, depth + 1))}
      </div>
    );
  };

  return (
    <>
      {notice && <div className="admin-notice" style={{ marginTop: 14, marginBottom: 0 }}>{notice}</div>}
      {error && <div className="error" style={{ marginTop: 14 }}>{error}</div>}

      <div className="oa-exp" style={{ gridTemplateColumns: sel ? "230px minmax(0,1fr) 380px" : "230px minmax(0,1fr)" }}>
        {/* ── 좌 pane — 스키마 내비 ── */}
        <nav className="oa-exp-nav">
          <button className={`oa-exp-all ${filter.mode === "objects" && !filter.node_type && !filter.property && !filter.q ? "on" : ""}`}
                  onClick={() => applyFilter({})}>
            {t("admin.exp.allObjects")}<span className="oa-exp-item-count">{totalObjects.toLocaleString()}</span>
          </button>

          {/* 클래스 */}
          <button className="oa-exp-sec-head" onClick={() => sec("classes")}>
            <span>{secOpen.classes ? "▾" : "▸"} {t("admin.exp.classes")}</span>
            <span className="oa-exp-item-count">{classes.length}</span>
          </button>
          {secOpen.classes && classes.map((c) => (
            <div key={c.type}>
              <div className="oa-exp-item-wrap">
                <button className={`oa-exp-item ${filter.node_type === c.type && !filter.property ? "on" : ""}`}
                        onClick={() => applyFilter({ node_type: c.type })}
                        title={c.declared?.description || ""}>
                  <span className="oa-exp-item-name">
                    {c.type}
                    {c.declared?.deprecated && <span className="oa-sch-badge">deprecated</span>}
                  </span>
                  <span className="oa-exp-item-count">{(c.count || 0).toLocaleString()}</span>
                </button>
                {!readOnly && (
                  <button className="oa-exp-expand oa-sch-editbtn" title={t("admin.sch.edit")}
                          onClick={() => { setEditClass(editClass === c.type ? "" : c.type); setEditPred(""); }}>✎</button>
                )}
                <button className="oa-exp-expand"
                        onClick={() => setOpenClass(openClass === c.type ? "" : c.type)}
                        title="properties">{openClass === c.type ? "▾" : "▸"}</button>
              </div>
              {!readOnly && editClass === c.type && (
                <SchemaTypeEditor base={base} cls={c} t={t} setError={setError}
                  onCancel={() => setEditClass("")}
                  onSaved={async () => {
                    setEditClass(""); await reloadSchema();
                    flash(t("admin.sch.typeSaved")); onChanged && onChanged();
                  }}
                  onRenamed={async (oldT, newT, n) => {
                    setEditClass("");
                    if (filter.node_type === oldT) applyFilter({});
                    await reloadSchema();
                    flash(t("admin.sch.renamed", { n })); onChanged && onChanged();
                  }} />
              )}
              {openClass === c.type && (
                <div className="oa-exp-props">
                  {Object.entries(c.properties || {}).sort((a, b) => b[1] - a[1]).slice(0, TOP_PROPS).map(([p, cnt]) => (
                    <button key={p} className={`oa-exp-propchip ${filter.node_type === c.type && filter.property === p ? "on" : ""}`}
                            onClick={() => applyFilter({ node_type: c.type, property: p })}>
                      {p} <b>{cnt}</b>
                    </button>
                  ))}
                  {Object.keys(c.properties || {}).length > TOP_PROPS && (
                    <span className="oa-exp-propchip" style={{ opacity: 0.6 }}>
                      +{Object.keys(c.properties).length - TOP_PROPS}</span>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* 술어 */}
          <button className="oa-exp-sec-head" onClick={() => sec("predicates")}>
            <span>{secOpen.predicates ? "▾" : "▸"} {t("admin.exp.predicates")}</span>
            <span className="oa-exp-item-count">{predicates.length}</span>
          </button>
          {secOpen.predicates && predicates.map((p) => (
            <div key={p.predicate}>
              <div className="oa-exp-item-wrap">
                <button className={`oa-exp-item ${filter.mode === "edges" && filter.predicate === p.predicate && !filter.source_type ? "on" : ""}`}
                        onClick={() => applyFilter({ mode: "edges", predicate: p.predicate })}
                        title={p.declared?.description || ""}>
                  <span className="oa-exp-item-name mono">
                    {p.predicate}
                    {p.declared && (p.declared.domain || p.declared.range) && (
                      <span className="oa-sch-sig">{p.declared.domain || "*"}→{p.declared.range || "*"}</span>
                    )}
                  </span>
                  <span className="oa-exp-item-count">{(p.count || 0).toLocaleString()}</span>
                </button>
                {!readOnly && (
                  <button className="oa-exp-expand oa-sch-editbtn" title={t("admin.sch.edit")}
                          onClick={() => { setEditPred(editPred === p.predicate ? "" : p.predicate); setEditClass(""); }}>✎</button>
                )}
                {(p.pairs || []).length > 0 && (
                  <button className="oa-exp-expand"
                          onClick={() => setOpenPred(openPred === p.predicate ? "" : p.predicate)}
                          title="signatures">{openPred === p.predicate ? "▾" : "▸"}</button>
                )}
              </div>
              {!readOnly && editPred === p.predicate && (
                <SchemaPredEditor base={base} pred={p} t={t} setError={setError}
                  onCancel={() => setEditPred("")}
                  onSaved={async () => {
                    setEditPred(""); await reloadSchema();
                    flash(t("admin.sch.predSaved")); onChanged && onChanged();
                  }} />
              )}
              {openPred === p.predicate && (
                <div className="oa-exp-props">
                  {(p.pairs || []).map((pr, i) => (
                    <button key={i} className={`oa-exp-propchip ${filter.predicate === p.predicate && filter.source_type === pr.source_type && filter.target_type === pr.target_type ? "on" : ""}`}
                            onClick={() => applyFilter({ mode: "edges", predicate: p.predicate, source_type: pr.source_type, target_type: pr.target_type })}>
                      {pr.source_type} → {pr.target_type} <b>×{pr.count}</b>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* is_a 계층 */}
          <button className="oa-exp-sec-head" onClick={() => sec("hierarchy")}>
            <span>{secOpen.hierarchy ? "▾" : "▸"} {t("admin.exp.hierarchy")}</span>
            <span className="oa-exp-item-count">{schema.hierarchy_edges || 0}</span>
          </button>
          {secOpen.hierarchy && (
            (schema.hierarchy_edges ? (
              <div className="oa-exp-tree">{hierarchy.map((r) => renderTree(r, 0))}</div>
            ) : (
              <p className="hint" style={{ padding: "4px 12px" }}>{t("admin.schema.hierarchy.empty")}</p>
            ))
          )}

          {/* 저장된 탐색 */}
          <button className="oa-exp-sec-head" onClick={() => sec("saved")}>
            <span>{secOpen.saved ? "▾" : "▸"} {t("admin.exp.saved")}</span>
            <span className="oa-exp-item-count">{views.length}</span>
          </button>
          {secOpen.saved && (
            <div style={{ padding: "2px 8px 8px" }}>
              {!savingView ? (
                <button className="ghost oa-exp-save-btn" onClick={() => setSavingView(true)}>
                  {t("admin.exp.saveCurrent")}</button>
              ) : (
                <div className="oa-exp-save-form">
                  <input type="text" value={viewName} placeholder={t("admin.exp.saveNamePh")}
                         onChange={(e) => setViewName(e.target.value)}
                         onKeyDown={(e) => { if (e.key === "Enter") saveView(); }} />
                  <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
                    <button onClick={saveView} disabled={!viewName.trim()}
                            style={{ padding: "5px 12px", fontSize: "0.76rem" }}>{t("admin.exp.save")}</button>
                    <button className="ghost" onClick={() => { setSavingView(false); setViewName(""); }}
                            style={{ padding: "5px 12px", fontSize: "0.76rem" }}>{t("admin.exp.cancel")}</button>
                  </div>
                </div>
              )}
              {views.length === 0 && <p className="hint" style={{ marginTop: 6 }}>{t("admin.exp.savedEmpty")}</p>}
              {views.map((v) => (
                <div key={v.id} className="oa-exp-saved-row">
                  <button className="oa-exp-saved-name" onClick={() => restoreView(v)}>{v.name}</button>
                  <button className="tree-toggle" title="✕" onClick={() => deleteView(v.id)}>✕</button>
                </div>
              ))}
            </div>
          )}
        </nav>

        {/* ── 센터 pane ── */}
        <section className="oa-exp-center">
          {/* 검색 + 새 노드 */}
          <div className="oa-exp-toolbar">
            <div className="oa-exp-search">
              <input type="text" value={qInput} placeholder={t("admin.exp.searchPh")}
                     onChange={(e) => onSearchInput(e.target.value)}
                     onFocus={() => setShowSuggest(!!qInput.trim())}
                     onBlur={() => setTimeout(() => setShowSuggest(false), 150)}
                     onKeyDown={(e) => { if (e.key === "Enter") runSearch(); }} />
              {showSuggest && (suggestClasses.length + suggestPreds.length + suggestNodes.length > 0) && (
                <div className="oa-exp-suggest">
                  {suggestClasses.map((c) => (
                    <button key={`c${c.type}`} className="oa-exp-suggest-item"
                            onMouseDown={() => applyFilter({ node_type: c.type })}>
                      <span className="oa-exp-suggest-kind">{t("admin.exp.suggest.class")}</span>
                      {c.type}<span className="oa-exp-item-count">{c.count}</span>
                    </button>
                  ))}
                  {suggestPreds.map((p) => (
                    <button key={`p${p.predicate}`} className="oa-exp-suggest-item"
                            onMouseDown={() => applyFilter({ mode: "edges", predicate: p.predicate })}>
                      <span className="oa-exp-suggest-kind">{t("admin.exp.suggest.predicate")}</span>
                      <span className="mono">{p.predicate}</span><span className="oa-exp-item-count">{p.count}</span>
                    </button>
                  ))}
                  {suggestNodes.map((n) => (
                    <button key={`n${n.node_id}`} className="oa-exp-suggest-item"
                            onMouseDown={() => openNode(n)}>
                      <span className="oa-exp-suggest-kind">{t("admin.exp.suggest.node")}</span>
                      {n.name || n.node_id}<span className="oa-exp-suggest-type">{n.type}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="oa-exp-tools">
              <div className="oa-exp-viewtoggle">
                <button className={`oa-exp-viewbtn ${viewMode === "cluster" ? "on" : ""}`}
                        onClick={() => { setViewMode("cluster");
                          if (graphNodeList.length === 0 && !graphBusy) loadStructure(); }}>
                  {t("admin.exp.view.cluster")}</button>
                <button className={`oa-exp-viewbtn ${viewMode === "graph" ? "on" : ""}`}
                        onClick={() => { setViewMode("graph");
                          if (graphNodeList.length === 0 && !graphBusy) loadStructure(); }}>
                  {t("admin.exp.view.graph")}</button>
                <button className={`oa-exp-viewbtn ${viewMode === "list" ? "on" : ""}`}
                        onClick={() => setViewMode("list")}>{t("admin.exp.view.list")}</button>
              </div>
              <span className="oa-exp-tools-sep" />
              <button className={`oa-exp-tool ${advOpen || filter.mode === "query" ? "on" : ""}`}
                      onClick={() => setAdvOpen((v) => !v)} title={t("admin.exp.adv.tip")}>
                {t("admin.exp.adv.toggle")}</button>
              <button className={`oa-exp-tool ${extractOpen ? "on" : ""}`}
                      onClick={() => { setExtractOpen((v) => !v); setExtractRes(null); }}
                      title={t("admin.exp.extract.tip")}>{t("admin.exp.extract.toggle")}</button>
              {viewMode === "list" && filter.mode === "objects" && !readOnly && (
                <button className="oa-exp-tool primary" onClick={() => setCreating(!creating)}>
                  {t("admin.exp.new")}</button>
              )}
            </div>
          </div>

          {/* ── 고급 검색(구조화): 프로퍼티 값 + 관계 제약 ── */}
          {advOpen && (
            <div className="oa-exp-adv">
              <div className="oa-exp-adv-row">
                <span className="oa-exp-adv-lbl">{t("admin.exp.adv.prop")}</span>
                <input className="oa-exp-adv-in" placeholder={t("admin.exp.adv.propKey")}
                       list="oa-exp-prop-list" value={qForm.prop_key}
                       onChange={(e) => setQForm({ ...qForm, prop_key: e.target.value })} />
                <select value={qForm.prop_op}
                        onChange={(e) => setQForm({ ...qForm, prop_op: e.target.value })}>
                  <option value="eq">=</option>
                  <option value="contains">{t("admin.exp.adv.contains")}</option>
                  <option value="exists">{t("admin.exp.adv.exists")}</option>
                </select>
                {qForm.prop_op !== "exists" && (
                  <input className="oa-exp-adv-in" placeholder={t("admin.exp.adv.propVal")}
                         value={qForm.prop_value}
                         onChange={(e) => setQForm({ ...qForm, prop_value: e.target.value })} />
                )}
              </div>
              <div className="oa-exp-adv-row">
                <span className="oa-exp-adv-lbl">{t("admin.exp.adv.rel")}</span>
                <select value={qForm.rel_direction}
                        onChange={(e) => setQForm({ ...qForm, rel_direction: e.target.value })}>
                  <option value="out">{t("admin.exp.adv.dirOut")}</option>
                  <option value="in">{t("admin.exp.adv.dirIn")}</option>
                </select>
                <select value={qForm.rel_predicate}
                        onChange={(e) => setQForm({ ...qForm, rel_predicate: e.target.value })}>
                  <option value="">{t("admin.exp.adv.anyPred")}</option>
                  {predicates.map((p) => <option key={p.predicate} value={p.predicate}>{p.predicate}</option>)}
                </select>
                <select value={qForm.rel_target_type}
                        onChange={(e) => setQForm({ ...qForm, rel_target_type: e.target.value })}>
                  <option value="">{t("admin.exp.adv.anyType")}</option>
                  {classes.map((c) => <option key={c.type} value={c.type}>{c.type}</option>)}
                </select>
                <input className="oa-exp-adv-in" placeholder={t("admin.exp.adv.relTarget")}
                       list="oa-exp-node-list" value={qForm.rel_target}
                       onChange={(e) => setQForm({ ...qForm, rel_target: e.target.value })} />
              </div>
              <div className="oa-exp-adv-actions">
                <button onClick={runQuery}>{t("admin.exp.adv.run")}</button>
                <button className="ghost" onClick={() => setQForm({ prop_key: "", prop_op: "eq", prop_value: "", rel_predicate: "", rel_direction: "out", rel_target_type: "", rel_target: "" })}>
                  {t("admin.exp.adv.clear")}</button>
                <span className="oa-exp-adv-hint">{t("admin.exp.adv.hint")}</span>
              </div>
              <datalist id="oa-exp-prop-list">
                <option value="source" /><option value="definition" /><option value="aliases" />
              </datalist>
            </div>
          )}

          {/* ── 추출(3+): 검색 코호트 → 학습데이터(RL/LLM) ── */}
          {extractOpen && (
            <div className="oa-exp-adv">
              <div className="oa-exp-adv-row">
                <span className="oa-exp-adv-lbl">{t("admin.exp.extract.scope")}</span>
                <span className="oa-exp-extract-scope">
                  {cohortFromFilter()
                    ? t("admin.exp.extract.cohort", { n: (nodes?.total ?? 0).toLocaleString() })
                    : t("admin.exp.extract.whole")}
                </span>
              </div>
              <div className="oa-exp-adv-row">
                <span className="oa-exp-adv-lbl">{t("admin.exp.extract.fmt")}</span>
                {["triples", "qa", "surface", "evidence"].map((f) => (
                  <label key={f} className="oa-exp-fmt">
                    <input type="checkbox" checked={extractFmts.includes(f)}
                           onChange={(e) => setExtractFmts((v) =>
                             e.target.checked ? [...v, f] : v.filter((x) => x !== f))} />
                    {f}</label>
                ))}
              </div>
              <div className="oa-exp-adv-actions">
                <button onClick={runExtract} disabled={extractBusy || !extractFmts.length}>
                  {extractBusy && <span className="spinner" />}{t("admin.exp.extract.preview")}</button>
                <a className="btn-link" href={extractDownloadUrl()} target="_blank" rel="noreferrer">
                  {t("admin.exp.extract.download")}</a>
                {extractRes && (
                  <span className="oa-exp-extract-res">
                    {extractRes.cohort_size != null &&
                      t("admin.exp.extract.fromCohort", { n: extractRes.cohort_size.toLocaleString() })}
                    {" "}{t("admin.exp.extract.rows", { n: extractRes.total.toLocaleString() })}
                    {" · "}{Object.entries(extractRes.counts).map(([k, v]) => `${k} ${v}`).join(" · ")}
                  </span>
                )}
              </div>
              <span className="oa-exp-adv-hint">{t("admin.exp.extract.hint")}</span>
            </div>
          )}

          {/* ── 타입-클러스터 2D 구조 뷰 (②b) — 진입 기본, 대용량도 읽힘 ── */}
          {viewMode === "cluster" ? (
            <div className="oa-exp-graph" ref={graph2dWrapRef}>
              {graphBusy && graphNodeList.length === 0 ? (
                <div className="empty-state" style={{ height: "100%", border: "none", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                  <span className="spinner" />{t("admin.loading")}
                </div>
              ) : (
                <ClusterView ref={clusterRef} nodes={graphNodeList} links={graphLinks}
                             selectedId={sel} t={t}
                             onSelect={(n) => n
                               ? openNode({ node_id: n.id, name: n.name, type: n.type })
                               : closeInspector()} />
              )}
              {graphNodeList.length > 0 && (
                <div className="oa-exp-graph-ctrl">
                  <button className="oa-exp-graph-ctrl-btn" title="Zoom in" onClick={() => clusterRef.current?.zoomIn()}>＋</button>
                  <button className="oa-exp-graph-ctrl-btn" title="Zoom out" onClick={() => clusterRef.current?.zoomOut()}>－</button>
                  <button className="oa-exp-graph-ctrl-btn" title="Fullscreen" onClick={() => toggleFullscreen(graph2dWrapRef.current)}>⛶</button>
                </div>
              )}
              {graphNodeList.length > 0 && (
                <div className="oa-exp-graph-count">
                  {t("admin.exp.graph.count", { n: graphNodeList.length })}
                  <button className="oa-exp-graph-reset" onClick={resetGraph}>{t("admin.exp.graph.reset")}</button>
                </div>
              )}
              <div className="oa-exp-graph-hint">{t("admin.exp.cluster.hint")}</div>
            </div>
          ) : viewMode === "graph" ? (
            <div className="oa-exp-graph" ref={graph3dWrapRef}>
              {graphNodeList.length === 0 ? (
                <div className="empty-state" style={{ height: "100%", border: "none", display: "flex", flexDirection: "column", gap: 8, alignItems: "center", justifyContent: "center" }}>
                  {graphBusy ? <><span className="spinner" />{t("admin.loading")}</>
                    : t("admin.exp.graph.anchorHint")}
                </div>
              ) : (
                <Graph3D ref={graph3dRef} data={graph3dData}
                         selectedId={sel} onSelect={onGraphSelect} hiddenTypes={[]}
                         emptyText={t("admin.exp.graph.empty")} />
              )}
              <div className="oa-exp-graph-ctrl">
                <button className="oa-exp-graph-ctrl-btn" title="Zoom in" onClick={() => graph3dRef.current?.zoomIn()}>＋</button>
                <button className="oa-exp-graph-ctrl-btn" title="Zoom out" onClick={() => graph3dRef.current?.zoomOut()}>－</button>
                <button className="oa-exp-graph-ctrl-btn" title="Fit" onClick={() => graph3dRef.current?.fit()}>◎</button>
                <button className="oa-exp-graph-ctrl-btn" title="Fullscreen" onClick={() => toggleFullscreen(graph3dWrapRef.current)}>⛶</button>
              </div>
              {graphNodeList.length > 0 && (
                <div className="oa-exp-graph-count">
                  {graphBusy && <span className="spinner" />}
                  {t("admin.exp.graph.count", { n: graphNodeList.length })}
                  <button className="oa-exp-graph-reset" onClick={resetGraph}>{t("admin.exp.graph.reset")}</button>
                </div>
              )}
              <div className="oa-exp-graph-hint">{t("admin.exp.graph.hint")}</div>
            </div>
          ) : (
          <>
          {/* 목록 모드 */}

          {/* 활성 필터 칩 */}
          {chips.length > 0 && (
            <div className="oa-exp-chips">
              {chips.map(([k, label]) => (
                <span key={k} className="oa-exp-chip">
                  {label}<button className="oa-exp-chip-x" onClick={() => clearField(k)}>✕</button>
                </span>
              ))}
            </div>
          )}

          {/* 새 노드 폼 */}
          {creating && !readOnly && filter.mode === "objects" && (
            <div className="card soft" style={{ marginTop: 12 }}>
              <h2 style={{ fontSize: "0.9rem" }}>{t("admin.nodes.create.title")}</h2>
              <div className="row" style={{ marginTop: 10 }}>
                <div>
                  <label>{t("admin.nodes.create.type")}</label>
                  <input type="text" list="oa-exp-type-list" value={cForm.node_type}
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
                  {createBusy && <span className="spinner" />}{t("admin.nodes.create.submit")}</button>
                <button className="ghost" onClick={() => setCreating(false)}>{t("admin.nodes.create.cancel")}</button>
              </div>
            </div>
          )}

          {/* 엣지 모드 헤더 */}
          {filter.mode === "edges" && (
            <button className="ghost oa-exp-back" onClick={() => applyFilter({ mode: "objects" })}>
              {t("admin.exp.backToObjects")}</button>
          )}

          {/* 목록 */}
          {listBusy && !nodes && !edges ? (
            <div className="empty-state"><span className="spinner" />{t("admin.loading")}</div>
          ) : filter.mode === "edges" ? (
            eItems.length === 0 ? (
              <p className="hint" style={{ marginTop: 12 }}>{t("admin.exp.edgesEmpty")}</p>
            ) : (
              <div className="oa-exp-table-wrap">
                <table>
                  <thead><tr>
                    <th>{t("admin.exp.th.source")}</th>
                    <th style={{ width: "22%" }}>{t("admin.exp.th.predicate")}</th>
                    <th>{t("admin.exp.th.target")}</th>
                    {!readOnly && <th style={{ width: 40 }}></th>}
                  </tr></thead>
                  <tbody>
                    {eItems.map((e, i) => (
                      <tr key={i}>
                        <td><button className="oa-exp-pivot"
                              onClick={() => openNode({ id: e.source, name: e.source_name, type: e.source_type })}>
                              {e.source_name || e.source}</button></td>
                        <td className="mono" style={{ color: "var(--accent)" }}>—{e.predicate}→</td>
                        <td><button className="oa-exp-pivot"
                              onClick={() => openNode({ id: e.target, name: e.target_name, type: e.target_type })}>
                              {e.target_name || e.target}</button></td>
                        {!readOnly && (
                          <td><button className="tree-toggle" title="✕"
                                onClick={() => removeEdgeFromList(e)}>✕</button></td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          ) : (
            <>
            {filter.mode === "objects" && !filter.q && (
              <div className="oa-exp-kindbar">
                {[["", "all"], ["class", "class"], ["instance", "instance"]].map(([k, lbl]) => (
                  <button key={lbl}
                    className={`oa-exp-kindbtn ${(filter.kind || "") === k ? "on" : ""}`}
                    onClick={() => patchFilter({ kind: k })}>
                    {t("admin.exp.kind." + lbl)}</button>
                ))}
              </div>
            )}
            {nItems.length === 0 ? (
              <p className="hint" style={{ marginTop: 12 }}>{t("admin.exp.empty")}</p>
            ) : (
              <>
                {facets && (
                  <div className="oa-exp-facets">
                    <span className="oa-exp-facet-src" title={t("admin.exp.facet.esTip")}>
                      {t("admin.exp.facet.es")}</span>
                    {Object.entries(facets.type || {}).sort((a, b) => b[1] - a[1]).slice(0, 8)
                      .map(([k, c]) => (
                        <button key={k}
                          className={`oa-exp-facet ${filter.node_type === k ? "on" : ""}`}
                          onClick={() => patchFilter({ node_type: filter.node_type === k ? "" : k })}>
                          {k}<span className="c">{c}</span></button>))}
                    {Object.entries(facets.trust || {}).sort((a, b) => b[1] - a[1])
                      .map(([k, c]) => (
                        <button key={"t_" + k}
                          className={`oa-exp-facet trust ${filter.trust === k ? "on" : ""}`}
                          onClick={() => patchFilter({ trust: filter.trust === k ? "" : k })}>
                          {k}<span className="c">{c}</span></button>))}
                  </div>
                )}
                <div className="oa-exp-table-wrap">
                  <table>
                    <thead><tr>
                      <th>{t("admin.nodes.th.name")}</th>
                      <th style={{ width: 74 }}>{t("admin.exp.th.kind")}</th>
                      <th>{t("admin.nodes.th.id")}</th>
                      <th>{t("admin.nodes.th.type")}</th>
                      <th style={{ width: 100 }}>{t("admin.nodes.th.trust")}</th>
                      {filter.property && <th>{t("admin.exp.th.value")}</th>}
                      <th style={{ textAlign: "right", width: 60 }}>
                        {searchSource === "es" ? t("admin.exp.th.score") : t("admin.exp.th.rels")}</th>
                    </tr></thead>
                    <tbody>
                      {nItems.map((n) => (
                        <tr key={n.node_id} onClick={() => openNode(n)}
                            className={sel === n.node_id ? "oa-row-sel" : ""}
                            style={{ cursor: "pointer" }}>
                          <td><b>{n.name || n.node_id}</b></td>
                          <td>
                            <span className={`oa-kind ${n.is_class ? "cls" : "inst"}`}>
                              {n.is_class ? t("admin.exp.kind.class") : t("admin.exp.kind.instance")}</span>
                          </td>
                          <td className="mono" style={{ fontSize: "0.72rem" }}>{n.node_id}</td>
                          <td>{n.type || "—"}</td>
                          <td><TrustPill trust={n.trust} /></td>
                          {filter.property && <td className="mono" style={{ fontSize: "0.74rem" }}>{n.prop_value ?? "—"}</td>}
                          <td className="mono" style={{ textAlign: "right" }}>
                            {n.score != null ? n.score.toFixed(2)
                              : (n.out_degree || 0) + (n.in_degree || 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {nTotal > 0 && (
                  <div className="oa-exp-page">
                    <span className="hint" style={{ marginTop: 0 }}>
                      {nCapped
                        ? t("admin.exp.pageCapped", { from, to, total: nTotal.toLocaleString() })
                        : t("admin.exp.page", { from, to, total: nTotal.toLocaleString() })}</span>
                    <button className="ghost" style={{ padding: "4px 12px" }}
                            disabled={listBusy || offset === 0}
                            onClick={() => setOffset(Math.max(0, offset - LIMIT))}>‹</button>
                    <button className="ghost" style={{ padding: "4px 12px" }}
                            disabled={listBusy || offset + LIMIT >= nTotal}
                            onClick={() => setOffset(offset + LIMIT)}>›</button>
                  </div>
                )}
              </>
            )}
            </>
          )}
          </>
          )}
        </section>

        {/* ── 우 pane — 인스펙터 ── */}
        {sel && (
          <Inspector key={sel} base={base} nodeId={sel} readOnly={readOnly}
                     types={types} targetOptions={nItems}
                     crumbs={crumbs} onPivot={(n) => openNode(n, true)} onCrumb={crumbTo}
                     onClose={closeInspector} onChanged={onRejected} onEdited={loadCenter}
                     flash={flash} setError={setError} />
        )}
      </div>

      <datalist id="oa-exp-type-list">{types.map((ty) => <option key={ty} value={ty} />)}</datalist>
    </>
  );

  // 엣지 모드 목록에서 직접 삭제
  async function removeEdgeFromList(e) {
    setError("");
    try {
      const p = new URLSearchParams({ source: e.source, predicate: e.predicate, target: e.target });
      const res = await fetch(`${base}/edges?${p.toString()}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await detailMsg(res));
      flash(t("admin.nodes.edges.removed"));
      await loadCenter();
    } catch (err) { setError(t("admin.nodes.edges.err", { e: err.message || err })); }
  }
}
