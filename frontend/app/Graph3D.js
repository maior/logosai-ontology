"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

// 채도 낮춘 절제 팔레트 (분석 도구 톤). 타입/술어에 데이터 기반 배정 — 하드코딩 없음.
// 선명 팔레트 — ClusterView.CLUSTER_PALETTE·대시보드와 통일(이전 muted 색은 어두웠음)
const NODE_PALETTE = ["#5b9bff", "#34d399", "#fbbf24", "#fb7185", "#a78bfa",
                      "#22d3ee", "#f472b6", "#4ade80", "#fb923c", "#60a5fa",
                      "#c084fc", "#2dd4bf", "#facc15", "#f87171", "#818cf8", "#e879f9"];
const EDGE_PALETTE = ["#5f6f9c", "#4f8a77", "#a5855f", "#936069",
                      "#75699a", "#916a82", "#5f7c98", "#71895f"];

const paletteMap = (keys, palette) => {
  const s = [...new Set(keys)].sort();
  const m = {};
  s.forEach((k, i) => { m[k] = palette[i % palette.length]; });
  return m;
};

/**
 * 3D 그래프 탐색기 — "분석 도구" 무광 톤 (Neo4j Bloom / Linkurious 계열).
 * - 무광 조명 구체(자체발광 최소) + 차분한 배경(별필드/맥동/네온 없음)
 * - 역할(구조 기반, 하드코딩 없음): degree tier 로 크기만 절제 반영
 *     hub(게이트웨이) 큼 / router 중간 / leaf(말단) 작음
 * - 라벨은 맥락적: 허브는 상시, 나머지는 선택 이웃일 때만 (+ 호버 툴팁)
 * - 엣지: 술어별 은은한 색 + 백본(hub 연결) 약간 굵게. 상시 애니메이션 없음.
 * - 선택: 이웃만 남기고 나머지 디밍(재질만 제자리 변경, 카메라 유지)
 */
const Graph3D = forwardRef(function Graph3D(
  { data, selectedId, onSelect, hiddenTypes = [], emptyText }, apiRef) {
  const containerRef = useRef(null);
  const graphRef = useRef(null);
  const THREERef = useRef(null);
  const SpriteTextRef = useRef(null);
  const roRef = useRef(null);
  const tweenRef = useRef(null);       // 선택 강조 tween (선택 시에만 잠깐 돎)
  const selAnimRef = useRef(null);     // 선택 노드 지속 breathe (선택 동안만 돎)
  const colorsRef = useRef({});
  const predColorRef = useRef({});
  const selRef = useRef({ id: null, neighbors: new Set() });
  const onSelectRef = useRef(onSelect);
  const dataRef = useRef(data);
  const hiddenRef = useRef(hiddenTypes);
  onSelectRef.current = onSelect;
  dataRef.current = data;
  hiddenRef.current = hiddenTypes;

  useImperativeHandle(apiRef, () => ({
    zoomIn: () => zoomBy(0.72),
    zoomOut: () => zoomBy(1.4),
    fit: () => graphRef.current?.zoomToFit(700, 90),
  }));

  function zoomBy(factor) {
    const g = graphRef.current;
    if (!g) return;
    const c = g.cameraPosition();
    g.cameraPosition(
      { x: c.x * factor, y: c.y * factor, z: c.z * factor }, undefined, 300);
  }

  const linkEnds = (l) => [l.source.id ?? l.source, l.target.id ?? l.target];
  const touches = (l) => {
    const { id } = selRef.current;
    if (!id) return false;
    const [s, t] = linkEnds(l);
    return s === id || t === id;
  };
  const nodeColorOf = (n) => colorsRef.current[n.type] || "#8390b4";
  const predColorOf = (l) => predColorRef.current[l.predicate] || "#5f6f9c";
  const roleSize = (role) => (role === "hub" ? 6.2 : role === "router" ? 4.4 : 3.4);

  // 무광 구체 + 라벨 1회 생성. 선택 강조는 tweenNodeStyles() 로 재질만 부드럽게 변경.
  function buildNodeObject(n) {
    const THREE = THREERef.current;
    const SpriteText = SpriteTextRef.current;
    const grp = new THREE.Group();
    const r = roleSize(n.__role);

    const mat = new THREE.MeshStandardMaterial({
      metalness: 0.1, roughness: 0.78, transparent: true,
    });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 24), mat);
    grp.add(mesh);

    const label = new SpriteText(n.name || n.id);
    label.textHeight = n.__role === "hub" ? 5.4 : 4.4;
    label.fontWeight = n.__role === "hub" ? "600" : "500";
    label.strokeColor = "#0a0e18";
    label.strokeWidth = 0.6;
    label.backgroundColor = false;
    if (label.material) label.material.depthWrite = false;
    label.position.set(0, r + 5.5, 0);
    grp.add(label);

    const viz = { mat, mesh, label, baseColor: nodeColorOf(n), role: n.__role, cur: null };
    n.__viz = viz;
    setNodeImmediate(n, viz);
    return grp;
  }

  // 선택/이웃 + 역할에 따른 목표 스타일 값 (무광, 디밍). 글로우/맥동 없음.
  function targetFor(n, v) {
    const sel = selRef.current;
    const hasSel = !!sel.id;
    const selected = n.id === sel.id;
    const near = !hasSel || selected || sel.neighbors.has(n.id);
    return {
      color: near ? v.baseColor : "#39435c",           // 비이웃은 회색으로 후퇴
      em: selected ? 0.4 : near ? 0.18 : 0.03,          // 자체발광 — 선명하게(이전 0.08 은 어두웠음)
      op: near ? 1 : 0.2,
      sc: selected ? 1.2 : 1,                           // 선택만 살짝 확대
      labelVisible: hasSel ? near : v.role === "hub",   // 허브 상시 / 선택 시 이웃
      labelColor: selected ? "#ffffff" : near ? "#dbe3f5" : "#8593b0",
    };
  }

  // 즉시 적용 (초기 생성·데이터 변경 — tween 없음)
  function setNodeImmediate(n, v) {
    const THREE = THREERef.current;
    const t = targetFor(n, v);
    v.mat.color.set(t.color);
    v.mat.emissive.set(v.baseColor);
    v.mat.emissiveIntensity = t.em;
    v.mat.opacity = t.op;
    v.mat.needsUpdate = true;
    v.mesh.scale.setScalar(t.sc);
    v.label.visible = t.labelVisible;
    v.label.color = t.labelColor;
    v.cur = { op: t.op, em: t.em, sc: t.sc, color: new THREE.Color(t.color) };
  }

  function applyNodeStyles() {
    const g = graphRef.current;
    if (!g) return;
    g.graphData().nodes.forEach((n) => n.__viz && setNodeImmediate(n, n.__viz));
  }

  // 선택 변경 시 강조를 ~380ms easeOut 로 부드럽게 보간 (하드 스냅 없음)
  function tweenNodeStyles() {
    const THREE = THREERef.current, g = graphRef.current;
    if (!THREE || !g) return;
    const nodes = g.graphData().nodes;
    nodes.forEach((n) => {
      const v = n.__viz;
      if (!v) return;
      const t = targetFor(n, v);
      if (!v.cur) v.cur = { op: t.op, em: t.em, sc: t.sc, color: new THREE.Color(t.color) };
      v.start = { op: v.cur.op, em: v.cur.em, sc: v.cur.sc, color: v.cur.color.clone() };
      v.tgt = { op: t.op, em: t.em, sc: t.sc, color: new THREE.Color(t.color) };
      v.label.visible = t.labelVisible;   // 라벨 토글은 즉시
      v.label.color = t.labelColor;
    });
    const dur = 380, t0 = performance.now();
    if (tweenRef.current) cancelAnimationFrame(tweenRef.current);
    const step = () => {
      const k = Math.min(1, (performance.now() - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);   // easeOutCubic
      nodes.forEach((n) => {
        const v = n.__viz;
        if (!v || !v.tgt) return;
        v.cur.op = v.start.op + (v.tgt.op - v.start.op) * e;
        v.cur.em = v.start.em + (v.tgt.em - v.start.em) * e;
        v.cur.sc = v.start.sc + (v.tgt.sc - v.start.sc) * e;
        v.mat.opacity = v.cur.op;
        v.mat.emissiveIntensity = v.cur.em;
        v.mat.needsUpdate = true;
        v.mesh.scale.setScalar(v.cur.sc);
        v.mat.color.copy(v.start.color).lerp(v.tgt.color, e);
        v.cur.color.copy(v.mat.color);
      });
      if (k < 1) tweenRef.current = requestAnimationFrame(step);
    };
    tweenRef.current = requestAnimationFrame(step);
  }

  // 선택된 "그 노드 하나"에만 지속되는 은은한 breathe (크기+발광 완만한 맥동)
  function startSelAnim() {
    if (selAnimRef.current) cancelAnimationFrame(selAnimRef.current);
    const id = selRef.current.id;
    if (!id) return;
    const loop = () => {
      const g = graphRef.current;
      if (!g || selRef.current.id !== id) return;   // 선택 해제/변경 시 자동 종료
      const n = g.graphData().nodes.find((x) => x.id === id);
      const v = n && n.__viz;
      if (v) {
        const k = 0.5 + 0.5 * Math.sin(performance.now() / 430);
        const sc = 1.16 + 0.14 * k;
        const em = 0.30 + 0.14 * k;
        v.mesh.scale.setScalar(sc); v.cur.sc = sc;
        v.mat.emissiveIntensity = em; v.cur.em = em;
        v.mat.needsUpdate = true;
      }
      selAnimRef.current = requestAnimationFrame(loop);
    };
    selAnimRef.current = requestAnimationFrame(loop);
  }

  // 엣지: 술어색 유지 + 선택 시 이웃 강조/나머지 페이드 (색 속성만, geometry 유지)
  function applyLinkStyles() {
    const g = graphRef.current;
    if (!g) return;
    const hasSel = () => !!selRef.current.id;
    g.linkColor((l) => (hasSel()
                        ? (touches(l) ? predColorOf(l) : "rgba(88,102,140,0.05)")
                        : predColorOf(l)))
     .linkDirectionalArrowColor((l) => (hasSel() && !touches(l)
                        ? "rgba(88,102,140,0.1)" : predColorOf(l)))
     .linkDirectionalParticles((l) => (touches(l) ? 2 : 0));   // 선택 엣지만 흐름
  }

  // 링크 기본 형태(1회): 백본(hub 연결) 약간 굵게. 파티클은 선택 시에만(위).
  function setupLinkBase() {
    const g = graphRef.current;
    if (!g) return;
    g.linkLabel((l) => l.predicate || "")
     .linkWidth((l) => (l.__backbone ? 0.9 : 0.45))
     .linkOpacity(0.6)
     .linkCurvature(0.08)
     .linkDirectionalArrowLength(2.2)
     .linkDirectionalArrowRelPos(1)
     .linkDirectionalParticleSpeed(0.008)
     .linkDirectionalParticleWidth(1.8)
     .linkDirectionalParticleColor((l) => predColorOf(l));
  }

  // ── 초기화 (mount 1회) ──
  useEffect(() => {
    let disposed = false;
    (async () => {
      const [fg, THREE, st] = await Promise.all([
        import("3d-force-graph"), import("three"), import("three-spritetext"),
      ]);
      if (disposed || !containerRef.current) return;
      const ForceGraph3D = fg.default;
      THREERef.current = THREE;
      SpriteTextRef.current = st.default;
      const el = containerRef.current;

      const g = ForceGraph3D()(el)
        .backgroundColor("#0e1420")           // 차분한 짙은 슬레이트
        .showNavInfo(false)
        .width(el.clientWidth || 900)
        .height(el.clientHeight || 600)
        .nodeLabel((n) => n.name || n.id)     // 호버 툴팁(라벨 미표시 노드용)
        .nodeRelSize(4)
        .nodeThreeObjectExtend(false)
        .nodeThreeObject((n) => buildNodeObject(n))
        .onNodeClick((n) => onSelectRef.current?.(n.id))   // 카메라 이동 없음
        .onBackgroundClick(() => onSelectRef.current?.(null));
      graphRef.current = g;

      // 간격 튜닝(1회) — 기본값(many-body ≈ -30, distanceMax=∞)은 노드가 전부
      // 서로 밀어내 넓게 흩어진다. distanceMax 로 반발을 국소화하고 링크 거리를
      // 줄여 적당한 거리에 모이게 한다. 선택과 무관(레이아웃 힘 설정일 뿐).
      g.d3Force("charge").strength(-55).distanceMax(190);
      g.d3Force("link").distance(28);

      setupLinkBase();

      // ── 차분한 원근·입체: 은은한 안개 + 무광 조명 (별필드 없음) ──
      const scene = g.scene();
      scene.fog = new THREE.FogExp2(0x0e1420, 0.0011);
      scene.add(new THREE.HemisphereLight(0xc8d4ee, 0x0a0e18, 0.75));
      const key = new THREE.DirectionalLight(0xffffff, 0.7);
      key.position.set(1, 1.2, 1);
      scene.add(key);
      const fill = new THREE.DirectionalLight(0x99aacc, 0.3);
      fill.position.set(-1, -0.5, -1);
      scene.add(fill);

      applyLinkStyles();
      pushData();

      const ro = new ResizeObserver(() => {
        if (!graphRef.current) return;
        graphRef.current.width(el.clientWidth).height(el.clientHeight);
      });
      ro.observe(el);
      roRef.current = ro;
    })();
    return () => {
      disposed = true;
      if (tweenRef.current) cancelAnimationFrame(tweenRef.current);
      if (selAnimRef.current) cancelAnimationFrame(selAnimRef.current);
      roRef.current?.disconnect();
      graphRef.current?._destructor?.();
      graphRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const hiddenKey = [...hiddenTypes].sort().join(",");

  // ── 데이터/필터 변경 → 갱신 (역할·술어색·백본 재계산) ──
  function pushData() {
    const g = graphRef.current;
    const d = dataRef.current;
    if (!g || !d) return;
    const hidden = new Set(hiddenRef.current);
    const nodes = d.nodes.filter((n) => !hidden.has(n.type)).map((n) => ({ ...n }));
    const visible = new Set(nodes.map((n) => n.id));
    const links = d.links
      .filter((l) => visible.has(l.source) && visible.has(l.target))
      .map((l) => ({ ...l }));
    colorsRef.current = paletteMap(nodes.map((n) => n.type), NODE_PALETTE);
    predColorRef.current = paletteMap(links.map((l) => l.predicate), EDGE_PALETTE);

    // degree → 역할 tier (구조 기반)
    const deg = {};
    links.forEach((l) => {
      deg[l.source] = (deg[l.source] || 0) + 1;
      deg[l.target] = (deg[l.target] || 0) + 1;
    });
    const degrees = nodes.map((n) => deg[n.id] || 0).sort((a, b) => a - b);
    const p90 = degrees.length
      ? degrees[Math.min(degrees.length - 1, Math.floor(0.9 * degrees.length))] : 0;
    const hubT = Math.max(3, p90);
    nodes.forEach((n) => {
      const dd = deg[n.id] || 0;
      n.__role = dd >= hubT ? "hub" : dd <= 1 ? "leaf" : "router";
    });
    links.forEach((l) => {
      l.__backbone = (deg[l.source] || 0) >= hubT || (deg[l.target] || 0) >= hubT;
    });

    const neighbors = {};
    links.forEach((l) => {
      (neighbors[l.source] = neighbors[l.source] || new Set()).add(l.target);
      (neighbors[l.target] = neighbors[l.target] || new Set()).add(l.source);
    });
    selRef.current.neighbors = neighbors[selRef.current.id] || new Set();

    g.graphData({ nodes, links });
    applyNodeStyles();
    applyLinkStyles();
  }

  useEffect(() => {
    pushData();
    if (graphRef.current) setTimeout(() => graphRef.current?.zoomToFit(700, 90), 700);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, hiddenKey]);

  // ── 선택 변경 → 제자리 강조 (카메라·배치 그대로) ──
  useEffect(() => {
    const g = graphRef.current;
    if (!g) return;
    const links = g.graphData().links;
    const near = new Set();
    if (selectedId) {
      links.forEach((l) => {
        const [s, t] = linkEnds(l);
        if (s === selectedId) near.add(t);
        if (t === selectedId) near.add(s);
      });
    }
    selRef.current = { id: selectedId, neighbors: near };
    tweenNodeStyles();   // 부드러운 강조 전환 (하드 스냅 X)
    applyLinkStyles();   // 선택 엣지 파티클 흐름
    if (selectedId) startSelAnim();   // 선택 노드 지속 breathe
    else if (selAnimRef.current) cancelAnimationFrame(selAnimRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative" }}>
      {(!data || !data.nodes?.length) && (
        <div className="empty-state" style={{ position: "absolute", inset: 0, color: "#aab3cc",
             display: "flex", alignItems: "center", justifyContent: "center" }}>
          {emptyText || "No nodes."}
        </div>
      )}
    </div>
  );
});

export default Graph3D;
