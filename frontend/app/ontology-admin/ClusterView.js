"use client";

// 2D 온톨로지 그래프 (v5 — Neo4j 스타일 라이브 force 시뮬레이션).
//   · 각 노드 = 원 + 자기 라벨(이름). 라벨이 노드에 붙어 **함께** 움직인다.
//   · d3-force(link·charge·collide)로 자연스럽게 배치·정착, 드래그하면 연결
//     노드가 물리로 반응. 배경 드래그/휠 = 전체 팬·줌. 클릭 = 인스펙터.
//   · 색 = 타입. 정적 클러스터(v3/v4)가 아니라 살아있는 그래프.
// d3 가 svg 를 직접 그린다(React 는 컨테이너·범례). 재구성은 데이터 변동 시만.

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import {
  select, zoom, drag, pointer,
  forceSimulation, forceLink, forceManyBody, forceCollide, forceCenter, forceX, forceY,
} from "d3";

export const CLUSTER_PALETTE = [
  "#5b9bff", "#34d399", "#fbbf24", "#fb7185", "#a78bfa", "#22d3ee",
  "#f472b6", "#4ade80", "#fb923c", "#60a5fa", "#c084fc", "#2dd4bf",
  "#facc15", "#f87171", "#818cf8", "#e879f9"];
const OTHER_COLOR = "#9aa4c0";
const W = 900, H = 620;

const ClusterView = forwardRef(function ClusterView(
  { nodes, links, onSelect, selectedId, t }, ref) {
  const svgRef = useRef(null);
  const zoomRef = useRef(null);   // d3 zoom behavior — 버튼 zoom in/out 용
  const selRef = useRef(onSelect);
  selRef.current = onSelect;

  // 부모(툴바)가 호출하는 zoom in/out — d3 zoom 을 프로그램적으로 scaleBy.
  useImperativeHandle(ref, () => ({
    zoomIn: () => { const el = svgRef.current, z = zoomRef.current;
      if (el && z) select(el).call(z.scaleBy, 1.3); },
    zoomOut: () => { const el = svgRef.current, z = zoomRef.current;
      if (el && z) select(el).call(z.scaleBy, 1 / 1.3); },
  }), []);

  const model = useMemo(() => {
    const items = (nodes || []).map((n) => ({ id: n.id, name: n.name || n.id,
      type: n.type || "기타", trust: n.trust || "unset" }));
    if (!items.length) return null;
    // 색: 타입 빈도 순위로 배정(상위=선명색, 그 외=회색 톤)
    const cnt = {}; items.forEach((n) => { cnt[n.type] = (cnt[n.type] || 0) + 1; });
    const ranked = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a]);
    const colorOf = (ty) => {
      const i = ranked.indexOf(ty);
      return i < CLUSTER_PALETTE.length ? CLUSTER_PALETTE[i] : OTHER_COLOR;
    };
    const idset = new Set(items.map((n) => n.id));
    const rawLinks = (links || []).filter((l) => idset.has(l.source) && idset.has(l.target));
    const deg = {};
    rawLinks.forEach((l) => { deg[l.source] = (deg[l.source] || 0) + 1; deg[l.target] = (deg[l.target] || 0) + 1; });
    // 시뮬은 배열을 변형(x/y 부여)하므로 매번 새 객체
    const simNodes = items.map((n) => ({ ...n, r: 4 + Math.sqrt(deg[n.id] || 0) * 1.7,
      color: colorOf(n.type) }));
    const simLinks = rawLinks.map((l) => ({ source: l.source, target: l.target }));
    // 범례용 상위 타입
    const legendTypes = ranked.slice(0, 14);
    return { simNodes, simLinks, colorOf, legendTypes, moreTypes: Math.max(0, ranked.length - 14) };
  }, [nodes, links]);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl || !model) return;
    const { simNodes, simLinks } = model;
    const svg = select(svgEl);
    svg.selectAll("*").remove();
    const root = svg.append("g");

    const link = root.append("g").attr("stroke", "#8fa0c8").attr("stroke-opacity", 0.28)
      .selectAll("line").data(simLinks).join("line").attr("stroke-width", 1);

    const node = root.append("g").selectAll("g.oa-node").data(simNodes, (d) => d.id)
      .join("g").attr("class", "oa-node");
    node.append("circle").attr("r", (d) => d.r).attr("fill", (d) => d.color)
      .attr("fill-opacity", 0.95).attr("stroke", "#fff").attr("stroke-width", 0.7)
      .attr("stroke-opacity", 0.5);
    node.append("text").attr("class", "oa-node-label").attr("text-anchor", "middle")
      .attr("dy", (d) => -d.r - 4).text((d) => d.name);
    node.append("title").text((d) => `${d.name} · ${d.type}${d.trust !== "unset" ? " · " + d.trust : ""}`);

    const sim = forceSimulation(simNodes)
      .force("link", forceLink(simLinks).id((d) => d.id).distance(46).strength(0.35))
      .force("charge", forceManyBody().strength(-150))
      .force("collide", forceCollide((d) => d.r + 9))
      .force("center", forceCenter(W / 2, H / 2))
      .force("x", forceX(W / 2).strength(0.03))
      .force("y", forceY(H / 2).strength(0.03))
      .on("tick", () => {
        link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
            .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
        node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });

    // 클릭 = 선택, 드래그 = 이동. 핵심: d3-drag 는 1px 만 움직여도 "drag" 를
    // 쏘므로(마우스 클릭의 미세 떨림 포함), 시작점 대비 5px 넘게 움직였을 때만
    // "이동"으로 간주한다. 그 전까지는 시뮬을 재가열하지 않아 노드가 제자리에
    // 머문다 → 정착된 그래프에서 클릭 선택이 흔들림 없이 잡힌다.
    let sx = 0, sy = 0, moved = false;
    node.style("cursor", "pointer").call(drag()
      .clickDistance(6)
      .on("start", (e, d) => { moved = false; sx = e.x; sy = e.y;
        if (e.sourceEvent) e.sourceEvent.stopPropagation(); })
      .on("drag", (e, d) => {
        if (!moved) {
          if (Math.hypot(e.x - sx, e.y - sy) < 5) return;   // 미세 이동 무시
          moved = true; if (!e.active) sim.alphaTarget(0.15).restart();
        }
        const [mx, my] = pointer(e.sourceEvent, root.node());
        d.fx = mx; d.fy = my; })
      .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null;
        if (!moved && selRef.current) selRef.current({ id: d.id, name: d.name, type: d.type }); }));

    const z = zoom().scaleExtent([0.2, 4]).on("zoom", (e) => root.attr("transform", e.transform));
    svg.call(z);
    zoomRef.current = z;   // 툴바 버튼 zoom in/out 에서 사용

    // 빈 공간 클릭 = 선택 해제. 노드는 <circle>/<text>(자식)가 타깃이라 여기 안
    // 걸리고, 배경만 svg 자신이 타깃이 된다. 재가열 없음(선택 상태만 바뀜).
    svg.on("click", (e) => { if (e.target === svgEl && selRef.current) selRef.current(null); });

    return () => { sim.stop(); svg.on(".zoom", null); svg.on("click", null); };
  }, [model]);

  // 선택 스포트라이트 (재구성 없이 스타일만) — 선택 노드+직접 이웃은 밝게,
  // 나머지는 흐리게(사라지지 않는다). 선택 없으면 전부 기본 상태로 복귀.
  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl || !model) return;
    const svg = select(svgEl);
    const eid = (v) => (v && v.id != null ? v.id : v);

    // 선택 노드의 직접 이웃 집합
    let near = null;
    if (selectedId) {
      near = new Set([selectedId]);
      model.simLinks.forEach((l) => {
        const s = eid(l.source), t = eid(l.target);
        if (s === selectedId) near.add(t);
        if (t === selectedId) near.add(s);
      });
    }
    const faded = (id) => selectedId != null && near && !near.has(id);

    const nodes = svg.selectAll("g.oa-node");
    nodes.select("circle")
      .attr("stroke-width", (d) => (d && d.id === selectedId ? 3 : 0.7))
      .attr("stroke-opacity", (d) => (d && faded(d.id) ? 0.25 : (d && d.id === selectedId ? 1 : 0.5)))
      .attr("fill-opacity", (d) => (d && faded(d.id) ? 0.12 : 0.95));
    nodes.select("text.oa-node-label")
      .attr("opacity", (d) => (d && faded(d.id) ? 0.1 : 1))
      .attr("font-weight", (d) => (d && d.id === selectedId ? 700 : null));
    nodes.classed("sel", (d) => d && d.id === selectedId);
    nodes.filter((d) => d && d.id === selectedId).raise();

    svg.selectAll("line")
      .attr("stroke-opacity", (l) => {
        if (selectedId == null) return 0.28;
        return eid(l.source) === selectedId || eid(l.target) === selectedId ? 0.75 : 0.05;
      })
      .attr("stroke-width", (l) =>
        selectedId != null && (eid(l.source) === selectedId || eid(l.target) === selectedId) ? 1.8 : 1);
  }, [selectedId, model]);

  if (!model) {
    return <div className="empty-state" style={{ height: "100%", border: "none" }}>
      {t("admin.exp.graph.anchorHint")}</div>;
  }
  return (
    <div className="oa-cluster">
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="oa-cluster-svg" role="img"
           preserveAspectRatio="xMidYMid meet" />
      <div className="oa-cluster-legend">
        {model.legendTypes.map((ty) => (
          <span key={ty}><span className="dot" style={{ background: model.colorOf(ty) }} />{ty}</span>
        ))}
        {model.moreTypes > 0 && <span className="oa-cluster-more">+{model.moreTypes}</span>}
      </div>
    </div>
  );
});

export default ClusterView;
