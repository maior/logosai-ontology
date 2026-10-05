"use client";

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import * as d3 from "d3";

// 타입별 색 (등장 순서 고정 배정 — 전문 팔레트)
const PALETTE = ["#4f46e5", "#059669", "#d97706", "#dc2626",
                 "#7c3aed", "#db2777", "#ea580c", "#0891b2"];

export function typeColor(types) {
  const sorted = [...new Set(types)].sort();
  const map = {};
  sorted.forEach((t, i) => { map[t] = PALETTE[i % PALETTE.length]; });
  return map;
}

/**
 * Neo4j 스타일 그래프 탐색기 (d3-force).
 * - hiddenTypes: 체크 해제된 타입은 노드·관계 모두 숨김
 * - 선택 노드: 이웃만 강조, 나머지는 흐리게 (시뮬레이션 재시작 없이 스타일만 갱신)
 * - 드래그=핀 고정, 더블클릭=해제, 휠 줌/팬
 */
const GraphExplorer = forwardRef(function GraphExplorer(
  { data, selectedId, onSelect, hiddenTypes = [], height = 600, emptyText }, apiRef) {
  const svgRef = useRef(null);
  const selRef = useRef({});   // d3 selections + 이웃 맵 (하이라이트용)
  const zoomRef = useRef(null);

  // 줌/전체보기 컨트롤 API — 부모(툴바 버튼)에서 호출
  useImperativeHandle(apiRef, () => ({
    zoomIn: () => {
      if (!zoomRef.current) return;
      d3.select(svgRef.current).transition().duration(200)
        .call(zoomRef.current.scaleBy, 1.4);
    },
    zoomOut: () => {
      if (!zoomRef.current) return;
      d3.select(svgRef.current).transition().duration(200)
        .call(zoomRef.current.scaleBy, 0.7);
    },
    fit: () => {
      // 전체 보기: 현재 노드 배치의 바운딩 박스를 화면에 맞춤
      if (!zoomRef.current || !svgRef.current) return;
      const svg = d3.select(svgRef.current);
      const layer = svg.select("g").node();
      if (!layer) return;
      const bbox = layer.getBBox();
      if (!bbox.width || !bbox.height) return;
      const w = svgRef.current.clientWidth, h = svgRef.current.clientHeight;
      const scale = Math.min(w / (bbox.width + 90), h / (bbox.height + 90), 2.5);
      const tx = w / 2 - scale * (bbox.x + bbox.width / 2);
      const ty = h / 2 - scale * (bbox.y + bbox.height / 2);
      svg.transition().duration(400).call(
        zoomRef.current.transform,
        d3.zoomIdentity.translate(tx, ty).scale(scale));
    },
  }));

  const colors = useMemo(
    () => typeColor((data?.nodes || []).map((n) => n.type)), [data]);

  const hiddenKey = [...hiddenTypes].sort().join(",");

  // ── 구조 변경 시에만 시뮬레이션 재구성 (선택 변경은 아래 효과가 처리) ──
  useEffect(() => {
    if (!data || !svgRef.current) return;
    const hidden = new Set(hiddenTypes);
    const nodes = data.nodes.filter((n) => !hidden.has(n.type)).map((n) => ({ ...n }));
    const visible = new Set(nodes.map((n) => n.id));
    const links = data.links
      .filter((l) => visible.has(l.source) && visible.has(l.target))
      .map((l) => ({ ...l }));

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    if (!nodes.length) return;
    const width = svgRef.current.clientWidth || 900;

    svg.append("defs").append("marker")
      .attr("id", "arrow").attr("viewBox", "0 -4 8 8")
      .attr("refX", 26).attr("refY", 0)
      .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
      .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", "#b8bfca");

    const zoomLayer = svg.append("g");
    const zoomBehavior = d3.zoom().scaleExtent([0.15, 5])
      .on("zoom", (e) => zoomLayer.attr("transform", e.transform));
    svg.call(zoomBehavior);
    zoomRef.current = zoomBehavior;

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(135).strength(0.55))
      .force("charge", d3.forceManyBody().strength(-460))
      .force("collide", d3.forceCollide(36))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = zoomLayer.append("g").selectAll("line").data(links).join("line")
      .attr("stroke", "#cfd6e0").attr("stroke-width", 1.4)
      .attr("marker-end", "url(#arrow)");

    const edgeLabel = zoomLayer.append("g").selectAll("text").data(links).join("text")
      .text((d) => d.predicate)
      .attr("font-size", 9).attr("fill", "#8a63d2").attr("text-anchor", "middle")
      .attr("font-family", "ui-monospace, monospace");

    const node = zoomLayer.append("g").selectAll("g").data(nodes).join("g")
      .style("cursor", "pointer");

    node.append("circle")
      .attr("r", 20)
      .attr("fill", (d) => colors[d.type] || "#98a2b3")
      .attr("stroke", "#fff").attr("stroke-width", 2)
      .style("filter", "drop-shadow(0 1px 2px rgba(16,24,40,0.25))");

    node.append("text")
      .text((d) => (d.name || d.id).slice(0, 4))
      .attr("text-anchor", "middle").attr("dy", 4)
      .attr("font-size", 10).attr("font-weight", 700).attr("fill", "#fff")
      .style("pointer-events", "none");

    node.append("text")
      .text((d) => (d.name || d.id).length > 16
        ? (d.name || d.id).slice(0, 15) + "…" : (d.name || d.id))
      .attr("text-anchor", "middle").attr("dy", 36)
      .attr("font-size", 10.5).attr("font-weight", 550).attr("fill", "#344054")
      .style("pointer-events", "none");

    node.append("title").text((d) =>
      `${d.type}: ${d.name}\n출처: ${d.source || "—"}`);

    node.on("click", (event, d) => { event.stopPropagation(); onSelect?.(d.id); });
    node.on("dblclick", (event, d) => {
      event.stopPropagation();
      d.fx = null; d.fy = null;
      simulation.alpha(0.3).restart();
    });
    node.call(d3.drag()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event) => { if (!event.active) simulation.alphaTarget(0); }));

    svg.on("click", () => onSelect?.(null));

    simulation.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      edgeLabel.attr("x", (d) => (d.source.x + d.target.x) / 2)
               .attr("y", (d) => (d.source.y + d.target.y) / 2 - 4);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    // 이웃 맵 (선택 하이라이트용)
    const neighbors = {};
    links.forEach((l) => {
      const s = l.source.id || l.source, t = l.target.id || l.target;
      (neighbors[s] = neighbors[s] || new Set()).add(t);
      (neighbors[t] = neighbors[t] || new Set()).add(s);
    });
    selRef.current = { node, link, edgeLabel, neighbors };

    return () => simulation.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, colors, hiddenKey, height]);

  // ── 선택 하이라이트: 시뮬레이션 재시작 없이 스타일만 (레이아웃 유지) ──
  useEffect(() => {
    const { node, link, edgeLabel, neighbors } = selRef.current;
    if (!node) return;
    if (!selectedId) {
      node.style("opacity", 1);
      node.select("circle").attr("stroke", "#fff").attr("stroke-width", 2);
      link.style("opacity", 1);
      edgeLabel.style("opacity", 1);
      return;
    }
    const near = neighbors[selectedId] || new Set();
    const isNear = (id) => id === selectedId || near.has(id);
    node.style("opacity", (d) => (isNear(d.id) ? 1 : 0.14));
    node.select("circle")
      .attr("stroke", (d) => (d.id === selectedId ? "#101828" : "#fff"))
      .attr("stroke-width", (d) => (d.id === selectedId ? 3.5 : 2));
    const touches = (l) => {
      const s = l.source.id || l.source, t = l.target.id || l.target;
      return s === selectedId || t === selectedId;
    };
    link.style("opacity", (l) => (touches(l) ? 1 : 0.08))
        .attr("stroke", (l) => (touches(l) ? "#8a63d2" : "#cfd6e0"));
    edgeLabel.style("opacity", (l) => (touches(l) ? 1 : 0.06));
  }, [selectedId, data, hiddenKey]);

  if (!data || !data.nodes.length) {
    return <div className="empty-state">{emptyText || "No nodes."}</div>;
  }
  return (
    <svg ref={svgRef} style={{ width: "100%", height: "100%", display: "block" }} />
  );
});

export default GraphExplorer;
