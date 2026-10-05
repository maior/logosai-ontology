"use client";

import { useEffect, useRef } from "react";
import "leaflet/dist/leaflet.css";

// 카테고리별 마커 색 (검증된 라이트 팔레트)
const CATEGORY_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e34948",
                         "#4a3aa7", "#e87ba4", "#eb6834", "#008300"];

export default function DataMap({ geo }) {
  const mapRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!geo || !geo.length || !containerRef.current) return;

    let cancelled = false;
    // Leaflet은 window에 의존 — 클라이언트에서만 동적 로드
    import("leaflet").then((L) => {
      if (cancelled) return;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }

      const map = L.map(containerRef.current, { scrollWheelZoom: false });
      mapRef.current = map;
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>',
        maxZoom: 18,
      }).addTo(map);

      const categories = [...new Set(geo.map((p) => p.category || "기타"))];
      const colorOf = (category) =>
        CATEGORY_COLORS[categories.indexOf(category || "기타") % CATEGORY_COLORS.length];

      const bounds = [];
      geo.forEach((point) => {
        bounds.push([point.lat, point.lng]);
        L.circleMarker([point.lat, point.lng], {
          radius: 9,
          color: "#2b2822",
          weight: 1.5,
          fillColor: colorOf(point.category),
          fillOpacity: 0.85,
        })
          .addTo(map)
          .bindPopup(
            `<b>${point.name}</b><br/>${point.type}` +
            (point.category ? ` · ${point.category}` : "") +
            (point.region ? `<br/>지역: ${point.region}` : ""));
      });
      map.fitBounds(bounds, { padding: [30, 30] });

      // 범례
      const legend = L.control({ position: "bottomright" });
      legend.onAdd = () => {
        const div = L.DomUtil.create("div");
        div.style.cssText =
          "background:#fdfcf7;border:1.5px solid #3a372e;border-radius:3px;" +
          "padding:8px 12px;font-size:12px;line-height:1.9;";
        div.innerHTML = categories.map((category) =>
          `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;` +
          `background:${colorOf(category)};margin-right:6px;"></span>${category}`
        ).join("<br/>");
        return div;
      };
      legend.addTo(map);
    });

    return () => {
      cancelled = true;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [geo]);

  if (!geo || !geo.length) return null;
  return (
    <div ref={containerRef}
         style={{ height: 440, marginTop: 14, border: "1.5px solid #3a372e",
                  borderRadius: 3, zIndex: 0 }} />
  );
}
