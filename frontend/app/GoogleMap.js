"use client";

import { useEffect, useRef } from "react";
import { MarkerClusterer } from "@googlemaps/markerclusterer";

const KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_KEY || "";
const PALETTE = ["#4f46e5", "#059669", "#d97706", "#dc2626",
                 "#7c3aed", "#db2777", "#ea580c", "#0891b2"];

let loaderPromise = null;
// Google Maps JS는 한 번 로드되면 언어 고정 → 첫 로드 언어로 결정.
// region도 함께 줘서 EN=미국(US)·KO=한국(KR) 지도 표기에 맞춘다.
function loadGoogleMaps(lang) {
  if (typeof window === "undefined") return Promise.reject();
  if (window.google?.maps) return Promise.resolve(window.google.maps);
  if (!loaderPromise) {
    const language = lang === "ko" ? "ko" : "en";
    const region = lang === "ko" ? "KR" : "US";
    loaderPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `https://maps.googleapis.com/maps/api/js?key=${KEY}&language=${language}&region=${region}`;
      script.async = true;
      script.onload = () => resolve(window.google.maps);
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }
  return loaderPromise;
}

/**
 * Google Maps 데이터맵.
 * - 카테고리별 색상 마커 + InfoWindow(그래프에서 보기 → onOpenNode)
 * - focusPoint({id,lat,lng}) 변경 시 해당 마커로 팬·줌 + InfoWindow 오픈
 */
export default function GoogleMap({ geo, focusPoint, onOpenNode, lang }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef({});   // node id → {marker, info}
  const infoRef = useRef(null);
  const focusRef = useRef(null);   // 지도 생성 전에 도착한 포커스 보관

  // 포커스 적용: 해당 위치로 팬 + 동네 수준 줌인 + 팝업
  const applyFocus = () => {
    const fp = focusRef.current;
    const maps = typeof window !== "undefined" ? window.google?.maps : null;
    if (!fp || !mapRef.current || !maps) return;
    mapRef.current.panTo({ lat: fp.lat, lng: fp.lng });
    mapRef.current.setZoom(15);
    const entry = markersRef.current[fp.id];
    if (entry) maps.event.trigger(entry.marker, "click");
  };

  // 지도 + 마커 구성
  useEffect(() => {
    if (!geo?.length || !containerRef.current) return;
    let cancelled = false;

    loadGoogleMaps(lang).then((maps) => {
      if (cancelled || !containerRef.current) return;

      const map = new maps.Map(containerRef.current, {
        mapTypeControl: false, streetViewControl: false, fullscreenControl: true,
        styles: [{ featureType: "poi", elementType: "labels", stylers: [{ visibility: "off" }] }],
      });
      mapRef.current = map;
      infoRef.current = new maps.InfoWindow();
      markersRef.current = {};

      const categories = [...new Set(geo.map((p) => p.category || "기타"))];
      const colorOf = (c) => PALETTE[categories.indexOf(c || "기타") % PALETTE.length];

      const bounds = new maps.LatLngBounds();
      const markers = [];
      geo.forEach((point) => {
        const position = { lat: point.lat, lng: point.lng };
        bounds.extend(position);
        const marker = new maps.Marker({
          position, title: point.name,
          icon: {
            path: maps.SymbolPath.CIRCLE, scale: 9,
            fillColor: colorOf(point.category), fillOpacity: 0.92,
            strokeColor: "#ffffff", strokeWeight: 2,
          },
        });
        marker.addListener("click", () => openInfo(maps, point, marker));
        markersRef.current[point.id] = { marker, point };
        markers.push(marker);
      });
      // 수천 마커 밀집 대응 — 줌아웃 시 숫자 클러스터로 묶음
      new MarkerClusterer({ map, markers });
      map.fitBounds(bounds, 48);
      // "지도에서 보기"로 진입한 경우: 지도 준비 직후 대기 포커스 적용
      if (focusRef.current) setTimeout(applyFocus, 350);
    }).catch(() => { /* 로드 실패 시 상위에서 안내 */ });

    function openInfo(maps, point, marker) {
      const el = document.createElement("div");
      el.style.cssText = "font-family:inherit;font-size:13px;line-height:1.6;min-width:180px;";
      el.innerHTML =
        `<b style="font-size:14px">${point.name}</b><br/>` +
        `<span style="color:#667085">${point.type}` +
        (point.category ? ` · ${point.category}` : "") +
        (point.region ? ` · ${point.region}` : "") + `</span><br/>`;
      const btn = document.createElement("button");
      btn.textContent = "그래프에서 보기 →";
      btn.style.cssText =
        "margin-top:7px;padding:4px 12px;border:1px solid #4f46e5;border-radius:6px;" +
        "background:#eef2ff;color:#4f46e5;font-size:12px;font-weight:600;cursor:pointer;";
      btn.onclick = () => onOpenNode?.(point.id);
      el.appendChild(btn);
      infoRef.current.setContent(el);
      infoRef.current.open({ map: mapRef.current, anchor: marker });
    }

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geo]);

  // 노드 상세 → "지도에서 보기" 포커스 (지도 미생성 시 focusRef에 보관 후 생성 시 적용)
  useEffect(() => {
    focusRef.current = focusPoint;
    applyFocus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusPoint]);

  return <div ref={containerRef} className="map-surface" />;
}

export const HAS_GOOGLE_KEY = !!KEY;
