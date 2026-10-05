"use client";

// 커버리지 지도 v3 — Overview(미니맵) → Zoom(창) → Detail(절 밴드 + 원문).
//
// v2 의 실패(사용자 지적): 절 밴드를 문서 전체만큼 세로로 쌓았다 — 51 밴드도
// 목록이지 지도가 아니고, 청크 1만이면 DOM 1만으로 죽는다. 시각화의 표준
// 처방은 Shneiderman 의 "Overview first, zoom and filter, details on demand":
//
//   ① 미니맵 밀도 스트립 — 문서 전체를 고정 폭 한 줄로 (binDensity, O(픽셀)).
//     청크가 몇 개든 화면 비용이 같다. 여기가 "지도"다.
//   ② 창(window) — 미니맵을 클릭한 구간 80칸만 절 밴드로 편다 (windowSlice).
//     DOM 은 항상 창 크기. 기본 창 = 가장 큰 빈 구간 (열자마자 문제 지점).
//   ③ 상세 — 칸 클릭 = 원문 + 연결 노드 (기존).
//
// 문서 선택기는 "빈 칸 많은 순" 정렬 — 관리자의 우선순위 그 자체다. 문서가
// 많아지면 행이 늘 뿐이고 8개를 넘으면 이름 필터가 나타난다.
//
// 원칙 유지: 결정적 레이아웃 · 색 = 측정값만 · 다크 토큰 위에서 그린다
// (.oa 스코프 — 하드코딩 라이트 배경 금지, 가독성 감사 실측 1.8:1 사고).

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  buildCoverageDocs, linkBucket, sectionRegions, coalesceRegions,
  unlinkedRuns, runOrders, binDensity, windowSlice,
} from "./coverageView.mjs";

const WINDOW = 80;               // 상세 창 크기 (칸) — DOM 상한이자 한 화면 분량
const CELL = 14, GAP = 3;
const BUCKET_FILL = ["#f0b48c", "#3f8f6f", "#1baf7a"];
const BUCKET_LABEL = ["미연결 0", "취약 1~2", "튼튼 3+"];
const INK = "var(--ink)";
const INK_SOFT = "var(--ink-2)";
const WARN = "#f2946a";
const LINE = "rgba(255,255,255,.08)";

/* bin 커버리지 → 미니맵 색. 0(빈 땅)이 가장 강하게 — 이 화면의 사냥감. */
function binColor(cov) {
  if (cov <= 0) return "#eb6834";
  if (cov < 0.5) return "#b98a5e";
  if (cov < 1) return "#3f8f6f";
  return "#1baf7a";
}

/* ① 미니맵 — 문서 전체가 한 줄. 클릭하면 창이 그 구간으로 간다. */
function Minimap({ bins, total, win, onJump }) {
  const W = 960, H = 34, TOP = 4;
  if (!bins.length) return null;
  const bw = W / bins.length;
  // 창 오버레이 좌표 (order → x 비례)
  const x1 = ((win.start - 1) / total) * W;
  const x2 = (Math.min(total, win.start + WINDOW - 1) / total) * W;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}
         role="img" aria-label="문서 전체 커버리지 미니맵">
      {bins.map((b, i) => (
        <rect key={i} x={i * bw} y={TOP} width={Math.max(bw - 0.4, 0.8)} height={H - TOP * 2}
              fill={binColor(b.coverage)} style={{ cursor: "pointer" }}
              onClick={() => onJump(b.startOrder)}>
          <title>{`#${b.startOrder}–${b.endOrder} · 연결 ${b.linked}/${b.count}` +
                  (b.sections.length ? `\n${b.sections.join(" · ")}` : "")}</title>
        </rect>
      ))}
      {/* 현재 창 — 지도 위의 '보고 있는 곳' */}
      <rect x={x1} y={1} width={Math.max(x2 - x1, 3)} height={H - 2} rx={3}
            fill="rgba(255,255,255,.14)" stroke="#fff" strokeWidth="1.2"
            pointerEvents="none" />
    </svg>
  );
}

/* 문서 선택기 한 줄 — 이름 + 미니 커버리지 바 + "N칸 빔". */
function DocRow({ doc, active, onClick }) {
  const pct = Math.round(doc.coverage * 100);
  return (
    <button onClick={onClick}
      style={{
        display: "grid", gridTemplateColumns: "minmax(180px,1fr) 160px 130px", gap: 12,
        alignItems: "center", width: "100%", textAlign: "left", padding: "8px 12px",
        borderRadius: 8, cursor: "pointer",
        border: active ? "2px solid #2a78d6" : `1px solid ${LINE}`,
        background: active ? "rgba(42,120,214,.14)" : "var(--surface)",
      }}>
      <span style={{ color: INK, fontWeight: active ? 700 : 500, fontSize: 13,
                     overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
            title={doc.source}>📄 {doc.source}</span>
      <span style={{ display: "block", height: 10, borderRadius: 5,
                     background: "rgba(240,180,140,.4)", overflow: "hidden" }}
            title={`연결 ${doc.linked} / ${doc.total}`}>
        <span style={{ display: "block", height: "100%", width: `${pct}%`,
                       background: "#1baf7a" }} />
      </span>
      <span style={{ color: INK_SOFT, fontVariantNumeric: "tabular-nums", fontSize: 12.5 }}>
        {pct}% · {doc.total - doc.linked}칸 빔
      </span>
    </button>
  );
}

/* ② 창 안의 절 지역 밴드 */
function Region({ region, highlight, selected, onSelect }) {
  const empty = region.linked === 0;
  const full = region.count > 0 && region.linked === region.count;
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "230px 1fr", gap: 10,
      padding: "7px 10px", alignItems: "start",
      background: empty ? "var(--red-soft)" : full ? "var(--green-soft)" : "transparent",
      borderLeft: `4px solid ${empty ? "#eb6834" : full ? "#1baf7a" : "rgba(255,255,255,.15)"}`,
      borderBottom: `1px solid ${LINE}`,
    }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ color: INK, fontSize: 12.5, fontWeight: 600,
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
             title={region.section || "(절 제목 없음)"}>
          {region.section || <i style={{ color: INK_SOFT }}>(절 제목 없음)</i>}
          {region.extraSections > 0 && (
            <span style={{ color: INK_SOFT, fontWeight: 400 }}> 외 {region.extraSections}절</span>
          )}
        </div>
        <div style={{ color: empty ? WARN : INK_SOFT, fontSize: 11.5,
                      fontVariantNumeric: "tabular-nums" }}>
          {region.linked}/{region.count} 연결{empty ? " — 빈 지역" : ""}
        </div>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: GAP }}>
        {region.chunks.map((c) => {
          const hot = highlight.has(c.order);
          const isSel = selected === c.chunkId;
          return (
            <span key={c.chunkId || c.order}
              onClick={() => onSelect(isSel ? null : c.chunkId)}
              title={`#${c.order}${c.section ? ` · §${c.section}` : ""} · 근거 ${c.links}개\n${c.textHead}`}
              style={{
                width: CELL, height: CELL, borderRadius: 3, cursor: "pointer",
                background: hot ? "#eb6834" : BUCKET_FILL[linkBucket(c.links)],
                outline: isSel ? "2px solid #fff" : "none",
                outlineOffset: 1, display: "inline-block",
              }} />
          );
        })}
      </div>
    </div>
  );
}

export default function CoverageTab({ t, api, namespace }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [docKey, setDocKey] = useState(null);
  const [docFilter, setDocFilter] = useState("");
  const [winStart, setWinStart] = useState(null);  // null = 기본(가장 큰 빈 구간)
  const [hotRun, setHotRun] = useState(null);
  const [sel, setSel] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    if (!namespace) return;
    let alive = true;
    setBusy(true); setErr(""); setData(null);
    setDocKey(null); setDocFilter(""); setWinStart(null);
    setHotRun(null); setSel(null); setDetail(null);
    fetch(`${api}/graphs/${encodeURIComponent(namespace)}/coverage-map`)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.detail || `HTTP ${r.status}`);
        return r.json();
      })
      .then((j) => { if (alive) setData(j); })
      .catch((e) => { if (alive) setErr(String(e?.message || e)); })
      .finally(() => { if (alive) setBusy(false); });
    return () => { alive = false; };
  }, [api, namespace]);

  // 문서는 "빈 칸 많은 순" — 관리자의 작업 우선순위. 이름 필터는 8개 초과 시.
  const docs = useMemo(() => {
    const list = (data ? buildCoverageDocs(data) : [])
      .sort((a, b) => (b.total - b.linked) - (a.total - a.linked) ||
                      a.source.localeCompare(b.source));
    const q = docFilter.trim().toLowerCase();
    return q ? list.filter((d) => d.source.toLowerCase().includes(q)) : list;
  }, [data, docFilter]);

  const doc = useMemo(() => {
    if (docs.length === 0) return null;
    return docs.find((d) => d.source === docKey) || docs[0];
  }, [docs, docKey]);

  const bins = useMemo(() => (doc ? binDensity(doc.chunks) : []), [doc]);
  const runs = useMemo(() => (doc ? unlinkedRuns(doc.chunks) : []), [doc]);

  // 창 시작: 명시 이동 > 기본(가장 큰 빈 구간 앞 8칸 — 문제 지점에서 연다)
  const clampStart = useCallback((s, total) =>
    Math.max(1, Math.min(Number(s) || 1, Math.max(1, total - WINDOW + 1))), []);
  const win = useMemo(() => {
    if (!doc) return { start: 1 };
    const base = winStart != null ? winStart
      : (runs[0] ? runs[0].start - 8 : 1);
    return { start: clampStart(base, doc.total) };
  }, [doc, winStart, runs, clampStart]);

  const windowChunks = useMemo(
    () => (doc ? windowSlice(doc.chunks, win.start, WINDOW) : []), [doc, win]);
  const regions = useMemo(
    () => coalesceRegions(sectionRegions(windowChunks)), [windowChunks]);
  const highlight = useMemo(() => (hotRun ? runOrders(hotRun) : new Set()), [hotRun]);

  const jump = useCallback((order) => {
    setWinStart(clampStart((Number(order) || 1) - Math.floor(WINDOW / 4),
                           doc ? doc.total : 1));
    setSel(null); setDetail(null);
  }, [doc, clampStart]);

  const pick = useCallback((chunkId) => {
    setSel(chunkId);
    setDetail(null);
    if (!chunkId) return;
    fetch(`${api}/graphs/${encodeURIComponent(namespace)}/chunks/${encodeURIComponent(chunkId)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [api, namespace]);

  const selChunk = sel && doc ? doc.chunks.find((c) => c.chunkId === sel) : null;
  const winEnd = doc ? Math.min(doc.total, win.start + WINDOW - 1) : 0;

  return (
    <>
      <section className="card">
        <h2>커버리지 지도 — 비어 있는 구간이 어느 절인가</h2>
        <p style={{ marginTop: 2, color: INK_SOFT, fontSize: 13, lineHeight: 1.55 }}>
          위 한 줄이 <b style={{ color: INK }}>문서 전체</b>(미니맵 — 청크가 몇 만
          개라도 이 한 줄이다), <b style={{ color: WARN }}>주황이 근거 없는 빈 땅</b>.
          미니맵을 클릭하면 아래 상세 창이 그 구간으로 가고, 상세의 칸을 클릭하면
          원문이 열린다. 열릴 때 창은 <b style={{ color: INK }}>가장 큰 빈 구간</b>에
          이미 가 있다.
        </p>
        <div style={{ display: "flex", gap: 14, marginTop: 8, flexWrap: "wrap" }}>
          {BUCKET_FILL.map((c, i) => (
            <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 12, height: 12, borderRadius: 3, background: c,
                             display: "inline-block" }} />
              <span style={{ color: INK_SOFT, fontSize: 12 }}>{BUCKET_LABEL[i]}</span>
            </span>
          ))}
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 12, height: 12, borderRadius: 3, background: "#eb6834",
                           display: "inline-block" }} />
            <span style={{ color: INK_SOFT, fontSize: 12 }}>빈 땅 / 선택 구간</span>
          </span>
        </div>

        {busy && <p className="hint" style={{ marginTop: 10 }}>불러오는 중…</p>}
        {err && <p className="hint" style={{ marginTop: 10 }}>⚠ {err}</p>}
        {!busy && !err && docs.length === 0 && !docFilter && (
          <p className="hint" style={{ marginTop: 10 }}>
            이 네임스페이스에 원문 청크가 없다 — 인제스트 탭에서 문서를 올리면 여기가 채워진다.
          </p>
        )}

        {(docs.length > 8 || docFilter) && (
          <input value={docFilter} onChange={(e) => setDocFilter(e.target.value)}
                 placeholder="문서 이름 필터…"
                 style={{ marginTop: 10, width: "100%", padding: "6px 10px",
                          borderRadius: 6, border: `1px solid ${LINE}`,
                          background: "var(--surface)", color: INK }} />
        )}
        {docs.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 12 }}>
            {docs.map((d) => (
              <DocRow key={d.source} doc={d} active={doc && d.source === doc.source}
                      onClick={() => { setDocKey(d.source); setWinStart(null);
                                       setHotRun(null); pick(null); }} />
            ))}
          </div>
        )}
      </section>

      {doc && (
        <section className="card">
          <h2 style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span>🗺 {doc.source}</span>
            <span className="badge">{Math.round(doc.coverage * 100)}%</span>
            <span style={{ color: INK_SOFT, fontWeight: 400, fontSize: 13 }}>
              청크 {doc.total} · 미연결 {doc.total - doc.linked}
            </span>
          </h2>
          {!doc.ordered && (
            <p style={{ color: WARN, fontSize: 12.5 }}>⚠ 이 문서는 문자 오프셋이 없어
              칸 순서가 원문 순서라고 보장되지 않는다 — 구간 판독에 주의.</p>
          )}

          {/* ① 미니맵 — 문서 전체 */}
          <Minimap bins={bins} total={doc.total} win={win} onJump={jump} />

          {/* 빈 구간 랭킹 — 클릭 = 창 이동 + 하이라이트 */}
          {runs.length > 0 && (
            <div style={{ margin: "8px 0 10px" }}>
              <span style={{ color: INK_SOFT, fontSize: 12.5, marginRight: 8 }}>
                가장 큰 빈 구간:
              </span>
              {runs.map((r) => {
                const on = hotRun && hotRun.start === r.start;
                return (
                  <button key={r.start} className="oa-rt-more"
                          style={{ marginRight: 6,
                                   ...(on ? { borderColor: "#eb6834", color: WARN } : {}) }}
                          onClick={() => {
                            setHotRun(on ? null : { start: r.start, count: r.count });
                            jump(r.start);
                          }}>
                    {r.sections[0] ? `${r.sections[0]} ` : ""}{r.count}칸
                  </button>
                );
              })}
            </div>
          )}

          {/* ② 창 도구줄 + 절 밴드 (창 안만 그린다) */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "4px 0 6px" }}>
            <button className="oa-rt-more" disabled={win.start <= 1}
                    onClick={() => jump(win.start - WINDOW + Math.floor(WINDOW / 4))}>◀ 이전</button>
            <span style={{ color: INK_SOFT, fontSize: 12.5, fontVariantNumeric: "tabular-nums" }}>
              #{win.start}–#{winEnd} 표시 중 (전체 {doc.total}칸)
            </span>
            <button className="oa-rt-more" disabled={winEnd >= doc.total}
                    onClick={() => jump(winEnd + 1 + Math.floor(WINDOW / 4))}>다음 ▶</button>
          </div>
          <div style={{ border: `1px solid ${LINE}`, borderRadius: 8, overflow: "hidden" }}>
            {regions.map((r, i) => (
              <Region key={`${r.section}-${r.start}-${i}`} region={r}
                      highlight={highlight} selected={sel} onSelect={pick} />
            ))}
          </div>

          {/* ③ 원문 상세 */}
          {selChunk && (
            <div style={{ marginTop: 10, padding: "12px 14px", borderRadius: 8,
                          border: `1.5px solid ${selChunk.links > 0 ? "#1baf7a" : "#eb6834"}`,
                          background: "var(--surface)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                <span className="mono" style={{ fontWeight: 700, color: INK }}>#{selChunk.order}</span>
                {selChunk.section && <span style={{ color: INK, fontWeight: 600 }}>§{selChunk.section}</span>}
                <span style={{ color: selChunk.links > 0 ? "#4cc9a4" : WARN, fontSize: 12.5 }}>
                  근거 링크 {selChunk.links}개</span>
                <button className="oa-rt-more" style={{ marginLeft: "auto" }}
                        onClick={() => pick(null)}>닫기</button>
              </div>
              {selChunk.nodeIds.length > 0 && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
                  {selChunk.nodeIds.map((n) => (
                    <span key={n} className="mono badge" title={n}
                          style={{ fontSize: 11.5 }}>{n}</span>
                  ))}
                </div>
              )}
              <p style={{ marginTop: 10, whiteSpace: "pre-wrap", color: INK,
                          fontSize: 13.5, lineHeight: 1.65 }}>
                {(detail && (detail.text || detail.chunk?.text)) || selChunk.textHead || "—"}
              </p>
              {selChunk.links === 0 && (
                <p style={{ marginTop: 8, color: WARN, fontSize: 12.5 }}>
                  이 청크는 어떤 노드의 근거도 아니다 — 개체가 실제로 없는 형식적
                  텍스트인지, 추출이 놓친 것인지 위 원문으로 판단하라. 놓친 것이면
                  검수 · 커버리지 검사(review/coverage)가 회복 경로다.
                </p>
              )}
            </div>
          )}
        </section>
      )}
    </>
  );
}
