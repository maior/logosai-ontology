"use client";

// 실험 탭 — "어떤 조합(채널×knob)이 최고 효율인가"를 파레토로 본다.
//
// 효율 = 품질(MRR) × 비용(질의 지연 p50)의 **파레토** — 단일 점수로 합치지
// 않는다(가중치를 지어내는 것). 백엔드: POST /experiments/run (Tier 0 러너,
// LLM 0콜) · GET /experiments (레코드, 최신 먼저) · GET /retrieval-config
// (현재 운영 설정 → ◆ 마커).
//
// 이 화면이 정직해야 하는 세 지점:
//   · 그래프 지문이 최신과 다른 레코드는 흐리게 — knob 결론이 그래프 상태에
//     두 번 뒤집힌 역사. 낡은 점을 지금의 점처럼 보여주면 그 역사를 반복한다.
//   · ⚠ small_sample — 1건=0.02 인 자로 knob 을 확정하지 않는다.
//   · measured=false·null 지표는 점이 되지 못한다 (뷰모델이 배제하고 수를 센다).

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  paretoView, comboTable, fingerprintGroups, latestGraphHash, currentMarker,
} from "./experimentView.mjs";
import { pctLabel } from "./goldenView.mjs";

// 채널 색 — 안내문("vector = 임베딩 단독 / graph = +온톨로지 / graph+prop =
// +확산")과 같은 순서의 시각 부호. 데이터 값이라 번역하지 않는다.
const CHANNEL_COLORS = {
  vector: "#8a94a8", graph: "#2a78d6", "graph+prop": "#1baf7a",
};
const channelColor = (axes) =>
  CHANNEL_COLORS[axes?.channel] || "#7a5cd6";

const axesLabel = (axes) =>
  Object.entries(axes || {}).map(([k, v]) => `${k}=${v}`).join(" · ") || "—";

/* 파레토 산점도 — 결정적 SVG, 물리 없음 (GoldenTab Timeline 패턴).
   x=비용(p50 ms), y=품질(MRR, 0..1 고정 — 스케일이 데이터를 따라 움직이면
   실행 간 비교가 착시가 된다). 프런티어 점은 채움 + 연결선, 지배점은 테두리만,
   stale 점은 흐리게 + title 경고, ◆ = 현재 운영 설정. */
function ParetoChart({ points, frontier, latestHash, current, t }) {
  if (!points.length) return null;
  const W = 960, H = 300, L = 56, R = 924, T = 18, B = 252;
  const maxX = Math.max(...points.map((p) => p.x), 1);
  const x = (v) => L + (v / maxX) * (R - L);
  const y = (v) => B - Math.max(0, Math.min(1, v)) * (B - T);
  const title = (p, stale) => [
    axesLabel(p.axes),
    `MRR ${p.y.toFixed(4)} · hit@1 ${p.hit1 == null ? "—" : p.hit1.toFixed(4)}`,
    `p50 ${p.x.toFixed(1)}ms · ${t("admin.ex.thCases")} ${p.cases ?? "—"}`,
    p.warnings.length ? `⚠ ${p.warnings.join(", ")}` : null,
    stale ? `⚠ ${t("admin.ex.stale")}` : null,
    p.graphHash ? `graph ${p.graphHash}` : null,
  ].filter(Boolean).join("\n");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}
         role="img" aria-label={t("admin.ex.pareto")}>
      {/* y 그리드 (MRR 0..1) */}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={R} y1={y(v)} y2={y(v)} stroke="rgba(255,255,255,.08)" />
          <text x={L - 6} y={y(v) + 3} textAnchor="end" fontSize="9"
                fill="var(--ink-2)">{v}</text>
        </g>
      ))}
      {/* x 눈금 (비용 ms) */}
      {[0, 0.5, 1].map((f) => (
        <text key={f} x={x(maxX * f)} y={H - 4} textAnchor="middle" fontSize="9"
              fill="var(--ink-2)">{(maxX * f).toFixed(0)}ms</text>
      ))}
      <text x={L} y={T - 6} fontSize="9" fill="var(--ink-2)">MRR</text>
      {/* 프런티어 연결선 — 비용 오름차순 */}
      {frontier.length > 1 && (
        <polyline fill="none" stroke="#1baf7a" strokeOpacity="0.45"
                  strokeWidth="1.4" strokeDasharray="5 4"
                  points={frontier.map((p) => `${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join(" ")} />
      )}
      {points.map((p, i) => {
        const stale = !!(latestHash && p.graphHash && p.graphHash !== latestHash);
        const isCur = current?.record && p.record === current.record;
        const col = channelColor(p.axes);
        return (
          <g key={`${p.runId}-${i}`} opacity={stale ? 0.35 : 1}>
            <circle cx={x(p.x)} cy={y(p.y)} r={5}
                    fill={p.onFrontier ? col : "none"}
                    stroke={col} strokeWidth={1.6}>
              <title>{title(p, stale)}</title>
            </circle>
            {/* ⚠ 표본 부족 배지 — 이 점으로 knob 을 확정하지 말라는 신호 */}
            {p.warnings.includes("small_sample") && (
              <text x={x(p.x) + 7} y={y(p.y) - 5} fontSize="9" fill="#eb6834">
                ⚠<title>{`small_sample — ${t("admin.ex.smallSample")}`}</title>
              </text>
            )}
            {isCur && (
              <text x={x(p.x)} y={y(p.y) - 9} textAnchor="middle" fontSize="11"
                    fill="#eb6834" fontWeight="700">
                ◆<title>{t("admin.ex.current")}</title>
              </text>
            )}
          </g>
        );
      })}
      {/* 채널 범례 — 데이터에 있는 채널만 */}
      {[...new Set(points.map((p) => p.axes?.channel).filter(Boolean))]
        .map((ch, i) => (
          <g key={ch}>
            <circle cx={R - 180 + i * 92} cy={T + 2} r={4}
                    fill={CHANNEL_COLORS[ch] || "#7a5cd6"} />
            <text x={R - 172 + i * 92} y={T + 5} fontSize="9.5"
                  fill="var(--ink-2)">{ch}</text>
          </g>
        ))}
    </svg>
  );
}

export default function ExperimentTab({ t, api, namespace }) {
  const [entries, setEntries] = useState([]);
  const [live, setLive] = useState(null);       // retrieval-config effective
  const [busy, setBusy] = useState(false);      // 목록 로드
  const [runBusy, setRunBusy] = useState(false);
  const [lastRun, setLastRun] = useState(null); // {run_id, combos}
  const [err, setErr] = useState("");
  const [k, setK] = useState(5);

  const call = useCallback(async (path, init) => {
    const r = await fetch(`${api}/graphs/${encodeURIComponent(namespace)}${path}`, init);
    if (!r.ok) {
      let detail = `HTTP ${r.status}`;
      try { const j = await r.json(); if (j?.detail) detail = j.detail; } catch { /* 본문 없음 */ }
      throw new Error(detail);
    }
    return r.json();
  }, [api, namespace]);

  const load = useCallback(async () => {
    if (!namespace) return;
    setBusy(true); setErr("");
    try {
      setEntries((await call("/experiments?limit=200&layer=retrieval"))?.entries || []);
    } catch (e) { setErr(String(e?.message || e)); setEntries([]); }
    finally { setBusy(false); }
    // 현재 운영 설정(◆ 마커) — 부수 정보라 실패해도 화면을 막지 않는다.
    try { setLive((await call("/retrieval-config"))?.effective || null); }
    catch { setLive(null); }
  }, [call, namespace]);

  useEffect(() => { load(); }, [load]);

  const run = useCallback(async () => {
    setRunBusy(true); setErr(""); setLastRun(null);
    try {
      // axes 생략 = 기본 격자 (채널 3종: vector/graph/graph+prop).
      // statuses 에 verified 포함 — PROJ-A 은 confirmed 0 이라 이게 없으면
      // 0 케이스로 "잰 것처럼 보이는 no_cases" 가 된다.
      const res = await call("/experiments/run", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ k, target: "evidence",
                               statuses: ["confirmed", "verified"] }),
      });
      setLastRun({ runId: res?.run_id || "", combos: res?.combos ?? 0 });
      await load();   // 방금 실행분이 레코드에 들어간다 — 바로 비교 가능하게
    } catch (e) { setErr(String(e?.message || e)); }
    finally { setRunBusy(false); }
  }, [call, k, load]);

  const pareto = useMemo(() => paretoView(entries), [entries]);
  const table = useMemo(() => comboTable(entries), [entries]);
  const latestHash = useMemo(
    () => latestGraphHash(fingerprintGroups(entries)), [entries]);
  const current = useMemo(
    () => (live ? currentMarker(entries, live) : null), [entries, live]);

  return (
    <>
      {/* ─── 실행 ─── */}
      <section className="card">
        <h2>{t("admin.ex.title")}
          {entries.length > 0 &&
            <span className="badge" style={{ marginLeft: 8 }}>{entries.length}</span>}
        </h2>
        <p className="hint" style={{ marginTop: 2 }}>{t("admin.ex.hint")}</p>
        <div style={{ display: "flex", gap: 10, alignItems: "center",
                      flexWrap: "wrap", marginTop: 12 }}>
          <button disabled={runBusy} onClick={run}>
            {runBusy && <span className="spinner" />}
            {runBusy ? t("admin.ex.running") : t("admin.ex.run")}
          </button>
          <label style={{ display: "flex", gap: 6, alignItems: "center",
                          margin: 0, fontSize: "0.78rem", color: "var(--ink-2)" }}>
            k
            <input type="number" min={1} max={50} value={k}
                   style={{ width: 64 }} aria-label="k"
                   onChange={(e) => {
                     const n = Number(e.target.value);
                     if (Number.isFinite(n)) setK(Math.max(1, Math.min(50, Math.trunc(n))));
                   }} />
          </label>
          <button className="ghost" disabled={busy} onClick={load}>
            {busy ? t("admin.loading") : t("admin.ex.reload")}</button>
        </div>
        {/* 채널 축 안내 — 점의 색이 곧 이 구분 */}
        <p className="hint" style={{ marginTop: 8 }}>{t("admin.ex.channels")}</p>
        {lastRun && (
          <p className="hint" style={{ marginTop: 6, color: "var(--ink)" }}>
            ✓ {t("admin.ex.ranDone", { n: lastRun.combos })}
            <span className="mono" style={{ marginLeft: 6, fontSize: "0.72rem" }}>
              {lastRun.runId}</span>
          </p>
        )}
        {err && <p style={{ marginTop: 8, color: "#f0857c", fontSize: "0.8rem" }}>⚠ {err}</p>}
      </section>

      {/* ─── 파레토 + 조합 표 ─── */}
      <section className="card">
        <h2>{t("admin.ex.pareto")}</h2>
        <p className="hint" style={{ marginTop: 2 }}>{t("admin.ex.paretoHint")}</p>

        {entries.length === 0 ? (
          <p className="hint" style={{ marginTop: 12 }}>
            {busy ? t("admin.loading") : t("admin.ex.empty")}</p>
        ) : (
          <>
            <ParetoChart points={pareto.points} frontier={pareto.frontier}
                         latestHash={latestHash} current={current} t={t} />
            {pareto.excluded > 0 && (
              <p className="hint" style={{ marginTop: 4 }}>
                {t("admin.ex.excluded", { n: pareto.excluded })}</p>
            )}
            {current && !current.record && (
              <p className="hint" style={{ marginTop: 4 }}>
                {t("admin.ex.noCurrent", { ch: current.liveChannel })}</p>
            )}

            <h3 style={{ fontSize: "0.85rem", marginTop: 18 }}>{t("admin.ex.combos")}</h3>
            <table style={{ marginTop: 8 }}>
              <thead><tr>
                {/* 축 열 동적 — 새 축(entry_k 격자 등)이 자동으로 열이 된다 */}
                {table.axisKeys.map((key) => <th key={key} className="mono">{key}</th>)}
                <th>hit@1</th><th>hit@k</th><th>MRR</th>
                <th>p50(ms)</th><th>embed</th><th>{t("admin.ex.thCases")}</th>
                <th>{t("admin.ex.thAt")}</th><th>{t("admin.ex.thGraph")}</th><th />
              </tr></thead>
              <tbody>
                {table.rows.map((row, i) => {
                  const stale = !!(latestHash && row.graphHash
                                   && row.graphHash !== latestHash);
                  return (
                    <tr key={`${row.runId}-${i}`}
                        style={stale ? { opacity: 0.45 } : undefined}
                        title={stale ? t("admin.ex.stale") : undefined}>
                      {table.axisKeys.map((key) => (
                        <td key={key} className="mono">
                          {row.axes[key] === undefined ? "—" : String(row.axes[key])}
                        </td>
                      ))}
                      <td className="mono">{pctLabel(row.hit1)}</td>
                      <td className="mono">{pctLabel(row.hitK)}</td>
                      <td className="mono">{row.mrr === null ? "—" : row.mrr.toFixed(3)}</td>
                      <td className="mono">{row.p50 === null ? "—" : row.p50.toFixed(1)}</td>
                      <td className="mono">{row.embed === null ? "—" : row.embed}</td>
                      <td className="mono">{row.cases === null ? "—" : row.cases}</td>
                      <td className="mono" style={{ fontSize: "0.72rem" }}>
                        {row.at.replace("T", " ")}</td>
                      <td className="mono" style={{ fontSize: "0.72rem" }}>{row.graphHash || "—"}</td>
                      <td>
                        {row.warnings.includes("small_sample") &&
                          <span title={`small_sample — ${t("admin.ex.smallSample")}`}
                                style={{ color: "#eb6834" }}>⚠</span>}
                        {!row.measured &&
                          <span title={t("admin.ex.notMeasured")}
                                style={{ color: "var(--ink-2)", marginLeft: 4 }}>∅</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
      </section>
    </>
  );
}
