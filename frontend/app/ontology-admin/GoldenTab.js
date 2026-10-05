"use client";

// 골든셋 — 검색을 채점하는 자(ruler)를 사람이 만드는 화면.
//
// 골든셋 = "이 질의엔 이 노드가 나와야 한다"의 **사람 확정** 목록. 이게 없으면
// RRF_K·entry_ratio 같은 상수를 바꿔도 나아졌는지 알 수 없다 — 그래서 이 탭이
// 정교한 검색 기법보다 먼저다.
//
// 이 화면이 정직해야 하는 두 지점:
//   · draft(LLM 초안) vs confirmed(사람 확정)를 섞지 않는다 — 자동 정답으로 채점하면
//     자기 채점이 된다. 평가 기본값은 confirmed 만.
//   · 채널 지표가 서로 같으면 **"이 측정은 두 채널을 구별하지 못한다"** 고 말한다.
//     숫자를 그냥 보여주면 튜닝 신호가 있는 것처럼 오해된다(라이브 실측이 이 상태였다).
//
// 백엔드 변경 0 — /qa · /qa/cases · /qa/generate · /qa/evaluate 는 이미 있다.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  normalizeCases, allTags, filterCases, normalizeEval,
  channelsIdentical, pctLabel, rankLabel,
  timelineSeries, configChanges,
} from "./goldenView.mjs";

const K_CHOICES = [5, 10];

/* 평가 이력 타임라인 — "그때 그 숫자"가 아니라 궤적. 지표 꺾은선 위에 설정
   변경 마커(주황 점선)를 겹쳐 "설정을 바꾼 자리"와 "지표가 움직인 자리"가
   같은 그림에 있게 한다 — 이 세션에서 회복 궤적 표를 손으로 세 번 만든
   그 자리의 자동화. 결정적 SVG (x = 실행 순서 — 평가는 불규칙 간격이라
   시간 비례축은 점들이 뭉친다). */
function Timeline({ entries, target }) {
  const points = timelineSeries(entries, { target });
  const marks = configChanges(points);
  if (points.length < 2) return null;
  const W = 960, H = 190, L = 46, R = 884, T = 14, B = 152;
  const x = (i) => L + (points.length === 1 ? 0 : (i * (R - L)) / (points.length - 1));
  const y = (v) => B - v * (B - T);
  const path = (key) => points
    .map((p, i) => (p[key] == null ? null : `${i === 0 || points[i - 1][key] == null ? "M" : "L"}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`))
    .filter(Boolean).join(" ");
  const SERIES = [
    { key: "hit1", label: "hit@1", color: "#2a78d6", w: 2 },
    { key: "hitk", label: "hit@k", color: "#8a94a8", w: 1.2 },
    { key: "mrr", label: "MRR", color: "#1baf7a", w: 2 },
  ];
  const day = (s) => String(s).slice(5, 16).replace("T", " ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}
         role="img" aria-label="평가 이력 타임라인">
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={R} y1={y(v)} y2={y(v)} stroke="rgba(255,255,255,.08)" />
          <text x={L - 6} y={y(v) + 3} textAnchor="end" fontSize="9"
                fill="var(--ink-2)">{v}</text>
        </g>
      ))}
      {/* 설정 변경 마커 — 인과를 물을 수 있는 자리 */}
      {marks.map((m) => (
        <g key={m.index}>
          <line x1={x(m.index)} x2={x(m.index)} y1={T - 4} y2={B}
                stroke="#eb6834" strokeWidth="1.1" strokeDasharray="4 4" opacity="0.8" />
          <text x={x(m.index)} y={T - 5} textAnchor="middle" fontSize="8.5" fill="#f2946a">⚙</text>
          <title>{`설정 변경: ${m.keys.join(", ")}`}</title>
        </g>
      ))}
      {SERIES.map((s) => (
        <path key={s.key} d={path(s.key)} fill="none" stroke={s.color}
              strokeWidth={s.w} strokeLinejoin="round" />
      ))}
      {/* 점 + tooltip — 값과 그때의 케이스 수 */}
      {points.map((p, i) => SERIES.map((s) => (p[s.key] == null ? null : (
        <circle key={`${i}-${s.key}`} cx={x(i)} cy={y(p[s.key])} r={3}
                fill={s.color}>
          <title>{`${day(p.at)} · ${s.label} ${p[s.key].toFixed(4)} (케이스 ${p.cases})`}</title>
        </circle>
      ))))}
      {/* 직접 라벨 — 마지막 점 옆 (범례 겸용) */}
      {SERIES.map((s) => {
        const lastIdx = [...points.keys()].reverse().find((i) => points[i][s.key] != null);
        if (lastIdx == null) return null;
        return (
          <text key={s.key} x={R + 8} y={y(points[lastIdx][s.key]) + 3}
                fontSize="10" fill={s.color}>{s.label}</text>
        );
      })}
      <text x={L} y={H - 4} fontSize="9" fill="var(--ink-2)">{day(points[0].at)}</text>
      <text x={R} y={H - 4} textAnchor="end" fontSize="9" fill="var(--ink-2)">
        {day(points[points.length - 1].at)}</text>
    </svg>
  );
}

export default function GoldenTab({ t, api, namespace }) {
  const [raw, setRaw] = useState(null);
  const [evalRaw, setEvalRaw] = useState(null);
  const [busy, setBusy] = useState(false);
  const [evalBusy, setEvalBusy] = useState(false);
  const [genBusy, setGenBusy] = useState(false);
  const [err, setErr] = useState("");
  const [status, setStatus] = useState("all");
  const [tag, setTag] = useState("");
  const [q, setQ] = useState("");
  const [k, setK] = useState(5);
  const [includeDrafts, setIncludeDrafts] = useState(false);
  // 채점 위치. node = 노드 랭킹(semantic vs retrieve — 구조상 같게 나온다),
  // chunk = 청크 회수(그래프 조건화의 효과가 실제로 드러나는 자리).
  const [target, setTarget] = useState("node");

  const call = useCallback(async (path, init) => {
    const r = await fetch(`${api}/graphs/${encodeURIComponent(namespace)}${path}`, init);
    if (!r.ok) {
      let detail = `HTTP ${r.status}`;
      try { const j = await r.json(); if (j?.detail) detail = j.detail; } catch { /* 본문 없음 */ }
      throw new Error(detail);
    }
    return r.json();
  }, [api, namespace]);

  // 평가 이력 — **설정 지문 + 지표**. 점수만 보면 "그때 그 숫자가 어떤 설정에서
  // 나온 것인가"를 알 수 없다. 임베더·entry_k·max_terms 가 전부 지표를 움직인다.
  const [history, setHistory] = useState([]);

  const load = useCallback(async () => {
    if (!namespace) return;
    setBusy(true); setErr("");
    try { setRaw(await call("/qa")); }
    catch (e) { setErr(String(e?.message || e)); setRaw(null); }
    finally { setBusy(false); }
  }, [call, namespace]);

  const loadHistory = useCallback(async () => {
    if (!namespace) return;
    // 이력 조회 실패가 골든셋 화면을 막지 않는다 — 부수 정보다.
    try { setHistory((await call("/qa/history?limit=30"))?.entries || []); }
    catch { setHistory([]); }
  }, [call, namespace]);

  useEffect(() => { load(); loadHistory(); }, [load, loadHistory]);

  const runEval = useCallback(async () => {
    setEvalBusy(true); setErr("");
    try {
      setEvalRaw(await call("/qa/evaluate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ k, include_drafts: includeDrafts, target }),
      }));
      loadHistory();     // 방금 실행분이 이력에 들어간다 — 바로 비교 가능하게
    } catch (e) { setErr(String(e?.message || e)); setEvalRaw(null); }
    finally { setEvalBusy(false); }
  }, [call, k, includeDrafts, target, loadHistory]);

  const confirmCase = useCallback(async (caseId) => {
    setErr("");
    try { await call(`/qa/cases/${encodeURIComponent(caseId)}/confirm`, { method: "POST" }); await load(); }
    catch (e) { setErr(String(e?.message || e)); }
  }, [call, load]);

  // LLM 초안 생성 — 노드당 1콜이라 비용이 있다. 상한을 화면에 밝히고 확인을 받는다.
  const generate = useCallback(async () => {
    if (!window.confirm(t("admin.gs.genConfirm"))) return;
    setGenBusy(true); setErr("");
    try { await call("/qa/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ limit: 10, per_node: 2 }) }); await load(); }
    catch (e) { setErr(String(e?.message || e)); }
    finally { setGenBusy(false); }
  }, [call, load, t]);

  const view = useMemo(() => normalizeCases(raw), [raw]);
  const tags = useMemo(() => allTags(view.cases), [view.cases]);
  const shown = useMemo(() => filterCases(view.cases, { status, tag, q }),
                        [view.cases, status, tag, q]);
  const ev = useMemo(() => (evalRaw ? normalizeEval(evalRaw) : null), [evalRaw]);
  const sameChannels = useMemo(() => (ev ? channelsIdentical(ev) : false), [ev]);

  return (
    <>
      {/* ─── 골든셋 현황 ─── */}
      <section className="card">
        <h2>{t("admin.gs.title")}
          <span className="badge" style={{ marginLeft: 8 }}>{view.total}</span></h2>
        <p className="hint" style={{ marginTop: 2 }}>{t("admin.gs.hint")}</p>

        <div className="oa-gs-counts">
          <div className="oa-gs-count">
            <div className="oa-gs-cnum">{view.counts.confirmed}</div>
            <div className="oa-gs-clbl">{t("admin.gs.confirmed")}</div>
            <div className="oa-gs-cdesc">{t("admin.gs.confirmedDesc")}</div>
          </div>
          <div className="oa-gs-count draft">
            <div className="oa-gs-cnum">{view.counts.draft}</div>
            <div className="oa-gs-clbl">{t("admin.gs.draft")}</div>
            <div className="oa-gs-cdesc">{t("admin.gs.draftDesc")}</div>
          </div>
        </div>

        <div className="oa-gs-bar">
          <input className="oa-gs-input" value={q} onChange={(e) => setQ(e.target.value)}
                 placeholder={t("admin.gs.search")} aria-label={t("admin.gs.search")} />
          <select className="oa-gs-sel" value={status} onChange={(e) => setStatus(e.target.value)}
                  aria-label={t("admin.gs.status")}>
            <option value="all">{t("admin.gs.allStatus")}</option>
            <option value="confirmed">{t("admin.gs.confirmed")}</option>
            <option value="draft">{t("admin.gs.draft")}</option>
          </select>
          <select className="oa-gs-sel" value={tag} onChange={(e) => setTag(e.target.value)}
                  aria-label={t("admin.gs.tag")}>
            <option value="">{t("admin.gs.allTags")}</option>
            {tags.map((tg) => <option key={tg} value={tg}>{tg}</option>)}
          </select>
          <button className="ghost" disabled={busy} onClick={load}>
            {busy ? t("admin.loading") : t("admin.gs.reload")}</button>
          <button className="ghost" disabled={genBusy} onClick={generate}>
            {genBusy ? t("admin.gs.generating") : t("admin.gs.generate")}</button>
        </div>

        {err && <p className="oa-gs-err">⚠ {err}</p>}

        {view.total === 0 ? (
          <p className="hint" style={{ marginTop: 12 }}>{t("admin.gs.empty")}</p>
        ) : (
          <table style={{ marginTop: 12 }}>
            <thead><tr>
              <th>{t("admin.gs.thQuery")}</th><th>{t("admin.gs.thExpected")}</th>
              <th>{t("admin.gs.thTags")}</th><th>{t("admin.gs.thStatus")}</th><th />
            </tr></thead>
            <tbody>
              {shown.map((c) => (
                <tr key={c.case_id}>
                  <td>{c.query}</td>
                  <td className="mono">{c.expected_node_id}
                    {Array.isArray(c.accepted) && c.accepted.length > 0 &&
                      <span className="oa-gs-acc">+{c.accepted.length}</span>}
                  </td>
                  <td>{(c.tags || []).map((tg) => (
                    <span key={tg} className="oa-gs-tag">{tg}</span>))}</td>
                  <td>
                    <span className={`oa-gs-st ${c.status || ""}`}>{c.status || "—"}</span>
                    {c.source && <span className="oa-gs-src">{c.source}</span>}
                  </td>
                  <td>
                    {c.status === "draft" &&
                      <button className="ghost" onClick={() => confirmCase(c.case_id)}>
                        {t("admin.gs.confirmBtn")}</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* ─── 평가 (채점) ─── */}
      <section className="card">
        <h2>{t("admin.gs.evalTitle")}</h2>
        <p className="hint" style={{ marginTop: 2 }}>{t("admin.gs.evalHint")}</p>

        <div className="oa-gs-bar">
          {/* 채점 위치 — 우리 차별점을 어디서 재는가 */}
          <div className="oa-gs-k" role="group" aria-label={t("admin.gs.target")}>
            <button className={`oa-gs-kbtn ${target === "node" ? "on" : ""}`}
                    onClick={() => setTarget("node")}>{t("admin.gs.targetNode")}</button>
            <button className={`oa-gs-kbtn ${target === "chunk" ? "on" : ""}`}
                    onClick={() => setTarget("chunk")}>{t("admin.gs.targetChunk")}</button>
          </div>
          <div className="oa-gs-k" role="group" aria-label="k">
            {K_CHOICES.map((n) => (
              <button key={n} className={`oa-gs-kbtn ${k === n ? "on" : ""}`}
                      onClick={() => setK(n)}>k={n}</button>
            ))}
          </div>
          <label className="oa-gs-chk">
            <input type="checkbox" checked={includeDrafts}
                   onChange={(e) => setIncludeDrafts(e.target.checked)} />
            {t("admin.gs.includeDrafts")}
          </label>
          <button disabled={evalBusy || view.total === 0} onClick={runEval}>
            {evalBusy ? t("admin.gs.evaluating") : t("admin.gs.evaluate")}</button>
        </div>
        {includeDrafts && <p className="hint">{t("admin.gs.draftWarn")}</p>}
        <p className="hint">{target === "chunk"
          ? t("admin.gs.targetChunkHint") : t("admin.gs.targetNodeHint")}</p>

        {ev && (
          <>
            {/* 채점된 문항 수와 스킵 수 — "몇 개를 실제로 쟀는가"를 숨기지 않는다 */}
            <div className="oa-gs-scope">
              <span className="oa-gs-scopenum">{ev.cases}</span>
              {t("admin.gs.scored")}
              {ev.skipped > 0 && (
                <span className="oa-gs-skip">
                  · {t("admin.gs.skipped")} <b>{ev.skipped}</b>
                  <span className="oa-gs-skipwhy">
                    {ev.target === "chunk" ? t("admin.gs.skipChunkWhy") : t("admin.gs.skipNodeWhy")}
                  </span>
                </span>
              )}
            </div>

            {/* 청크 라벨이 아예 없으면 지표가 전부 0 — 측정 불가임을 명시한다 */}
            {ev.cases === 0 && (
              <div className="oa-gs-warn">
                <b>{t("admin.gs.noneScoredTitle")}</b>
                <p>{ev.target === "chunk"
                  ? t("admin.gs.noneScoredChunk") : t("admin.gs.noneScoredNode")}</p>
              </div>
            )}

            {/* 채널 지표가 같으면 = 이 측정이 두 채널을 구별하지 못한다 */}
            {sameChannels && ev.cases > 0 && (
              <div className="oa-gs-warn">
                <b>{t("admin.gs.sameTitle")}</b>
                <p>{t("admin.gs.sameBody")}</p>
              </div>
            )}

            <table style={{ marginTop: 12 }}>
              <thead><tr>
                <th>{t("admin.gs.thChannel")}</th><th>{t("admin.gs.thCases")}</th>
                <th>hit@1</th><th>{`hit@${ev.k}`}</th><th>MRR</th>
              </tr></thead>
              <tbody>
                {ev.channels.map((c) => (
                  <tr key={c.name}>
                    <td><b>{c.name}</b></td>
                    <td className="mono">{c.cases}</td>
                    <td className="mono">{pctLabel(c.hit1)}</td>
                    <td className="mono">{pctLabel(c.hitK)}</td>
                    <td className="mono">{c.mrr === null ? "—" : c.mrr.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {ev.byTag.length > 0 && (
              <>
                <h3 className="oa-gs-h3">{t("admin.gs.byTag")}</h3>
                <p className="hint">{t("admin.gs.byTagHint")}</p>
                <table>
                  <thead><tr>
                    <th>{t("admin.gs.thTag")}</th><th>{t("admin.gs.thChannel")}</th>
                    <th>{t("admin.gs.thCases")}</th><th>hit@1</th>
                    <th>{`hit@${ev.k}`}</th><th>MRR</th>
                  </tr></thead>
                  <tbody>
                    {ev.byTag.flatMap((tg) => tg.channels.map((c, i) => (
                      <tr key={`${tg.tag}-${c.name}`}>
                        <td>{i === 0 ? <b>{tg.tag}</b> : ""}</td>
                        <td>{c.name}</td>
                        <td className="mono">{c.cases}</td>
                        <td className="mono">{pctLabel(c.hit1)}</td>
                        <td className="mono">{pctLabel(c.hitK)}</td>
                        <td className="mono">{c.mrr === null ? "—" : c.mrr.toFixed(3)}</td>
                      </tr>
                    )))}
                  </tbody>
                </table>
              </>
            )}

            {ev.perCase.length > 0 && (
              <>
                <h3 className="oa-gs-h3">{t("admin.gs.perCase")}</h3>
                <p className="hint">{t("admin.gs.perCaseHint")}</p>
                <table>
                  {/* 채널명은 타깃에 따라 다르다(node: semantic·retrieve /
                      chunk: chunk·retrieve) — 하드코딩하면 청크 채점에서 빈 칸이 된다 */}
                  <thead><tr>
                    <th>{t("admin.gs.thQuery")}</th><th>{t("admin.gs.thExpected")}</th>
                    {ev.channels.map((ch) => <th key={ch.name}>{ch.name}</th>)}
                  </tr></thead>
                  <tbody>
                    {ev.perCase.map((c) => (
                      <tr key={c.case_id}>
                        <td>{c.query}</td>
                        <td className="mono">{c.expected}</td>
                        {ev.channels.map((ch) => {
                          const lbl = rankLabel(c[`${ch.name}_rank`]);
                          return (
                            <td key={ch.name}
                                className={`mono oa-gs-rank ${lbl === "✗" ? "miss" : ""}`}>
                              {lbl}</td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </>
        )}

        {/* 평가 이력 — 설정 지문과 함께. 점수만 보면 "그때 그 숫자가 어떤
            설정에서 나온 것인가"를 잃고 비교가 불가능해진다 (실제로 두 번 데였다:
            max_terms 8→2 를 되돌렸고, 임베더도 채널 하나만 재고 결론냈다). */}
        {history.length > 0 && (
          <>
            <h3 className="oa-gs-h3">{t("admin.gs.history")}</h3>
            <p className="hint">{t("admin.gs.historyHint")}</p>
            {/* 궤적 차트 — 현재 선택된 target 의 retrieve 채널. 주황 점선 = 설정
                변경(임베더·knob·확산). 점 hover = 값·케이스 수, ⚙ hover = 바뀐 키. */}
            <Timeline entries={history} target={target} />
            <table>
              <thead><tr>
                <th>{t("admin.gs.thWhen")}</th><th>{t("admin.gs.thTarget")}</th>
                <th>k</th><th>{t("admin.gs.thCases")}</th>
                <th>{t("admin.gs.thChannel")}</th>
                <th>hit@1</th><th>hit@k</th><th>MRR</th>
                <th>{t("admin.sys.model")}</th><th>knob</th>
              </tr></thead>
              <tbody>
                {history.flatMap((h) => {
                  const cfg = h.config || {};
                  const model = cfg.node_model === cfg.chunk_model
                    ? (cfg.node_model || "—")
                    : `${cfg.node_model || "?"} / ${cfg.chunk_model || "?"}`;
                  const knob = [
                    cfg.entry_k != null ? `k${cfg.entry_k}` : null,
                    cfg.max_terms != null ? `t${cfg.max_terms}` : null,
                    cfg.entry_ratio != null ? `r${cfg.entry_ratio}` : null,
                    cfg.propagation_channel ? `prop${cfg.propagation_weight}` : null,
                  ].filter(Boolean).join(" ");
                  const names = Object.keys(h.channels || {});
                  if (!names.length) return [];
                  return names.map((name, i) => {
                    const m = h.channels[name] || {};
                    const hk = Object.keys(m).find((x) => x.startsWith("hit@")
                                                   && x !== "hit@1");
                    return (
                      <tr key={`${h.at}-${h.target}-${name}`}>
                        <td className="mono">{i === 0 ? (h.at || "").replace("T", " ") : ""}</td>
                        <td>{i === 0 ? h.target : ""}</td>
                        <td className="mono">{i === 0 ? h.k : ""}</td>
                        <td className="mono">{i === 0 ? h.cases : ""}</td>
                        <td>{name}</td>
                        <td className="mono">{pctLabel(m["hit@1"])}</td>
                        <td className="mono">{pctLabel(hk ? m[hk] : null)}</td>
                        <td className="mono">{m.mrr == null ? "—" : m.mrr.toFixed(3)}</td>
                        <td className="mono" style={{ fontSize: "0.72rem" }}>
                          {i === 0 ? model : ""}</td>
                        <td className="mono" style={{ fontSize: "0.72rem" }}>
                          {i === 0 ? knob : ""}</td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </>
        )}
      </section>
    </>
  );
}
