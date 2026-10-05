"use client";

// 수집(Ingest) 섹션 — 데이터를 온톨로지에 넣는 유일한 입구.
//   업로드 → 감식(analyze) → 확인 게이트 → 수집(build/ingest, 백그라운드 job)
// 백엔드 계약(server/router.py):
//   POST /datasets                     multipart files → {dataset_id, files, total_bytes}
//   POST /datasets/{id}/analyze        → {files[], plan[], total_estimated_llm_calls}
//   POST /datasets/{id}/ingest         {namespace, plan, save} → {job_id}
//   GET  /jobs/{id}                    → {status, progress{stage,current,total}, report, error}
// 원칙: protected(default) 네임스페이스는 백엔드가 400 으로 막는다 — 여기서도 사전 차단.
//       plan 은 analyze 초안을 그대로 제출하되, 신뢰등급만 운영자가 덮어쓸 수 있게 한다
//       (온톨로지의 차별점 = trust provenance 이므로 수집 시점에 명시).

import { useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_ONTOLOGY_API || "http://localhost:9274";
const BASE = `${API}/api/v1/ontology`;

const TRUST_OPTIONS = ["authoritative", "unknown", "summary"];
const SPECIES_TONE = {
  records: "ok", prose: "info", articled: "info",
  seed_ontology: "accent", skip: "muted",
};

const detailMsg = async (res) => {
  try {
    const d = (await res.json()).detail;
    if (typeof d === "string" && d) return d;
    if (d) return JSON.stringify(d);
  } catch { /* 비 JSON → statusText */ }
  return res.statusText;
};

export default function IngestTab({ t, namespace, namespaces = [], protectedNamespaces = [], onComplete, flash }) {
  const [files, setFiles] = useState([]);          // 선택된 File[]
  const [dataset, setDataset] = useState(null);    // {dataset_id, files, total_bytes}
  const [analysis, setAnalysis] = useState(null);  // {files[], plan[], total_estimated_llm_calls}
  const [plan, setPlan] = useState([]);            // 편집 가능한 plan (trust override)
  const [targetNs, setTargetNs] = useState(namespace || "");
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [extractionMode, setExtractionMode] = useState("exhaustive"); // exhaustive | topic
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const pollRef = useRef(null);
  const inputRef = useRef(null);

  const nsProtected = protectedNamespaces.includes(targetNs.trim());
  const speciesLabel = (s) => t(`admin.ingest.species.${s}`) || s;

  const reset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setFiles([]); setDataset(null); setAnalysis(null); setPlan([]);
    setJob(null); setError(""); setBusy(false);
  };

  const addFiles = (list) => {
    const incoming = Array.from(list || []);
    if (!incoming.length) return;
    // 새 파일을 고르면 이전 감식 결과는 무효
    setDataset(null); setAnalysis(null); setPlan([]); setJob(null); setError("");
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => f.name + f.size));
      return [...prev, ...incoming.filter((f) => !seen.has(f.name + f.size))];
    });
  };

  const removeFile = (i) =>
    setFiles((prev) => prev.filter((_, idx) => idx !== i));

  // 업로드 → 감식 을 하나의 사용자 행위로 묶는다 (분리하면 중간 상태가 의미 없음)
  const uploadAndAnalyze = async () => {
    if (!files.length || busy) return;
    setBusy(true); setError(""); setJob(null);
    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const up = await fetch(`${BASE}/datasets`, { method: "POST", body: form });
      if (!up.ok) throw new Error(await detailMsg(up));
      const ds = await up.json();
      setDataset(ds);

      const an = await fetch(`${BASE}/datasets/${ds.dataset_id}/analyze`, { method: "POST" });
      if (!an.ok) throw new Error(await detailMsg(an));
      const result = await an.json();
      setAnalysis(result);
      setPlan(result.plan || []);
    } catch (e) {
      setError(t("admin.ingest.err.analyze", { e: e.message || e }));
    }
    setBusy(false);
  };

  const setTrust = (i, trust) =>
    setPlan((prev) => prev.map((p, idx) => (idx === i ? { ...p, trust } : p)));

  const start = async () => {
    const ns = targetNs.trim();
    if (!ns || !dataset || !plan.length || busy || nsProtected) return;
    setBusy(true); setError(""); setJob(null);
    try {
      const res = await fetch(`${BASE}/datasets/${dataset.dataset_id}/ingest`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ namespace: ns, plan, save: true, schema_mode: "auto",
                               extraction_mode: extractionMode }),
      });
      if (!res.ok) throw new Error(await detailMsg(res));
      const { job_id } = await res.json();
      setJob({ status: "queued" });
      pollRef.current = setInterval(async () => {
        try {
          const j = await fetch(`${BASE}/jobs/${job_id}`).then((r) => r.json());
          setJob(j);
          if (j.status === "completed" || j.status === "failed") {
            clearInterval(pollRef.current);
            setBusy(false);
            if (j.status === "completed") {
              flash?.(t("admin.ingest.done", { ns }));
              onComplete?.(ns);
            } else {
              setError(t("admin.ingest.err.job", { e: j.error || "unknown" }));
            }
          }
        } catch { /* 폴링 실패는 다음 tick 에 재시도 */ }
      }, 1500);
    } catch (e) {
      setError(t("admin.ingest.err.start", { e: e.message || e }));
      setBusy(false);
    }
  };

  const running = job && (job.status === "queued" || job.status === "running");
  const prog = job?.progress || {};
  const pct = prog.total ? Math.round((prog.current / prog.total) * 100) : null;

  return (
    <div className="oa-ingest">
      <div className="oa-context">
        <div>
          <h1 className="oa-h1">{t("admin.ingest.title")}</h1>
          <p className="oa-sub">{t("admin.ingest.subtitle")}</p>
        </div>
        {(dataset || files.length > 0) && !running && (
          <button className="ghost" onClick={reset}>{t("admin.ingest.reset")}</button>
        )}
      </div>

      {error && <div className="error" style={{ marginBottom: 14 }}>{error}</div>}

      {/* 진행 중 job — 다른 UI 를 가리고 진행률만 */}
      {job && (
        <div className="card" style={{ marginTop: 14 }}>
          <h2 style={{ fontSize: "0.9rem" }}>{t("admin.ingest.progress.title")}</h2>
          {running ? (
            <>
              <div className="oa-ing-stage">
                <span className="spinner" />
                {prog.stage
                  ? t(`admin.ingest.stage.${prog.stage}`) || prog.stage
                  : t("admin.ingest.stage.queued")}
                {pct !== null && <span className="oa-ing-count"> · {prog.current}/{prog.total}</span>}
              </div>
              <div className="oa-ing-bar">
                <span style={pct !== null
                  ? { width: `${pct}%` }
                  : { width: "100%", opacity: 0.35 }} />
              </div>
            </>
          ) : job.status === "completed" ? (
            <div className="oa-ing-report">
              ✓ {t("admin.ingest.progress.done")}
              {/* B2: 커버리지 expectation 게이트 — 경고이지 실패가 아니다 */}
              {job.report?.coverage_gate?.status === "warn" && (
                <div style={{ marginTop: 8, padding: "8px 10px", borderRadius: 8,
                              border: "1px solid #f2946a",
                              color: "var(--ink)", fontSize: 12.5 }}>
                  <b style={{ color: "#f2946a" }}>⚠ 커버리지 기대 미달</b>
                  <span style={{ color: "var(--ink-2)" }}> — 인제스트는 완료됐다. </span>
                  {(job.report.coverage_gate.warnings || [])
                    .filter((w) => w.measured)
                    .map((w) => (
                      <div key={w.key} style={{ color: "var(--ink-2)" }}>
                        · {w.message || `${w.key}: ${w.actual} (기대 ${w.expected})`}
                      </div>
                    ))}
                  {job.report.coverage_gate.auto_check?.gaps > 0 && (
                    <div style={{ color: "var(--ink-2)" }}>
                      · 자동 커버리지 검사: gap {job.report.coverage_gate.auto_check.gaps}건이
                      검수 큐에 준비됨
                    </div>
                  )}
                </div>
              )}
              {job.report && (
                <div className="oa-ing-report-nums">
                  {Object.entries(job.report)
                    .filter(([, v]) => typeof v === "number")
                    .map(([k, v]) => (
                      <span key={k} className="oa-ing-num">
                        <b>{v.toLocaleString()}</b> {k}
                      </span>
                    ))}
                </div>
              )}
              <button style={{ marginTop: 12 }} onClick={reset}>
                {t("admin.ingest.progress.again")}
              </button>
            </div>
          ) : null}
        </div>
      )}

      {/* 1) 파일 선택 */}
      {!job && (
        <div
          className={`oa-drop ${dragOver ? "over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); addFiles(e.dataTransfer.files); }}
          onClick={() => inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" multiple style={{ display: "none" }}
                 onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
          <div className="oa-drop-icon">⬆</div>
          <div className="oa-drop-main">{t("admin.ingest.drop")}</div>
          <div className="oa-drop-sub">{t("admin.ingest.dropHint")}</div>
        </div>
      )}

      {!job && files.length > 0 && (
        <div className="card soft" style={{ marginTop: 14 }}>
          <div className="oa-filelist">
            {files.map((f, i) => (
              <div key={f.name + i} className="oa-file">
                <span className="oa-file-name">{f.name}</span>
                <span className="oa-file-size">{(f.size / 1024).toFixed(1)} KB</span>
                {!dataset && (
                  <button className="oa-file-x" title={t("admin.ingest.remove")}
                          onClick={() => removeFile(i)}>✕</button>
                )}
              </div>
            ))}
          </div>
          {!analysis && (
            <button style={{ marginTop: 12 }} onClick={uploadAndAnalyze} disabled={busy}>
              {busy && <span className="spinner" />}
              {busy ? t("admin.ingest.analyzing") : t("admin.ingest.analyze")}
            </button>
          )}
        </div>
      )}

      {/* 2) 감식 결과 + 확인 게이트 */}
      {!job && analysis && (
        <div className="card" style={{ marginTop: 14 }}>
          <h2 style={{ fontSize: "0.9rem" }}>{t("admin.ingest.analysis.title")}</h2>
          <table style={{ marginTop: 8 }}>
            <thead><tr>
              <th>{t("admin.ingest.th.file")}</th>
              <th>{t("admin.ingest.th.species")}</th>
              <th style={{ textAlign: "right" }}>{t("admin.ingest.th.units")}</th>
              <th style={{ textAlign: "right" }}>{t("admin.ingest.th.llm")}</th>
              <th style={{ width: "22%" }}>{t("admin.ingest.th.trust")}</th>
            </tr></thead>
            <tbody>
              {analysis.files.map((f, i) => {
                const p = plan[i] || {};
                const units = f.species === "records" || f.species === "seed_ontology"
                  ? f.record_count : f.chunk_count;
                const unitKind = f.species === "records" || f.species === "seed_ontology"
                  ? t("admin.ingest.units.records") : t("admin.ingest.units.chunks");
                return (
                  <tr key={f.filename}>
                    <td>
                      <b>{f.filename}</b>
                      <span className="oa-file-fmt">{f.format}</span>
                      {f.error && <div className="status-failed">{f.error}</div>}
                    </td>
                    <td><span className={`badge tone-${SPECIES_TONE[f.species] || "muted"}`}>
                      {speciesLabel(f.species)}</span></td>
                    <td className="mono" style={{ textAlign: "right" }}>
                      {(units || 0).toLocaleString()} <span className="oa-unit">{unitKind}</span></td>
                    <td className="mono" style={{ textAlign: "right" }}>
                      {(f.estimated_llm_calls || 0).toLocaleString()}</td>
                    <td>
                      <select value={p.trust || f.filename_meta?.trust || "unknown"}
                              onChange={(e) => setTrust(i, e.target.value)}>
                        {TRUST_OPTIONS.map((tr) =>
                          <option key={tr} value={tr}>{tr}</option>)}
                      </select>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div className="oa-costgate">
            <span>{t("admin.ingest.estCost", { n: (analysis.total_estimated_llm_calls || 0).toLocaleString() })}</span>
          </div>

          {/* 3) 대상 네임스페이스 + 시작 */}
          <div className="oa-ing-target">
            <label>{t("admin.ingest.target")}</label>
            <input type="text" value={targetNs} placeholder={t("admin.ingest.targetPh")}
                   list="oa-ns-list"
                   onChange={(e) => setTargetNs(e.target.value)} />
            <datalist id="oa-ns-list">
              {namespaces
                .filter((n) => !protectedNamespaces.includes(n))
                .map((n) => <option key={n} value={n} />)}
            </datalist>
          </div>
          {nsProtected && (
            <div className="oa-warn">{t("admin.ingest.protectedWarn", { ns: targetNs.trim() })}</div>
          )}
          <p className="oa-hint">{t("admin.ingest.targetHint")}</p>

          {/* 벡터화 전 교감 — 무엇으로 벡터화되는지 미리보기 */}
          {analysis?.ingest_config && (() => {
            const cfg = analysis.ingest_config;
            const ns = targetNs.trim() || "{namespace}";
            const sub = (s) => (s || "").replace("{namespace}", ns);
            return (
              <div className="oa-ing-preview">
                <div className="oa-ing-preview-title">{t("admin.ingest.preview.title")}</div>
                <div className="oa-ing-preview-grid">
                  <div><span>{t("admin.ingest.preview.embModel")}</span>
                    <b>{cfg.embedding_model}</b>
                    {!cfg.embedder_available &&
                      <em className="oa-ing-preview-warn"> · {t("admin.ingest.preview.embOff")}</em>}</div>
                  <div><span>{t("admin.ingest.preview.chunk")}</span>
                    <b>{cfg.chunk_size} / {cfg.overlap} · {cfg.segment_mode}</b></div>
                  <div><span>{t("admin.ingest.preview.keyword")}</span>
                    <b className={cfg.keyword_search ? "oa-index-ok" : "oa-index-warn"}>
                      {cfg.keyword_search ? t("admin.ingest.preview.kwOn") : t("admin.ingest.preview.kwOff")}</b></div>
                  <div><span>{t("admin.ingest.preview.objIndex")}</span>
                    <b className="mono">{sub(cfg.object_index)}</b></div>
                  <div><span>{t("admin.ingest.preview.chunkIndex")}</span>
                    <b className="mono">{sub(cfg.chunk_index)}</b></div>
                </div>
                <p className="oa-hint" style={{ marginTop: 6 }}>{t("admin.ingest.preview.hint")}</p>
              </div>
            );
          })()}

          {/* 추출 방식(텍스트/약관 경로) — 전체 vs 토픽 회수 */}
          <div className="oa-ing-mode">
            <label>{t("admin.ingest.mode.label")}</label>
            <div className="oa-exp-kindbar" style={{ margin: "6px 0 0" }}>
              {["exhaustive", "topic"].map((m) => (
                <button key={m}
                  className={`oa-exp-kindbtn ${extractionMode === m ? "on" : ""}`}
                  onClick={() => setExtractionMode(m)}>
                  {t(`admin.ingest.mode.${m}`)}</button>
              ))}
            </div>
            <p className="oa-hint" style={{ marginTop: 6 }}>
              {t(`admin.ingest.mode.${extractionMode}Hint`)}</p>
          </div>

          <button style={{ marginTop: 12 }} onClick={start}
                  disabled={busy || !targetNs.trim() || nsProtected}>
            {t("admin.ingest.start")}
          </button>
        </div>
      )}
    </div>
  );
}
