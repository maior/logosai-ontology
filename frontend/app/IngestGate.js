"use client";

// 업로드 → 감식(analyze) → 확인 게이트 → 실행(ingest) 3단계 빌드 플로우.
// 백엔드: POST /datasets → POST /datasets/{id}/analyze → POST /datasets/{id}/ingest → GET /jobs/{id}
import { useEffect, useRef, useState } from "react";
import { useT } from "./i18n";

const ACCEPT_EXTS = ".pdf,.csv,.json,.jsonld,.hwp,.doc,.docx,.md,.txt";

const SPECIES_COLORS = {
  records: "#2563eb",       // 파랑
  articled: "#7c3aed",      // 보라
  prose: "#6b7280",         // 회색
  seed_ontology: "#059669", // 초록
};
const ROLE_COLORS = { identity: "#2563eb", node_candidate: "#7c3aed", attribute: "#98a2b3" };

const pill = (bg) => ({
  display: "inline-block", padding: "2px 9px", borderRadius: 999,
  background: bg || "#98a2b3", color: "#fff", fontSize: "0.68rem",
  fontWeight: 700, letterSpacing: "0.3px",
});
const smallInput = { padding: "5px 8px", fontSize: "0.8rem" };
const smallBtn = { padding: "4px 10px", fontSize: "0.74rem" };

// analyze 응답(files + plan)을 편집 가능한 per-file 엔트리로 변환
function initEntries(analysis) {
  const planByFile = Object.fromEntries((analysis.plan || []).map((p) => [p.filename, p]));
  return (analysis.files || []).map((f) => {
    const p = planByFile[f.filename] || {};
    const mapping = p.mapping || f.mapping_proposal || null;
    const hierarchy = p.hierarchy || f.hierarchy_proposal || null;
    return {
      profile: f,
      route: p.route || f.species || "prose",
      trust: p.trust || f.filename_meta?.trust || "unknown",
      recordsPath: p.records_path ?? f.records_path ?? "",
      mapping: mapping
        ? { node_type: mapping.node_type || "", name_field: mapping.name_field || "",
            type_field: mapping.type_field || null,
            relations: (mapping.relations || []).map((r) => ({ ...r })) }
        : { node_type: "", name_field: (f.fields || [])[0] || "", type_field: null, relations: [] },
      hierarchy: hierarchy
        ? { path: hierarchy.path || "", child_field: hierarchy.child_field || "",
            parent_field: hierarchy.parent_field || "", node_type: hierarchy.node_type || "" }
        : null,
      hierarchyOn: !!hierarchy,
    };
  });
}

export default function IngestGate({ api, onComplete }) {
  const { t } = useT();
  const [step, setStep] = useState(1);
  const [files, setFiles] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [gateNs, setGateNs] = useState("");
  const [datasetId, setDatasetId] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [entries, setEntries] = useState([]);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const validateNs = () => {
    const ns = gateNs.trim();
    if (!ns) { setError(t("gate.err.nsEmpty")); return null; }
    if (ns === "default") { setError(t("gate.err.nsDefault")); return null; }
    return ns;
  };

  const analyze = async (dsId) => {
    setAnalyzing(true); setError("");
    try {
      const res = await fetch(`${api}/api/v1/ontology/datasets/${dsId}/analyze`, { method: "POST" });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setAnalysis(data);
      setEntries(initEntries(data));
    } catch (e) { setError(t("gate.err.analyze", { e: e.message || e })); }
    setAnalyzing(false);
  };

  const uploadAndAnalyze = async () => {
    if (!files.length || !validateNs()) return;
    setBusy(true); setError("");
    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const res = await fetch(`${api}/api/v1/ontology/datasets`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const ds = await res.json();
      setDatasetId(ds.dataset_id);
      setStep(2);
      await analyze(ds.dataset_id);
    } catch (e) { setError(t("err.upload", { e: e.message || e })); }
    setBusy(false);
  };

  // ── 엔트리 편집 헬퍼 ──
  const patchEntry = (i, patch) =>
    setEntries((prev) => prev.map((e, j) => (j === i ? { ...e, ...patch } : e)));
  const patchMapping = (i, patch) =>
    setEntries((prev) => prev.map((e, j) =>
      j === i ? { ...e, mapping: { ...e.mapping, ...patch } } : e));
  const patchHierarchy = (i, patch) =>
    setEntries((prev) => prev.map((e, j) =>
      j === i ? { ...e, hierarchy: { ...(e.hierarchy || {}), ...patch } } : e));
  const patchRelation = (i, ri, patch) =>
    setEntries((prev) => prev.map((e, j) => j === i
      ? { ...e, mapping: { ...e.mapping,
          relations: e.mapping.relations.map((r, k) => (k === ri ? { ...r, ...patch } : r)) } }
      : e));
  const removeRelation = (i, ri) =>
    setEntries((prev) => prev.map((e, j) => j === i
      ? { ...e, mapping: { ...e.mapping, relations: e.mapping.relations.filter((_, k) => k !== ri) } }
      : e));
  const addRelation = (i) =>
    setEntries((prev) => prev.map((e, j) => j === i
      ? { ...e, mapping: { ...e.mapping,
          relations: [...e.mapping.relations, { field: (e.profile.fields || [])[0] || "", predicate: "", target_type: "" }] } }
      : e));

  // 편집된 UI 상태 → PlanEntry 배열 (skip 제외, 그대로 제출)
  const assemblePlan = () =>
    entries.filter((e) => e.route !== "skip").map((e) => {
      const entry = { filename: e.profile.filename, route: e.route, trust: e.trust };
      if (e.route === "records") {
        entry.records_path = e.recordsPath || "";
        entry.mapping = {
          node_type: e.mapping.node_type,
          name_field: e.mapping.name_field,
          type_field: e.mapping.type_field || null,
          relations: e.mapping.relations.filter((r) => r.field && r.predicate),
        };
        entry.hierarchy = e.hierarchyOn && e.hierarchy ? e.hierarchy : null;
      }
      return entry;
    });

  const totalCalls = entries
    .filter((e) => e.route !== "skip")
    .reduce((s, e) => s + (e.profile.estimated_llm_calls || 0), 0);

  const startIngest = async () => {
    const ns = validateNs();
    if (!ns) return;
    setBusy(true); setError(""); setJob(null);
    try {
      const res = await fetch(`${api}/api/v1/ontology/datasets/${datasetId}/ingest`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ namespace: ns, plan: assemblePlan(), save: true, schema_mode: "auto" }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const { job_id } = await res.json();
      setStep(3);
      pollRef.current = setInterval(async () => {
        try {
          const j = await fetch(`${api}/api/v1/ontology/jobs/${job_id}`).then((r) => r.json());
          setJob(j);
          if (j.status === "completed" || j.status === "failed") {
            clearInterval(pollRef.current);
            setBusy(false);
          }
        } catch { /* transient poll error — keep polling */ }
      }, 1500);
    } catch (e) { setError(t("gate.err.ingest", { e: e.message || e })); setBusy(false); }
  };

  const reset = () => {
    clearInterval(pollRef.current);
    setStep(1); setFiles([]); setDatasetId(null); setAnalysis(null);
    setEntries([]); setJob(null); setBusy(false); setError("");
  };

  // ── Step 1: 업로드 ──
  const step1 = (
    <section className="card">
      <h2>{t("gate.step1")} <span className="hint-inline">{t("gate.formats")}</span></h2>
      <div className={`dropzone ${dragActive ? "active" : ""}`}
           onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
           onDragLeave={() => setDragActive(false)}
           onDrop={(e) => { e.preventDefault(); setDragActive(false); setFiles([...e.dataTransfer.files]); }}
           onClick={() => document.getElementById("gate-file-input").click()}>
        {files.length
          ? t("build.dropzone.files", { n: files.length, names: files.map((f) => f.name).join(", ") })
          : t("build.dropzone")}
      </div>
      <input id="gate-file-input" type="file" multiple accept={ACCEPT_EXTS}
             style={{ display: "none" }} onChange={(e) => setFiles([...e.target.files])} />
      <div className="row" style={{ marginTop: 14 }}>
        <div style={{ flex: 2 }}>
          <label>{t("gate.namespace")}</label>
          <input type="text" value={gateNs} placeholder={t("gate.namespace.ph")}
                 onChange={(e) => setGateNs(e.target.value)} />
        </div>
        <div style={{ flex: 1, display: "flex", alignItems: "flex-end", paddingBottom: 2 }}>
          <button onClick={uploadAndAnalyze} disabled={!files.length || busy}>
            {busy && <span className="spinner" />}
            {busy ? t("gate.analyzing") : t("gate.uploadAnalyze")}
          </button>
        </div>
      </div>
      <p className="hint">{t("gate.step1.hint")}</p>
    </section>
  );

  // ── Step 2: 확인 게이트 ──
  const fileCard = (e, i) => {
    const f = e.profile;
    const fields = f.fields || [];
    const isRecords = e.route === "records";
    const meta = f.filename_meta;
    return (
      <section className="card" key={f.filename}
               style={e.route === "skip" ? { opacity: 0.55 } : undefined}>
        {/* 헤더 */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <b style={{ fontSize: "0.94rem" }}>{f.filename}</b>
          <span style={pill(SPECIES_COLORS[f.species])}>{f.species}</span>
          <span className="hint-inline">{f.format}</span>
          {meta && (
            <span className="hint-inline">
              {[meta.doc_kind, meta.entity, meta.version].filter(Boolean).join(" · ")}
            </span>
          )}
        </div>
        {f.error && <div className="error" style={{ marginTop: 8 }}>{f.error}</div>}

        {/* trust / route */}
        <div className="row" style={{ marginTop: 12 }}>
          <div>
            <label>{t("gate.trust")}</label>
            <select value={e.trust || "unknown"} style={smallInput}
                    onChange={(ev) => patchEntry(i, { trust: ev.target.value })}>
              <option value="authoritative">{t("gate.trust.authoritative")}</option>
              <option value="unknown">{t("gate.trust.unknown")}</option>
              <option value="summary">{t("gate.trust.summary")}</option>
            </select>
            <div className="hint">{t("gate.trust.hint")}</div>
          </div>
          <div>
            <label>{t("gate.route")}</label>
            <select value={e.route} style={smallInput}
                    onChange={(ev) => patchEntry(i, { route: ev.target.value })}>
              <option value="records">{t("gate.route.records")}</option>
              <option value="articled">{t("gate.route.articled")}</option>
              <option value="prose">{t("gate.route.prose")}</option>
              <option value="seed_ontology">{t("gate.route.seed")}</option>
              <option value="skip">{t("gate.route.skip")}</option>
            </select>
          </div>
          <div style={{ display: "flex", alignItems: "flex-end", paddingBottom: 6,
                        color: "var(--muted)", fontSize: "0.8rem" }}>
            {isRecords
              ? t("gate.records", { n: f.record_count })
              : t("gate.chunks", { n: f.chunk_count, c: f.estimated_llm_calls })}
          </div>
        </div>

        {/* records: 필드 표 + 매핑 편집 */}
        {isRecords && e.route !== "skip" && (
          <>
            {fields.length > 0 && (
              <table>
                <thead><tr>
                  <th>{t("gate.th.field")}</th><th>{t("gate.th.stats")}</th><th>{t("gate.th.role")}</th>
                </tr></thead>
                <tbody>
                  {fields.map((fd) => {
                    const st = f.field_stats?.[fd] || {};
                    const role = f.field_roles?.[fd] || "attribute";
                    return (
                      <tr key={fd}>
                        <td className="mono">{fd}</td>
                        <td className="mono">{st.distinct ?? "—"} / {st.present ?? "—"}</td>
                        <td><span style={pill(ROLE_COLORS[role])}>{role}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            {f.mapping_error && !f.mapping_proposal && (
              <div className="error" style={{ marginTop: 10 }}>
                {t("gate.mapping.error", { e: f.mapping_error })}
              </div>
            )}
            <h4 style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--faint)",
                         margin: "14px 0 6px", letterSpacing: "0.4px" }}>
              {t("gate.mapping")}
            </h4>
            <div className="row">
              <div>
                <label>{t("gate.mapping.nodeType")}</label>
                <input type="text" style={smallInput} value={e.mapping.node_type}
                       onChange={(ev) => patchMapping(i, { node_type: ev.target.value })} />
              </div>
              <div>
                <label>{t("gate.mapping.nameField")}</label>
                <select style={smallInput} value={e.mapping.name_field}
                        onChange={(ev) => patchMapping(i, { name_field: ev.target.value })}>
                  {!fields.includes(e.mapping.name_field) && (
                    <option value={e.mapping.name_field}>{e.mapping.name_field || "—"}</option>)}
                  {fields.map((fd) => <option key={fd} value={fd}>{fd}</option>)}
                </select>
              </div>
              <div>
                <label>{t("gate.mapping.typeField")}</label>
                <select style={smallInput} value={e.mapping.type_field || ""}
                        onChange={(ev) => patchMapping(i, { type_field: ev.target.value || null })}>
                  <option value="">{t("gate.mapping.none")}</option>
                  {fields.map((fd) => <option key={fd} value={fd}>{fd}</option>)}
                </select>
              </div>
            </div>
            <label style={{ marginTop: 12 }}>{t("gate.mapping.relations")}</label>
            {e.mapping.relations.map((r, ri) => (
              <div key={ri} style={{ display: "flex", gap: 8, marginBottom: 6, alignItems: "center" }}>
                <select style={{ ...smallInput, flex: 1 }} value={r.field}
                        onChange={(ev) => patchRelation(i, ri, { field: ev.target.value })}>
                  {!fields.includes(r.field) && <option value={r.field}>{r.field || "—"}</option>}
                  {fields.map((fd) => <option key={fd} value={fd}>{fd}</option>)}
                </select>
                <input type="text" style={{ ...smallInput, flex: 1 }} value={r.predicate}
                       placeholder={t("gate.mapping.predicate")}
                       onChange={(ev) => patchRelation(i, ri, { predicate: ev.target.value })} />
                <input type="text" style={{ ...smallInput, flex: 1 }} value={r.target_type}
                       placeholder={t("gate.mapping.targetType")}
                       onChange={(ev) => patchRelation(i, ri, { target_type: ev.target.value })} />
                <button className="ghost" style={smallBtn}
                        onClick={() => removeRelation(i, ri)}>✕</button>
              </div>
            ))}
            <button className="ghost" style={smallBtn} onClick={() => addRelation(i)}>
              {t("gate.mapping.addRel")}
            </button>

            {/* 계층 제안 */}
            {(e.hierarchy || f.hierarchy_proposal) && (
              <div style={{ marginTop: 14, padding: "10px 12px", border: "1px solid var(--border)",
                            borderRadius: 8, background: "var(--accent-soft, #f6f7ff)" }}>
                <label style={{ fontWeight: 600, marginBottom: 8 }}>
                  <input type="checkbox" checked={e.hierarchyOn}
                         onChange={(ev) => patchEntry(i, {
                           hierarchyOn: ev.target.checked,
                           hierarchy: e.hierarchy || {
                             path: f.hierarchy_proposal?.path || "",
                             child_field: f.hierarchy_proposal?.child_field || "",
                             parent_field: f.hierarchy_proposal?.parent_field || "",
                             node_type: f.hierarchy_proposal?.node_type || "" },
                         })}
                         style={{ width: "auto", marginRight: 6 }} />
                  {t("gate.hierarchy.use")} <span className="hint-inline">{t("gate.hierarchy")}</span>
                </label>
                {e.hierarchyOn && e.hierarchy && (
                  <div className="row" style={{ marginTop: 8 }}>
                    {[["path", "gate.hierarchy.path"], ["child_field", "gate.hierarchy.childField"],
                      ["parent_field", "gate.hierarchy.parentField"], ["node_type", "gate.hierarchy.nodeType"]]
                      .map(([k, label]) => (
                        <div key={k}>
                          <label>{t(label)}</label>
                          <input type="text" style={smallInput} value={e.hierarchy[k] || ""}
                                 onChange={(ev) => patchHierarchy(i, { [k]: ev.target.value })} />
                        </div>
                      ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </section>
    );
  };

  const step2 = (
    <>
      {analyzing ? (
        <section className="card" style={{ display: "flex", alignItems: "center", gap: 10,
                                           color: "var(--muted)" }}>
          <span className="spinner" />{t("gate.analyzing")}
        </section>
      ) : analysis && (
        <>
          <section className="card soft" style={{ display: "flex", alignItems: "center",
                                                  gap: 14, flexWrap: "wrap" }}>
            <b>{t("gate.step2")}</b>
            <span style={pill("#4f46e5")}>{t("gate.cost", { n: totalCalls })}</span>
            <span className="hint-inline">{datasetId} → <b>{gateNs.trim()}</b></span>
          </section>
          {entries.map(fileCard)}
          <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
            <button className="ghost" disabled={busy || analyzing}
                    onClick={() => analyze(datasetId)}>{t("gate.reanalyze")}</button>
            <button onClick={startIngest}
                    disabled={busy || !entries.some((e) => e.route !== "skip")}>
              {t("gate.build")}
            </button>
          </div>
        </>
      )}
    </>
  );

  // ── Step 3: 실행 ──
  const step3 = (
    <section className="card">
      <h2>{t("gate.step3")}</h2>
      <div style={{ fontSize: "0.86rem", marginTop: 8 }}>
        {(!job || job.status === "queued" || job.status === "running") && <span className="spinner" />}
        {t("build.status")}{" "}
        <span className={`status-${job?.status || "running"}`}>{job?.status || "queued"}</span>
        {job?.status === "running" && job.progress?.stage && (
          <span style={{ marginLeft: 10, color: "var(--muted)" }}>
            {job.progress.stage} {job.progress.source || ""}
          </span>)}
      </div>
      {job?.report?.files?.length > 0 && (
        <table>
          <thead><tr>
            <th>{t("gate.th.file")}</th><th>{t("gate.th.route")}</th>
            <th>{t("gate.th.entities")}</th><th>{t("gate.th.relations")}</th><th>{t("gate.th.error")}</th>
          </tr></thead>
          <tbody>
            {job.report.files.map((fr) => (
              <tr key={fr.filename}>
                <td>{fr.filename}</td>
                <td><span style={pill(SPECIES_COLORS[fr.route])}>{fr.route}</span></td>
                <td className="mono">+{fr.entities_added}</td>
                <td className="mono">+{fr.relations_added}</td>
                <td className="error" style={{ fontSize: "0.76rem" }}>{fr.error || ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {job?.status === "failed" && (
        <div className="error" style={{ marginTop: 12 }}>
          {t("gate.failed")}{job.error ? ` — ${job.error}` : ""}
        </div>
      )}
      {job?.status === "completed" && (
        <div style={{ marginTop: 14 }}>
          <span className="status-completed">{t("gate.done")}</span>
          <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
            <button onClick={() => onComplete?.(gateNs.trim())}>{t("gate.viewGraph")}</button>
            <button className="ghost" onClick={reset}>{t("gate.newBuild")}</button>
          </div>
        </div>
      )}
      {job?.status === "failed" && (
        <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
          <button className="ghost" onClick={() => setStep(2)}>{t("gate.backToConfirm")}</button>
          <button className="ghost" onClick={reset}>{t("gate.newBuild")}</button>
        </div>
      )}
    </section>
  );

  return (
    <>
      {error && <div className="error card soft" style={{ marginBottom: 14 }}>{error}</div>}
      {step === 1 && step1}
      {step === 2 && step2}
      {step === 3 && step3}
    </>
  );
}
