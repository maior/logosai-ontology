"use client";

// 스키마 탭 — GET /graphs/{ns}/schema 한 번으로 관측된 스키마를 보여준다.
// 클래스(타입별 인스턴스·프로퍼티 사용) · 술어(관측 시그니처) · is_a 계층 트리.
// 데이터 값(타입명·술어)은 번역하지 않는다 — i18n 규약.

import { useEffect, useState } from "react";
import { useT } from "../i18n";

const TOP_PROPS = 8;

const detailMsg = async (res) => {
  try {
    const d = (await res.json()).detail;
    if (typeof d === "string" && d) return d;
    if (d) return JSON.stringify(d);
  } catch { /* 본문이 JSON 이 아니면 statusText 폴백 */ }
  return res.statusText;
};

function PropChips({ properties }) {
  const entries = Object.entries(properties || {}).sort((a, b) => b[1] - a[1]);
  const shown = entries.slice(0, TOP_PROPS);
  const rest = entries.length - shown.length;
  return (
    <>
      {shown.map(([p, c]) => (
        <span key={p} className="alias-chip">
          {p} <b style={{ marginLeft: 4 }}>{c}</b>
        </span>
      ))}
      {rest > 0 && (
        <span className="alias-chip" style={{ borderStyle: "dashed" }}
              title={entries.slice(TOP_PROPS).map(([p, c]) => `${p} ${c}`).join(" · ")}>
          +{rest}
        </span>
      )}
    </>
  );
}

function TreeNode({ node, depth, collapsed, onToggle }) {
  const kids = node.children || [];
  const isCollapsed = !!collapsed[node.id];
  return (
    <div>
      <div className="tree-row" style={{ paddingLeft: depth * 18 }}>
        {kids.length > 0 ? (
          <button className="tree-toggle" onClick={() => onToggle(node.id)}>
            {isCollapsed ? "▸" : "▾"}
          </button>
        ) : (
          <span style={{ width: 20, display: "inline-block" }} />
        )}
        <span>{node.name || node.id}</span>
        {kids.length > 0 && (
          <span style={{ fontSize: "0.7rem", color: "var(--faint)" }}>({kids.length})</span>
        )}
      </div>
      {!isCollapsed && kids.map((c) => (
        <TreeNode key={c.id} node={c} depth={depth + 1}
                  collapsed={collapsed} onToggle={onToggle} />
      ))}
    </div>
  );
}

export default function SchemaTab({ api, namespace }) {
  const { t } = useT();
  const [schema, setSchema] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [collapsed, setCollapsed] = useState({});

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true); setError("");
      try {
        const res = await fetch(`${api}/graphs/${namespace}/schema`);
        if (!res.ok) throw new Error(await detailMsg(res));
        const body = await res.json();
        if (alive) setSchema(body);
      } catch (e) {
        if (alive) setError(t("admin.schema.err", { e: e.message || e }));
      }
      if (alive) setLoading(false);
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, namespace]);

  const onToggle = (id) => setCollapsed((c) => ({ ...c, [id]: !c[id] }));

  if (error) return <div className="error" style={{ marginTop: 18 }}>{error}</div>;
  if (loading || !schema) {
    return (
      <div className="empty-state" style={{ marginTop: 18 }}>
        <span className="spinner" />{t("admin.loading")}
      </div>
    );
  }

  const classes = schema.classes || [];
  const predicates = schema.predicates || [];

  return (
    <>
      {/* 클래스 — 인스턴스 수 내림차순 (백엔드 정렬 계약) */}
      <section className="card">
        <h2>{t("admin.schema.classes.title")}
          <span className="badge" style={{ marginLeft: 8 }}>{classes.length}</span></h2>
        <p className="hint" style={{ marginTop: 2 }}>{t("admin.schema.classes.hint")}</p>
        {classes.length === 0
          ? <p className="hint" style={{ marginTop: 10 }}>{t("admin.schema.empty")}</p>
          : <table>
              <thead><tr>
                <th>{t("admin.schema.th.class")}</th>
                <th style={{ textAlign: "right", width: 100 }}>{t("admin.schema.th.count")}</th>
                <th>{t("admin.schema.th.props")}</th>
              </tr></thead>
              <tbody>
                {classes.map((c) => (
                  <tr key={c.type}>
                    <td><b>{c.type}</b></td>
                    <td className="mono" style={{ textAlign: "right" }}>
                      {(c.count || 0).toLocaleString()}</td>
                    <td><PropChips properties={c.properties} /></td>
                  </tr>
                ))}
              </tbody>
            </table>}
      </section>

      {/* 술어 — 관측 시그니처 (source_type → target_type) */}
      <section className="card">
        <h2>{t("admin.schema.predicates.title")}
          <span className="badge" style={{ marginLeft: 8 }}>{predicates.length}</span></h2>
        <p className="hint" style={{ marginTop: 2 }}>{t("admin.schema.predicates.hint")}</p>
        {predicates.length === 0
          ? <p className="hint" style={{ marginTop: 10 }}>{t("admin.schema.empty")}</p>
          : <table>
              <thead><tr>
                <th>{t("admin.schema.th.predicate")}</th>
                <th style={{ textAlign: "right", width: 100 }}>{t("admin.schema.th.uses")}</th>
                <th>{t("admin.schema.th.signatures")}</th>
              </tr></thead>
              <tbody>
                {predicates.map((p) => (
                  <tr key={p.predicate}>
                    <td className="mono">{p.predicate}</td>
                    <td className="mono" style={{ textAlign: "right" }}>
                      {(p.count || 0).toLocaleString()}</td>
                    <td>
                      {(p.pairs || []).map((pair, i) => (
                        <span key={i} className="alias-chip" style={{ borderStyle: "dashed" }}>
                          {pair.source_type} → {pair.target_type}
                          <b style={{ marginLeft: 4 }}>×{pair.count}</b>
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>}
      </section>

      {/* is_a 계층 트리 */}
      <section className="card">
        <h2>{t("admin.schema.hierarchy.title")}
          <span className="badge" style={{ marginLeft: 8 }}>{schema.hierarchy_edges || 0}</span></h2>
        {!schema.hierarchy_edges
          ? <p className="hint" style={{ marginTop: 10 }}>{t("admin.schema.hierarchy.empty")}</p>
          : <>
              <p className="hint" style={{ marginTop: 2 }}>{t("admin.schema.hierarchy.hint")}</p>
              <div style={{ marginTop: 10 }}>
                {(schema.hierarchy || []).map((root) => (
                  <TreeNode key={root.id} node={root} depth={0}
                            collapsed={collapsed} onToggle={onToggle} />
                ))}
              </div>
            </>}
      </section>
    </>
  );
}
