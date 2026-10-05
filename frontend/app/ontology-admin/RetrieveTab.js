"use client";

// 검색 Explain — 그래프-조건부 검색(/retrieve)을 "왜 이 결과인지"까지 펼쳐 보이는 탭.
//
// 여기가 온톨로지의 색깔이 가장 진하게 드러나는 화면이다. 일반 검색 UI 는 결과 목록을
// 주인공으로 두지만, 사실의 층에서 주인공은 **근거 사슬** 세 층이다:
//   ① 확장  — 질의가 어떤 개념을 거쳐 어떻게 넓어졌나 (진입 노드 · is_a/인접 · 확장어)
//   ② 채널  — 각 히트가 청크(임베딩·BM25)로 왔나, 그래프(개념)로 왔나, 둘 다인가
//   ③ 인용  — 원문의 어디인가 (제N조 · 출처 · 문자 오프셋 · trust)
// 이 셋이 보이지 않으면 결과를 신뢰할 근거가 없다 (router.py:660-663 의 계약).
//
// 백엔드 변경 0 — /retrieve 가 이미 돌려주는 것을 화면에 꺼내 놓을 뿐이다.

import { useCallback, useMemo, useState } from "react";
import {
  normalizeRetrieve, channelBadges, citationLabel,
  offsetLabel, maxScore, scoreBarPct, viaLabel,
  buildRetrieveGraph,
} from "./retrieveView.mjs";

/* 질의 중심 서브그래프 — /retrieve 응답이 이미 들고 있는 연결을 그래프로 편다.
   탐색 탭의 개체-개체 엣지는 성기지만(실측 고립 62.9%였다) 연결의 실체는
   근거 링크(노드↔청크)가 나른다 — 그게 이 그림이다.

   5단 고정 레이아웃(물리 시뮬레이션 없음 — 결정적이어야 같은 질의를 두 번
   보면 같은 그림이 나온다):
     질의 → 진입 노드 → 확장 노드 → 근거 청크 → 문서(파일)
   진입과 확장을 열로 갈라 확장 hop 이 실제 단계로 보이게 했고, 끝단에 문서
   층을 둬 근거가 결국 어느 파일에서 왔는지가 그림에서 끝난다.
   청크를 클릭하면 원문 상세(onSelectChunk)가 아래 패널로 열린다. */
function QueryGraph({ graph, selected, onSelectChunk }) {
  const [hover, setHover] = useState(null);
  const ROW = 34, TOP = 26;
  const X = { query: 68, entry: 278, expanded: 492, chunk: 716, doc: 1032 };
  const PILL = 75;   // 노드 pill 반너비
  const CW = 85;     // 청크 rect 반너비
  const DW = 105;    // 문서 rect 반너비

  const entries = graph.nodes.filter((n) => n.kind === "entry");
  const expandeds = graph.nodes.filter((n) => n.kind === "expanded");
  const yMap = (list, key) => new Map(list.map((v, i) => [key(v), TOP + i * ROW]));
  const entryY = yMap(entries, (n) => n.id);
  const expandedY = yMap(expandeds, (n) => n.id);
  const chunkY = yMap(graph.chunks, (c) => c.chunkId);
  const docY = yMap(graph.documents, (d) => d.source);
  const nodeY = (id) => entryY.get(id) ?? expandedY.get(id);
  // 근거 엣지의 출발 x 는 그 노드가 어느 열에 있느냐에 달렸다.
  const nodeRightX = (id) => (entryY.has(id) ? X.entry : X.expanded) + PILL;

  const rows = Math.max(entries.length, expandeds.length,
                        graph.chunks.length, graph.documents.length, 1);
  const H = TOP + rows * ROW + 14;
  const queryY = TOP + ((rows - 1) * ROW) / 2;
  const curve = (x1, y1, x2, y2) =>
    `M${x1},${y1} C${x1 + (x2 - x1) * 0.45},${y1} ${x2 - (x2 - x1) * 0.45},${y2} ${x2},${y2}`;
  // hover 시 관련 요소만 진하게 — 왜 연결됐는지 따라가 볼 수 있게.
  const dim = (ids) => (hover && !ids.includes(hover) ? 0.12 : 0.85);
  const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

  return (
    <svg viewBox={`0 0 1160 ${H}`} style={{ width: "100%", height: "auto", minWidth: 900 }}
         role="img" aria-label="질의 중심 연결 그래프">
      {/* 열 제목 */}
      <text x={X.query} y={12} textAnchor="middle" fontSize="10" fill="#8a94a8">질의</text>
      <text x={X.entry} y={12} textAnchor="middle" fontSize="10" fill="#8a94a8">진입 노드 (임베딩)</text>
      <text x={X.expanded} y={12} textAnchor="middle" fontSize="10" fill="#8a94a8">확장 노드 (온톨로지)</text>
      <text x={X.chunk} y={12} textAnchor="middle" fontSize="10" fill="#8a94a8">근거 청크 (클릭=원문)</text>
      <text x={X.doc} y={12} textAnchor="middle" fontSize="10" fill="#8a94a8">문서 (파일)</text>

      {/* 질의 → 진입 */}
      {entries.map((n) => (
        <path key={`q-${n.id}`} d={curve(X.query + 60, queryY, X.entry - PILL, entryY.get(n.id))}
              fill="none" stroke="#a5aec2" strokeWidth="1.4" opacity={dim([n.id])} />
      ))}
      {/* 확장 엣지: 진입 열 → 확장 열 (via 라벨) */}
      {graph.edges.expansion.map((e, i) => {
        const y1 = nodeY(e.from), y2 = expandedY.get(e.to);
        if (y1 == null || y2 == null) return null;
        return (
          <g key={`x-${i}`} opacity={dim([e.from, e.to])}>
            <path d={curve(nodeRightX(e.from), y1, X.expanded - PILL, y2)}
                  fill="none" stroke="#7c3aed" strokeWidth="1.3"
                  strokeDasharray={/sameAs|propagation/.test(e.via) ? "4 4" : "none"} />
            <text x={(X.entry + X.expanded) / 2} y={(y1 + y2) / 2 - 4}
                  textAnchor="middle" fontSize="9" fill="#7c3aed">{e.via}</text>
          </g>
        );
      })}
      {/* 근거 엣지: 노드(진입/확장 어느 열이든) → 청크 */}
      {graph.edges.evidence.map((e, i) => {
        const y1 = nodeY(e.node), y2 = chunkY.get(e.chunk);
        if (y1 == null || y2 == null) return null;
        return <path key={`e-${i}`} d={curve(nodeRightX(e.node), y1, X.chunk - CW, y2)}
                     fill="none" stroke="#1baf7a" strokeWidth="1.3"
                     opacity={dim([e.node, e.chunk])} />;
      })}
      {/* 임베딩 단독 청크: 질의에서 점선 (그래프 경유 아님을 구분) */}
      {graph.edges.direct.map((e, i) => {
        const y2 = chunkY.get(e.chunk);
        if (y2 == null) return null;
        return <path key={`d-${i}`} d={curve(X.query + 60, queryY, X.chunk - CW, y2)}
                     fill="none" stroke="#a5aec2" strokeWidth="1.2" strokeDasharray="4 4"
                     opacity={dim([e.chunk])} />;
      })}
      {/* 청크 → 문서: 근거가 결국 어느 파일에서 왔는가 */}
      {graph.chunks.map((c) => {
        const y1 = chunkY.get(c.chunkId), y2 = docY.get(c.source);
        if (y1 == null || y2 == null) return null;
        return <path key={`f-${c.chunkId}`} d={curve(X.chunk + CW, y1, X.doc - DW, y2)}
                     fill="none" stroke="#c3cad8" strokeWidth="1.1"
                     opacity={dim([c.chunkId, c.source])} />;
      })}

      {/* 질의 박스 */}
      <g>
        <rect x={X.query - 60} y={queryY - 13} width={120} height={26} rx={13}
              fill="#1e293b" />
        <text x={X.query} y={queryY + 4} textAnchor="middle" fontSize="11" fill="#e2e8f0">
          {trunc(graph.query, 11)}</text>
      </g>
      {/* 노드 (진입 열 + 확장 열) */}
      {graph.nodes.map((n) => {
        const cx = n.kind === "entry" ? X.entry : X.expanded;
        const cy = nodeY(n.id);
        return (
          <g key={n.id} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}
             style={{ cursor: "default" }} opacity={dim([n.id])}>
            <rect x={cx - PILL} y={cy - 11} width={PILL * 2} height={22} rx={11}
                  fill={n.kind === "entry" ? "#2a78d6" : "#fff"}
                  stroke="#2a78d6" strokeWidth={n.kind === "entry" ? 0 : 1.6} />
            <text x={cx} y={cy + 4} textAnchor="middle" fontSize="10.5"
                  fill={n.kind === "entry" ? "#fff" : "#2a78d6"}>
              {trunc(n.label, 14)}{n.kind === "entry" && n.score ? ` ${n.score.toFixed(2)}` : ""}
            </text>
            <title>{n.id}{n.via ? ` (via ${n.via})` : ""}</title>
          </g>
        );
      })}
      {/* 청크 — 클릭하면 아래 원문 패널이 열린다 */}
      {graph.chunks.map((c) => {
        const isSel = selected === c.chunkId;
        return (
          <g key={c.chunkId} onMouseEnter={() => setHover(c.chunkId)}
             onMouseLeave={() => setHover(null)}
             onClick={() => onSelectChunk && onSelectChunk(isSel ? null : c.chunkId)}
             style={{ cursor: "pointer" }} opacity={dim([c.chunkId, c.source])}>
            <rect x={X.chunk - CW} y={chunkY.get(c.chunkId) - 11} width={CW * 2} height={22} rx={5}
                  fill={isSel ? "#1baf7a" : "#e4f7f0"} stroke="#1baf7a"
                  strokeWidth={isSel ? 2.2 : 1.4} />
            <text x={X.chunk - CW + 8} y={chunkY.get(c.chunkId) + 4} fontSize="10"
                  fill={isSel ? "#fff" : "#1a2233"}>
              {c.rank}. {trunc(c.label, 20)}</text>
            <title>{c.source}{c.section ? ` §${c.section}` : ""} — 클릭하면 원문 표시</title>
          </g>
        );
      })}
      {/* 문서(파일) — 끝단 */}
      {graph.documents.map((d) => (
        <g key={d.source} onMouseEnter={() => setHover(d.source)}
           onMouseLeave={() => setHover(null)}
           opacity={dim([d.source,
                         ...graph.chunks.filter((c) => c.source === d.source)
                           .map((c) => c.chunkId)])}>
          <rect x={X.doc - DW} y={docY.get(d.source) - 12} width={DW * 2} height={24} rx={5}
                fill="#f4f0e6" stroke="#eb6834" strokeWidth="1.4" />
          <text x={X.doc - DW + 8} y={docY.get(d.source) + 4} fontSize="10" fill="#1a2233">
            📄 {trunc(d.source, 18)} ({d.count})</text>
          <title>{d.source} — 표시된 근거 청크 {d.count}개</title>
        </g>
      ))}
    </svg>
  );
}

const TOP_K_CHOICES = [5, 10, 20];
const EXCERPT_LIMIT = 220;   // 발췌 클램프 — 조문 전문은 토글로

// 채널 배지 색: 청크=벡터/키워드 계열, 그래프=개념 계열. 두 개 다면 융합.
const CH_CLASS = { chunk: "chunk", graph: "graph" };

export default function RetrieveTab({ t, api, namespace }) {
  const [q, setQ] = useState("");
  const [topK, setTopK] = useState(10);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [open, setOpen] = useState({});   // chunk_id → 전문 펼침
  const [saved, setSaved] = useState({}); // node_id → ok|dup|err (골든 케이스 등록 결과)
  const [selChunk, setSelChunk] = useState(null); // 연결 그래프에서 클릭한 청크 (원문 패널)

  const run = useCallback(async () => {
    const query = q.trim();
    if (!query || !namespace) return;   // 빈 질의는 400 — 아예 부르지 않는다
    setBusy(true); setErr("");
    try {
      const url = `${api}/graphs/${encodeURIComponent(namespace)}/retrieve`
        + `?query=${encodeURIComponent(query)}&top_k=${topK}`;
      const r = await fetch(url);
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { const j = await r.json(); if (j?.detail) detail = j.detail; } catch { /* 본문 없음 */ }
        setErr(detail); setData(null);
      } else {
        setData(await r.json());
        setOpen({});
        setSelChunk(null);   // 새 검색이면 이전 검색의 원문 패널을 남기지 않는다
      }
    } catch (e) {
      setErr(String(e?.message || e)); setData(null);
    } finally {
      setBusy(false);
    }
  }, [api, namespace, q, topK]);

  // 검색 → 라벨 루프: 히트에 달린 노드를 "이 질의의 정답"으로 골든셋에 등록한다.
  // 골든셋을 빈 화면에서 상상해 채우는 것보다, 실제 검색 결과를 보며 확정하는 편이
  // 정답의 근거가 분명하다(등록 시 질의는 방금 검색한 그 질의).
  // body 는 노드 라벨({expected_node_id}) 또는 청크 라벨({expected_chunk_id}).
  // 청크 라벨이 중요한 이유: 노드 채점은 두 채널이 구조상 같게 나와 튜닝 신호가
  // 없고, 그래프 조건화의 효과는 **청크 회수**에서 드러난다(실측: 같은 질의에
  // 겹침 0/5 까지 갈린다). 그 자리를 재려면 정답도 청크여야 한다.
  const addCase = useCallback(async (key, body) => {
    const query = q.trim();
    if (!query || !key) return;
    try {
      const r = await fetch(`${api}/graphs/${encodeURIComponent(namespace)}/qa/cases`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, ...body }),
      });
      // 409 = 같은 질의가 이미 있다 — 실패가 아니라 "이미 라벨됨"이다.
      setSaved((s) => ({ ...s, [key]: r.ok ? "ok" : (r.status === 409 ? "dup" : "err") }));
    } catch {
      setSaved((s) => ({ ...s, [key]: "err" }));
    }
  }, [api, namespace, q]);

  const view = useMemo(() => (data ? normalizeRetrieve(data) : null), [data]);
  const graphView = useMemo(() => (data ? buildRetrieveGraph(data) : null), [data]);
  const top = useMemo(() => (view ? maxScore(view.hits) : 0), [view]);

  const onKey = (e) => { if (e.key === "Enter") run(); };

  return (
    <>
      {/* ─── 질의 바 ─── */}
      <section className="card">
        <h2>{t("admin.rt.title")}</h2>
        <p className="hint" style={{ marginTop: 2 }}>{t("admin.rt.hint")}</p>

        <div className="oa-rt-bar">
          <input className="oa-rt-input" value={q} onChange={(e) => setQ(e.target.value)}
                 onKeyDown={onKey} placeholder={t("admin.rt.placeholder")}
                 aria-label={t("admin.rt.title")} />
          <div className="oa-rt-k" role="group" aria-label="top_k">
            {TOP_K_CHOICES.map((n) => (
              <button key={n} className={`oa-rt-kbtn ${topK === n ? "on" : ""}`}
                      onClick={() => setTopK(n)}>{n}</button>
            ))}
          </div>
          <button disabled={busy || !q.trim()} onClick={run}>
            {busy ? t("admin.rt.running") : t("admin.rt.run")}
          </button>
        </div>

        {err && <p className="oa-rt-err">⚠ {err}</p>}

        {/* 측정된 품질 — 지어내지 않는다. RRF·코사인은 보정 안 된 값이라
            "신뢰도"로 포장할 수 없고, 보여줄 수 있는 것은 골든셋 측정치 +
            측정 시점 + stale(측정 후 설정 변경) 뿐이다. */}
        {data?.quality && (
          data.quality.measured ? (
            <p className="hint" style={{ marginTop: 8 }}>
              측정된 품질 (골든셋 {data.quality.cases}건, {String(data.quality.measured_at).replace("T", " ")}):
              {" "}hit@1 <b>{(data.quality.retrieve?.["hit@1"] ?? 0).toFixed(3)}</b>
              {" "}· hit@{data.quality.k} <b>{(data.quality.retrieve?.[`hit@${data.quality.k}`] ?? 0).toFixed(3)}</b>
              {" "}· MRR <b>{(data.quality.retrieve?.mrr ?? 0).toFixed(3)}</b>
              {data.quality.stale && (
                <span style={{ color: "var(--warn, #b45309)" }}>
                  {" "}⚠ 측정 후 설정 변경됨: {data.quality.stale_keys.join(", ")} — 재평가 필요</span>
              )}
            </p>
          ) : (
            <p className="hint" style={{ marginTop: 8 }}>⚠ {data.quality.reason}</p>
          )
        )}
      </section>

      {/* ─── ① 확장 — 왜 이 질의로 찾았나 ─── */}
      {view && (
        <section className="card">
          <h2>{t("admin.rt.expansion")}</h2>
          <p className="hint" style={{ marginTop: 2 }}>{t("admin.rt.expansionHint")}</p>

          <div className="oa-rt-qflow">
            <div className="oa-rt-qbox">
              <div className="oa-rt-qlbl">{t("admin.rt.original")}</div>
              <div className="oa-rt-qval mono">{view.query || "—"}</div>
            </div>
            <div className="oa-rt-qarrow">→</div>
            <div className="oa-rt-qbox grow">
              <div className="oa-rt-qlbl">{t("admin.rt.expanded")}</div>
              <div className="oa-rt-qval mono">{view.expandedQuery || "—"}</div>
            </div>
          </div>

          {view.terms.length > 0 ? (
            <div className="oa-rt-terms">
              <span className="oa-rt-termlbl">{t("admin.rt.addedTerms")}</span>
              {view.terms.map((tm) => (
                <span key={tm} className="oa-rt-term">{tm}</span>
              ))}
            </div>
          ) : (
            <p className="hint" style={{ marginTop: 8 }}>{t("admin.rt.noTerms")}</p>
          )}

          <div className="oa-rt-nodes">
            <div className="oa-rt-nodecol">
              <div className="oa-rt-nodehd">
                {t("admin.rt.entryNodes")}
                <span className="badge" style={{ marginLeft: 6 }}>{view.entryNodes.length}</span>
              </div>
              {view.entryNodes.length === 0
                ? <p className="hint">{t("admin.rt.none")}</p>
                : view.entryNodes.map((n, i) => (
                    <div className="oa-rt-node" key={`${n.node_id}-${i}`}>
                      <span className="oa-rt-nodeid mono">{n.node_id}</span>
                      {Number.isFinite(Number(n.score)) &&
                        <span className="oa-rt-nodescore mono">{Number(n.score).toFixed(3)}</span>}
                    </div>
                  ))}
            </div>
            <div className="oa-rt-nodecol">
              <div className="oa-rt-nodehd">
                {t("admin.rt.expandedNodes")}
                <span className="badge" style={{ marginLeft: 6 }}>{view.expandedNodes.length}</span>
              </div>
              {view.expandedNodes.length === 0
                ? <p className="hint">{t("admin.rt.none")}</p>
                : view.expandedNodes.map((n, i) => (
                    <div className="oa-rt-node" key={`${n.node_id}-${i}`}>
                      <span className="oa-rt-nodeid mono">{n.name || n.node_id}</span>
                      {n.via && <span className="oa-rt-via">{n.via}</span>}
                    </div>
                  ))}
            </div>
          </div>
        </section>
      )}

      {/* ─── ①.5 질의 중심 연결 그래프 ─── */}
      {view && graphView && (graphView.nodes.length > 0 || graphView.chunks.length > 0) && (
        <section className="card">
          <h2>연결 그래프 — 이 질의가 무엇을 거쳐 근거에 닿았나</h2>
          <p className="hint" style={{ marginTop: 2 }}>
            질의 → 진입 노드(임베딩, 채움+점수) → 확장 노드(온톨로지, 보라 선이 via —
            sameAs·확산은 점선) → 근거 청크(초록 선 = 근거 링크) → 문서(파일).
            회색 점선은 그래프를 거치지 않고 임베딩만으로 걸린 청크.
            마우스를 올리면 연결만 남고, <b>청크를 클릭하면 원문이 아래에 열린다</b>.
            {graphView.dropped.chunks > 0 && ` (청크 ${graphView.dropped.chunks}개는 지면상 생략 — 아래 히트 목록에는 전부 있음)`}
          </p>
          <QueryGraph graph={graphView} selected={selChunk} onSelectChunk={setSelChunk} />
          {/* 클릭한 청크의 원문 — 그래프에서 근거 사슬 끝(원문)까지 한 화면에서 닿게 */}
          {selChunk && (() => {
            const c = graphView.chunks.find((x) => x.chunkId === selChunk);
            if (!c) return null;
            const offset = c.charStart != null && c.charEnd != null
              ? `${c.charStart.toLocaleString("en-US")}–${c.charEnd.toLocaleString("en-US")}` : "";
            return (
              <div style={{ marginTop: 10, padding: "10px 14px", borderRadius: 8,
                            // var(--green-soft): 콘솔은 다크 스코프(.oa)라 하드코딩
                            // 라이트 배경은 다크 잉크(#b7bfd0)와 1.8:1 로 깨진다
                            // (가독성 감사 실측). 다크 토큰 위에서 그린다.
                            border: "1.5px solid #1baf7a", background: "var(--green-soft)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                  <span className="mono" style={{ fontWeight: 700 }}>#{c.rank}</span>
                  <span>📄 {c.source}{c.section ? ` · §${c.section}` : ""}</span>
                  {offset && <span className="hint mono">오프셋 {offset}</span>}
                  {c.score > 0 && <span className="hint mono">score {c.score.toFixed(4)}</span>}
                  {c.channels.map((ch) => (
                    <span key={ch} className={`oa-rt-ch ${CH_CLASS[ch] || ""}`}>{ch}</span>
                  ))}
                  <button className="oa-rt-more" style={{ marginLeft: "auto" }}
                          onClick={() => setSelChunk(null)}>닫기</button>
                </div>
                <p className="oa-rt-text" style={{ marginTop: 8, whiteSpace: "pre-wrap" }}>
                  {c.text || "(이 응답에 원문 텍스트가 없습니다 — 아래 히트 목록을 확인하세요)"}
                </p>
              </div>
            );
          })()}
        </section>
      )}

      {/* ─── ②③ 히트 — 어느 채널로 왔나 + 원문 어디인가 ─── */}
      {view && (
        <section className="card">
          <h2>{t("admin.rt.hits")}
            <span className="badge" style={{ marginLeft: 8 }}>{view.hits.length}</span></h2>
          <p className="hint" style={{ marginTop: 2 }}>{t("admin.rt.hitsHint")}</p>

          {view.hits.length === 0 ? (
            <p className="hint" style={{ marginTop: 10 }}>{t("admin.rt.noHits")}</p>
          ) : view.hits.map((h, i) => {
            const chs = channelBadges(h);
            const off = offsetLabel(h);
            const via = viaLabel(h);
            const text = String(h.text || "");
            const isOpen = !!open[h.chunk_id];
            const clamped = !isOpen && text.length > EXCERPT_LIMIT;
            return (
              <div className="oa-rt-hit" key={h.chunk_id || i}>
                <div className="oa-rt-hithd">
                  <span className="oa-rt-rank mono">#{i + 1}</span>
                  <span className="oa-rt-cite">{citationLabel(h) || "—"}</span>
                  <span className="oa-rt-chs">
                    {chs.map((c) => (
                      <span key={c} className={`oa-rt-ch ${CH_CLASS[c] || ""}`}>{c}</span>
                    ))}
                    {chs.length > 1 && <span className="oa-rt-ch fuse">RRF</span>}
                  </span>
                </div>

                <div className="oa-rt-meta">
                  {/* 그래프 채널이 이 청크를 데려온 노드들 — 우리 색깔의 핵심 근거 */}
                  {via && <span className="oa-rt-m oa-rt-via-nodes" title={via}>
                    {t("admin.rt.via")} <b>{via}</b></span>}
                  {off && <span className="oa-rt-m mono" title={t("admin.rt.offsetTip")}>
                    {t("admin.rt.offset")} {off}</span>}
                  {h.trust && <span className={`oa-rt-trust ${h.trust}`}>{h.trust}</span>}
                  {Array.isArray(h.node_ids) && h.node_ids.length > 0 &&
                    <span className="oa-rt-m">{t("admin.rt.nodes")} {h.node_ids.length}</span>}
                </div>

                {/* 이 청크가 이 질의의 정답 근거인가 — 청크 단위 채점의 재료 */}
                <button className={`oa-rt-golden ${saved[h.chunk_id] || ""}`}
                        title={t("admin.rt.labelChunkTip")}
                        onClick={() => addCase(h.chunk_id, { expected_chunk_id: h.chunk_id })}>
                  {saved[h.chunk_id] === "ok" ? `✓ ${t("admin.rt.labeled")}`
                    : saved[h.chunk_id] === "dup" ? t("admin.rt.alreadyLabeled")
                    : saved[h.chunk_id] === "err" ? `⚠ ${t("admin.rt.labelFail")}`
                    : t("admin.rt.labelChunk")}
                </button>

                <div className="oa-rt-score" title={`score ${h.score}`}>
                  <div className="oa-rt-scorebar" style={{ width: `${scoreBarPct(h, top)}%` }} />
                  <span className="oa-rt-scoreval mono">
                    {Number.isFinite(Number(h.score)) ? Number(h.score).toFixed(4) : "—"}</span>
                </div>

                <p className="oa-rt-text">
                  {clamped ? `${text.slice(0, EXCERPT_LIMIT)}…` : (text || "—")}
                </p>
                {text.length > EXCERPT_LIMIT && (
                  <button className="oa-rt-more"
                          onClick={() => setOpen((o) => ({ ...o, [h.chunk_id]: !isOpen }))}>
                    {isOpen ? t("admin.rt.less") : t("admin.rt.moreBtn")}
                  </button>
                )}

                {Array.isArray(h.node_ids) && h.node_ids.length > 0 && (
                  <div className="oa-rt-nids">
                    <span className="oa-rt-nidlbl">{t("admin.rt.labelAs")}</span>
                    {h.node_ids.map((nid) => (
                      <button key={nid} className={`oa-rt-nid mono ${saved[nid] || ""}`}
                              title={t("admin.rt.labelTip")}
                              onClick={() => addCase(nid, { expected_node_id: nid })}>
                        {nid}
                        {saved[nid] === "ok" && <span className="oa-rt-nidmark"> ✓</span>}
                        {saved[nid] === "dup" && <span className="oa-rt-nidmark"> ·이미</span>}
                        {saved[nid] === "err" && <span className="oa-rt-nidmark"> ⚠</span>}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </section>
      )}
    </>
  );
}
