/**
 * coverageView — /coverage-map 응답을 커버리지 지도 표현으로 바꾸는 순수 함수들.
 *
 * 화면의 질문은 "커버리지 몇 %"가 아니라 **"비어 있는 구간이 어느 절인가"** 다 —
 * 그 답(빈 구간 랭킹)을 여기서 계산한다. 렌더 없이 단위 테스트한다
 * (retrieveView 와 같은 이유 — 이 변환이 곧 판독 규칙이라 회귀가 나면
 * 관리자가 엉뚱한 절을 회복 타깃으로 잡는다).
 *
 * 모든 함수는 **던지지 않는다** — 빈 네임스페이스·필드 누락은 정상 상태다.
 */

const isObj = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const arr = (v) => (Array.isArray(v) ? v : []);

/** /coverage-map 응답 → 안전 정규화된 문서 목록. */
export function buildCoverageDocs(resp) {
  const r = isObj(resp) ? resp : {};
  return arr(r.documents).filter(isObj).map((d) => ({
    source: String(d.source || "?"),
    total: Number(d.total) || 0,
    linked: Number(d.linked) || 0,
    coverage: Number(d.coverage) || 0,
    ordered: d.ordered !== false,   // 명시적 false 만 불신 — 누락은 신뢰
    chunks: arr(d.chunks).filter(isObj).map((c, i) => ({
      chunkId: String(c.chunk_id || ""),
      order: Number(c.order) || i + 1,
      section: String(c.section || "").trim(),
      links: Number(c.links) || 0,
      nodeIds: arr(c.node_ids).map(String),
      textHead: String(c.text_head || ""),
    })),
  }));
}

/**
 * 링크 수 → 채색 버킷. 0 = 미연결(문제 — 사냥 대상) / 1 = 약함(1~2) / 2 = 튼튼(3+).
 * 경계는 실측 습관에서: 근거 1~2개는 노드 하나가 지워지면 다시 고아가 되는 취약 상태.
 */
export function linkBucket(links) {
  const n = Number(links) || 0;
  if (n <= 0) return 0;
  return n <= 2 ? 1 : 2;
}

/** 연속 같은 section 구간 — 스트립의 절 경계 표시용. [{section, start, count}] */
export function sectionSpans(chunks) {
  const out = [];
  for (const c of arr(chunks)) {
    const sec = String(c.section || "");
    const last = out[out.length - 1];
    if (last && last.section === sec) last.count += 1;
    else out.push({ section: sec, start: Number(c.order) || 0, count: 1 });
  }
  return out;
}

/**
 * 절(section)을 지역으로 — "지도"의 구획.
 *
 * 균일 격자는 달력 히트맵으로 읽히고 문서 구조를 잃는다(사용자 피드백:
 * "지도처럼 보이지 않는다"). 지도의 지리는 이 데이터에서 **목차**다 — 연속
 * 같은 절을 하나의 지역(band)으로 묶고, 지역마다 이름·청크·커버리지를 들려
 * 준다. 빈 지역(coverage 0)이 시각적으로 "빈 땅"이 되는 것이 목표.
 *
 * 절 이름 없는 연속 구간도 지역이다(section="") — 빼면 그 청크들이 지도에서
 * 사라져 "전부 어딘가에 속함"이라는 거짓말이 된다. 표시는 컴포넌트 몫.
 */
export function sectionRegions(chunks) {
  const out = [];
  for (const c of arr(chunks)) {
    // 상위 절로 묶는다: "보안요구사항 > 4.2 시큐어 코딩" → "보안요구사항".
    // 실측(PROJ-A RFP): 절 라벨이 청크와 거의 1:1(286/348)이라 원 라벨 그대로는
    // 밴드 286줄 — 지도가 아니라 목차 덤프가 된다. 하위 라벨은 칸 tooltip 이 나른다.
    const sec = String(c.section || "").split(">")[0].trim();
    let last = out[out.length - 1];
    if (!last || last.section !== sec) {
      last = { section: sec, start: Number(c.order) || 0,
               count: 0, linked: 0, coverage: 0, chunks: [] };
      out.push(last);
    }
    last.count += 1;
    if ((Number(c.links) || 0) > 0) last.linked += 1;
    last.chunks.push(c);
  }
  for (const r of out) r.coverage = r.count ? r.linked / r.count : 0;
  return out;
}

/**
 * 작은 연속 지역 병합 — 지도의 축척.
 *
 * 상위 절로 묶어도 코퍼스에 따라(RFP 처럼 평평한 제목 체계) 1~2칸짜리 지역이
 * 수백 개 남는다. 연속한 작은 지역들을 minChunks 에 찰 때까지 하나의 밴드로
 * 합치고, 라벨은 "첫절 외 N절"로 정직하게 만든다(extraSections). 큰 지역은
 * 그대로 — 합치는 것은 축척이지 정보 삭제가 아니다(칸·통계는 전부 보존).
 * 결정적: 같은 입력 = 같은 병합.
 */
export function coalesceRegions(regions, { minChunks = 6 } = {}) {
  const out = [];
  let acc = null;
  for (const r of arr(regions)) {
    if (acc) {
      acc.count += r.count;
      acc.linked += r.linked;
      acc.chunks = acc.chunks.concat(r.chunks);
      acc.extraSections += 1;
    } else if (r.count >= minChunks) {
      out.push({ ...r, extraSections: 0 });
      continue;
    } else {
      acc = { ...r, chunks: [...r.chunks], extraSections: 0 };
    }
    if (acc.count >= minChunks) {
      out.push(acc); acc = null;
    }
  }
  if (acc) out.push(acc);   // 꼬리 지역 — 작아도 버리지 않는다
  for (const r of out) r.coverage = r.count ? r.linked / r.count : 0;
  return out;
}

/**
 * 미연결 연속 구간 랭킹 — 이 화면의 존재 이유.
 * links=0 이 연속으로 이어진 구간을 길이 내림차순으로 (동률은 앞선 위치 우선 —
 * 결정적). 구간이 걸친 절 이름을 함께 — "몇 번째 칸"이 아니라 "어느 절"이
 * 회복 배치의 언어다. min 미만 짧은 구멍은 잡음이라 뺀다(기본 2).
 */
export function unlinkedRuns(chunks, { top = 5, min = 2 } = {}) {
  const list = arr(chunks);
  const runs = [];
  let run = null;
  for (const c of list) {
    if ((Number(c.links) || 0) === 0) {
      if (!run) run = { start: Number(c.order) || 0, count: 0, _secs: new Set() };
      run.count += 1;
      if (c.section) run._secs.add(String(c.section));
    } else if (run) {
      runs.push(run); run = null;
    }
  }
  if (run) runs.push(run);
  return runs
    .filter((r) => r.count >= min)
    .sort((a, b) => b.count - a.count || a.start - b.start)
    .slice(0, top)
    .map((r) => ({ start: r.start, count: r.count,
                   sections: [...r._secs].slice(0, 4) }));
}

/**
 * 미니맵 밀도 비닝 — 문서 전체를 고정 수의 픽셀 열로.
 *
 * 규모 독립성이 존재 이유다: 밴드 목록은 청크 수에 비례해 자라서(실측 51 밴드,
 * 1만 청크면 사용 불가) 지도가 못 된다. 문서가 몇 청크든 화면 비용은 bins 로
 * 고정된다(코드 에디터 미니맵·유전체 커버리지 트랙과 같은 구조). 각 bin 은
 * 자기 구간의 커버리지와 대표 절 이름을 들고, 클릭하면 상세 창이 그리로 간다.
 * 청크가 bins 보다 적으면 bin = 청크 (1:1).
 */
export function binDensity(chunks, bins = 220) {
  const list = arr(chunks);
  if (!list.length) return [];
  const n = Math.max(1, Math.min(Math.floor(bins) || 1, list.length));
  const out = [];
  for (let i = 0; i < n; i += 1) {
    const lo = Math.floor((i * list.length) / n);
    const hi = Math.floor(((i + 1) * list.length) / n);
    const slice = list.slice(lo, hi);
    if (!slice.length) continue;
    const linked = slice.reduce((k, c) => k + ((Number(c.links) || 0) > 0 ? 1 : 0), 0);
    const secs = [];
    for (const c of slice) {
      const s = String(c.section || "").split(">")[0].trim();
      if (s && !secs.includes(s)) secs.push(s);
      if (secs.length >= 3) break;
    }
    out.push({
      startOrder: Number(slice[0].order) || lo + 1,
      endOrder: Number(slice[slice.length - 1].order) || hi,
      count: slice.length, linked,
      coverage: linked / slice.length,
      sections: secs,
    });
  }
  return out;
}

/**
 * 상세 창 절단 — order 기준 [start, start+size). 창 밖은 그리지 않는다
 * (DOM 비용을 창 크기로 고정). clamp 는 호출자가 문서 길이로 한다.
 */
export function windowSlice(chunks, start, size) {
  const s = Math.max(1, Number(start) || 1);
  const e = s + Math.max(1, Number(size) || 1);
  return arr(chunks).filter((c) => {
    const o = Number(c.order) || 0;
    return o >= s && o < e;
  });
}

/** 청크 order → run 포함 여부 집합 (하이라이트용). */
export function runOrders(run) {
  if (!isObj(run)) return new Set();
  const out = new Set();
  for (let i = 0; i < (Number(run.count) || 0); i += 1) out.add(run.start + i);
  return out;
}
