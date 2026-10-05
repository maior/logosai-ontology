// coverageView — 커버리지 지도 뷰모델 (순수, node --test 로 파일 명시 실행)
//
// 판독 규칙이 여기 있다: 버킷 경계(0/1~2/3+), 절 경계, 그리고 화면의 존재
// 이유인 "미연결 연속 구간 랭킹". 회귀가 나면 관리자가 엉뚱한 절을 회복
// 타깃으로 잡는다.
import test from "node:test";
import assert from "node:assert/strict";
import {
  buildCoverageDocs, linkBucket, sectionSpans, sectionRegions,
  coalesceRegions, unlinkedRuns, runOrders, binDensity, windowSlice,
} from "../app/ontology-admin/coverageView.mjs";

const RESP = {
  documents: [{
    source: "요구서.txt", total: 6, linked: 3, coverage: 0.5, ordered: true,
    chunks: [
      { chunk_id: "c1", order: 1, section: "1장", links: 3, node_ids: ["T:a"], text_head: "머리" },
      { chunk_id: "c2", order: 2, section: "1장", links: 1 },
      { chunk_id: "c3", order: 3, section: "2장", links: 0 },
      { chunk_id: "c4", order: 4, section: "2장", links: 0 },
      { chunk_id: "c5", order: 5, section: "3장", links: 0 },
      { chunk_id: "c6", order: 6, section: "3장", links: 2 },
    ],
  }],
};

test("buildCoverageDocs: 정규화 + 필드 기본값", () => {
  const docs = buildCoverageDocs(RESP);
  assert.equal(docs.length, 1);
  const d = docs[0];
  assert.equal(d.total, 6);
  assert.equal(d.chunks[0].textHead, "머리");
  assert.deepEqual(d.chunks[0].nodeIds, ["T:a"]);
  assert.equal(d.chunks[1].textHead, "");   // 누락 필드는 안전 기본값
});

test("buildCoverageDocs: 쓰레기 입력은 빈 목록 (절대 던지지 않음)", () => {
  for (const bad of [null, undefined, {}, { documents: "x" }, { documents: [3] }]) {
    assert.deepEqual(buildCoverageDocs(bad), []);
  }
});

test("buildCoverageDocs: ordered 는 명시적 false 만 불신", () => {
  assert.equal(buildCoverageDocs({ documents: [{ source: "a", chunks: [] }] })[0].ordered, true);
  assert.equal(buildCoverageDocs({ documents: [{ source: "a", ordered: false, chunks: [] }] })[0].ordered, false);
});

test("linkBucket: 0 / 1~2 / 3+ 경계", () => {
  assert.equal(linkBucket(0), 0);
  assert.equal(linkBucket(1), 1);
  assert.equal(linkBucket(2), 1);
  assert.equal(linkBucket(3), 2);
  assert.equal(linkBucket(undefined), 0);
});

test("sectionSpans: 연속 같은 절을 구간으로", () => {
  const d = buildCoverageDocs(RESP)[0];
  assert.deepEqual(sectionSpans(d.chunks), [
    { section: "1장", start: 1, count: 2 },
    { section: "2장", start: 3, count: 2 },
    { section: "3장", start: 5, count: 2 },
  ]);
});

test("sectionRegions: 절을 지역으로 — 이름·청크·커버리지", () => {
  const d = buildCoverageDocs(RESP)[0];
  const regions = sectionRegions(d.chunks);
  assert.deepEqual(regions.map((r) => [r.section, r.count, r.linked]),
    [["1장", 2, 2], ["2장", 2, 0], ["3장", 2, 1]]);
  assert.equal(regions[1].coverage, 0);           // 빈 지역
  assert.equal(regions[2].coverage, 0.5);
  assert.equal(regions[0].chunks[0].chunkId, "c1"); // 지역이 청크를 들고 다닌다
});

test("sectionRegions: 절 이름 없는 구간도 지역이다 (지도에서 사라지면 거짓)", () => {
  const regions = sectionRegions([
    { order: 1, section: "", links: 0 },
    { order: 2, section: "", links: 1 },
    { order: 3, section: "1장", links: 0 },
  ]);
  assert.deepEqual(regions.map((r) => [r.section, r.count]), [["", 2], ["1장", 1]]);
});

test("sectionRegions: 같은 절이 떨어져 두 번 나오면 별개 지역 (문서 순서 보존)", () => {
  const regions = sectionRegions([
    { order: 1, section: "부록", links: 1 },
    { order: 2, section: "본문", links: 1 },
    { order: 3, section: "부록", links: 0 },
  ]);
  assert.deepEqual(regions.map((r) => r.section), ["부록", "본문", "부록"]);
});

test("sectionRegions: 하위 절(' > ')은 상위 절로 묶인다", () => {
  const regions = sectionRegions([
    { order: 1, section: "보안요구사항 > 4.1 개요", links: 1 },
    { order: 2, section: "보안요구사항 > 4.2 시큐어 코딩", links: 0 },
  ]);
  assert.deepEqual(regions.map((r) => [r.section, r.count]), [["보안요구사항", 2]]);
});

test("coalesceRegions: 작은 연속 지역을 minChunks 까지 병합, 라벨은 '외 N절'", () => {
  const regions = sectionRegions([
    { order: 1, section: "가", links: 1 },
    { order: 2, section: "나", links: 0 },
    { order: 3, section: "다", links: 0 },
    { order: 4, section: "라", links: 1 },
  ]);
  const bands = coalesceRegions(regions, { minChunks: 3 });
  assert.equal(bands.length, 2);
  assert.equal(bands[0].section, "가");
  assert.equal(bands[0].extraSections, 2);         // 나·다 흡수
  assert.equal(bands[0].count, 3);
  assert.equal(bands[0].linked, 1);                // 통계 보존
  assert.equal(bands[1].section, "라");            // 꼬리 지역 — 작아도 버리지 않음
  assert.equal(bands[1].extraSections, 0);
});

test("coalesceRegions: 큰 지역은 그대로 통과 (병합은 축척이지 삭제가 아니다)", () => {
  const big = { section: "본문", start: 1, count: 10, linked: 4,
                chunks: Array.from({ length: 10 }, (_, i) => ({ order: i + 1 })) };
  const bands = coalesceRegions([big], { minChunks: 6 });
  assert.equal(bands.length, 1);
  assert.equal(bands[0].count, 10);
  assert.equal(bands[0].extraSections, 0);
  assert.equal(bands[0].chunks.length, 10);        // 칸 보존
});

test("coalesceRegions: 결정적 + 칸 전량 보존", () => {
  const regions = sectionRegions(Array.from({ length: 9 }, (_, i) => (
    { order: i + 1, section: `s${i}`, links: i % 2 })));
  const bands = coalesceRegions(regions, { minChunks: 4 });
  const total = bands.reduce((n, b) => n + b.chunks.length, 0);
  assert.equal(total, 9);
  assert.deepEqual(bands, coalesceRegions(regions, { minChunks: 4 }));
});

test("unlinkedRuns: 미연결 연속 구간을 길이순으로, 걸친 절 이름과 함께", () => {
  const d = buildCoverageDocs(RESP)[0];
  const runs = unlinkedRuns(d.chunks);
  // c3,c4,c5 가 연속 3칸 (2장·3장에 걸침)
  assert.deepEqual(runs, [{ start: 3, count: 3, sections: ["2장", "3장"] }]);
});

test("unlinkedRuns: min 미만 구멍은 잡음으로 제외, 동률은 앞선 위치 우선", () => {
  const chunks = [
    { order: 1, links: 0 },                       // 1칸 — min=2 미달
    { order: 2, links: 5 },
    { order: 3, links: 0 }, { order: 4, links: 0 },  // 2칸
    { order: 5, links: 5 },
    { order: 6, links: 0 }, { order: 7, links: 0 },  // 2칸 (동률 — 뒤)
  ];
  const runs = unlinkedRuns(chunks);
  assert.deepEqual(runs.map((r) => r.start), [3, 6]);
});

test("unlinkedRuns: 문서 끝에서 끝나는 구간도 잡는다", () => {
  const runs = unlinkedRuns([{ order: 1, links: 1 }, { order: 2, links: 0 }, { order: 3, links: 0 }]);
  assert.deepEqual(runs, [{ start: 2, count: 2, sections: [] }]);
});

test("binDensity: 규모 독립 — 청크가 몇 개든 bin 수는 고정 상한", () => {
  const many = Array.from({ length: 1000 }, (_, i) => (
    { order: i + 1, links: i % 4 === 0 ? 0 : 1, section: `s${Math.floor(i / 100)}` }));
  const bins = binDensity(many, 200);
  assert.equal(bins.length, 200);
  const total = bins.reduce((n, b) => n + b.count, 0);
  assert.equal(total, 1000);                       // 청크 전량이 어떤 bin 에 속한다
});

test("binDensity: 청크가 bins 보다 적으면 1:1", () => {
  const bins = binDensity([{ order: 1, links: 0 }, { order: 2, links: 3 }], 220);
  assert.equal(bins.length, 2);
  assert.equal(bins[0].coverage, 0);
  assert.equal(bins[1].coverage, 1);
});

test("binDensity: bin 은 구간 order 와 대표 절을 들고 다닌다", () => {
  const chunks = [
    { order: 1, links: 0, section: "1장 > 1.1" },
    { order: 2, links: 1, section: "1장 > 1.2" },
    { order: 3, links: 1, section: "2장" },
    { order: 4, links: 0, section: "2장" },
  ];
  const bins = binDensity(chunks, 2);
  assert.deepEqual([bins[0].startOrder, bins[0].endOrder], [1, 2]);
  assert.equal(bins[0].coverage, 0.5);
  assert.deepEqual(bins[0].sections, ["1장"]);     // 상위 절로 정규화
  assert.deepEqual(bins[1].sections, ["2장"]);
});

test("binDensity: 빈 입력·쓰레기는 빈 배열", () => {
  assert.deepEqual(binDensity([], 100), []);
  assert.deepEqual(binDensity(null, 100), []);
});

test("binDensity: 결정적", () => {
  const chunks = Array.from({ length: 97 }, (_, i) => ({ order: i + 1, links: i % 3 }));
  assert.deepEqual(binDensity(chunks, 40), binDensity(chunks, 40));
});

test("windowSlice: [start, start+size) 창 절단", () => {
  const chunks = Array.from({ length: 10 }, (_, i) => ({ order: i + 1 }));
  const w = windowSlice(chunks, 3, 4);
  assert.deepEqual(w.map((c) => c.order), [3, 4, 5, 6]);
  assert.deepEqual(windowSlice(chunks, 9, 5).map((c) => c.order), [9, 10]); // 끝 넘어도 안전
  assert.deepEqual(windowSlice([], 1, 5), []);
});

test("runOrders: 하이라이트 집합", () => {
  assert.deepEqual([...runOrders({ start: 3, count: 3 })], [3, 4, 5]);
  assert.equal(runOrders(null).size, 0);
});

test("결정적 — 같은 입력이면 같은 결과", () => {
  const d = buildCoverageDocs(RESP)[0];
  assert.deepEqual(unlinkedRuns(d.chunks), unlinkedRuns(d.chunks));
  assert.deepEqual(buildCoverageDocs(RESP), buildCoverageDocs(RESP));
});
