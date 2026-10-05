"""
⑥ 데이터셋 큐레이터 — build_training_dataset 산출물의 자동 품질 게이트.

프로젝트의 원래 목표는 파인튜닝용 학습 데이터 추출이다. build_training_dataset
은 환각 0 인 행을 만들지만 품질은 고르지 않다: 중복, 너무 짧은 원문, 한
문서의 지배, summary 신뢰 등급 행의 혼입. aicoach 는 이 게이트를 수동
(curated=true)으로 운영했고, 이 모듈은 그것을 자동화한다.

고정하는 계약:
1. **결정적 큐레이션은 LLM 0콜** — dedup/min_input_chars/max_per_source/
   exclude_trust 는 순수 함수. 순서 보존, 첫 등장 승리, 입력 불변.
2. **no silent caps** — 떨어진 행은 전부 이유별로 센다.
   kept + sum(dropped) == 입력 행 수.
3. LLM 품질 게이트의 출력은 불신한다 — 판정 불가(잘못된 index/score)는
   행을 **보존**한다. 판정 불가로 데이터를 버리지 않는다.
4. 배치 LLM 실패 → 배치 전체 보존 + quality_skipped 집계. 품질 게이트가
   죽는다고 데이터가 사라지면 안 된다. 절대 raise 하지 않는다.
"""

import asyncio
import copy
import json
import uuid

import pytest

from ontology.core.dataset_curator import QualityScorer, curate_rows


def run(coro):
    return asyncio.run(coro)


# ─── 행 헬퍼 (build_training_dataset 의 실제 행 모양) ─────────────────

def ev_row(text, source="약관.md", trust="", chunk_id="c1"):
    return {"format": "evidence",
            "instruction": "다음 원문에서 지식 그래프 개체를 추출하세요.",
            "input": text,
            "output": [{"name": "청약철회", "type": "Clause"}],
            "source": source, "chunk_id": chunk_id,
            "char_start": 0, "char_end": len(text), "trust": trust}


def qa_row(instruction, output, source="약관.md"):
    return {"format": "qa", "instruction": instruction, "output": output,
            "source": source}


def triple_row(subj, pred, obj, source="약관.md"):
    return {"format": "triple", "subject": subj, "predicate": pred,
            "object": obj, "subject_type": "Clause", "object_type": "Regulation",
            "source": source}


def surface_row(inp, out, source="약관.md"):
    return {"format": "surface", "input": inp, "output": out, "source": source}


def total_dropped(report):
    return sum(report["dropped"].values())


# ─── 1. 중복 제거 — 포맷별 의미 페이로드 키 ──────────────────────────

class TestDedup:
    def test_evidence_dedup_squashes_whitespace_first_wins(self):
        """공백·개행만 다른 원문은 같은 학습 신호다 — 내용 기준 중복."""
        rows = [ev_row("청약철회권은 15일 이내에 행사할 수 있다."),
                ev_row("청약철회권은  15일\n이내에 행사할 수 있다.")]
        kept, report = curate_rows(rows)
        assert len(kept) == 1
        assert kept[0]["input"] == rows[0]["input"]  # 첫 등장 승리
        assert report["dropped"]["duplicate"] == 1

    def test_qa_dedup_keys_on_instruction_and_output(self):
        rows = [qa_row("청약철회란?", "15일 이내 무르는 권리"),
                qa_row("청약철회란?", "15일 이내 무르는 권리"),
                qa_row("청약철회란?", "다른 답변")]  # output 다르면 중복 아님
        kept, report = curate_rows(rows)
        assert len(kept) == 2
        assert report["dropped"]["duplicate"] == 1

    def test_triple_dedup_keys_on_spo(self):
        """출처가 달라도 s/p/o 가 같으면 같은 관계 트리플이다."""
        rows = [triple_row("청약철회", "citesRegulation", "금소법", source="a.md"),
                triple_row("청약철회", "citesRegulation", "금소법", source="b.md"),
                triple_row("청약철회", "citesRegulation", "다른법")]
        kept, report = curate_rows(rows)
        assert len(kept) == 2
        assert report["dropped"]["duplicate"] == 1

    def test_surface_dedup_keys_on_input_output(self):
        rows = [surface_row("철회", "청약철회"), surface_row("철회", "청약철회"),
                surface_row("철회", "계약철회")]
        kept, _ = curate_rows(rows)
        assert len(kept) == 2

    def test_dedup_can_be_disabled(self):
        rows = [qa_row("q", "a"), qa_row("q", "a")]
        kept, report = curate_rows(rows, dedup=False)
        assert len(kept) == 2
        assert report["dropped"].get("duplicate", 0) == 0

    def test_order_preserved(self):
        rows = [qa_row(f"q{i}", "a") for i in range(5)]
        kept, _ = curate_rows(rows)
        assert [r["instruction"] for r in kept] == [f"q{i}" for i in range(5)]


# ─── 2. 최소 원문 길이 / 문서별 상한 / trust 배제 ────────────────────

class TestFilters:
    def test_min_input_chars_drops_only_input_bearing_rows(self):
        """너무 짧은 원문은 학습 신호가 약하다 — 단, input 이 없는 행
        (qa/triple)은 이 필터의 대상이 아니다."""
        rows = [ev_row("짧다"),
                ev_row("충분히 긴 원문 문장으로 학습 신호가 있는 행이다."),
                qa_row("input 없는 qa 행은 안 건드린다", "answer"),
                triple_row("s", "p", "o")]
        kept, report = curate_rows(rows, min_input_chars=10)
        assert len(kept) == 3
        assert report["dropped"]["too_short"] == 1
        assert all(r["format"] != "evidence" or len(r["input"]) >= 10
                   for r in kept)

    def test_min_input_chars_measures_squashed_length(self):
        """공백을 부풀려 길이를 채운 원문은 짧은 원문이다."""
        rows = [ev_row("짧 " * 20)]  # squash 후 "짧 짧 ..." 약 39자? no
        # squash("짧 짧 ... ") 는 여전히 39자 — 대신 개행 부풀림으로 검증
        rows = [ev_row("짧다\n\n\n\n\n\n\n\n\n\n\n\n")]
        kept, report = curate_rows(rows, min_input_chars=10)
        assert kept == []
        assert report["dropped"]["too_short"] == 1

    def test_max_per_source_caps_and_keeps_first_n(self):
        """한 문서가 데이터셋을 지배하면 모델이 그 문서를 외운다."""
        rows = [qa_row(f"q{i}", "a", source="지배문서.md") for i in range(4)]
        rows += [qa_row("other", "a", source="다른문서.md")]
        kept, report = curate_rows(rows, max_per_source=2)
        dominant = [r for r in kept if r["source"] == "지배문서.md"]
        assert [r["instruction"] for r in dominant] == ["q0", "q1"]  # 앞 N개
        assert report["dropped"]["source_capped"] == 2
        assert len(kept) == 3

    def test_exclude_trust_drops_summary_rows(self):
        """boilerplate(요약서)에서 뽑은 쌍을 원본 쌍과 섞지 않는다
        (aicoach layer 원칙)."""
        rows = [ev_row("원본 약관의 문장이다.", trust="authoritative"),
                ev_row("요약서의 문장이다.", trust="summary"),
                qa_row("trust 필드 없는 행", "무관")]
        kept, report = curate_rows(rows, exclude_trust=["summary"])
        assert len(kept) == 2
        assert all(r.get("trust") != "summary" for r in kept)
        assert report["dropped"]["trust_excluded"] == 1


# ─── 3. 리포트 — no silent caps ──────────────────────────────────────

class TestReport:
    def test_every_drop_is_attributed(self):
        """kept + 이유별 drop 합 == 입력 행 수. 조용히 사라지는 행은 없다."""
        rows = [ev_row("충분히 긴 원본 약관 문장이다.", trust="authoritative"),
                ev_row("충분히  긴 원본\n약관 문장이다.", trust="authoritative"),  # dup
                ev_row("짧다", trust="authoritative"),                             # too_short
                ev_row("요약서에서 뽑은 충분히 긴 문장.", trust="summary"),        # trust
                qa_row("q1", "a", source="지배.md"),
                qa_row("q2", "a", source="지배.md"),
                qa_row("q3", "a", source="지배.md")]                               # capped
        kept, report = curate_rows(rows, min_input_chars=10, max_per_source=2,
                                   exclude_trust=["summary"])
        assert report["kept"] == len(kept)
        assert report["kept"] + total_dropped(report) == len(rows)
        assert report["dropped"]["duplicate"] == 1
        assert report["dropped"]["too_short"] == 1
        assert report["dropped"]["trust_excluded"] == 1
        assert report["dropped"]["source_capped"] == 1

    def test_empty_rows_ok(self):
        kept, report = curate_rows([])
        assert kept == []
        assert report["kept"] == 0
        assert total_dropped(report) == 0

    def test_input_rows_are_not_mutated(self):
        rows = [ev_row("원문 문장이다."), qa_row("q", "a")]
        snapshot = copy.deepcopy(rows)
        curate_rows(rows, dedup=True, min_input_chars=3, max_per_source=1,
                    exclude_trust=["summary"])
        assert rows == snapshot


# ─── 4. LLM 품질 게이트 — 출력 불신 ──────────────────────────────────

def scores_llm(scores):
    """배치 크기와 무관하게 주어진 (index, score) 목록을 돌려주는 fake."""
    def fn(prompt):
        return json.dumps({"scores": [{"index": i, "score": s, "reason": "r"}
                                      for i, s in scores]})
    return fn


class TestQualityScorer:
    def test_below_threshold_rows_dropped_as_low_quality(self):
        rows = [qa_row("좋은 쌍", "a"), qa_row("나쁜 쌍", "b"), qa_row("보통 쌍", "c")]
        scorer = QualityScorer(llm_fn=scores_llm([(0, 5), (1, 2), (2, 3)]))
        kept, report = run(scorer.score_rows(rows, threshold=3))
        assert [r["instruction"] for r in kept] == ["좋은 쌍", "보통 쌍"]
        assert report["low_quality"] == 1

    def test_invalid_or_missing_entries_keep_rows(self):
        """판정 불가로 데이터를 버리지 않는다 — index 범위 밖, score 비정상,
        누락 항목은 전부 '보존'이다."""
        rows = [qa_row("q0", "a"), qa_row("q1", "a"), qa_row("q2", "a")]

        def weird(prompt):
            return json.dumps({"scores": [
                {"index": 99, "score": 1, "reason": "범위 밖 index"},
                {"index": 0, "score": "높음", "reason": "score 가 int 아님"},
                {"index": 1, "score": 7, "reason": "score 범위 밖"},
                # index 2 는 아예 누락
            ]})

        scorer = QualityScorer(llm_fn=weird)
        kept, report = run(scorer.score_rows(rows, threshold=3))
        assert len(kept) == 3
        assert report["low_quality"] == 0

    def test_batch_llm_failure_keeps_batch_and_counts_skipped(self):
        """품질 게이트가 죽는다고 데이터가 사라지면 안 된다."""
        rows = [qa_row(f"q{i}", "a") for i in range(3)]

        def broken(prompt):
            raise RuntimeError("LLM down")

        scorer = QualityScorer(llm_fn=broken)
        kept, report = run(scorer.score_rows(rows))
        assert len(kept) == 3
        assert report["quality_skipped"] == 3

    def test_non_json_response_keeps_batch_as_skipped(self):
        rows = [qa_row("q", "a")]
        scorer = QualityScorer(llm_fn=lambda p: "json 아님")
        kept, report = run(scorer.score_rows(rows))  # raise 하지 않는다
        assert len(kept) == 1
        assert report["quality_skipped"] == 1

    def test_batches_ceil_of_n_over_batch_size(self):
        """LLM 콜 수 = ceil(n / batch_size) — 행마다 부르면 비용이 폭발한다."""
        rows = [qa_row(f"q{i}", "a") for i in range(25)]
        calls = {"n": 0}

        def counting(prompt):
            calls["n"] += 1
            return json.dumps({"scores": []})

        scorer = QualityScorer(llm_fn=counting)
        kept, _ = run(scorer.score_rows(rows, batch_size=10))
        assert calls["n"] == 3
        assert len(kept) == 25  # 항목 누락 → 전부 보존

    def test_partial_batch_failure_only_skips_that_batch(self):
        """앞 배치가 죽어도 뒤 배치는 정상 채점된다."""
        rows = [qa_row(f"q{i}", "a") for i in range(4)]
        calls = {"n": 0}

        def flaky(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first batch down")
            return json.dumps({"scores": [{"index": 0, "score": 1, "reason": "r"},
                                          {"index": 1, "score": 5, "reason": "r"}]})

        scorer = QualityScorer(llm_fn=flaky)
        kept, report = run(scorer.score_rows(rows, threshold=3, batch_size=2))
        assert report["quality_skipped"] == 2   # 배치 1 보존
        assert report["low_quality"] == 1       # 배치 2 의 q2 탈락
        assert [r["instruction"] for r in kept] == ["q0", "q1", "q3"]


# ─── 5. 서비스 + 라우트 관통 ─────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_chunk_stores():
    from ontology.core.chunk_store import reset_chunk_stores
    reset_chunk_stores()
    yield
    reset_chunk_stores()


@pytest.fixture
def dataset_ns(tmp_path):
    """KG + ChunkStore 를 직접 구성 — 원본(authoritative) 청크 1 +
    요약서(summary) 청크 1. 빌더를 돌리지 않아 결정적이고 빠르다."""
    import ontology.core.chunk_store as cs
    from ontology.builder.models import Chunk
    from ontology.core.chunk_store import ChunkStore
    from ontology.engines.knowledge_graph_clean import (KnowledgeGraphEngine,
                                                        _kg_instances)

    ns = f"curator_{uuid.uuid4().hex[:8]}"
    kg = KnowledgeGraphEngine(fast_mode=True, namespace=ns)
    _kg_instances[ns] = kg
    kg.graph.add_node("Clause:청약철회", name="청약철회", type="Clause",
                      source="약관.md")
    kg.graph.add_node("Clause:보장요약", name="보장요약", type="Clause",
                      source="요약서.md")

    store = ChunkStore(namespace=ns, path=tmp_path / "chunks.jsonl")
    cs._stores[ns] = store
    store.add(Chunk(text="청약철회권은 보험증권을 받은 날부터 15일 이내에 행사할 수 있다.",
                    source="약관.md", index=0),
              node_ids=["Clause:청약철회"], trust="authoritative")
    store.add(Chunk(text="보장내용 요약: 암진단비를 지급한다.",
                    source="요약서.md", index=0),
              node_ids=["Clause:보장요약"], trust="summary")
    yield ns
    _kg_instances.pop(ns, None)


class TestServiceIntegration:
    def test_evidence_rows_carry_chunk_trust(self, dataset_ns, tmp_path):
        """exclude_trust 가 동작하려면 evidence 행에 trust 가 실려야 한다."""
        from ontology.server.service import OntologyBuilderService
        service = OntologyBuilderService(data_dir=tmp_path / "ds")
        result = service.build_training_dataset(dataset_ns, formats=["evidence"])
        trusts = {r["source"]: r["trust"] for r in result["rows"]}
        assert trusts["약관.md"] == "authoritative"
        assert trusts["요약서.md"] == "summary"

    def test_build_curated_dataset_reports_and_recounts(self, dataset_ns, tmp_path):
        from ontology.server.service import OntologyBuilderService
        service = OntologyBuilderService(data_dir=tmp_path / "ds")
        result = run(service.build_curated_dataset(
            dataset_ns, formats=["evidence"],
            curate={"exclude_trust": ["summary"]}))
        assert result["curation"]["dropped"]["trust_excluded"] == 1
        assert result["total"] == 1                      # 재계산된 total
        assert result["counts"] == {"evidence": 1}       # 재계산된 counts
        assert all(r["trust"] != "summary" for r in result["rows"])

    def test_build_curated_dataset_llm_quality_uses_service_llm_fn(
            self, dataset_ns, tmp_path):
        from ontology.server.service import OntologyBuilderService

        def all_bad(prompt):
            return json.dumps({"scores": [{"index": i, "score": 1, "reason": "r"}
                                          for i in range(10)]})

        service = OntologyBuilderService(data_dir=tmp_path / "ds",
                                         llm_fn=all_bad)
        result = run(service.build_curated_dataset(
            dataset_ns, formats=["evidence"],
            curate={"dedup": False, "llm_quality": True}))
        assert result["total"] == 0
        assert result["curation"]["dropped"]["low_quality"] == 2


@pytest.fixture
def client(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ontology.server import router as server_router
    from ontology.server.service import OntologyBuilderService

    service = OntologyBuilderService(data_dir=tmp_path / "api")
    app = FastAPI()
    app.include_router(server_router.router, prefix="/api/v1/ontology")
    app.dependency_overrides[server_router.get_ontology_service] = lambda: service
    return TestClient(app)


class TestDatasetRoute:
    def test_post_with_curate_returns_curation_report(self, dataset_ns, client):
        response = client.post(
            f"/api/v1/ontology/graphs/{dataset_ns}/dataset",
            json={"formats": ["evidence"],
                  "curate": {"exclude_trust": ["summary"]}})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["curation"]["dropped"]["trust_excluded"] == 1
        assert body["total"] == len(body["rows"]) == 1

    def test_post_without_curate_preserves_contract(self, dataset_ns, client):
        """옵트인 — curate 없이는 기존 응답 모양 그대로 (curation 키 없음)."""
        response = client.post(
            f"/api/v1/ontology/graphs/{dataset_ns}/dataset",
            json={"formats": ["evidence"]})
        assert response.status_code == 200, response.text
        body = response.json()
        assert "curation" not in body
        assert body["total"] == 2
