"""커버리지 expectation 게이트 — 설정 + 평가 (core/coverage_expectations.py).

핵심 계약 (변이 사살로 고정):
- **전역 기본값은 전부 None (unconfigured)** — 측정 없이 감으로 정한 임계를
  강제하지 않는다.
- **경계값은 통과** — min 은 >=, max 는 <=. 임계와 정확히 같으면 위반이 아니다.
- **0/0 정직** — 분모 0 이면 measured:false 로 보고하되 warn 으로 만들지 않는다.
  "측정 안 됨"과 "미달"의 구별이 이 함수의 존재 이유다.
- 비율의 분자·분모가 바뀌면 (orphan/nodes ↔ unlinked/chunks) 즉시 잡힌다 —
  비대칭 픽스처(고아 2/노드 10 vs 미연결 3/청크 20)로 사살한다.
"""
import pytest

from ontology.core.coverage_expectations import (KEYS, THRESHOLD_KEYS,
                                                 CoverageExpectations,
                                                 effective_config,
                                                 evaluate_expectations,
                                                 reset_coverage_expectations,
                                                 validate_overrides)

# 비대칭 픽스처: orphan_ratio = 2/10 = 0.2, unlinked_ratio = 3/20 = 0.15.
# 두 비율이 다르고, 분자·분모를 어떤 조합으로 바꿔치기해도 (10/2=5.0,
# 3/10=0.3, 2/20=0.1, 20/3≈6.67) 원값과 겹치지 않는다.
METRICS = {
    "extraction_coverage": 0.9,
    "orphan_node_count": 2,
    "node_count": 10,
    "unlinked_chunk_count": 3,
    "chunk_count": 20,
    "dangling_node_ref_count": 0,
}


class TestEffectiveConfig:
    def test_defaults_are_all_none(self):
        """전역 기본값은 전부 unconfigured — 지어낸 임계를 강제하지 않는다."""
        cfg = effective_config({})
        assert set(cfg) == set(KEYS)
        assert all(v is None for v in cfg.values())

    def test_partial_override(self):
        cfg = effective_config({"min_extraction_coverage": 0.6})
        assert cfg["min_extraction_coverage"] == 0.6
        assert cfg["max_orphan_ratio"] is None    # 나머지는 unconfigured

    def test_unknown_keys_are_dropped(self):
        assert "bogus" not in effective_config({"bogus": 1})


class TestValidation:
    @pytest.mark.parametrize("bad", [
        {"min_extraction_coverage": -0.1}, {"min_extraction_coverage": 1.5},
        {"max_orphan_ratio": 2.0}, {"max_unlinked_ratio": -0.5},
        {"max_dangling_refs": -1}, {"max_dangling_refs": 1.5},
        {"auto_coverage_limit": 0}, {"auto_coverage_limit": 201},
        {"auto_coverage_check": "yes"}, {"auto_coverage_check": 1},
        {"min_extraction_coverage": "0.5"},
        {"max_dangling_refs": True},          # bool 은 int 하위형 함정
    ])
    def test_rejects_bad_values(self, bad):
        ok, why = validate_overrides(bad)
        assert ok is False and why

    @pytest.mark.parametrize("good", [
        {}, {"min_extraction_coverage": 0.0}, {"min_extraction_coverage": 1.0},
        {"max_orphan_ratio": 0.5}, {"max_dangling_refs": 0},
        {"max_dangling_refs": 1000},          # 상한 없음
        {"auto_coverage_check": True}, {"auto_coverage_limit": 1},
        {"auto_coverage_limit": 200}, {"min_extraction_coverage": None},
    ])
    def test_accepts_good_values(self, good):
        assert validate_overrides(good)[0] is True

    def test_unknown_key_is_rejected_loudly(self):
        """조용히 버리는 것과 다르다 — 쓰기 API 는 오타를 알려줘야 한다."""
        ok, why = validate_overrides({"min_extraction_coverge": 0.5})
        assert ok is False and "min_extraction_coverge" in why


class TestEvaluateUnconfigured:
    def test_all_none_thresholds_is_unconfigured(self):
        result = evaluate_expectations(METRICS, effective_config({}))
        assert result["status"] == "unconfigured"
        assert result["warnings"] == []

    def test_auto_keys_alone_do_not_configure(self):
        """auto_* 는 동작 설정이지 임계가 아니다 — 임계 없이는 unconfigured."""
        thresholds = effective_config({"auto_coverage_check": True,
                                       "auto_coverage_limit": 50})
        assert evaluate_expectations(METRICS, thresholds)["status"] == \
            "unconfigured"

    @pytest.mark.parametrize("metrics,thresholds", [
        (None, None), ("garbage", 123), ([], {}), ({}, "x"),
    ])
    def test_garbage_input_never_raises(self, metrics, thresholds):
        result = evaluate_expectations(metrics, thresholds)
        assert result == {"status": "unconfigured", "warnings": [],
                          "metrics": {}}


class TestEvaluateDirections:
    """min 을 max 처럼(또는 반대로) 취급하는 mutant 사살."""

    def test_min_coverage_above_threshold_is_ok(self):
        """커버리지 0.9 > 임계 0.6 → ok (min 을 max 로 뒤집으면 여기서 warn)."""
        result = evaluate_expectations(
            METRICS, {"min_extraction_coverage": 0.6})
        assert result["status"] == "ok"
        assert result["warnings"] == []

    def test_min_coverage_below_threshold_warns(self):
        """커버리지 0.5 < 임계 0.6 → warn."""
        metrics = dict(METRICS, extraction_coverage=0.5)
        result = evaluate_expectations(
            metrics, {"min_extraction_coverage": 0.6})
        assert result["status"] == "warn"
        [w] = result["warnings"]
        assert w["key"] == "min_extraction_coverage"
        assert w["measured"] is True
        assert w["expected"] == 0.6 and w["actual"] == 0.5
        assert w["message"]

    def test_max_ratio_below_threshold_is_ok(self):
        result = evaluate_expectations(METRICS, {"max_orphan_ratio": 0.5})
        assert result["status"] == "ok"

    def test_max_ratio_above_threshold_warns(self):
        result = evaluate_expectations(METRICS, {"max_orphan_ratio": 0.1})
        assert result["status"] == "warn"
        [w] = result["warnings"]
        assert w["key"] == "max_orphan_ratio" and w["actual"] == 0.2

    def test_max_dangling_refs_violation(self):
        metrics = dict(METRICS, dangling_node_ref_count=3)
        result = evaluate_expectations(metrics, {"max_dangling_refs": 2})
        assert result["status"] == "warn"
        assert result["warnings"][0]["key"] == "max_dangling_refs"


class TestEvaluateBoundaries:
    """경계값: 임계와 **정확히 같으면 통과** — 비교 방향/포함 뒤집기 mutant 사살."""

    def test_min_equal_passes(self):
        """coverage 0.9 == min 0.9 → ok (>= 를 > 로 바꾸면 여기서 warn)."""
        result = evaluate_expectations(
            METRICS, {"min_extraction_coverage": 0.9})
        assert result["status"] == "ok" and result["warnings"] == []

    def test_max_ratio_equal_passes(self):
        """orphan_ratio 0.2 == max 0.2 → ok (<= 를 < 로 바꾸면 warn)."""
        result = evaluate_expectations(METRICS, {"max_orphan_ratio": 0.2})
        assert result["status"] == "ok" and result["warnings"] == []

    def test_max_dangling_equal_passes(self):
        """dangling 0 == max 0 → ok."""
        result = evaluate_expectations(METRICS, {"max_dangling_refs": 0})
        assert result["status"] == "ok"

    def test_just_past_boundary_warns(self):
        """경계 바로 밖은 warn — '경계 통과'가 '전부 통과' mutant 가 아님을 확인."""
        metrics = dict(METRICS, dangling_node_ref_count=1)
        result = evaluate_expectations(metrics, {"max_dangling_refs": 0})
        assert result["status"] == "warn"


class TestZeroDenominatorHonesty:
    """0/0 은 1.0 도 0.0 도 아니다 — measured:false 로 보고하고 warn 하지 않는다."""

    def test_zero_nodes_orphan_ratio_is_unmeasured(self):
        """노드 0 → orphan_ratio None (0/0→1.0 이나 0.0 mutant 사살)."""
        metrics = dict(METRICS, orphan_node_count=0, node_count=0)
        result = evaluate_expectations(metrics, {"max_orphan_ratio": 0.1})
        assert result["status"] == "ok"           # warn 아님
        [w] = result["warnings"]
        assert w == {"key": "max_orphan_ratio", "measured": False,
                     "reason": "no_data"}
        assert result["metrics"]["orphan_ratio"] is None

    def test_zero_chunks_unlinked_ratio_is_unmeasured(self):
        metrics = dict(METRICS, unlinked_chunk_count=0, chunk_count=0)
        result = evaluate_expectations(metrics, {"max_unlinked_ratio": 0.1})
        assert result["status"] == "ok"
        assert result["warnings"][0]["measured"] is False
        assert result["metrics"]["unlinked_ratio"] is None

    def test_missing_coverage_is_unmeasured_not_failing(self):
        metrics = dict(METRICS, extraction_coverage=None)
        result = evaluate_expectations(
            metrics, {"min_extraction_coverage": 0.6})
        assert result["status"] == "ok"
        assert result["warnings"][0] == {"key": "min_extraction_coverage",
                                         "measured": False,
                                         "reason": "no_data"}

    def test_unmeasured_does_not_mask_a_real_violation(self):
        """측정 불가 하나 + 실제 위반 하나 → status 는 warn 이어야 한다."""
        metrics = dict(METRICS, node_count=0, orphan_node_count=0,
                       extraction_coverage=0.1)
        result = evaluate_expectations(
            metrics, {"max_orphan_ratio": 0.1,
                      "min_extraction_coverage": 0.6})
        assert result["status"] == "warn"
        by_key = {w["key"]: w for w in result["warnings"]}
        assert by_key["max_orphan_ratio"]["measured"] is False
        assert by_key["min_extraction_coverage"]["measured"] is True


class TestRatioWiring:
    """orphan_ratio 와 unlinked_ratio 의 분자·분모 바꿔치기 mutant 사살.

    비대칭 픽스처: 고아 2/노드 10 = 0.2, 미연결 3/청크 20 = 0.15. 어떤
    바꿔치기(2/20, 3/10, 10/2, 20/3)도 원값과 겹치지 않는다.
    """

    def test_computed_ratios_are_exact(self):
        result = evaluate_expectations(METRICS, {"max_orphan_ratio": 1.0})
        assert result["metrics"]["orphan_ratio"] == pytest.approx(0.2)
        assert result["metrics"]["unlinked_ratio"] == pytest.approx(0.15)

    def test_orphan_threshold_targets_orphan_ratio(self):
        """임계 0.18: orphan(0.2) 만 위반, unlinked(0.15) 는 통과."""
        result = evaluate_expectations(
            METRICS, {"max_orphan_ratio": 0.18, "max_unlinked_ratio": 0.18})
        assert result["status"] == "warn"
        [w] = result["warnings"]
        assert w["key"] == "max_orphan_ratio"
        assert w["actual"] == pytest.approx(0.2)

    def test_unlinked_threshold_targets_unlinked_ratio(self):
        """임계 0.12: unlinked(0.15) 만 위반, orphan 은 임계 미설정."""
        result = evaluate_expectations(METRICS, {"max_unlinked_ratio": 0.12})
        [w] = result["warnings"]
        assert w["key"] == "max_unlinked_ratio"
        assert w["actual"] == pytest.approx(0.15)


class TestPersistence:
    def test_records_and_reloads(self, tmp_path):
        c = CoverageExpectations(namespace="ns", path=tmp_path / "c.json")
        c.set({"min_extraction_coverage": 0.6}, actor="t")
        again = CoverageExpectations(namespace="ns", path=tmp_path / "c.json")
        assert again.load_from_disk()
        assert again.overrides()["min_extraction_coverage"] == 0.6

    def test_set_merges_not_replaces(self, tmp_path):
        c = CoverageExpectations(namespace="ns", path=tmp_path / "c.json")
        c.set({"min_extraction_coverage": 0.6}, actor="t")
        c.set({"max_dangling_refs": 5}, actor="t")
        assert c.overrides() == {"min_extraction_coverage": 0.6,
                                 "max_dangling_refs": 5}

    def test_none_clears_a_key(self, tmp_path):
        """unconfigured 로 되돌리는 길 — 없으면 한 번 설정하면 영구히 굳는다."""
        c = CoverageExpectations(namespace="ns", path=tmp_path / "c.json")
        c.set({"max_orphan_ratio": 0.3}, actor="t")
        c.set({"max_orphan_ratio": None}, actor="t")
        assert "max_orphan_ratio" not in c.overrides()
        assert c.effective()["max_orphan_ratio"] is None

    def test_evaluate_uses_effective(self, tmp_path):
        c = CoverageExpectations(namespace="ns", path=tmp_path / "c.json")
        assert c.evaluate(METRICS)["status"] == "unconfigured"
        c.set({"max_orphan_ratio": 0.1}, actor="t")
        assert c.evaluate(METRICS)["status"] == "warn"

    def test_write_failure_does_not_raise(self, tmp_path):
        c = CoverageExpectations(namespace="ns", path=tmp_path / "no" / "c.json")
        c.set({"max_dangling_refs": 5}, actor="t")   # 디렉터리 없음
        assert c.overrides()["max_dangling_refs"] == 5   # 메모리는 유지

    def test_corrupt_file_degrades_to_defaults(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text("{broken", encoding="utf-8")
        c = CoverageExpectations(namespace="ns", path=p)
        c.load_from_disk()
        assert c.overrides() == {}

    def test_singletons_per_namespace(self):
        from ontology.core.coverage_expectations import \
            get_coverage_expectations
        reset_coverage_expectations()
        assert get_coverage_expectations("a") is not \
            get_coverage_expectations("b")
        assert get_coverage_expectations("a") is get_coverage_expectations("a")

    def test_default_path_uses_namespace(self):
        c = CoverageExpectations(namespace="pathns")
        assert c.path.name == "coverageexpectations_pathns.json"


class TestSpecShape:
    def test_threshold_keys_are_subset_of_keys(self):
        assert set(THRESHOLD_KEYS) <= set(KEYS)
