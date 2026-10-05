"""네임스페이스별 검색 설정 — knob 이 전역 상수여서 못 하던 것.

**실측이 요구했다.** 확산 채널의 가치가 두 네임스페이스에서 정반대로 나왔다:

    네임스페이스        커버리지   evidence hit@1: 확산 off → on
    ────────────────────────────────────────────────────────────
    ins_cancer_demo     100%       동률
    PROJ-A                25%       0.3871→0.4516 · 0.6154→0.6615

두 자 모두에서 재현됐다 — **확산의 가치는 그래프가 성길 때 나타난다**(설계 의도와
일치). 그런데 `DEFAULT_PROPAGATION_CHANNEL` 이 전역 상수라 한쪽에 맞추면 다른 쪽이
틀린다. 게다가 `service.retrieve(ns, query, top_k)` 가 플래그를 받지 않아
**라이브에서는 아예 켤 수 없었다** — 측정만 되고 실사용에 못 쓰는 상태였다.

설계 규정:
- **부분 오버라이드.** 미설정 키는 전역 기본값을 쓴다 — 전부 명시하도록 요구하면
  기본값이 바뀔 때 네임스페이스마다 낡은 값이 굳는다.
- **값을 검증한다.** 잘못된 값이 조용히 통과하면 검색이 이상해지는데 원인을
  찾기 어렵다(entry_k=0 이면 진입 노드가 0개).
- **측정과 라이브가 같은 설정을 쓴다.** 그러지 않으면 지표가 라이브를 설명하지
  못한다 — 이 세션에서 가장 많이 데인 부류.
- **설정 지문도 네임스페이스별이어야 한다.** eval_history 가 전역 기본값만
  기록하면 이력이 거짓이 된다.
"""
import json

import pytest

from ontology.core.retrieval_config import (RetrievalConfig, effective_config,
                                            reset_retrieval_configs,
                                            validate_overrides)


class TestEffectiveConfig:
    def test_unset_falls_back_to_global_defaults(self):
        """미설정이면 오늘과 완전히 같아야 한다 — 하위호환 관문."""
        from ontology.core import graph_retrieval as gr
        cfg = effective_config({})
        assert cfg["entry_k"] == gr.DEFAULT_ENTRY_K
        assert cfg["max_terms"] == gr.DEFAULT_MAX_TERMS
        assert cfg["propagation_channel"] == gr.DEFAULT_PROPAGATION_CHANNEL

    def test_partial_override(self):
        """전부 명시하도록 요구하면 기본값이 바뀔 때 낡은 값이 굳는다."""
        from ontology.core import graph_retrieval as gr
        cfg = effective_config({"propagation_channel": True})
        assert cfg["propagation_channel"] is True
        assert cfg["entry_k"] == gr.DEFAULT_ENTRY_K      # 나머지는 기본값

    def test_unknown_keys_are_dropped(self):
        """오타 키를 통과시키면 search() 가 TypeError 로 죽는다."""
        assert "bogus" not in effective_config({"bogus": 1})

    def test_returns_only_search_kwargs(self):
        """search()/expand() 에 그대로 넘길 수 있어야 한다."""
        cfg = effective_config({})
        assert set(cfg) == {"entry_k", "entry_ratio", "min_entry_score",
                            "max_terms", "use_propagation",
                            "propagation_channel", "propagation_top",
                            "propagation_weight"}


class TestValidation:
    @pytest.mark.parametrize("bad", [
        {"entry_k": 0}, {"entry_k": -1}, {"entry_k": 1000},
        {"entry_ratio": -0.1}, {"entry_ratio": 1.5},
        {"max_terms": -1}, {"propagation_top": 0},
        {"propagation_weight": -0.5},
        {"propagation_channel": "yes"},
        {"entry_k": "3"},
    ])
    def test_rejects_bad_values(self, bad):
        ok, why = validate_overrides(bad)
        assert ok is False and why

    @pytest.mark.parametrize("good", [
        {}, {"entry_k": 1}, {"entry_k": 20}, {"entry_ratio": 0.0},
        {"entry_ratio": 1.0}, {"max_terms": 0}, {"propagation_channel": True},
        {"use_propagation": False}, {"propagation_weight": 0.0},
        {"min_entry_score": 0.0},
    ])
    def test_accepts_good_values(self, good):
        assert validate_overrides(good)[0] is True

    def test_unknown_key_is_rejected_loudly(self):
        """조용히 버리는 것과 다르다 — 쓰기 API 는 오타를 알려줘야 한다."""
        ok, why = validate_overrides({"entry_kk": 3})
        assert ok is False and "entry_kk" in why


class TestPersistence:
    def test_records_and_reloads(self, tmp_path):
        c = RetrievalConfig(namespace="ns", path=tmp_path / "c.json")
        c.set({"propagation_channel": True}, actor="t")
        again = RetrievalConfig(namespace="ns", path=tmp_path / "c.json")
        assert again.load_from_disk()
        assert again.overrides()["propagation_channel"] is True

    def test_set_merges_not_replaces(self, tmp_path):
        """한 키만 바꾸려고 부르는 게 정상 사용이다 — 나머지가 날아가면 안 된다."""
        c = RetrievalConfig(namespace="ns", path=tmp_path / "c.json")
        c.set({"propagation_channel": True}, actor="t")
        c.set({"entry_k": 5}, actor="t")
        assert c.overrides() == {"propagation_channel": True, "entry_k": 5}

    def test_none_clears_a_key(self, tmp_path):
        """전역 기본값으로 되돌리는 길 — 없으면 한 번 설정하면 영구히 굳는다."""
        c = RetrievalConfig(namespace="ns", path=tmp_path / "c.json")
        c.set({"entry_k": 5}, actor="t")
        c.set({"entry_k": None}, actor="t")
        assert "entry_k" not in c.overrides()

    def test_effective_reflects_overrides(self, tmp_path):
        c = RetrievalConfig(namespace="ns", path=tmp_path / "c.json")
        c.set({"entry_k": 5}, actor="t")
        assert c.effective()["entry_k"] == 5

    def test_write_failure_does_not_raise(self, tmp_path):
        c = RetrievalConfig(namespace="ns", path=tmp_path / "no" / "c.json")
        c.set({"entry_k": 5}, actor="t")          # 디렉터리 없음
        assert c.overrides()["entry_k"] == 5      # 메모리는 유지

    def test_corrupt_file_degrades_to_defaults(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text("{broken", encoding="utf-8")
        c = RetrievalConfig(namespace="ns", path=p)
        c.load_from_disk()
        assert c.overrides() == {}

    def test_singletons_per_namespace(self):
        from ontology.core.retrieval_config import get_retrieval_config
        reset_retrieval_configs()
        assert get_retrieval_config("a") is not get_retrieval_config("b")
        assert get_retrieval_config("a") is get_retrieval_config("a")


class TestFingerprintIsNamespaceAware:
    """설정 지문이 전역 기본값만 기록하면 **이력이 거짓**이 된다.

    eval_history 의 목적이 "그때 그 숫자가 어떤 설정에서 나왔나"인데, 네임스페이스
    오버라이드를 무시하면 그 목적을 정면으로 배신한다.
    """

    def test_fingerprint_reflects_namespace_override(self, tmp_path,
                                                     monkeypatch):
        import ontology.core.retrieval_config as rc
        from ontology.core.eval_history import config_fingerprint
        monkeypatch.setattr(rc, "_DEFAULT_DATA_DIR", tmp_path)
        rc.reset_retrieval_configs()
        rc.get_retrieval_config("fpns").set({"entry_k": 5}, actor="t")
        assert config_fingerprint("fpns")["entry_k"] == 5

    def test_fingerprint_without_namespace_uses_globals(self):
        from ontology.core import graph_retrieval as gr
        from ontology.core.eval_history import config_fingerprint
        assert config_fingerprint()["entry_k"] == gr.DEFAULT_ENTRY_K
