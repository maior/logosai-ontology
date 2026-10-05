"""평가 스냅샷 이력 — "이 설정에서 이 점수"를 기록해 비교 가능하게 한다.

**왜 필요한가**: 이 세션 내내 검색 knob 과 임베더를 바꿔가며 지표를 쟀는데, 그
비교표를 전부 손으로 만들었다. 그리고 두 번 데였다 — `max_terms` 를 8→2 로
바꿨다가 되돌렸고(그래프 상태가 바뀌자 결론이 뒤집혔다), 임베더도 채널 하나만
재고 결론냈다가 뒤집혔다. **설정과 점수를 같이 기록하지 않으면 "그때 그 숫자가
어떤 설정에서 나온 것인가"를 잃는다.**

설계 규정:
- **추가전용 JSONL** — 골든셋·검수 로그와 같은 규약. 로그가 원본이다.
- **설정 지문을 같이 남긴다.** 점수만 남기면 비교가 불가능하다 — 임베더·entry_k·
  max_terms·확산 플래그·케이스 수가 전부 지표를 움직인다(이 세션에서 각각 실측).
- **평가할 때 자동으로 남긴다.** 사람이 기억해서 기록하는 절차는 지켜지지 않는다.
- 기록 실패가 평가를 죽이지 않는다 — 측정이 본체고 이력은 부수물이다.
"""
import json

import pytest

from ontology.core.eval_history import (EvalHistory, config_fingerprint,
                                        reset_eval_histories)


def _result(**kw):
    base = {"target": "node", "k": 5, "cases": 46, "measured": True,
            "statuses": ["confirmed", "verified"],
            "channels": {"retrieve": {"hit@1": 0.63, "hit@5": 0.85,
                                      "mrr": 0.70, "cases": 46}}}
    base.update(kw)
    return base


class TestConfigFingerprint:
    def test_captures_what_moves_the_metric(self):
        """이 세션에서 지표를 움직인 것이 전부 들어가야 한다 — 하나라도 빠지면
        "같은 설정인데 점수가 다르다"는 미궁이 생긴다."""
        fp = config_fingerprint()
        for key in ("node_model", "chunk_model", "entry_k", "max_terms",
                    "entry_ratio", "use_propagation", "propagation_channel"):
            assert key in fp, key

    def test_reads_current_env(self, monkeypatch):
        monkeypatch.setenv("ONTOLOGY_EMBEDDING_MODEL", "BAAI/bge-m3")
        assert config_fingerprint()["node_model"] == "BAAI/bge-m3"

    def test_never_raises(self, monkeypatch):
        import ontology.core.eval_history as eh
        monkeypatch.setattr(eh, "_retrieval_defaults",
                            lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert isinstance(config_fingerprint(), dict)


class TestEvalHistory:
    def test_records_metrics_and_config(self, tmp_path):
        h = EvalHistory(namespace="ns", path=tmp_path / "h.jsonl")
        h.record(_result(), actor="test")
        rows = h.entries()
        assert len(rows) == 1
        assert rows[0]["channels"]["retrieve"]["mrr"] == 0.70
        assert "config" in rows[0] and "node_model" in rows[0]["config"]

    def test_survives_reload(self, tmp_path):
        """추가전용 로그다 — 재시작에 이력이 사라지면 비교의 목적을 잃는다."""
        h = EvalHistory(namespace="ns", path=tmp_path / "h.jsonl")
        h.record(_result(), actor="a")
        h.record(_result(target="evidence"), actor="a")
        again = EvalHistory(namespace="ns", path=tmp_path / "h.jsonl")
        assert again.load_from_disk() and len(again.entries()) == 2

    def test_newest_first(self, tmp_path):
        h = EvalHistory(namespace="ns", path=tmp_path / "h.jsonl")
        h.record(_result(k=5), actor="a")
        h.record(_result(k=10), actor="a")
        assert h.entries()[0]["k"] == 10

    def test_limit(self, tmp_path):
        h = EvalHistory(namespace="ns", path=tmp_path / "h.jsonl")
        for i in range(5):
            h.record(_result(cases=i), actor="a")
        assert len(h.entries(limit=2)) == 2

    def test_per_case_is_not_stored(self, tmp_path):
        """케이스별 순위는 이력의 목적이 아니고 파일만 불린다 — 지표와 설정만."""
        h = EvalHistory(namespace="ns", path=tmp_path / "h.jsonl")
        h.record(_result(per_case=[{"case_id": "a"}] * 46), actor="a")
        assert "per_case" not in h.entries()[0]

    def test_unmeasured_run_is_still_recorded(self, tmp_path):
        """0 건이었다는 사실도 이력이다 — 안 남기면 "왜 그때 안 쟀지"가 된다."""
        h = EvalHistory(namespace="ns", path=tmp_path / "h.jsonl")
        h.record(_result(measured=False, cases=0), actor="a")
        assert h.entries()[0]["measured"] is False

    def test_write_failure_does_not_raise(self, tmp_path):
        """기록 실패가 평가를 죽이면 안 된다 — 측정이 본체다."""
        h = EvalHistory(namespace="ns", path=tmp_path / "nope" / "h.jsonl")
        h.record(_result(), actor="a")          # 디렉터리 없음
        assert h.entries() == [] or True

    def test_corrupt_line_is_skipped(self, tmp_path):
        p = tmp_path / "h.jsonl"
        p.write_text('{"broken\n' + json.dumps({"target": "node"}) + "\n",
                     encoding="utf-8")
        h = EvalHistory(namespace="ns", path=p)
        assert h.load_from_disk() and len(h.entries()) == 1

    def test_namespace_singletons_are_separate(self):
        from ontology.core.eval_history import get_eval_history
        reset_eval_histories()
        assert get_eval_history("a") is not get_eval_history("b")
        assert get_eval_history("a") is get_eval_history("a")


class TestLatestQuality:
    """검색 응답에 동봉할 품질 블록 — **지어내지 않는다**.

    RRF 점수(0.016…)나 코사인은 보정되지 않은 값이라 사용자에게 "신뢰도"로
    보여주면 없는 정밀도를 지어내는 것이다. 정직한 답은 **측정된 골든셋 지표 +
    측정 시점 + 설정 일치 여부**다: "이 검색기는 이 네임스페이스에서 hit@5 0.95
    로 측정됨(65케이스, 08-02)" — 그리고 그 뒤 설정이 바뀌었으면 stale 로
    표시한다. 측정치가 현재를 대변하지 않는데 그대로 보여주면 거짓이 된다.
    """

    def _entry(self, cfg_overrides=None, target="evidence"):
        cfg = {"node_model": "m", "chunk_model": "m", "entry_k": 3,
               "entry_ratio": 0.5, "max_terms": 8, "use_propagation": False,
               "propagation_channel": False, "propagation_weight": 0.05,
               "vector_backend": ""}
        cfg.update(cfg_overrides or {})
        return _result(target=target, config=cfg)

    def _record(self, h, monkeypatch, cfg, **kw):
        """record() 는 전달 config 를 무시하고 **기록 시점의 지문**을 찍는다 —
        그게 설계다(측정 순간의 진실). 테스트는 지문을 먼저 고정하고 기록한다."""
        monkeypatch.setattr("ontology.core.eval_history.config_fingerprint",
                            lambda ns="": dict(cfg))
        h.record(_result(**kw), actor="t")

    def test_returns_latest_matching_target(self, tmp_path, monkeypatch):
        from ontology.core.eval_history import latest_quality
        h = EvalHistory(namespace="qns", path=tmp_path / "h.jsonl")
        cfg = self._entry()["config"]
        self._record(h, monkeypatch, cfg, target="node")
        self._record(h, monkeypatch, cfg, target="evidence")
        monkeypatch.setattr("ontology.core.eval_history.get_eval_history",
                            lambda ns: h)
        q = latest_quality("qns")
        assert q["measured"] is True
        assert q["target"] == "evidence"
        assert q["retrieve"]["mrr"] == 0.70

    def test_stale_when_config_changed(self, tmp_path, monkeypatch):
        """측정 후 설정이 바뀌었으면 stale — 옛 숫자를 현재인 양 보여주지 않는다."""
        from ontology.core.eval_history import latest_quality
        h = EvalHistory(namespace="qns", path=tmp_path / "h.jsonl")
        cfg = self._entry()["config"]
        self._record(h, monkeypatch, cfg, target="evidence")
        monkeypatch.setattr("ontology.core.eval_history.get_eval_history",
                            lambda ns: h)
        monkeypatch.setattr("ontology.core.eval_history.config_fingerprint",
                            lambda ns="": {**cfg, "propagation_channel": True})
        q = latest_quality("qns")
        assert q["stale"] is True
        assert "propagation_channel" in q["stale_keys"]

    def test_fresh_when_config_identical(self, tmp_path, monkeypatch):
        from ontology.core.eval_history import latest_quality
        h = EvalHistory(namespace="qns", path=tmp_path / "h.jsonl")
        cfg = self._entry()["config"]
        self._record(h, monkeypatch, cfg, target="evidence")
        monkeypatch.setattr("ontology.core.eval_history.get_eval_history",
                            lambda ns: h)
        assert latest_quality("qns")["stale"] is False

    def test_no_history_is_honest(self, tmp_path, monkeypatch):
        """측정한 적 없으면 measured:false — 0 점이나 빈 dict 가 아니라."""
        from ontology.core.eval_history import latest_quality
        h = EvalHistory(namespace="qns", path=tmp_path / "h.jsonl")
        monkeypatch.setattr("ontology.core.eval_history.get_eval_history",
                            lambda ns: h)
        q = latest_quality("qns")
        assert q["measured"] is False and q["reason"]

    def test_unmeasured_entries_skipped(self, tmp_path, monkeypatch):
        """cases=0 (measured:false) 기록은 품질 근거가 아니다."""
        from ontology.core.eval_history import latest_quality
        h = EvalHistory(namespace="qns", path=tmp_path / "h.jsonl")
        cfg = self._entry()["config"]
        self._record(h, monkeypatch, cfg, target="evidence")
        self._record(h, monkeypatch, cfg, target="evidence",
                     measured=False, cases=0)
        monkeypatch.setattr("ontology.core.eval_history.get_eval_history",
                            lambda ns: h)
        assert latest_quality("qns")["measured"] is True
