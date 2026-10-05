"""Unit tests for required_resources on capability_gap (P1.2).

P1.2 adds a `required_resources` field to the capability_gap that QueryPlanner
emits, so logos_api can later place the generated agent on a resource-matched
ACP node (affinity placement). These tests cover the deterministic, pure parts:
  - normalize_capability_gap: guarantees a clean list `required_resources`
  - detect_explicit_capability_gap: safety-net dict includes the field

The LLM prompt change (asking the model to populate required_resources) is not
unit-tested here — it is non-deterministic — but normalization guarantees the
field always exists downstream regardless of what the LLM returns.

Run: cd Logos && python -m pytest ontology/orchestrator/test_capability_gap_resources.py -v
"""

import os
import sys

_LOGOS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _LOGOS)                       # for `import ontology`
sys.path.insert(0, os.path.join(_LOGOS, "ontology"))  # ontology-internal `from core.models import`

from ontology.orchestrator.query_planner import (  # noqa: E402
    detect_explicit_capability_gap,
    normalize_capability_gap,
)


# ── normalize_capability_gap ────────────────────────────────────────────

def test_normalize_none_passes_through():
    assert normalize_capability_gap(None) is None


def test_normalize_adds_missing_required_resources():
    gap = {"detected": True, "missing_capabilities": ["mastodon_api"]}
    out = normalize_capability_gap(gap)
    assert out["required_resources"] == []


def test_normalize_preserves_existing_resources():
    gap = {"detected": True, "required_resources": ["desktop:kakaotalk", "region:kr"]}
    out = normalize_capability_gap(gap)
    assert out["required_resources"] == ["desktop:kakaotalk", "region:kr"]


def test_normalize_coerces_non_list_to_empty():
    gap = {"detected": True, "required_resources": "desktop:kakaotalk"}
    out = normalize_capability_gap(gap)
    assert out["required_resources"] == []


def test_normalize_strips_blanks_and_casts_to_str():
    gap = {"detected": True, "required_resources": ["api:mastodon", "  ", "", "region:kr"]}
    out = normalize_capability_gap(gap)
    assert out["required_resources"] == ["api:mastodon", "region:kr"]


# ── detect_explicit_capability_gap (safety net) ─────────────────────────

def test_explicit_gap_includes_required_resources_key():
    gap = detect_explicit_capability_gap("Mastodon 에이전트 만들어줘")
    assert gap is not None
    assert gap["detected"] is True
    assert "required_resources" in gap
    assert gap["required_resources"] == []


def test_non_creation_query_returns_none():
    assert detect_explicit_capability_gap("오늘 서울 날씨 알려줘") is None


# Direct runner — repo-root stray Logos/__init__.py breaks pytest collection
# (see memory: reference_acp_pytest_stray_init), so run as a plain script.
if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    _failed = 0
    for _t in _tests:
        try:
            _t()
            print(f"  PASS  {_t.__name__}")
        except Exception as e:  # noqa: BLE001
            _failed += 1
            print(f"  FAIL  {_t.__name__}: {e}")
    print(f"\n{len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)
