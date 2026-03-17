"""Conftest that fixes sys.path for standalone ontology testing.

Logos/__init__.py is an old SDK artifact with broken imports.
We block it via a dummy sys.modules entry, then keep Logos/ in
sys.path so that `from ontology.ml.config import ...` resolves
correctly (Python finds ontology/ under Logos/).
"""
import sys
import os
import types

_ontology_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_logos_root = os.path.dirname(_ontology_root)

# 1. Block Logos/__init__.py from being loaded by inserting a dummy module.
#    This prevents the broken `from .agent import LogosAIAgent` inside it.
for mod_name in list(sys.modules.keys()):
    if mod_name == "Logos" or mod_name.startswith("Logos."):
        del sys.modules[mod_name]

_dummy_logos = types.ModuleType("Logos")
_dummy_logos.__path__ = [_logos_root]
_dummy_logos.__file__ = os.path.join(_logos_root, "__init__.py")
_dummy_logos.__package__ = "Logos"
sys.modules["Logos"] = _dummy_logos

# 2. Ensure Logos/ is in sys.path so `from ontology.x import y` works.
#    (ontology/ lives at Logos/ontology/)
if _logos_root not in sys.path:
    sys.path.insert(0, _logos_root)
