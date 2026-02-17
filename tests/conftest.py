"""Conftest that fixes sys.path for standalone ontology testing."""
import sys
import os

# Fix: Remove Logos/ parent from sys.path and import path to prevent
# Logos/__init__.py from being loaded (it's an old SDK artifact)
_ontology_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_logos_root = os.path.dirname(_ontology_root)

# Block the parent Logos package from being importable
if _logos_root in sys.path:
    sys.path.remove(_logos_root)

# Ensure ontology root is importable
if _ontology_root not in sys.path:
    sys.path.insert(0, _ontology_root)

# Pre-empt: if 'Logos' is already in sys.modules, remove it
for mod_name in list(sys.modules.keys()):
    if mod_name == 'Logos' or mod_name.startswith('Logos.'):
        del sys.modules[mod_name]
