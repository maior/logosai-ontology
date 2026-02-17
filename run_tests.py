#!/usr/bin/env python3
"""
Run ontology tests.

Usage:
    python run_tests.py          # Run all tests
    python run_tests.py -v       # Verbose output
"""
import sys
import os

# Fix import path: ensure ontology/ is the package root, not Logos/
ontology_root = os.path.dirname(os.path.abspath(__file__))
logos_root = os.path.dirname(ontology_root)

# Remove Logos/ parent from path (its __init__.py conflicts)
sys.path = [p for p in sys.path if os.path.abspath(p) != logos_root]

# Add ontology root
if ontology_root not in sys.path:
    sys.path.insert(0, ontology_root)

# Clear any cached Logos module
for mod in list(sys.modules.keys()):
    if mod == "Logos" or mod.startswith("Logos."):
        del sys.modules[mod]

if __name__ == "__main__":
    import pytest
    args = sys.argv[1:] or ["-v"]
    sys.exit(pytest.main([*args, "tests/", f"--rootdir={ontology_root}"]))
