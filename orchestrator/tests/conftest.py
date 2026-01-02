"""
pytest configuration for orchestrator tests

Ensures proper import paths to avoid circular import issues.
"""

import sys
import os

# Add orchestrator directory to path before pytest imports anything else
_orchestrator_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ontology_dir = os.path.dirname(_orchestrator_dir)
_project_root = os.path.dirname(_ontology_dir)

# Add paths in order of priority
for path in [_orchestrator_dir, _ontology_dir, _project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Prevent pytest from collecting logosai tests
collect_ignore_glob = ["**/logosai/**"]
