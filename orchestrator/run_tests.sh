#!/bin/bash
#
# Workflow Orchestrator Test Runner
#
# Run this script after ANY modification to the orchestrator module
# to ensure no regressions.
#
# Usage:
#   ./ontology/orchestrator/run_tests.sh          # Run all tests
#   ./ontology/orchestrator/run_tests.sh quick    # Run unit tests only
#   ./ontology/orchestrator/run_tests.sh full     # Run with verbose output
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"

echo "============================================================"
echo "🧪 Workflow Orchestrator Test Suite"
echo "============================================================"
echo ""

MODE="${1:-all}"

case "$MODE" in
    quick)
        echo "Running quick unit tests..."
        python "$SCRIPT_DIR/tests/test_models_standalone.py"
        ;;
    full)
        echo "Running full test suite..."
        echo ""

        echo "📦 Unit Tests (models)..."
        python "$SCRIPT_DIR/tests/test_models_standalone.py"

        echo ""
        echo "🔗 Integration Tests..."
        python "$SCRIPT_DIR/test_orchestrator.py"
        ;;
    integration)
        echo "Running integration tests..."
        python "$SCRIPT_DIR/test_orchestrator.py"
        ;;
    all|*)
        echo "Running all tests..."
        echo ""

        echo "📦 Unit Tests (models)..."
        python "$SCRIPT_DIR/tests/test_models_standalone.py"

        echo ""
        echo "🔗 Integration Tests..."
        python "$SCRIPT_DIR/test_orchestrator.py"

        echo ""
        echo "============================================================"
        echo "✅ All tests completed successfully!"
        echo "============================================================"
        ;;
esac
