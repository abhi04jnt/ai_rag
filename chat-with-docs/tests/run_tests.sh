#!/bin/bash
# Test runner script for chat-with-docs

set -e

echo "🧪 Chat With Your Docs - Test Suite"
echo "===================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "Consider activating a virtual environment first"
    echo ""
fi

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python -c "import pytest, httpx" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install -e . -q
    echo "✅ Dependencies installed"
}

echo ""
echo "🔧 Test Configuration:"
echo "  - Python: $(python --version)"
echo "  - Pytest: $(pytest --version | head -n 1)"
echo "  - Working Dir: $(pwd)"
echo ""

# Parse command line arguments
TEST_TYPE="${1:-all}"
VERBOSE="${2:--v}"

case "$TEST_TYPE" in
    unit)
        echo "🏃 Running Unit Tests..."
        pytest tests/test_unit.py $VERBOSE
        ;;
    e2e)
        echo "🏃 Running E2E Tests..."
        echo "⚠️  Note: E2E tests require OpenAI API key and may take several minutes"
        pytest tests/test_e2e.py $VERBOSE
        ;;
    eval)
        echo "🏃 Running RAG Evaluation Tests..."
        echo "⚠️  Note: Requires indexed documents and OpenAI API key"
        pytest tests/test_evaluation.py $VERBOSE
        ;;
    quick)
        echo "🏃 Running Quick Tests (unit only)..."
        pytest tests/test_unit.py -m "not slow" $VERBOSE
        ;;
    all)
        echo "🏃 Running All Tests..."
        pytest tests/ $VERBOSE
        ;;
    coverage)
        echo "🏃 Running Tests with Coverage..."
        pytest tests/ --cov=src --cov-report=html --cov-report=term $VERBOSE
        echo ""
        echo "📊 Coverage report: htmlcov/index.html"
        ;;
    *)
        echo "Usage: $0 [unit|e2e|eval|quick|all|coverage] [pytest-options]"
        echo ""
        echo "Options:"
        echo "  unit     - Run unit tests only"
        echo "  e2e      - Run end-to-end tests only"
        echo "  eval     - Run RAG evaluation tests (quality metrics)"
        echo "  quick    - Run quick tests (excludes slow tests)"
        echo "  all      - Run all tests (default)"
        echo "  coverage - Run all tests with coverage report"
        echo ""
        echo "Examples:"
        echo "  $0 unit"
        echo "  $0 e2e -v"
        echo "  $0 eval"
        echo "  $0 coverage"
        exit 1
        ;;
esac

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
