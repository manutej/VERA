#!/bin/bash
# VERA Smoke Test - Quick validation that basic functionality works
# Usage: ./tests/manual/smoke_test.sh

set -e  # Exit on error

echo "🔥 VERA Smoke Test"
echo "=================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
PASS=0
FAIL=0

# Helper function
test_step() {
    echo -n "$1... "
}

test_pass() {
    echo -e "${GREEN}✅ PASS${NC}"
    ((PASS++))
}

test_fail() {
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Error: $1"
    ((FAIL++))
}

# Change to project root
cd "$(dirname "$0")/../.."

echo "📍 Working directory: $(pwd)"
echo ""

# Test 1: Go version
test_step "Checking Go installation"
if command -v go &> /dev/null; then
    GO_VERSION=$(go version | awk '{print $3}')
    test_pass
    echo "   Version: $GO_VERSION"
else
    test_fail "Go not installed. Run: brew install go"
fi

# Test 2: Go module
test_step "Verifying go.mod"
if [ -f "go.mod" ]; then
    MODULE=$(head -1 go.mod | awk '{print $2}')
    test_pass
    echo "   Module: $MODULE"
else
    test_fail "go.mod not found"
fi

# Test 3: Dependencies
test_step "Checking dependencies"
if go mod download 2>&1 | grep -q "go: downloading"; then
    test_pass
elif go mod verify &> /dev/null; then
    test_pass
    echo "   All dependencies present"
else
    test_fail "Dependency download failed"
fi

# Test 4: Compilation
test_step "Compiling code"
if go build ./... &> /dev/null; then
    test_pass
else
    test_fail "Compilation failed. Run: go build ./..."
fi

# Test 5: Property tests
test_step "Running property tests (100,000 iterations)"
if go test ./tests/property -v 2>&1 | grep -q "PASS"; then
    test_pass
    PROP_TIME=$(go test ./tests/property 2>&1 | grep "ok" | awk '{print $4}')
    echo "   Runtime: $PROP_TIME"
else
    test_fail "Property tests failed"
fi

# Test 6: Unit tests
test_step "Running unit tests (pkg/core)"
if go test ./pkg/core 2>&1 | grep -q "PASS"; then
    test_pass
    COVERAGE=$(go test ./pkg/core -cover 2>&1 | grep "coverage:" | awk '{print $2}')
    echo "   Coverage: $COVERAGE"
else
    test_fail "Unit tests failed"
fi

# Test 7: Ollama check
test_step "Checking Ollama availability"
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    test_pass
    MODEL=$(curl -s http://localhost:11434/api/tags | grep -o "nomic-embed-text" | head -1)
    if [ -n "$MODEL" ]; then
        echo "   Model: $MODEL available"
    else
        echo -e "   ${YELLOW}⚠️  Warning: nomic-embed-text not found. Run: ollama pull nomic-embed-text${NC}"
    fi
else
    test_fail "Ollama not running. Start with: ollama serve"
fi

# Test 8: API key check
test_step "Checking ANTHROPIC_API_KEY"
if [ -n "$ANTHROPIC_API_KEY" ]; then
    KEY_PREFIX=$(echo $ANTHROPIC_API_KEY | cut -c1-15)
    if [[ $KEY_PREFIX == sk-ant-api03-* ]]; then
        test_pass
        echo "   Key format: ${KEY_PREFIX}***"
    else
        test_fail "API key format incorrect (should start with sk-ant-api03-)"
    fi
else
    test_fail "ANTHROPIC_API_KEY not set. Export it first."
fi

# Summary
echo ""
echo "===================="
echo "📊 Test Summary"
echo "===================="
echo -e "${GREEN}Passed: $PASS${NC}"
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Failed: $FAIL${NC}"
fi

if [ $FAIL -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All smoke tests passed!${NC}"
    echo "Your VERA environment is ready."
    echo ""
    echo "Next steps:"
    echo "  1. Run integration tests: go test ./tests/integration -v"
    echo "  2. Run manual tests: go test ./tests/manual -v"
    echo "  3. Start M4 implementation"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    echo "Review errors above and fix before proceeding."
    echo ""
    echo "Common fixes:"
    echo "  - Install Go: brew install go"
    echo "  - Start Ollama: ollama serve"
    echo "  - Pull model: ollama pull nomic-embed-text"
    echo "  - Set API key: export ANTHROPIC_API_KEY='sk-ant-api03-...'"
    exit 1
fi
