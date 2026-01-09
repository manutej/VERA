# VERA Manual Testing Guide

**Purpose**: Verify that VERA actually works beyond unit tests
**Status**: M1-M3 Complete, M4-M6 Pending
**Date**: 2026-01-07

---

## Why Manual Testing Matters

**The Problem**: Unit tests passing ≠ system working

Unit tests verify:
- ✅ Individual functions work in isolation
- ✅ Categorical laws hold mathematically
- ✅ Error handling works correctly

**But they DON'T verify**:
- ❌ Real LLM APIs are accessible
- ❌ Document parsing works on actual files
- ❌ End-to-end pipelines compose correctly
- ❌ Performance meets latency budgets

**This guide** provides step-by-step instructions to verify VERA actually works in the real world.

---

## Prerequisites

### Required Software

| Tool | Version | Installation | Purpose |
|------|---------|--------------|---------|
| **Go** | ≥1.25 | `brew install go` | Run tests, compile code |
| **Ollama** | ≥0.13 | `brew install ollama` | Local embeddings |
| **nomic-embed-text** | v1.5 | `ollama pull nomic-embed-text` | Embedding model |
| **Git** | Any | Pre-installed on Mac | Version control |

### Required API Keys

| Service | Key Name | Where to Get | Cost |
|---------|----------|--------------|------|
| **Anthropic** | `ANTHROPIC_API_KEY` | https://console.anthropic.com | $3-15 per 1M tokens |

**Set API Key**:
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
# Or add to ~/.zshrc for persistence
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.zshrc
```

**⚠️ Security**: Never commit API keys to git! They should only exist in:
- Environment variables (export command)
- Shell profile (~/.zshrc)
- `.env` file (with `.env` in .gitignore)

---

## Test Suite Overview

| Test Level | What It Verifies | Time | When to Run |
|------------|------------------|------|-------------|
| **Smoke Test** | Basic functionality works | 30s | After every code change |
| **Integration Test** | Real APIs work | 10s | Before committing |
| **Functional Test** | End-to-end scenarios | 5-10min | Before deploying |
| **Performance Test** | Latency budgets met | 2-5min | Weekly or before release |

---

## Level 1: Smoke Test (30 seconds)

**Goal**: Verify the most basic functionality works

### Step 1.1: Clone and Setup

```bash
cd /Users/manu/Documents/LUXOR/VERA

# Verify Go installation
go version
# Expected: go version go1.25.3 darwin/arm64 (or similar)

# Verify dependencies
go mod download
# Expected: No errors, dependencies downloaded
```

**✅ Success Criteria**: Commands complete without errors

---

### Step 1.2: Run Property Tests

```bash
go test ./tests/property -v
```

**Expected Output**:
```
=== RUN   TestFunctorLaws
+ Functor Law 1: Map(id) = id: OK, passed 100 tests.
+ Functor Law 2: Map(g ∘ f) = Map(g) ∘ Map(f): OK, passed 100 tests.
+ Map propagates errors unchanged: OK, passed 100 tests.
--- PASS: TestFunctorLaws (0.00s)

=== RUN   TestMonadLaws
+ Monad Law 1: FlatMap(Ok(a), f) = f(a): OK, passed 100 tests.
+ Monad Law 2: FlatMap(m, Ok) = m: OK, passed 100 tests.
+ Monad Law 3: Associativity: OK, passed 100 tests.
+ FlatMap propagates errors unchanged: OK, passed 100 tests.
--- PASS: TestMonadLaws (0.00s)

=== RUN   TestPipelineCompositionLaws
+ Pipeline Composition: Associativity: OK, passed 100 tests.
+ Pipeline Composition: Left Identity: OK, passed 100 tests.
+ Pipeline Composition: Right Identity: OK, passed 100 tests.
--- PASS: TestPipelineCompositionLaws (0.00s)

PASS
ok  	github.com/manu/vera/tests/property	0.376s
```

**✅ Success Criteria**:
- All 10 laws pass
- Runtime < 1 second
- No errors or panics

**❌ If Failed**:
- Check Go version (must be ≥1.25)
- Run `go clean -testcache` to clear cache
- Check for compilation errors

---

### Step 1.3: Run Unit Tests

```bash
go test ./pkg/core -v
```

**Expected Output**:
```
=== RUN   TestResultConstructors
--- PASS: TestResultConstructors (0.00s)

=== RUN   TestResultConvenienceFunctions
--- PASS: TestResultConvenienceFunctions (0.00s)

[... 77 more tests ...]

PASS
coverage: 89.5% of statements
ok  	github.com/manu/vera/pkg/core	0.274s
```

**✅ Success Criteria**:
- 79 unit tests pass
- Coverage ≥ 89%
- Runtime < 1 second

---

## Level 2: Integration Tests (10 seconds)

**Goal**: Verify real APIs work (Ollama + Anthropic)

### Step 2.1: Verify Ollama is Running

```bash
# Start Ollama service (if not running)
ollama serve &

# Check service is running
curl http://localhost:11434/api/tags

# Expected output: JSON with list of models
# {"models":[{"name":"nomic-embed-text:latest",...}]}
```

**✅ Success Criteria**: Ollama responds with model list

**❌ If Failed**:
```bash
# Check if Ollama is installed
which ollama

# If not installed:
brew install ollama

# Pull the model
ollama pull nomic-embed-text

# Verify model is downloaded
ollama list
# Expected: nomic-embed-text listed
```

---

### Step 2.2: Test Ollama Embeddings

```bash
go test ./tests/integration -run TestOllamaEmbeddings -v
```

**Expected Output**:
```
=== RUN   TestOllamaEmbeddings
=== RUN   TestOllamaEmbeddings/single_text_embedding
    providers_test.go:126: ✅ Single embedding: 768 dims, 6965.00ms latency, 10 tokens
=== RUN   TestOllamaEmbeddings/batch_embedding
    providers_test.go:145: ✅ Batch embedding: 3 texts, 115.00ms latency
=== RUN   TestOllamaEmbeddings/matryoshka_truncation
    providers_test.go:161: ✅ Matryoshka truncation: 512 dims (99.5% quality)
=== RUN   TestOllamaEmbeddings/error:_empty_texts
    providers_test.go:169: ✅ Correctly rejected empty texts
=== RUN   TestOllamaEmbeddings/error:_dimensions_too_large
    providers_test.go:177: ✅ Correctly rejected dimensions > 768
=== RUN   TestOllamaEmbeddings/timeout_handling
    providers_test.go:186: ✅ Correctly handled context timeout
--- PASS: TestOllamaEmbeddings (7.10s)
PASS
```

**✅ Success Criteria**:
- All 6 sub-tests pass
- First embedding takes ~7s (model loading)
- Subsequent embeddings take < 200ms
- 768-dimensional vectors returned

**❌ If Failed**:
| Error | Cause | Fix |
|-------|-------|-----|
| `connection refused` | Ollama not running | Run `ollama serve` |
| `model not found` | Model not downloaded | Run `ollama pull nomic-embed-text` |
| `timeout` | Model loading slow | Increase timeout or wait for first load |

---

### Step 2.3: Test Anthropic Completions

```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY
# Expected: sk-ant-api03-... (your actual key)

# Run Anthropic tests
go test ./tests/integration -run TestAnthropicCompletion -v
```

**Expected Output**:
```
=== RUN   TestAnthropicCompletion
=== RUN   TestAnthropicCompletion/basic_completion
    providers_test.go:244: ✅ Completion: '4' (2647.00ms, 20→5 tokens)
=== RUN   TestAnthropicCompletion/system_prompt
    providers_test.go:268: ✅ System prompt completion: 'Your favorite color is blue.'
=== RUN   TestAnthropicCompletion/error:_empty_prompt
    providers_test.go:283: ✅ Correctly rejected empty prompt
=== RUN   TestAnthropicCompletion/timeout_handling
    providers_test.go:300: ✅ Correctly handled context timeout
--- PASS: TestAnthropicCompletion (5.09s)
PASS
```

**✅ Success Criteria**:
- All 4 sub-tests pass
- API responds in < 3 seconds
- Math completion returns "4"
- System prompt injection works

**❌ If Failed**:
| Error | Cause | Fix |
|-------|-------|-----|
| `API key not found` | Key not set | `export ANTHROPIC_API_KEY="sk-..."` |
| `invalid API key` | Wrong key | Get new key from console.anthropic.com |
| `rate limit` | Too many requests | Wait 60s and retry |
| `timeout` | Network slow | Increase timeout in test |

---

### Step 2.4: Test Document Parsing

```bash
go test ./tests/integration -run TestMarkdownParsing -v
```

**Expected Output**:
```
=== RUN   TestMarkdownParsing
=== RUN   TestMarkdownParsing/parse_sample.md
    ingestion_test.go:42: ✅ Parse sample.md: 217 words, 1866 chars, 0.28ms
=== RUN   TestMarkdownParsing/parse_short.md
    ingestion_test.go:65: ✅ Parse short.md: 23 words, 0.069ms
=== RUN   TestMarkdownParsing/file_not_found
    ingestion_test.go:80: ✅ File not found: [VALIDATION] error
=== RUN   TestMarkdownParsing/wrong_format
    ingestion_test.go:95: ✅ Wrong format: [VALIDATION] error
=== RUN   TestMarkdownParsing/context_cancellation
    ingestion_test.go:110: ✅ Context cancellation handled
--- PASS: TestMarkdownParsing (0.00s)
PASS
```

**✅ Success Criteria**:
- All 5 sub-tests pass
- Parsing speed < 1ms per file
- Error handling correct

---

## Level 3: Functional Tests (5-10 minutes)

**Goal**: Test realistic end-to-end scenarios

### Test 3.1: Embedding Pipeline Test

**Scenario**: Process a document through the complete embedding pipeline

Create test file `tests/manual/embedding_pipeline_test.go`:

```go
package manual

import (
	"context"
	"testing"
	"time"

	"github.com/manu/vera/pkg/core"
	"github.com/manu/vera/pkg/ingestion"
	"github.com/manu/vera/pkg/providers"
)

func TestEmbeddingPipeline(t *testing.T) {
	// Skip in CI (requires Ollama running)
	if testing.Short() {
		t.Skip("Skipping manual test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Step 1: Parse a Markdown document
	parser := ingestion.NewMarkdownParser()
	parseResult := parser.Parse(ctx, "../../tests/fixtures/sample.md")

	if parseResult.IsErr() {
		t.Fatalf("Parse failed: %v", parseResult.UnwrapErr())
	}

	doc := parseResult.Unwrap()
	t.Logf("✅ Parsed document: %d words, %s format", doc.WordCount(), doc.Format)

	// Step 2: Generate embeddings for document content
	embedder := providers.NewOllamaEmbeddingProvider("http://localhost:11434", "nomic-embed-text", 768)

	request := providers.EmbeddingRequest{
		Texts: []string{doc.Content},
	}

	embedResult := embedder.Embed(ctx, request)
	if embedResult.IsErr() {
		t.Fatalf("Embedding failed: %v", embedResult.UnwrapErr())
	}

	response := embedResult.Unwrap()
	t.Logf("✅ Generated embedding: %d dimensions, %.2fms latency",
		len(response.Embeddings[0]), response.LatencyMs)

	// Step 3: Verify embedding quality
	embedding := response.Embeddings[0]

	if len(embedding) != 768 {
		t.Errorf("Expected 768 dimensions, got %d", len(embedding))
	}

	// Verify L2 norm is ~1.0 (normalized)
	var norm float64
	for _, val := range embedding {
		norm += float64(val) * float64(val)
	}
	norm = norm

	if norm < 0.99 || norm > 1.01 {
		t.Errorf("Expected L2 norm ~1.0, got %.4f (not normalized)", norm)
	}

	t.Logf("✅ Embedding quality verified: L2 norm = %.4f", norm)

	// Step 4: Verify pipeline composition
	// Parse → Embed pipeline (using Pipeline operators)
	parsePipeline := core.NewPipeline(func(ctx context.Context, path string) core.Result[ingestion.Document] {
		return parser.Parse(ctx, path)
	})

	embedPipeline := core.NewPipeline(func(ctx context.Context, doc ingestion.Document) core.Result[[]float32] {
		req := providers.EmbeddingRequest{Texts: []string{doc.Content}}
		result := embedder.Embed(ctx, req)
		return core.Map(result, func(resp providers.EmbeddingResponse) []float32 {
			return resp.Embeddings[0]
		})
	})

	// Compose pipelines with → operator
	composed := core.Then(parsePipeline, embedPipeline)

	finalResult := composed.Run(ctx, "../../tests/fixtures/sample.md")
	if finalResult.IsErr() {
		t.Fatalf("Composed pipeline failed: %v", finalResult.UnwrapErr())
	}

	finalEmbedding := finalResult.Unwrap()
	t.Logf("✅ Pipeline composition works: %d-dim embedding from file path", len(finalEmbedding))
}
```

**Run the test**:
```bash
go test ./tests/manual -run TestEmbeddingPipeline -v
```

**✅ Success Criteria**:
- Document parses successfully
- Embedding generates 768 dimensions
- L2 norm ≈ 1.0 (normalized)
- Pipeline composition works
- Total time < 30 seconds

---

### Test 3.2: LLM Completion Test

**Scenario**: Generate a completion and verify response quality

Create `tests/manual/llm_completion_test.go`:

```go
package manual

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/manu/vera/pkg/providers"
)

func TestLLMCompletion(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping manual test")
	}

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Fatal("ANTHROPIC_API_KEY not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	llm := providers.NewAnthropicProvider(apiKey, "claude-sonnet-4-20250514")

	// Test 1: Basic reasoning
	t.Run("mathematical_reasoning", func(t *testing.T) {
		request := providers.CompletionRequest{
			Prompt:      "What is 2 + 2? Respond with ONLY the number, no explanation.",
			Temperature: 0.0,
			MaxTokens:   10,
		}

		result := llm.Complete(ctx, request)
		if result.IsErr() {
			t.Fatalf("Completion failed: %v", result.UnwrapErr())
		}

		response := result.Unwrap()
		answer := strings.TrimSpace(response.Content)

		if answer != "4" {
			t.Errorf("Expected '4', got '%s'", answer)
		}

		t.Logf("✅ Math reasoning: '%s' (%.0fms, %d tokens)",
			answer, response.LatencyMs, response.Usage.TotalTokens)
	})

	// Test 2: Document analysis
	t.Run("document_analysis", func(t *testing.T) {
		document := `
		# Product Specification
		The widget costs $49.99 and ships within 5-7 business days.
		Color options: Red, Blue, Green.
		`

		request := providers.CompletionRequest{
			SystemPrompt: "You are a precise information extractor. Answer questions based ONLY on the provided document.",
			Prompt:       "What is the price? Respond with ONLY the price, no explanation.",
			Temperature:  0.0,
			MaxTokens:    20,
		}

		result := llm.Complete(ctx, request)
		if result.IsErr() {
			t.Fatalf("Completion failed: %v", result.UnwrapErr())
		}

		response := result.Unwrap()
		answer := strings.TrimSpace(response.Content)

		if !strings.Contains(answer, "49.99") {
			t.Errorf("Expected price to contain '49.99', got '%s'", answer)
		}

		t.Logf("✅ Document analysis: '%s' (%.0fms)", answer, response.LatencyMs)
	})

	// Test 3: Token tracking
	t.Run("token_tracking", func(t *testing.T) {
		request := providers.CompletionRequest{
			Prompt:      "Write exactly 10 words.",
			Temperature: 0.0,
			MaxTokens:   50,
		}

		result := llm.Complete(ctx, request)
		if result.IsErr() {
			t.Fatalf("Completion failed: %v", result.UnwrapErr())
		}

		response := result.Unwrap()

		if response.Usage.InputTokens == 0 {
			t.Error("InputTokens should be > 0")
		}
		if response.Usage.OutputTokens == 0 {
			t.Error("OutputTokens should be > 0")
		}
		if response.Usage.TotalTokens == 0 {
			t.Error("TotalTokens should be > 0")
		}

		t.Logf("✅ Token tracking: %d input + %d output = %d total",
			response.Usage.InputTokens, response.Usage.OutputTokens, response.Usage.TotalTokens)
	})
}
```

**Run the test**:
```bash
go test ./tests/manual -run TestLLMCompletion -v
```

**✅ Success Criteria**:
- All 3 sub-tests pass
- Math reasoning returns "4"
- Document analysis extracts price
- Token tracking works
- Total time < 30 seconds

---

## Level 4: Performance Tests (2-5 minutes)

**Goal**: Verify latency budgets are met

### Test 4.1: Latency Budget Validation

Create `tests/manual/performance_test.go`:

```go
package manual

import (
	"context"
	"testing"
	"time"

	"github.com/manu/vera/pkg/providers"
)

func TestLatencyBudgets(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test")
	}

	ctx := context.Background()

	// Test 1: Single embedding latency (after warmup)
	t.Run("embedding_latency", func(t *testing.T) {
		embedder := providers.NewOllamaEmbeddingProvider("http://localhost:11434", "nomic-embed-text", 768)

		// Warmup (first call loads model)
		warmupCtx, _ := context.WithTimeout(ctx, 10*time.Second)
		_ = embedder.Embed(warmupCtx, providers.EmbeddingRequest{Texts: []string{"warmup"}})

		// Actual test
		start := time.Now()
		result := embedder.Embed(ctx, providers.EmbeddingRequest{
			Texts: []string{"The quick brown fox jumps over the lazy dog."},
		})
		latency := time.Since(start)

		if result.IsErr() {
			t.Fatalf("Embedding failed: %v", result.UnwrapErr())
		}

		// Budget: 100ms for single query embedding
		if latency > 100*time.Millisecond {
			t.Errorf("Embedding latency %v exceeds budget 100ms", latency)
		}

		t.Logf("✅ Embedding latency: %v (budget: 100ms)", latency)
	})

	// Test 2: Batch embedding efficiency
	t.Run("batch_embedding_efficiency", func(t *testing.T) {
		embedder := providers.NewOllamaEmbeddingProvider("http://localhost:11434", "nomic-embed-text", 768)

		texts := make([]string, 10)
		for i := range texts {
			texts[i] = "Sample text for batch embedding test."
		}

		start := time.Now()
		result := embedder.Embed(ctx, providers.EmbeddingRequest{Texts: texts})
		latency := time.Since(start)

		if result.IsErr() {
			t.Fatalf("Batch embedding failed: %v", result.UnwrapErr())
		}

		// Budget: 800ms for batch of 10 documents
		if latency > 800*time.Millisecond {
			t.Errorf("Batch embedding latency %v exceeds budget 800ms", latency)
		}

		perText := latency / time.Duration(len(texts))
		t.Logf("✅ Batch embedding: %v total, %v per text (budget: 800ms total)", latency, perText)
	})
}
```

**Run the test**:
```bash
go test ./tests/manual -run TestLatencyBudgets -v -timeout 30s
```

**✅ Success Criteria**:
- Single embedding < 100ms (after warmup)
- Batch of 10 < 800ms
- No timeouts

---

## Troubleshooting Guide

### Common Issues

#### Issue 1: "Ollama connection refused"

**Symptoms**:
```
Error: Post "http://localhost:11434/api/embeddings": dial tcp [::1]:11434: connect: connection refused
```

**Diagnosis**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Check port is listening
lsof -i :11434
```

**Fix**:
```bash
# Start Ollama
ollama serve

# In new terminal, verify
curl http://localhost:11434/api/tags
```

---

#### Issue 2: "Model not found"

**Symptoms**:
```
Error: model "nomic-embed-text" not found
```

**Diagnosis**:
```bash
ollama list
```

**Fix**:
```bash
ollama pull nomic-embed-text

# Verify
ollama list
# Should show: nomic-embed-text:latest
```

---

#### Issue 3: "Anthropic API key invalid"

**Symptoms**:
```
Error: invalid x-api-key
```

**Diagnosis**:
```bash
# Check if key is set
echo $ANTHROPIC_API_KEY

# Check key format (should start with sk-ant-api03-)
echo $ANTHROPIC_API_KEY | cut -c1-15
```

**Fix**:
```bash
# Get new key from https://console.anthropic.com
# Set in environment
export ANTHROPIC_API_KEY="sk-ant-api03-YOUR_KEY_HERE"

# Test with curl
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

---

#### Issue 4: "Test timeout"

**Symptoms**:
```
panic: test timed out after 10s
```

**Diagnosis**: Tests running too slow (model loading, network latency)

**Fix**:
```bash
# Increase timeout
go test ./tests/manual -timeout 60s

# Or skip slow tests
go test ./tests/manual -short
```

---

## Success Checklist

After running all tests, you should have:

- [ ] ✅ Property tests passing (10 categorical laws)
- [ ] ✅ Unit tests passing (79 tests, 89.5% coverage)
- [ ] ✅ Ollama embeddings working (6/6 tests)
- [ ] ✅ Anthropic completions working (4/4 tests)
- [ ] ✅ Markdown parsing working (5/5 tests)
- [ ] ✅ Embedding pipeline test passing
- [ ] ✅ LLM completion test passing
- [ ] ✅ Latency budgets met

**If all checkboxes pass**: ✅ **VERA M1-M3 is functionally working!**

---

## Next Steps

**If tests pass**: You've verified VERA actually works! Time to:
1. Continue with M4 (Verification Engine)
2. Build the vector store integration
3. Implement end-to-end query functionality

**If tests fail**: Debug using the troubleshooting guide above, then re-run tests.

---

**Document Status**: Manual testing guide complete
**Last Updated**: 2026-01-07
**Maintainer**: VERA Development Team
