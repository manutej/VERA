package integration_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/manu/vera/pkg/providers"
)

// TestOllamaEmbeddings tests Ollama embedding generation with nomic-embed-text.
//
// Prerequisites:
//   - Ollama running (brew services start ollama)
//   - nomic-embed-text model downloaded (ollama pull nomic-embed-text)
//
// What this tests:
//   - Single text embedding
//   - Batch embedding
//   - Matryoshka dimension truncation (512 dims)
//   - L2 normalization
//   - Error handling (empty texts, invalid dimensions)
//   - Observability (latency tracking)
func TestOllamaEmbeddings(t *testing.T) {
	provider := providers.NewOllamaEmbeddingProvider("", "")

	t.Run("single text embedding", func(t *testing.T) {
		ctx := context.Background()
		request := providers.EmbeddingRequest{
			Texts:      []string{"VERA verifies evidence-grounded reasoning"},
			Dimensions: 768, // Native dimensions
			Normalize:  true,
		}

		result := provider.Embed(ctx, request)
		if result.IsErr() {
			t.Fatalf("Embed failed: %v", result.Error())
		}

		response := result.Unwrap()

		// Verify response structure
		if len(response.Embeddings) != 1 {
			t.Errorf("Expected 1 embedding, got %d", len(response.Embeddings))
		}

		embedding := response.Embeddings[0]
		if len(embedding) != 768 {
			t.Errorf("Expected 768 dimensions, got %d", len(embedding))
		}

		// Verify L2 normalization (||v|| should be ≈ 1.0)
		var sumSquares float32
		for _, v := range embedding {
			sumSquares += v * v
		}
		norm := float32(sumSquares)
		if norm < 0.99 || norm > 1.01 {
			t.Errorf("Expected L2 norm ≈ 1.0, got %.4f", norm)
		}

		// Verify observability
		if response.LatencyMs == 0 {
			t.Error("Expected non-zero latency")
		}
		if response.Model != "nomic-embed-text" {
			t.Errorf("Expected model 'nomic-embed-text', got '%s'", response.Model)
		}
		if response.Usage.PromptTokens == 0 {
			t.Error("Expected non-zero prompt tokens")
		}

		t.Logf("✅ Single embedding: %d dims, %.2fms latency, %d tokens",
			len(embedding), float64(response.LatencyMs), response.Usage.PromptTokens)
	})

	t.Run("batch embedding", func(t *testing.T) {
		ctx := context.Background()
		request := providers.EmbeddingRequest{
			Texts: []string{
				"VERA verifies evidence-grounded reasoning",
				"Categorical verification transcends traditional RAG",
				"Natural transformation ensures grounding quality",
			},
			Dimensions: 768,
			Normalize:  true,
		}

		result := provider.Embed(ctx, request)
		if result.IsErr() {
			t.Fatalf("Batch embed failed: %v", result.Error())
		}

		response := result.Unwrap()

		// Verify batch size
		if len(response.Embeddings) != 3 {
			t.Errorf("Expected 3 embeddings, got %d", len(response.Embeddings))
		}

		// Verify all embeddings have correct dimensions
		for i, embedding := range response.Embeddings {
			if len(embedding) != 768 {
				t.Errorf("Embedding %d: expected 768 dims, got %d", i, len(embedding))
			}
		}

		t.Logf("✅ Batch embedding: %d texts, %.2fms latency",
			len(response.Embeddings), float64(response.LatencyMs))
	})

	t.Run("matryoshka truncation", func(t *testing.T) {
		ctx := context.Background()
		request := providers.EmbeddingRequest{
			Texts:      []string{"VERA verification"},
			Dimensions: 512, // Matryoshka: 99.5% quality, 33% faster
			Normalize:  true,
		}

		result := provider.Embed(ctx, request)
		if result.IsErr() {
			t.Fatalf("Matryoshka embed failed: %v", result.Error())
		}

		response := result.Unwrap()
		embedding := response.Embeddings[0]

		// Verify truncated dimensions
		if len(embedding) != 512 {
			t.Errorf("Expected 512 dims (Matryoshka), got %d", len(embedding))
		}

		t.Logf("✅ Matryoshka truncation: 512 dims (99.5%% quality)")
	})

	t.Run("error: empty texts", func(t *testing.T) {
		ctx := context.Background()
		request := providers.EmbeddingRequest{
			Texts:      []string{},
			Dimensions: 768,
		}

		result := provider.Embed(ctx, request)
		if result.IsOk() {
			t.Error("Expected error for empty texts, got success")
		}

		err := result.Error()
		if err == nil {
			t.Fatal("Expected error object")
		}

		t.Logf("✅ Correctly rejected empty texts: %v", err)
	})

	t.Run("error: dimensions too large", func(t *testing.T) {
		ctx := context.Background()
		request := providers.EmbeddingRequest{
			Texts:      []string{"test"},
			Dimensions: 1024, // Exceeds nomic-embed-text max (768)
		}

		result := provider.Embed(ctx, request)
		if result.IsOk() {
			t.Error("Expected error for dimensions > 768, got success")
		}

		t.Logf("✅ Correctly rejected dimensions > 768")
	})

	t.Run("timeout handling", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
		defer cancel()

		request := providers.EmbeddingRequest{
			Texts:      []string{"test"},
			Dimensions: 768,
		}

		result := provider.Embed(ctx, request)
		if result.IsOk() {
			t.Error("Expected timeout error, got success")
		}

		t.Logf("✅ Correctly handled context timeout")
	})
}

// TestAnthropicCompletion tests Anthropic Claude completions.
//
// Prerequisites:
//   - ANTHROPIC_API_KEY environment variable set
//
// What this tests:
//   - Basic completion
//   - System prompt
//   - Temperature control
//   - Token tracking
//   - Error handling (empty prompt)
//   - Observability (latency, tokens)
func TestAnthropicCompletion(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping Anthropic tests")
	}

	provider := providers.NewAnthropicProvider(apiKey, "")

	t.Run("basic completion", func(t *testing.T) {
		ctx := context.Background()
		request := providers.CompletionRequest{
			Prompt:      "What is 2+2? Answer with just the number.",
			MaxTokens:   10,
			Temperature: 0.0, // Deterministic
		}

		result := provider.Complete(ctx, request)
		if result.IsErr() {
			t.Fatalf("Complete failed: %v", result.Error())
		}

		response := result.Unwrap()

		// Verify response structure
		if response.Text == "" {
			t.Error("Expected non-empty completion text")
		}

		// Verify observability
		if response.LatencyMs == 0 {
			t.Error("Expected non-zero latency")
		}
		if response.Usage.PromptTokens == 0 {
			t.Error("Expected non-zero prompt tokens")
		}
		if response.Usage.CompletionTokens == 0 {
			t.Error("Expected non-zero completion tokens")
		}
		if response.Model == "" {
			t.Error("Expected non-empty model name")
		}

		t.Logf("✅ Completion: '%s' (%.2fms, %d→%d tokens)",
			response.Text, float64(response.LatencyMs),
			response.Usage.PromptTokens, response.Usage.CompletionTokens)
	})

	t.Run("system prompt", func(t *testing.T) {
		ctx := context.Background()
		request := providers.CompletionRequest{
			Prompt:       "What is my favorite color?",
			SystemPrompt: "You are a helpful assistant. The user's favorite color is blue.",
			MaxTokens:    20,
			Temperature:  0.0,
		}

		result := provider.Complete(ctx, request)
		if result.IsErr() {
			t.Fatalf("Complete with system prompt failed: %v", result.Error())
		}

		response := result.Unwrap()
		if response.Text == "" {
			t.Error("Expected non-empty completion")
		}

		t.Logf("✅ System prompt completion: '%s'", response.Text)
	})

	t.Run("error: empty prompt", func(t *testing.T) {
		ctx := context.Background()
		request := providers.CompletionRequest{
			Prompt:    "",
			MaxTokens: 10,
		}

		result := provider.Complete(ctx, request)
		if result.IsOk() {
			t.Error("Expected error for empty prompt, got success")
		}

		t.Logf("✅ Correctly rejected empty prompt")
	})

	t.Run("timeout handling", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
		defer cancel()

		request := providers.CompletionRequest{
			Prompt:    "test",
			MaxTokens: 10,
		}

		result := provider.Complete(ctx, request)
		if result.IsOk() {
			t.Error("Expected timeout error, got success")
		}

		t.Logf("✅ Correctly handled context timeout")
	})
}

// TestProviderNames tests provider introspection methods.
func TestProviderNames(t *testing.T) {
	t.Run("ollama name", func(t *testing.T) {
		provider := providers.NewOllamaEmbeddingProvider("", "")
		if provider.Name() != "ollama" {
			t.Errorf("Expected name 'ollama', got '%s'", provider.Name())
		}
		if provider.Dimensions() != 768 {
			t.Errorf("Expected 768 dimensions, got %d", provider.Dimensions())
		}
	})

	t.Run("anthropic name", func(t *testing.T) {
		provider := providers.NewAnthropicProvider("test-key", "")
		if provider.Name() != "anthropic" {
			t.Errorf("Expected name 'anthropic', got '%s'", provider.Name())
		}
	})
}
