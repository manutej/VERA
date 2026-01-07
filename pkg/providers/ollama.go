package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"github.com/manu/vera/pkg/core"
)

// OllamaEmbeddingProvider implements EmbeddingProvider for Ollama's nomic-embed-text.
//
// Architecture:
//   - Uses Ollama HTTP API (http://localhost:11434)
//   - Supports nomic-embed-text-v1.5 (768 dimensions, Apache 2.0)
//   - Matryoshka dimension truncation (512 = 99.5% quality, 33% faster)
//   - L2 normalization for cosine similarity
//
// Performance (from ADR-0024):
//   - Native: 768 dims, 100% quality
//   - Matryoshka 512: 512 dims, 99.5% quality, 33% faster
//   - Throughput: ~1000 embeddings/sec (batched)
//
// Configuration:
//   - Base URL: http://localhost:11434 (default Ollama port)
//   - Model: nomic-embed-text (137M parameters)
//   - Timeout: 30 seconds
type OllamaEmbeddingProvider struct {
	baseURL    string
	httpClient *http.Client
	model      string
}

// Ollama API request structure
type ollamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// Ollama API response structure
type ollamaEmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// NewOllamaEmbeddingProvider creates an Ollama embedding provider.
//
// Parameters:
//   - baseURL: Ollama server URL (default: "http://localhost:11434")
//   - model: Embedding model name (default: "nomic-embed-text")
//
// Returns:
//   - EmbeddingProvider implementation
//
// Usage:
//
//	provider := NewOllamaEmbeddingProvider("", "")
func NewOllamaEmbeddingProvider(baseURL, model string) *OllamaEmbeddingProvider {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	if model == "" {
		model = "nomic-embed-text"
	}

	return &OllamaEmbeddingProvider{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second, // 30s timeout for embeddings
		},
		model: model,
	}
}

// Embed generates vector embeddings via Ollama API.
//
// Implementation notes:
//   - Batches requests (one HTTP call per text)
//   - Applies Matryoshka truncation if Dimensions < 768
//   - Applies L2 normalization if Normalize = true
//   - Tracks latency for observability
//   - Returns ErrorKindProvider for API failures (triggers retry logic)
//   - Returns ErrorKindValidation for invalid input (fail fast)
func (p *OllamaEmbeddingProvider) Embed(ctx context.Context, request EmbeddingRequest) core.Result[EmbeddingResponse] {
	startTime := time.Now()

	// Validate input
	if len(request.Texts) == 0 {
		return core.Err[EmbeddingResponse](
			core.NewError(core.ErrorKindValidation, "texts cannot be empty", nil),
		)
	}

	// Validate dimensions (nomic-embed-text native = 768)
	dimensions := request.Dimensions
	if dimensions == 0 {
		dimensions = 768 // Default to native
	}
	if dimensions > 768 {
		return core.Err[EmbeddingResponse](
			core.NewError(
				core.ErrorKindValidation,
				fmt.Sprintf("dimensions %d exceeds nomic-embed-text max (768)", dimensions),
				nil,
			),
		)
	}

	// Process each text (Ollama API only supports single text per request)
	embeddings := make([][]float32, 0, len(request.Texts))
	totalTokens := 0

	for i, text := range request.Texts {
		if text == "" {
			return core.Err[EmbeddingResponse](
				core.NewError(
					core.ErrorKindValidation,
					fmt.Sprintf("text at index %d is empty", i),
					nil,
				),
			)
		}

		// Build API request
		apiReq := ollamaEmbedRequest{
			Model:  p.model,
			Prompt: text,
		}

		// Marshal request body
		bodyBytes, err := json.Marshal(apiReq)
		if err != nil {
			return core.Err[EmbeddingResponse](
				core.NewError(core.ErrorKindInternal, "failed to marshal request", err),
			)
		}

		// Create HTTP request
		httpReq, err := http.NewRequestWithContext(
			ctx,
			"POST",
			p.baseURL+"/api/embeddings",
			bytes.NewReader(bodyBytes),
		)
		if err != nil {
			return core.Err[EmbeddingResponse](
				core.NewError(core.ErrorKindInternal, "failed to create HTTP request", err),
			)
		}

		// Set headers
		httpReq.Header.Set("Content-Type", "application/json")

		// Execute request
		httpResp, err := p.httpClient.Do(httpReq)
		if err != nil {
			return core.Err[EmbeddingResponse](
				core.NewError(
					core.ErrorKindProvider,
					"Ollama API request failed",
					err,
				).WithContext("provider", "ollama").WithContext("model", p.model),
			)
		}
		defer httpResp.Body.Close()

		// Read response body
		respBody, err := io.ReadAll(httpResp.Body)
		if err != nil {
			return core.Err[EmbeddingResponse](
				core.NewError(core.ErrorKindProvider, "failed to read response", err),
			)
		}

		// Handle HTTP errors
		if httpResp.StatusCode != http.StatusOK {
			return core.Err[EmbeddingResponse](
				core.NewError(
					core.ErrorKindProvider,
					fmt.Sprintf("Ollama API error (status %d): %s", httpResp.StatusCode, string(respBody)),
					nil,
				).WithContext("status_code", httpResp.StatusCode),
			)
		}

		// Parse successful response
		var apiResp ollamaEmbedResponse
		if err := json.Unmarshal(respBody, &apiResp); err != nil {
			return core.Err[EmbeddingResponse](
				core.NewError(core.ErrorKindInternal, "failed to parse response", err),
			)
		}

		// Verify embedding dimension
		if len(apiResp.Embedding) != 768 {
			return core.Err[EmbeddingResponse](
				core.NewError(
					core.ErrorKindProvider,
					fmt.Sprintf("unexpected embedding dimension: got %d, expected 768", len(apiResp.Embedding)),
					nil,
				),
			)
		}

		// Apply Matryoshka truncation if needed
		embedding := apiResp.Embedding
		if dimensions < 768 {
			embedding = embedding[:dimensions]
		}

		// Apply L2 normalization if requested
		if request.Normalize {
			embedding = normalizeL2(embedding)
		}

		embeddings = append(embeddings, embedding)

		// Approximate token count (rough estimate: 1 token ≈ 4 chars)
		totalTokens += len(text) / 4
	}

	// Calculate latency
	latencyMs := time.Since(startTime).Milliseconds()

	// Build response
	response := EmbeddingResponse{
		Embeddings: embeddings,
		Usage: TokenUsage{
			PromptTokens:     totalTokens,
			CompletionTokens: 0, // Embeddings don't generate completion tokens
			TotalTokens:      totalTokens,
		},
		LatencyMs: latencyMs,
		Model:     p.model,
	}

	return core.Ok(response)
}

// Name returns the provider name for logging and metrics.
func (p *OllamaEmbeddingProvider) Name() string {
	return "ollama"
}

// Dimensions returns the native embedding dimension (before Matryoshka truncation).
func (p *OllamaEmbeddingProvider) Dimensions() int {
	return 768 // nomic-embed-text native dimensions
}

// normalizeL2 applies L2 normalization to a vector for cosine similarity.
// This ensures ||v|| = 1, making dot product equivalent to cosine similarity.
func normalizeL2(vec []float32) []float32 {
	// Calculate L2 norm: sqrt(sum(v_i^2))
	var sumSquares float32
	for _, v := range vec {
		sumSquares += v * v
	}
	norm := float32(math.Sqrt(float64(sumSquares)))

	// Avoid division by zero
	if norm == 0 {
		return vec
	}

	// Normalize: v' = v / ||v||
	normalized := make([]float32, len(vec))
	for i, v := range vec {
		normalized[i] = v / norm
	}

	return normalized
}
