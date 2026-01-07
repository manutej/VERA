package providers

import (
	"context"

	"github.com/manu/vera/pkg/core"
)

// CompletionProvider defines the interface for LLM completion services.
//
// Design Principles (Constitution Article III: Provider Agnosticism):
//   - Interface decoupled from implementation (supports Anthropic, OpenAI, Ollama, etc.)
//   - Returns Result[T] for type-safe error handling
//   - Context propagation for cancellation and timeouts
//   - Observable by default (providers track tokens and latency)
//
// Implementations:
//   - AnthropicProvider (Claude Sonnet) - PRIMARY
//   - OpenAIProvider (GPT-4) - Alternative
//   - OllamaProvider (local models) - Development/testing
//
// Usage:
//
//	provider := NewAnthropicProvider(apiKey)
//	request := CompletionRequest{
//	    Prompt: "Explain quantum entanglement",
//	    MaxTokens: 1000,
//	    Temperature: 0.7,
//	}
//	result := provider.Complete(ctx, request)
type CompletionProvider interface {
	// Complete generates a text completion from the given prompt.
	//
	// Parameters:
	//   - ctx: Context for cancellation and timeouts
	//   - request: Completion parameters (prompt, temperature, max tokens, etc.)
	//
	// Returns:
	//   - Ok(CompletionResponse) on success with generated text and metadata
	//   - Err(VERAError) on failure (ErrorKindProvider for API failures)
	//
	// Error Handling:
	//   - Rate limits: Return ErrorKindProvider (trigger exponential backoff)
	//   - Invalid input: Return ErrorKindValidation (fail fast)
	//   - Network errors: Return ErrorKindProvider (retry with timeout)
	Complete(ctx context.Context, request CompletionRequest) core.Result[CompletionResponse]

	// Name returns the provider name for logging and metrics.
	Name() string
}

// EmbeddingProvider defines the interface for text embedding services.
//
// Design Principles (Constitution Article III: Provider Agnosticism):
//   - Interface decoupled from implementation (supports Ollama, Voyage AI, OpenAI, etc.)
//   - Returns Result[T] for type-safe error handling
//   - Supports batch embedding for efficiency
//   - Dimension-agnostic (512 or 768 via Matryoshka)
//
// Implementations:
//   - OllamaProvider (nomic-embed-text-v1.5) - PRIMARY ($0/month, self-hosted)
//   - VoyageProvider (voyage-2) - Alternative
//   - OpenAIProvider (text-embedding-3-small) - Alternative
//
// Usage:
//
//	provider := NewOllamaEmbeddingProvider("http://localhost:11434")
//	request := EmbeddingRequest{
//	    Texts: []string{"Document chunk 1", "Document chunk 2"},
//	    Dimensions: 512, // Matryoshka truncation (99.5% of 768 quality)
//	}
//	result := provider.Embed(ctx, request)
type EmbeddingProvider interface {
	// Embed generates vector embeddings for the given texts.
	//
	// Parameters:
	//   - ctx: Context for cancellation and timeouts
	//   - request: Embedding parameters (texts, dimensions)
	//
	// Returns:
	//   - Ok(EmbeddingResponse) on success with vectors and metadata
	//   - Err(VERAError) on failure (ErrorKindProvider for API failures)
	//
	// Performance:
	//   - Batch size: Recommended 32-128 texts per request
	//   - Dimensions: 512 (Matryoshka) recommended for speed/quality balance
	//   - Latency: < 100ms for 32 texts @ 512 dims (Ollama local)
	//
	// Error Handling:
	//   - Empty texts: Return ErrorKindValidation (fail fast)
	//   - Service down: Return ErrorKindProvider (retry or fallback)
	//   - Invalid dimensions: Return ErrorKindValidation (fail fast)
	Embed(ctx context.Context, request EmbeddingRequest) core.Result[EmbeddingResponse]

	// Name returns the provider name for logging and metrics.
	Name() string

	// Dimensions returns the native embedding dimension (before Matryoshka truncation).
	// For nomic-embed-text-v1.5: 768
	// For voyage-2: 1024
	Dimensions() int
}

// CompletionRequest encapsulates parameters for text completion.
type CompletionRequest struct {
	// Prompt is the input text to complete.
	Prompt string

	// SystemPrompt optionally sets the system message (behavior instructions).
	// Example: "You are a helpful assistant that explains technical concepts simply."
	SystemPrompt string

	// MaxTokens limits the completion length (default: 1000).
	MaxTokens int

	// Temperature controls randomness (0.0 = deterministic, 1.0 = creative).
	// Recommended: 0.0-0.3 for factual answers, 0.7-0.9 for creative writing.
	Temperature float64

	// TopP controls nucleus sampling (alternative to temperature).
	// Recommended: 0.95 for balanced output.
	TopP float64

	// Stop sequences that terminate generation early.
	Stop []string
}

// CompletionResponse contains the generated completion and metadata.
type CompletionResponse struct {
	// Text is the generated completion.
	Text string

	// FinishReason indicates why generation stopped ("stop", "length", "error").
	FinishReason string

	// Usage tracks token consumption.
	Usage TokenUsage

	// Latency measures generation time (for observability).
	LatencyMs int64

	// Model identifies the specific model used (e.g., "claude-sonnet-4").
	Model string
}

// EmbeddingRequest encapsulates parameters for vector embedding.
type EmbeddingRequest struct {
	// Texts are the input strings to embed (batch).
	Texts []string

	// Dimensions specifies output vector size (optional, uses native if 0).
	// For Matryoshka models like nomic-embed-text:
	//   - 512: 99.5% of 768 quality, 33% faster
	//   - 768: Full quality (native)
	Dimensions int

	// Normalize indicates whether to L2-normalize embeddings (default: true).
	// Normalized embeddings enable cosine similarity via dot product.
	Normalize bool
}

// EmbeddingResponse contains the generated embeddings and metadata.
type EmbeddingResponse struct {
	// Embeddings are the vector representations (one per input text).
	// Dimensions match request.Dimensions (or provider.Dimensions() if unspecified).
	Embeddings [][]float32

	// Usage tracks token consumption (if provider supports it).
	Usage TokenUsage

	// Latency measures embedding time (for observability).
	LatencyMs int64

	// Model identifies the specific model used (e.g., "nomic-embed-text-v1.5").
	Model string
}

// TokenUsage tracks token consumption for cost estimation and observability.
//
// Observability (Constitution Article IX: Observable by Default):
//   - All provider operations track tokens and latency
//   - Enables cost analysis (tokens × price per token)
//   - Enables performance monitoring (tokens per second)
type TokenUsage struct {
	// PromptTokens counts input tokens.
	PromptTokens int

	// CompletionTokens counts output tokens (0 for embeddings).
	CompletionTokens int

	// TotalTokens is the sum of prompt + completion tokens.
	TotalTokens int
}
