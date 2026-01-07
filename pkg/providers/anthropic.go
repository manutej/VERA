package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/manu/vera/pkg/core"
)

// AnthropicProvider implements CompletionProvider for Claude models.
//
// Architecture:
//   - Uses Anthropic Messages API (https://docs.anthropic.com/claude/reference/messages)
//   - Supports Claude Sonnet 4 (primary), Haiku, Opus
//   - Rate limits: 50 requests/min (tier 1), 5000 requests/min (tier 4)
//   - Token tracking for cost estimation
//
// Cost (as of 2024):
//   - Claude Sonnet 4: $3 input / $15 output per 1M tokens
//   - Average VERA query: ~2K input + ~500 output = $0.0135/query
//   - Monthly estimate (1000 queries): $13.50
//
// Configuration:
//   - ANTHROPIC_API_KEY environment variable (required)
//   - Default model: claude-sonnet-4-20250514
//   - Default max tokens: 1000
//   - Default temperature: 0.0 (deterministic)
type AnthropicProvider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	model      string
}

// Anthropic Messages API request structure
type anthropicRequest struct {
	Model       string              `json:"model"`
	Messages    []anthropicMessage  `json:"messages"`
	System      string              `json:"system,omitempty"`
	MaxTokens   int                 `json:"max_tokens"`
	Temperature float64             `json:"temperature,omitempty"`
	TopP        float64             `json:"top_p,omitempty"`
	Stop        []string            `json:"stop_sequences,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`    // "user" or "assistant"
	Content string `json:"content"` // Message text
}

// Anthropic Messages API response structure
type anthropicResponse struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	Role         string             `json:"role"`
	Content      []anthropicContent `json:"content"`
	Model        string             `json:"model"`
	StopReason   string             `json:"stop_reason"`
	Usage        anthropicUsage     `json:"usage"`
}

type anthropicContent struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Anthropic error response structure
type anthropicError struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// NewAnthropicProvider creates an Anthropic completion provider.
//
// Parameters:
//   - apiKey: Anthropic API key (from ANTHROPIC_API_KEY env var)
//   - model: Claude model name (default: "claude-sonnet-4-20250514")
//
// Returns:
//   - CompletionProvider implementation
//
// Usage:
//
//	provider := NewAnthropicProvider(os.Getenv("ANTHROPIC_API_KEY"), "")
func NewAnthropicProvider(apiKey, model string) *AnthropicProvider {
	if model == "" {
		model = "claude-sonnet-4-20250514" // Default to Claude Sonnet 4
	}

	return &AnthropicProvider{
		apiKey:  apiKey,
		baseURL: "https://api.anthropic.com/v1",
		httpClient: &http.Client{
			Timeout: 60 * time.Second, // 60s timeout for long completions
		},
		model: model,
	}
}

// Complete generates text completion via Anthropic Messages API.
//
// Implementation notes:
//   - Uses Messages API (not legacy Text API)
//   - Tracks tokens via usage object
//   - Measures latency for observability
//   - Returns ErrorKindProvider for API failures (triggers retry logic)
//   - Returns ErrorKindValidation for invalid input (fail fast)
func (p *AnthropicProvider) Complete(ctx context.Context, request CompletionRequest) core.Result[CompletionResponse] {
	startTime := time.Now()

	// Validate input
	if request.Prompt == "" {
		return core.Err[CompletionResponse](
			core.NewError(core.ErrorKindValidation, "prompt cannot be empty", nil),
		)
	}

	// Build Anthropic API request
	maxTokens := request.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1000 // Default
	}

	apiReq := anthropicRequest{
		Model: p.model,
		Messages: []anthropicMessage{
			{Role: "user", Content: request.Prompt},
		},
		System:      request.SystemPrompt,
		MaxTokens:   maxTokens,
		Temperature: request.Temperature,
		TopP:        request.TopP,
		Stop:        request.Stop,
	}

	// Marshal request body
	bodyBytes, err := json.Marshal(apiReq)
	if err != nil {
		return core.Err[CompletionResponse](
			core.NewError(core.ErrorKindInternal, "failed to marshal request", err),
		)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(
		ctx,
		"POST",
		p.baseURL+"/messages",
		bytes.NewReader(bodyBytes),
	)
	if err != nil {
		return core.Err[CompletionResponse](
			core.NewError(core.ErrorKindInternal, "failed to create HTTP request", err),
		)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-API-Key", p.apiKey)
	httpReq.Header.Set("Anthropic-Version", "2023-06-01")

	// Execute request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return core.Err[CompletionResponse](
			core.NewError(
				core.ErrorKindProvider,
				"API request failed",
				err,
			).WithContext("provider", "anthropic").WithContext("model", p.model),
		)
	}
	defer httpResp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return core.Err[CompletionResponse](
			core.NewError(core.ErrorKindProvider, "failed to read response", err),
		)
	}

	// Handle HTTP errors
	if httpResp.StatusCode != http.StatusOK {
		var apiErr anthropicError
		if err := json.Unmarshal(respBody, &apiErr); err != nil {
			return core.Err[CompletionResponse](
				core.NewError(
					core.ErrorKindProvider,
					fmt.Sprintf("API error (status %d)", httpResp.StatusCode),
					nil,
				).WithContext("status_code", httpResp.StatusCode),
			)
		}

		kind := core.ErrorKindProvider
		if httpResp.StatusCode == http.StatusBadRequest {
			kind = core.ErrorKindValidation // 400 = invalid input
		}

		return core.Err[CompletionResponse](
			core.NewError(
				kind,
				apiErr.Error.Message,
				nil,
			).WithContext("error_type", apiErr.Error.Type).
				WithContext("status_code", httpResp.StatusCode),
		)
	}

	// Parse successful response
	var apiResp anthropicResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return core.Err[CompletionResponse](
			core.NewError(core.ErrorKindInternal, "failed to parse response", err),
		)
	}

	// Extract completion text (first content block)
	text := ""
	if len(apiResp.Content) > 0 {
		text = apiResp.Content[0].Text
	}

	// Calculate latency
	latencyMs := time.Since(startTime).Milliseconds()

	// Build response
	response := CompletionResponse{
		Text:         text,
		FinishReason: apiResp.StopReason,
		Usage: TokenUsage{
			PromptTokens:     apiResp.Usage.InputTokens,
			CompletionTokens: apiResp.Usage.OutputTokens,
			TotalTokens:      apiResp.Usage.InputTokens + apiResp.Usage.OutputTokens,
		},
		LatencyMs: latencyMs,
		Model:     apiResp.Model,
	}

	return core.Ok(response)
}

// Name returns the provider name for logging and metrics.
func (p *AnthropicProvider) Name() string {
	return "anthropic"
}
