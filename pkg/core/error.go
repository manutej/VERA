package core

import "fmt"

// ErrorKind represents categorical error classifications in VERA.
// Each kind corresponds to a distinct failure mode with specific handling semantics.
type ErrorKind int

const (
	// ErrorKindValidation indicates invalid input data (e.g., empty prompt, invalid file path).
	// Handling: Return 400 status in API layer, do not retry.
	ErrorKindValidation ErrorKind = iota

	// ErrorKindProvider indicates LLM/Embedding provider failure (e.g., API rate limit, 503).
	// Handling: Trigger circuit breaker, exponential backoff, consider fallback provider.
	ErrorKindProvider

	// ErrorKindIngestion indicates document parsing failure (e.g., corrupt PDF, unsupported format).
	// Handling: Skip document, log warning, continue batch processing.
	ErrorKindIngestion

	// ErrorKindRetrieval indicates retrieval pipeline failure (e.g., embedding service down, no results).
	// Handling: Return degraded response, log error, consider cached results.
	ErrorKindRetrieval

	// ErrorKindVerification indicates verification engine failure (e.g., NLI service unavailable).
	// Handling: Return unverified answer with warning, use local ONNX fallback if available.
	ErrorKindVerification

	// ErrorKindConfiguration indicates configuration error (e.g., missing API key, invalid config file).
	// Handling: Fail fast at startup, do not retry, require user intervention.
	ErrorKindConfiguration

	// ErrorKindInternal indicates internal system error (e.g., unexpected nil, assertion failure).
	// Handling: Log with stack trace, return 500 status, investigate immediately.
	ErrorKindInternal
)

// String returns human-readable error kind name.
func (k ErrorKind) String() string {
	switch k {
	case ErrorKindValidation:
		return "VALIDATION"
	case ErrorKindProvider:
		return "PROVIDER"
	case ErrorKindIngestion:
		return "INGESTION"
	case ErrorKindRetrieval:
		return "RETRIEVAL"
	case ErrorKindVerification:
		return "VERIFICATION"
	case ErrorKindConfiguration:
		return "CONFIGURATION"
	case ErrorKindInternal:
		return "INTERNAL"
	default:
		return "UNKNOWN"
	}
}

// VERAError represents a structured error with categorical classification and context.
//
// Design:
//   - Kind: Categorical error type enabling type-safe error handling
//   - Message: Human-readable description
//   - Cause: Wrapped underlying error (supports errors.Is, errors.As)
//   - Context: Structured metadata (file paths, token counts, request IDs)
//
// Example:
//
//	err := NewError(ErrorKindProvider, "Anthropic API rate limit exceeded", httpErr).
//	    WithContext("model", "claude-sonnet-4").
//	    WithContext("tokens_requested", 4000)
type VERAError struct {
	Kind    ErrorKind
	Message string
	Cause   error
	Context map[string]any
}

// Error implements the error interface.
// Format: [KIND] message: cause
func (e *VERAError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Kind, e.Message, e.Cause)
	}
	return fmt.Sprintf("[%s] %s", e.Kind, e.Message)
}

// Unwrap returns the underlying cause error.
// This enables Go 1.13+ errors.Is and errors.As functionality.
func (e *VERAError) Unwrap() error {
	return e.Cause
}

// NewError creates a VERAError with kind, message, and optional cause.
func NewError(kind ErrorKind, message string, cause error) *VERAError {
	return &VERAError{
		Kind:    kind,
		Message: message,
		Cause:   cause,
		Context: make(map[string]any),
	}
}

// WithContext adds contextual metadata to the error.
// Returns the error for method chaining.
//
// Example:
//
//	err := NewError(ErrorKindRetrieval, "no documents found", nil).
//	    WithContext("query", "legal compliance").
//	    WithContext("top_k", 5)
func (e *VERAError) WithContext(key string, value any) *VERAError {
	e.Context[key] = value
	return e
}

// Is enables errors.Is matching on error kind.
//
// Example:
//
//	if errors.Is(err, NewError(ErrorKindProvider, "", nil)) {
//	    // Handle provider errors specifically
//	}
func (e *VERAError) Is(target error) bool {
	t, ok := target.(*VERAError)
	if !ok {
		return false
	}
	return e.Kind == t.Kind
}
