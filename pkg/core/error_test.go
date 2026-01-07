package core_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/manu/vera/pkg/core"
)

// TestNewError verifies VERAError constructor.
func TestNewError(t *testing.T) {
	cause := errors.New("underlying error")
	err := core.NewError(core.ErrorKindProvider, "API rate limit", cause)

	if err.Kind != core.ErrorKindProvider {
		t.Errorf("expected Kind %d, got %d", core.ErrorKindProvider, err.Kind)
	}

	if err.Message != "API rate limit" {
		t.Errorf("expected Message 'API rate limit', got %q", err.Message)
	}

	if !errors.Is(err.Cause, cause) {
		t.Errorf("expected Cause %v, got %v", cause, err.Cause)
	}

	if err.Context == nil {
		t.Error("expected Context map initialized, got nil")
	}
}

// TestVERAError_Error verifies Error method formats correctly.
func TestVERAError_Error(t *testing.T) {
	tests := []struct {
		name     string
		err      *core.VERAError
		contains []string
	}{
		{
			name: "with cause",
			err: core.NewError(
				core.ErrorKindProvider,
				"API failed",
				errors.New("timeout"),
			),
			contains: []string{"API failed", "timeout", "PROVIDER"},
		},
		{
			name:     "without cause",
			err:      core.NewError(core.ErrorKindValidation, "Invalid input", nil),
			contains: []string{"Invalid input", "VALIDATION"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errStr := tt.err.Error()
			for _, substr := range tt.contains {
				if !strings.Contains(errStr, substr) {
					t.Errorf("expected Error() to contain %q, got %q", substr, errStr)
				}
			}
		})
	}
}

// TestVERAError_Unwrap verifies Unwrap returns cause.
func TestVERAError_Unwrap(t *testing.T) {
	cause := errors.New("root cause")
	err := core.NewError(core.ErrorKindInternal, "Wrapper", cause)

	unwrapped := err.Unwrap()

	if !errors.Is(unwrapped, cause) {
		t.Errorf("expected Unwrap() to return %v, got %v", cause, unwrapped)
	}
}

// TestVERAError_Unwrap_NoCause verifies Unwrap returns nil when no cause.
func TestVERAError_Unwrap_NoCause(t *testing.T) {
	err := core.NewError(core.ErrorKindValidation, "No cause", nil)

	unwrapped := err.Unwrap()

	if unwrapped != nil {
		t.Errorf("expected nil for no cause, got %v", unwrapped)
	}
}

// TestVERAError_WithContext verifies WithContext adds metadata.
func TestVERAError_WithContext(t *testing.T) {
	err := core.NewError(core.ErrorKindProvider, "API failed", nil)

	// WithContext should be chainable
	err = err.WithContext("provider", "anthropic").
		WithContext("retry_after", 60).
		WithContext("request_id", "req_123")

	if err.Context["provider"] != "anthropic" {
		t.Errorf("expected provider = 'anthropic', got %v", err.Context["provider"])
	}

	if err.Context["retry_after"] != 60 {
		t.Errorf("expected retry_after = 60, got %v", err.Context["retry_after"])
	}

	if err.Context["request_id"] != "req_123" {
		t.Errorf("expected request_id = 'req_123', got %v", err.Context["request_id"])
	}
}

// TestVERAError_Is verifies Is method matches on ErrorKind.
func TestVERAError_Is(t *testing.T) {
	err1 := core.NewError(core.ErrorKindProvider, "API failed", nil)
	err2 := core.NewError(core.ErrorKindProvider, "Different message", nil)
	err3 := core.NewError(core.ErrorKindValidation, "Wrong kind", nil)

	// Same kind should match
	if !err1.Is(err2) {
		t.Error("expected Is(err2) = true for same ErrorKind")
	}

	// Different kind should not match
	if err1.Is(err3) {
		t.Error("expected Is(err3) = false for different ErrorKind")
	}
}

// TestVERAError_Is_NonVERAError verifies Is returns false for non-VERAError.
func TestVERAError_Is_NonVERAError(t *testing.T) {
	veraErr := core.NewError(core.ErrorKindProvider, "API failed", nil)
	stdErr := errors.New("standard error")

	if veraErr.Is(stdErr) {
		t.Error("expected Is(stdErr) = false for non-VERAError")
	}
}

// TestErrorsIs_Integration verifies errors.Is works with VERAError.
func TestErrorsIs_Integration(t *testing.T) {
	cause := errors.New("root cause")
	err := core.NewError(core.ErrorKindProvider, "API failed", cause)

	// errors.Is should find cause through Unwrap chain
	if !errors.Is(err, cause) {
		t.Error("expected errors.Is to find cause through Unwrap")
	}

	// errors.Is should match on ErrorKind via Is method
	sameKind := core.NewError(core.ErrorKindProvider, "Different message", nil)
	if !errors.Is(err, sameKind) {
		t.Error("expected errors.Is to match on ErrorKind")
	}
}

// TestErrorKind_Values verifies all ErrorKind constants are distinct.
func TestErrorKind_Values(t *testing.T) {
	kinds := []core.ErrorKind{
		core.ErrorKindValidation,
		core.ErrorKindProvider,
		core.ErrorKindIngestion,
		core.ErrorKindRetrieval,
		core.ErrorKindVerification,
		core.ErrorKindConfiguration,
		core.ErrorKindInternal,
	}

	// All should be distinct
	seen := make(map[core.ErrorKind]bool)
	for _, kind := range kinds {
		if seen[kind] {
			t.Errorf("duplicate ErrorKind value: %d", kind)
		}
		seen[kind] = true
	}

	if len(seen) != 7 {
		t.Errorf("expected 7 distinct ErrorKind values, got %d", len(seen))
	}
}

// TestErrorKind_String verifies ErrorKind string representation.
func TestErrorKind_String(t *testing.T) {
	tests := []struct {
		kind     core.ErrorKind
		expected string
	}{
		{core.ErrorKindValidation, "VALIDATION"},
		{core.ErrorKindProvider, "PROVIDER"},
		{core.ErrorKindIngestion, "INGESTION"},
		{core.ErrorKindRetrieval, "RETRIEVAL"},
		{core.ErrorKindVerification, "VERIFICATION"},
		{core.ErrorKindConfiguration, "CONFIGURATION"},
		{core.ErrorKindInternal, "INTERNAL"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			err := core.NewError(tt.kind, "test", nil)
			errStr := err.Error()
			if !strings.Contains(errStr, tt.expected) {
				t.Errorf("expected Error() to contain %q for kind %d, got %q",
					tt.expected, tt.kind, errStr)
			}
		})
	}
}

// TestVERAError_ContextPreservation verifies context is preserved through error wrapping.
func TestVERAError_ContextPreservation(t *testing.T) {
	err := core.NewError(core.ErrorKindProvider, "Original error", nil).
		WithContext("attempt", 1).
		WithContext("endpoint", "/api/v1/chat")

	// Wrap in another VERAError
	wrapped := core.NewError(core.ErrorKindInternal, "Wrapper error", err)

	// Original context should be accessible through Unwrap
	unwrapped := wrapped.Unwrap().(*core.VERAError)
	if unwrapped.Context["attempt"] != 1 {
		t.Errorf("expected context preserved through wrapping, got %v", unwrapped.Context)
	}
}

// TestVERAError_ChainableContext verifies WithContext returns same error for chaining.
func TestVERAError_ChainableContext(t *testing.T) {
	err1 := core.NewError(core.ErrorKindProvider, "test", nil)
	err2 := err1.WithContext("key", "value")

	// Should return the same error instance (chainable)
	if err1 != err2 {
		t.Error("expected WithContext to return same error instance for chaining")
	}
}

// TestVERAError_MultipleContextUpdates verifies context can be updated multiple times.
func TestVERAError_MultipleContextUpdates(t *testing.T) {
	err := core.NewError(core.ErrorKindProvider, "test", nil)

	err.WithContext("counter", 1)
	err.WithContext("counter", 2) // Update same key
	err.WithContext("counter", 3)

	if err.Context["counter"] != 3 {
		t.Errorf("expected counter = 3 (last update), got %v", err.Context["counter"])
	}
}
