package core_test

import (
	"errors"
	"testing"

	"github.com/manu/vera/pkg/core"
)

// TestCollect_AllOk verifies Collect returns Ok([]T) when all Results are Ok.
func TestCollect_AllOk(t *testing.T) {
	results := []core.Result[int]{
		core.Ok(1),
		core.Ok(2),
		core.Ok(3),
	}

	collected := core.Collect(results)

	if !collected.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", collected.Error())
	}

	values := collected.Unwrap()
	if len(values) != 3 {
		t.Fatalf("expected 3 values, got %d", len(values))
	}

	expected := []int{1, 2, 3}
	for i, v := range values {
		if v != expected[i] {
			t.Errorf("values[%d] = %d, want %d", i, v, expected[i])
		}
	}
}

// TestCollect_SomeErr verifies Collect returns Err on first error.
func TestCollect_SomeErr(t *testing.T) {
	firstErr := errors.New("first error")
	results := []core.Result[int]{
		core.Ok(1),
		core.Err[int](firstErr),
		core.Ok(3),
	}

	collected := core.Collect(results)

	if !collected.IsErr() {
		t.Fatalf("expected Err, got Ok: %v", collected.Unwrap())
	}

	if !errors.Is(collected.Error(), firstErr) {
		t.Errorf("expected first error, got %v", collected.Error())
	}
}

// TestCollect_Empty verifies Collect handles empty slice.
func TestCollect_Empty(t *testing.T) {
	results := []core.Result[int]{}
	collected := core.Collect(results)

	if !collected.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", collected.Error())
	}

	values := collected.Unwrap()
	if len(values) != 0 {
		t.Errorf("expected empty slice, got %d values", len(values))
	}
}

// TestPartition_Mixed verifies Partition separates Ok and Err results.
func TestPartition_Mixed(t *testing.T) {
	err1 := errors.New("error 1")
	err2 := errors.New("error 2")
	results := []core.Result[int]{
		core.Ok(1),
		core.Err[int](err1),
		core.Ok(3),
		core.Err[int](err2),
		core.Ok(5),
	}

	values, errs := core.Partition(results)

	if len(values) != 3 {
		t.Errorf("expected 3 values, got %d", len(values))
	}
	expectedValues := []int{1, 3, 5}
	for i, v := range values {
		if v != expectedValues[i] {
			t.Errorf("values[%d] = %d, want %d", i, v, expectedValues[i])
		}
	}

	if len(errs) != 2 {
		t.Errorf("expected 2 errors, got %d", len(errs))
	}
	if !errors.Is(errs[0], err1) {
		t.Errorf("errs[0] = %v, want %v", errs[0], err1)
	}
	if !errors.Is(errs[1], err2) {
		t.Errorf("errs[1] = %v, want %v", errs[1], err2)
	}
}

// TestPartition_AllOk verifies Partition with no errors.
func TestPartition_AllOk(t *testing.T) {
	results := []core.Result[int]{core.Ok(1), core.Ok(2), core.Ok(3)}
	values, errs := core.Partition(results)

	if len(values) != 3 {
		t.Errorf("expected 3 values, got %d", len(values))
	}
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %d", len(errs))
	}
}

// TestPartition_AllErr verifies Partition with no successes.
func TestPartition_AllErr(t *testing.T) {
	results := []core.Result[int]{
		core.Err[int](errors.New("e1")),
		core.Err[int](errors.New("e2")),
	}
	values, errs := core.Partition(results)

	if len(values) != 0 {
		t.Errorf("expected 0 values, got %d", len(values))
	}
	if len(errs) != 2 {
		t.Errorf("expected 2 errors, got %d", len(errs))
	}
}

// TestTry_Success verifies Try wraps successful (T, error) function.
func TestTry_Success(t *testing.T) {
	result := core.Try(func() (int, error) {
		return 42, nil
	})

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 42 {
		t.Errorf("expected 42, got %d", result.Unwrap())
	}
}

// TestTry_Error verifies Try wraps failing (T, error) function.
func TestTry_Error(t *testing.T) {
	expectedErr := errors.New("failure")
	result := core.Try(func() (int, error) {
		return 0, expectedErr
	})

	if !result.IsErr() {
		t.Fatalf("expected Err, got Ok: %v", result.Unwrap())
	}

	if !errors.Is(result.Error(), expectedErr) {
		t.Errorf("expected %v, got %v", expectedErr, result.Error())
	}
}

// TestOrElse_Ok verifies OrElse doesn't call recovery function on Ok.
func TestOrElse_Ok(t *testing.T) {
	called := false
	result := core.Ok(42)
	recovered := core.OrElse(result, func(err error) core.Result[int] {
		called = true
		return core.Ok(0)
	})

	if called {
		t.Error("OrElse should not call recovery function on Ok")
	}

	if !recovered.IsOk() || recovered.Unwrap() != 42 {
		t.Errorf("expected Ok(42), got %v", recovered)
	}
}

// TestOrElse_Err verifies OrElse calls recovery function on Err.
func TestOrElse_Err(t *testing.T) {
	called := false
	result := core.Err[int](errors.New("failed"))
	recovered := core.OrElse(result, func(err error) core.Result[int] {
		called = true
		return core.Ok(99)
	})

	if !called {
		t.Error("OrElse should call recovery function on Err")
	}

	if !recovered.IsOk() || recovered.Unwrap() != 99 {
		t.Errorf("expected Ok(99), got %v", recovered)
	}
}
