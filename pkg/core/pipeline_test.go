package core_test

import (
	"context"
	"errors"
	"testing"

	"github.com/manu/vera/pkg/core"
)

// TestIf_TrueBranch verifies If executes ifTrue pipeline when predicate is true.
func TestIf_TrueBranch(t *testing.T) {
	ctx := context.Background()

	ifTrue := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 2) // double
	})

	ifFalse := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 3) // triple
	})

	pipeline := core.If(
		func(ctx context.Context, x int) bool { return x > 5 },
		ifTrue,
		ifFalse,
	)

	result := pipeline.Execute(ctx, 10) // 10 > 5, should double

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 20 {
		t.Errorf("expected 20 (doubled), got %d", result.Unwrap())
	}
}

// TestIf_FalseBranch verifies If executes ifFalse pipeline when predicate is false.
func TestIf_FalseBranch(t *testing.T) {
	ctx := context.Background()

	ifTrue := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 2) // double
	})

	ifFalse := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 3) // triple
	})

	pipeline := core.If(
		func(ctx context.Context, x int) bool { return x > 5 },
		ifTrue,
		ifFalse,
	)

	result := pipeline.Execute(ctx, 3) // 3 <= 5, should triple

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 9 {
		t.Errorf("expected 9 (tripled), got %d", result.Unwrap())
	}
}

// TestIf_ErrorInSelectedBranch verifies If propagates errors from selected branch.
func TestIf_ErrorInSelectedBranch(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("ifTrue failed")

	ifTrue := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Err[int](expectedErr)
	})

	ifFalse := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 3)
	})

	pipeline := core.If(
		func(ctx context.Context, x int) bool { return true }, // always true
		ifTrue,
		ifFalse,
	)

	result := pipeline.Execute(ctx, 10)

	if !result.IsErr() {
		t.Fatalf("expected Err, got Ok: %v", result.Unwrap())
	}

	if !errors.Is(result.Error(), expectedErr) {
		t.Errorf("expected %v, got %v", expectedErr, result.Error())
	}
}

// TestUntil_PredicateBecomesTrue verifies Until stops when predicate becomes true.
func TestUntil_PredicateBecomesTrue(t *testing.T) {
	ctx := context.Background()

	// Pipeline that doubles the value
	double := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 2)
	})

	// Stop when value >= 100
	pipeline := core.Until(
		func(ctx context.Context, x int) bool { return x >= 100 },
		double,
		10, // max iterations
	)

	result := pipeline.Execute(ctx, 5)
	// 5 -> 10 -> 20 -> 40 -> 80 -> 160 (5 iterations)

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 160 {
		t.Errorf("expected 160, got %d", result.Unwrap())
	}
}

// TestUntil_MaxIterations verifies Until stops at max iterations.
func TestUntil_MaxIterations(t *testing.T) {
	ctx := context.Background()

	// Pipeline that increments by 1
	increment := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x + 1)
	})

	// Never-true predicate (would loop forever without max iterations)
	pipeline := core.Until(
		func(ctx context.Context, x int) bool { return false },
		increment,
		5, // max 5 iterations
	)

	result := pipeline.Execute(ctx, 0)
	// 0 -> 1 -> 2 -> 3 -> 4 -> 5 (5 iterations, then stop)

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 5 {
		t.Errorf("expected 5 (after 5 iterations), got %d", result.Unwrap())
	}
}

// TestUntil_PredicateInitiallyTrue verifies Until returns immediately if predicate is initially true.
func TestUntil_PredicateInitiallyTrue(t *testing.T) {
	ctx := context.Background()

	// Pipeline that should never execute
	double := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		t.Error("Pipeline should not execute when predicate is initially true")
		return core.Ok(x * 2)
	})

	// Already true predicate
	pipeline := core.Until(
		func(ctx context.Context, x int) bool { return true },
		double,
		10,
	)

	result := pipeline.Execute(ctx, 42)

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 42 {
		t.Errorf("expected 42 (unchanged), got %d", result.Unwrap())
	}
}

// TestUntil_ErrorInIteration verifies Until propagates errors from pipeline.
func TestUntil_ErrorInIteration(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("iteration failed")

	iterations := 0
	failOnThird := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		iterations++
		if iterations == 3 {
			return core.Err[int](expectedErr)
		}
		return core.Ok(x + 1)
	})

	pipeline := core.Until(
		func(ctx context.Context, x int) bool { return false }, // never true
		failOnThird,
		10,
	)

	result := pipeline.Execute(ctx, 0)

	if !result.IsErr() {
		t.Fatalf("expected Err, got Ok: %v", result.Unwrap())
	}

	if !errors.Is(result.Error(), expectedErr) {
		t.Errorf("expected %v, got %v", expectedErr, result.Error())
	}

	if iterations != 3 {
		t.Errorf("expected 3 iterations before error, got %d", iterations)
	}
}

// TestUntil_ContextCancellation verifies Until respects context cancellation.
func TestUntil_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	iterations := 0
	cancelOnSecond := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		iterations++
		if iterations == 2 {
			cancel() // Cancel context on second iteration
		}
		return core.Ok(x + 1)
	})

	pipeline := core.Until(
		func(ctx context.Context, x int) bool { return false }, // never true
		cancelOnSecond,
		10,
	)

	result := pipeline.Execute(ctx, 0)

	if !result.IsErr() {
		t.Fatalf("expected Err (context canceled), got Ok: %v", result.Unwrap())
	}

	if !errors.Is(result.Error(), context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", result.Error())
	}
}

// TestConditional_PredicateTrue verifies Conditional executes pipeline when true.
func TestConditional_PredicateTrue(t *testing.T) {
	ctx := context.Background()

	double := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 2)
	})

	pipeline := core.Conditional(
		func(ctx context.Context, x int) bool { return x > 5 },
		double,
	)

	result := pipeline.Execute(ctx, 10)

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 20 {
		t.Errorf("expected 20 (doubled), got %d", result.Unwrap())
	}
}

// TestConditional_PredicateFalse verifies Conditional returns input unchanged when false.
func TestConditional_PredicateFalse(t *testing.T) {
	ctx := context.Background()

	double := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		t.Error("Pipeline should not execute when predicate is false")
		return core.Ok(x * 2)
	})

	pipeline := core.Conditional(
		func(ctx context.Context, x int) bool { return x > 5 },
		double,
	)

	result := pipeline.Execute(ctx, 3) // 3 <= 5, should not execute

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 3 {
		t.Errorf("expected 3 (unchanged), got %d", result.Unwrap())
	}
}

// TestSequence_Success verifies Sequence chains two pipelines correctly.
func TestSequence_Success(t *testing.T) {
	ctx := context.Background()

	add10 := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x + 10)
	})

	double := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 2)
	})

	pipeline := core.Sequence(add10, double)
	result := pipeline.Execute(ctx, 5)
	// 5 -> add 10 -> 15 -> double -> 30

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 30 {
		t.Errorf("expected 30, got %d", result.Unwrap())
	}
}

// TestSequence_FirstError verifies Sequence propagates error from first pipeline.
func TestSequence_FirstError(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("first failed")

	failFirst := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Err[int](expectedErr)
	})

	double := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		t.Error("Second pipeline should not execute when first fails")
		return core.Ok(x * 2)
	})

	pipeline := core.Sequence(failFirst, double)
	result := pipeline.Execute(ctx, 5)

	if !result.IsErr() {
		t.Fatalf("expected Err, got Ok: %v", result.Unwrap())
	}

	if !errors.Is(result.Error(), expectedErr) {
		t.Errorf("expected %v, got %v", expectedErr, result.Error())
	}
}

// TestSequence_SecondError verifies Sequence propagates error from second pipeline.
func TestSequence_SecondError(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("second failed")

	add10 := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x + 10)
	})

	failSecond := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Err[int](expectedErr)
	})

	pipeline := core.Sequence(add10, failSecond)
	result := pipeline.Execute(ctx, 5)

	if !result.IsErr() {
		t.Fatalf("expected Err, got Ok: %v", result.Unwrap())
	}

	if !errors.Is(result.Error(), expectedErr) {
		t.Errorf("expected %v, got %v", expectedErr, result.Error())
	}
}

// TestParallel_Success verifies Parallel executes two pipelines concurrently.
func TestParallel_Success(t *testing.T) {
	ctx := context.Background()

	add10 := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x + 10)
	})

	double := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 2)
	})

	pipeline := core.Parallel(add10, double)
	result := pipeline.Execute(ctx, 5)
	// 5 || add10 -> 15, 5 || double -> 10

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	results := result.Unwrap()
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	r1, ok1 := results[0].(int)
	r2, ok2 := results[1].(int)

	if !ok1 || !ok2 {
		t.Fatalf("expected both results to be int, got %T and %T", results[0], results[1])
	}

	if r1 != 15 {
		t.Errorf("expected first result = 15, got %d", r1)
	}

	if r2 != 10 {
		t.Errorf("expected second result = 10, got %d", r2)
	}
}

// TestParallel_FirstError verifies Parallel returns error when first pipeline fails.
func TestParallel_FirstError(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("first failed")

	failFirst := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Err[int](expectedErr)
	})

	double := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x * 2)
	})

	pipeline := core.Parallel(failFirst, double)
	result := pipeline.Execute(ctx, 5)

	if !result.IsErr() {
		t.Fatalf("expected Err, got Ok: %v", result.Unwrap())
	}

	if !errors.Is(result.Error(), expectedErr) {
		t.Errorf("expected %v, got %v", expectedErr, result.Error())
	}
}

// TestParallel_SecondError verifies Parallel returns error when second pipeline fails.
func TestParallel_SecondError(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("second failed")

	add10 := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Ok(x + 10)
	})

	failSecond := core.PipelineFunc[int, int](func(ctx context.Context, x int) core.Result[int] {
		return core.Err[int](expectedErr)
	})

	pipeline := core.Parallel(add10, failSecond)
	result := pipeline.Execute(ctx, 5)

	if !result.IsErr() {
		t.Fatalf("expected Err, got Ok: %v", result.Unwrap())
	}

	if !errors.Is(result.Error(), expectedErr) {
		t.Errorf("expected %v, got %v", expectedErr, result.Error())
	}
}

// TestIdentity_Success verifies Identity returns input unchanged.
func TestIdentity_Success(t *testing.T) {
	ctx := context.Background()

	pipeline := core.Identity[int]()
	result := pipeline.Execute(ctx, 42)

	if !result.IsOk() {
		t.Fatalf("expected Ok, got Err: %v", result.Error())
	}

	if result.Unwrap() != 42 {
		t.Errorf("expected 42 (unchanged), got %d", result.Unwrap())
	}
}
