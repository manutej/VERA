package core

import "context"

// Pipeline[In, Out] represents a composable transformation with error handling.
//
// Categorical Properties:
//   - Identity: Pipeline that returns input unchanged
//   - Composition: Sequence(p1, p2) chains pipelines sequentially
//   - Parallel: Executes pipelines concurrently
//
// Design:
// All pipelines return Result[Out] instead of (Out, error) tuples.
// This enables functional composition without explicit error checking.
//
// Example:
//
//	parsePDF := PipelineFunc[string, Document](func(path string) Result[Document] {
//	    return pdfParser.Parse(path)
//	})
//
//	extractText := PipelineFunc[Document, string](func(doc Document) Result[string] {
//	    return Ok(doc.Content)
//	})
//
//	pipeline := Sequence(parsePDF, extractText)
//	result := pipeline.Execute("contract.pdf")
type Pipeline[In, Out any] interface {
	// Execute runs the pipeline transformation on input.
	// Returns Result[Out] containing either the transformed value or an error.
	Execute(ctx context.Context, input In) Result[Out]
}

// PipelineFunc wraps a function as a Pipeline for convenient construction.
//
// Example:
//
//	double := PipelineFunc[int, int](func(ctx context.Context, x int) Result[int] {
//	    return Ok(x * 2)
//	})
type PipelineFunc[In, Out any] func(context.Context, In) Result[Out]

// Execute implements Pipeline interface for PipelineFunc.
func (f PipelineFunc[In, Out]) Execute(ctx context.Context, input In) Result[Out] {
	return f(ctx, input)
}

// Sequence composes two pipelines sequentially (→ operator in VERA DSL).
//
// Categorical Law (Associativity):
//
//	Sequence(Sequence(p1, p2), p3) ≅ Sequence(p1, Sequence(p2, p3))
//
// Usage:
//
//	pipeline := Sequence(parsePDF, extractText)
//	// Equivalent to: input |> parsePDF |> extractText
//
// Error Handling:
// If p1 fails, returns Err[C] without executing p2.
// If p1 succeeds but p2 fails, returns p2's error.
func Sequence[A, B, C any](p1 Pipeline[A, B], p2 Pipeline[B, C]) Pipeline[A, C] {
	return PipelineFunc[A, C](func(ctx context.Context, input A) Result[C] {
		// Execute p1
		result1 := p1.Execute(ctx, input)
		if result1.IsErr() {
			return Err[C](result1.Error())
		}

		// Execute p2 with p1's output
		return p2.Execute(ctx, result1.Unwrap())
	})
}

// Parallel executes two pipelines concurrently and combines results (|| operator).
//
// Implementation:
// Uses goroutines + channels for concurrent execution.
// Waits for both pipelines to complete before returning.
//
// Error Handling:
// If either pipeline fails, returns the first error encountered.
//
// Example:
//
//	bm25 := PipelineFunc[string, []Doc](func(ctx context.Context, q string) Result[[]Doc] {
//	    return bm25Retriever.Search(ctx, q)
//	})
//
//	dense := PipelineFunc[string, []Doc](func(ctx context.Context, q string) Result[[]Doc] {
//	    return denseRetriever.Search(ctx, q)
//	})
//
//	hybrid := Parallel(bm25, dense)
//	// Executes BM25 and dense search concurrently
func Parallel[In, Out1, Out2 any](p1 Pipeline[In, Out1], p2 Pipeline[In, Out2]) Pipeline[In, [2]any] {
	return PipelineFunc[In, [2]any](func(ctx context.Context, input In) Result[[2]any] {
		// Channels for results
		ch1 := make(chan Result[Out1], 1)
		ch2 := make(chan Result[Out2], 1)

		// Execute p1 concurrently
		go func() {
			ch1 <- p1.Execute(ctx, input)
		}()

		// Execute p2 concurrently
		go func() {
			ch2 <- p2.Execute(ctx, input)
		}()

		// Wait for both results
		r1, r2 := <-ch1, <-ch2

		// Check for errors
		if r1.IsErr() {
			return Err[[2]any](r1.Error())
		}
		if r2.IsErr() {
			return Err[[2]any](r2.Error())
		}

		// Return combined results
		return Ok([2]any{r1.Unwrap(), r2.Unwrap()})
	})
}

// Identity returns a pipeline that passes input through unchanged.
//
// Categorical Law (Left/Right Identity):
//
//	Sequence(Identity[T](), p) ≅ p
//	Sequence(p, Identity[T]()) ≅ p
//
// Example:
//
//	noop := Identity[int]()
//	result := noop.Execute(ctx, 42)  // Returns Ok(42)
func Identity[T any]() Pipeline[T, T] {
	return PipelineFunc[T, T](func(ctx context.Context, input T) Result[T] {
		return Ok(input)
	})
}

// Conditional executes pipeline if predicate is true, otherwise returns input unchanged.
//
// Example:
//
//	pipeline := Conditional(
//	    func(ctx context.Context, x int) bool { return x > 0 },
//	    doublePositive,
//	)
func Conditional[T any](predicate func(context.Context, T) bool, pipeline Pipeline[T, T]) Pipeline[T, T] {
	return PipelineFunc[T, T](func(ctx context.Context, input T) Result[T] {
		if predicate(ctx, input) {
			return pipeline.Execute(ctx, input)
		}
		return Ok(input)
	})
}

// If executes ifTrue pipeline if predicate is true, otherwise executes ifFalse pipeline.
// This provides branching logic within pipeline composition.
//
// Example:
//
//	validateLength := If(
//	    func(ctx context.Context, s string) bool { return len(s) > 100 },
//	    longDocumentPipeline,
//	    shortDocumentPipeline,
//	)
//
// Error Handling:
// Executes only the selected branch - errors from the non-selected branch are not possible.
func If[T any](
	predicate func(context.Context, T) bool,
	ifTrue Pipeline[T, T],
	ifFalse Pipeline[T, T],
) Pipeline[T, T] {
	return PipelineFunc[T, T](func(ctx context.Context, input T) Result[T] {
		if predicate(ctx, input) {
			return ifTrue.Execute(ctx, input)
		}
		return ifFalse.Execute(ctx, input)
	})
}

// Until repeatedly executes pipeline until predicate becomes true or max iterations reached.
// This provides looping logic within pipeline composition.
//
// Parameters:
//   - predicate: Returns true when loop should terminate
//   - pipeline: Transformation to apply on each iteration
//   - maxIterations: Safety limit to prevent infinite loops (0 = no limit, use with caution)
//
// Returns:
//   - Ok(T) with final value when predicate becomes true or max iterations reached
//   - Err if any iteration fails
//
// Example:
//
//	refineUntilQuality := Until(
//	    func(ctx context.Context, doc Document) bool { return doc.Quality >= 0.9 },
//	    refinementPipeline,
//	    10, // max 10 refinement passes
//	)
//
// Warning:
// Setting maxIterations=0 creates potential infinite loop. Only use when predicate is guaranteed to become true.
func Until[T any](
	predicate func(context.Context, T) bool,
	pipeline Pipeline[T, T],
	maxIterations int,
) Pipeline[T, T] {
	return PipelineFunc[T, T](func(ctx context.Context, input T) Result[T] {
		current := input
		iterations := 0

		for {
			// Check termination conditions
			if predicate(ctx, current) {
				return Ok(current)
			}

			if maxIterations > 0 && iterations >= maxIterations {
				// Max iterations reached, return current value
				return Ok(current)
			}

			// Check context cancellation
			select {
			case <-ctx.Done():
				return Err[T](ctx.Err())
			default:
			}

			// Execute pipeline iteration
			result := pipeline.Execute(ctx, current)
			if result.IsErr() {
				return result
			}

			current = result.Unwrap()
			iterations++
		}
	})
}
