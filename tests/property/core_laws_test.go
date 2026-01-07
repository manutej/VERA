package property

import (
	"context"
	"errors"
	"testing"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
	"github.com/manu/vera/pkg/core"
)

// TestFunctorLaws verifies that Result[T] satisfies functor laws.
//
// Functor Law 1 (Identity): fmap id = id
//   Mapping the identity function should be equivalent to not mapping at all.
//
// Functor Law 2 (Composition): fmap (g ∘ f) = fmap g ∘ fmap f
//   Mapping a composition should be equivalent to composing two maps.
//
// These laws ensure Map behaves predictably and enables safe refactoring.
func TestFunctorLaws(t *testing.T) {
	properties := gopter.NewProperties(nil)

	// Law 1: Identity
	// Map(id) should equal identity function
	properties.Property("Functor Law 1: Map(id) = id", prop.ForAll(
		func(x int) bool {
			r := core.Ok(x)

			// Map identity function
			mapped := core.Map(r, func(v int) int { return v })

			// Should return same value
			return mapped.IsOk() && mapped.Unwrap() == x
		},
		gen.Int(),
	))

	// Law 2: Composition
	// Map(g ∘ f) should equal Map(g) ∘ Map(f)
	properties.Property("Functor Law 2: Map(g ∘ f) = Map(g) ∘ Map(f)", prop.ForAll(
		func(x int) bool {
			// Define f: int -> int (add 1)
			f := func(v int) int { return v + 1 }
			// Define g: int -> int (multiply by 2)
			g := func(v int) int { return v * 2 }

			r := core.Ok(x)

			// Left side: Map(g ∘ f) - compose first, then map
			composed := core.Map(r, func(v int) int { return g(f(v)) })

			// Right side: Map(g) ∘ Map(f) - map twice
			chained := core.Map(core.Map(r, f), g)

			// Both should produce same result
			return composed.IsOk() && chained.IsOk() &&
				composed.Unwrap() == chained.Unwrap()
		},
		gen.Int(),
	))

	// Additional: Error propagation through Map
	properties.Property("Map propagates errors unchanged", prop.ForAll(
		func(msg string) bool {
			err := errors.New(msg)
			r := core.Err[int](err)

			// Map should not execute function on error
			mapped := core.Map(r, func(v int) int { return v * 2 })

			// Error should propagate
			return mapped.IsErr() && mapped.Error().Error() == msg
		},
		gen.AnyString(),
	))

	// Run all functor property tests
	properties.TestingRun(t, gopter.ConsoleReporter(false))
}

// TestMonadLaws verifies that Result[T] satisfies monad laws.
//
// Monad Law 1 (Left Identity): return a >>= f = f a
//   Wrapping a value and binding should equal applying the function directly.
//
// Monad Law 2 (Right Identity): m >>= return = m
//   Binding to the return function should not change the monad.
//
// Monad Law 3 (Associativity): (m >>= f) >>= g = m >>= (\x -> f x >>= g)
//   The order of binding operations should not matter.
//
// These laws ensure FlatMap enables safe sequential composition.
func TestMonadLaws(t *testing.T) {
	properties := gopter.NewProperties(nil)

	// Law 1: Left Identity
	// FlatMap(Ok(a), f) should equal f(a)
	properties.Property("Monad Law 1: FlatMap(Ok(a), f) = f(a)", prop.ForAll(
		func(x int) bool {
			// Define f: int -> Result[int]
			f := func(v int) core.Result[int] {
				return core.Ok(v * 2)
			}

			// Left side: FlatMap(Ok(x), f)
			left := core.FlatMap(core.Ok(x), f)

			// Right side: f(x)
			right := f(x)

			// Both should produce same result
			return left.IsOk() && right.IsOk() &&
				left.Unwrap() == right.Unwrap()
		},
		gen.Int(),
	))

	// Law 2: Right Identity
	// FlatMap(m, Ok) should equal m
	properties.Property("Monad Law 2: FlatMap(m, Ok) = m", prop.ForAll(
		func(x int) bool {
			m := core.Ok(x)

			// Bind to return (Ok) function
			bound := core.FlatMap(m, func(v int) core.Result[int] {
				return core.Ok(v)
			})

			// Should be equivalent to original monad
			return bound.IsOk() && bound.Unwrap() == x
		},
		gen.Int(),
	))

	// Law 3: Associativity
	// FlatMap(FlatMap(m, f), g) should equal FlatMap(m, func(x) { FlatMap(f(x), g) })
	properties.Property("Monad Law 3: Associativity", prop.ForAll(
		func(x int) bool {
			// Define f: int -> Result[int]
			f := func(v int) core.Result[int] {
				return core.Ok(v + 1)
			}
			// Define g: int -> Result[int]
			g := func(v int) core.Result[int] {
				return core.Ok(v * 2)
			}

			m := core.Ok(x)

			// Left side: (m >>= f) >>= g
			left := core.FlatMap(core.FlatMap(m, f), g)

			// Right side: m >>= (\x -> f x >>= g)
			right := core.FlatMap(m, func(v int) core.Result[int] {
				return core.FlatMap(f(v), g)
			})

			// Both should produce same result
			return left.IsOk() && right.IsOk() &&
				left.Unwrap() == right.Unwrap()
		},
		gen.Int(),
	))

	// Additional: Error propagation through FlatMap
	properties.Property("FlatMap propagates errors unchanged", prop.ForAll(
		func(msg string) bool {
			err := errors.New(msg)
			r := core.Err[int](err)

			// FlatMap should not execute function on error
			bound := core.FlatMap(r, func(v int) core.Result[int] {
				return core.Ok(v * 2)
			})

			// Error should propagate
			return bound.IsErr() && bound.Error().Error() == msg
		},
		gen.AnyString(),
	))

	// Run all monad property tests
	properties.TestingRun(t, gopter.ConsoleReporter(false))
}

// TestPipelineCompositionLaws verifies Pipeline composition satisfies categorical laws.
//
// Composition Law (Associativity):
//   Sequence(Sequence(p1, p2), p3) ≅ Sequence(p1, Sequence(p2, p3))
//
// Identity Law:
//   Sequence(Identity(), p) ≅ p
//   Sequence(p, Identity()) ≅ p
func TestPipelineCompositionLaws(t *testing.T) {
	properties := gopter.NewProperties(nil)

	ctx := context.Background()

	// Associativity: (p1 → p2) → p3 = p1 → (p2 → p3)
	properties.Property("Pipeline Composition: Associativity", prop.ForAll(
		func(x int) bool {
			// Define pipelines
			p1 := core.PipelineFunc[int, int](func(_ context.Context, v int) core.Result[int] {
				return core.Ok(v + 1)
			})
			p2 := core.PipelineFunc[int, int](func(_ context.Context, v int) core.Result[int] {
				return core.Ok(v * 2)
			})
			p3 := core.PipelineFunc[int, int](func(_ context.Context, v int) core.Result[int] {
				return core.Ok(v - 3)
			})

			// Left: (p1 → p2) → p3
			left := core.Sequence(core.Sequence(p1, p2), p3)
			leftResult := left.Execute(ctx, x)

			// Right: p1 → (p2 → p3)
			right := core.Sequence(p1, core.Sequence(p2, p3))
			rightResult := right.Execute(ctx, x)

			// Both should produce same result
			return leftResult.IsOk() && rightResult.IsOk() &&
				leftResult.Unwrap() == rightResult.Unwrap()
		},
		gen.Int(),
	))

	// Left Identity: Identity → p = p
	properties.Property("Pipeline Composition: Left Identity", prop.ForAll(
		func(x int) bool {
			p := core.PipelineFunc[int, int](func(_ context.Context, v int) core.Result[int] {
				return core.Ok(v * 2)
			})

			// Identity → p
			composed := core.Sequence(core.Identity[int](), p)
			composedResult := composed.Execute(ctx, x)

			// Direct p
			directResult := p.Execute(ctx, x)

			// Should produce same result
			return composedResult.IsOk() && directResult.IsOk() &&
				composedResult.Unwrap() == directResult.Unwrap()
		},
		gen.Int(),
	))

	// Right Identity: p → Identity = p
	properties.Property("Pipeline Composition: Right Identity", prop.ForAll(
		func(x int) bool {
			p := core.PipelineFunc[int, int](func(_ context.Context, v int) core.Result[int] {
				return core.Ok(v * 2)
			})

			// p → Identity
			composed := core.Sequence(p, core.Identity[int]())
			composedResult := composed.Execute(ctx, x)

			// Direct p
			directResult := p.Execute(ctx, x)

			// Should produce same result
			return composedResult.IsOk() && directResult.IsOk() &&
				composedResult.Unwrap() == directResult.Unwrap()
		},
		gen.Int(),
	))

	// Run all pipeline property tests
	properties.TestingRun(t, gopter.ConsoleReporter(false))
}
