package core

import "fmt"

// Result[T] represents a computation that may succeed with value T or fail with error.
// This monad provides type-safe error handling without exceptions.
//
// Categorical Properties:
//   - Functor: Map preserves composition (fmap (g ∘ f) = fmap g ∘ fmap f)
//   - Monad: FlatMap satisfies identity and associativity laws
//
// Usage:
//
//	func divide(a, b int) Result[int] {
//	    if b == 0 {
//	        return Err[int](errors.New("division by zero"))
//	    }
//	    return Ok(a / b)
//	}
type Result[T any] struct {
	value *T
	err   error
}

// Ok creates a successful Result containing value.
func Ok[T any](value T) Result[T] {
	return Result[T]{value: &value, err: nil}
}

// Err creates a failed Result containing error.
func Err[T any](err error) Result[T] {
	return Result[T]{value: nil, err: err}
}

// IsOk returns true if Result contains a value (not an error).
func (r Result[T]) IsOk() bool {
	return r.err == nil
}

// IsErr returns true if Result contains an error (not a value).
func (r Result[T]) IsErr() bool {
	return r.err != nil
}

// Unwrap returns the value or panics if Result contains an error.
// Only use when you are certain the Result is Ok (e.g., after IsOk check).
//
// For safer alternatives, use UnwrapOr or match on IsOk/IsErr.
func (r Result[T]) Unwrap() T {
	if r.IsErr() {
		panic(fmt.Sprintf("called Unwrap on Err: %v", r.err))
	}
	return *r.value
}

// UnwrapOr returns the value if Ok, or defaultValue if Err.
// This is the safe alternative to Unwrap.
func (r Result[T]) UnwrapOr(defaultValue T) T {
	if r.IsErr() {
		return defaultValue
	}
	return *r.value
}

// Error returns the error if Err, or nil if Ok.
func (r Result[T]) Error() error {
	return r.err
}

// Map applies function f to the value if Ok (Functor operation).
// If Err, propagates the error unchanged.
//
// Functor Law 1 (Identity): Map(id) = id
// Functor Law 2 (Composition): Map(g ∘ f) = Map(g) ∘ Map(f)
//
// Example:
//
//	result := Ok(5).Map(func(x int) any { return x * 2 })
//	// result.Unwrap() == 10
func Map[T, U any](r Result[T], f func(T) U) Result[U] {
	if r.IsErr() {
		return Err[U](r.err)
	}
	return Ok(f(*r.value))
}

// FlatMap applies function f that returns Result[U] (Monad operation).
// If Err, propagates the error unchanged.
//
// Monad Law 1 (Left Identity): FlatMap(Ok(a), f) = f(a)
// Monad Law 2 (Right Identity): FlatMap(m, Ok) = m
// Monad Law 3 (Associativity): FlatMap(FlatMap(m, f), g) = FlatMap(m, func(x) { FlatMap(f(x), g) })
//
// Example:
//
//	divide := func(a, b int) Result[int] {
//	    if b == 0 { return Err[int](errors.New("div by zero")) }
//	    return Ok(a / b)
//	}
//	result := FlatMap(Ok(10), func(x int) Result[int] { return divide(x, 2) })
//	// result.Unwrap() == 5
func FlatMap[T, U any](r Result[T], f func(T) Result[U]) Result[U] {
	if r.IsErr() {
		return Err[U](r.err)
	}
	return f(*r.value)
}

// AndThen is an alias for FlatMap (more readable in pipeline contexts).
func AndThen[T, U any](r Result[T], f func(T) Result[U]) Result[U] {
	return FlatMap(r, f)
}

// OrElse returns the Result if Ok, or calls f with the error if Err.
// This enables error recovery patterns.
//
// Example:
//
//	result := Err[int](errors.New("failed"))
//	recovered := OrElse(result, func(err error) Result[int] {
//	    log.Println("Recovering from:", err)
//	    return Ok(0)
//	})
func OrElse[T any](r Result[T], f func(error) Result[T]) Result[T] {
	if r.IsOk() {
		return r
	}
	return f(r.err)
}

// Collect aggregates a slice of Results into a single Result containing a slice.
// Returns Ok([]T) if all Results are Ok, or Err with the first error encountered.
//
// This is useful for batch operations where you want all-or-nothing semantics.
//
// Example:
//
//	results := []Result[int]{Ok(1), Ok(2), Ok(3)}
//	collected := Collect(results)
//	// collected.Unwrap() == []int{1, 2, 3}
//
//	results := []Result[int]{Ok(1), Err[int](errors.New("fail")), Ok(3)}
//	collected := Collect(results)
//	// collected.IsErr() == true
func Collect[T any](results []Result[T]) Result[[]T] {
	values := make([]T, 0, len(results))
	for _, r := range results {
		if r.IsErr() {
			return Err[[]T](r.err)
		}
		values = append(values, *r.value)
	}
	return Ok(values)
}

// Partition separates a slice of Results into successful values and errors.
// Unlike Collect, this doesn't short-circuit on first error.
//
// Returns (values, errors) where values contains all Ok results and errors contains all Err results.
//
// Example:
//
//	results := []Result[int]{Ok(1), Err[int](errors.New("e1")), Ok(3), Err[int](errors.New("e2"))}
//	values, errs := Partition(results)
//	// values == []int{1, 3}
//	// errs == []error{errors.New("e1"), errors.New("e2")}
func Partition[T any](results []Result[T]) ([]T, []error) {
	values := make([]T, 0, len(results))
	errs := make([]error, 0, len(results))
	for _, r := range results {
		if r.IsOk() {
			values = append(values, *r.value)
		} else {
			errs = append(errs, r.err)
		}
	}
	return values, errs
}

// Try adapts a traditional Go (T, error) function to Result[T].
// This is useful for wrapping existing Go APIs into the Result monad.
//
// Example:
//
//	result := Try(func() (int, error) {
//	    return strconv.Atoi("42")
//	})
//	// result.Unwrap() == 42
//
//	result := Try(func() (int, error) {
//	    return strconv.Atoi("invalid")
//	})
//	// result.IsErr() == true
func Try[T any](f func() (T, error)) Result[T] {
	value, err := f()
	if err != nil {
		return Err[T](err)
	}
	return Ok(value)
}
