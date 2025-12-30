# ADR-002: Go with Functional Programming Patterns

**Status**: Proposed
**Date**: 2025-12-29
**Context**: VERA Categorical Verification System - Language & Paradigm Decision

## Context

VERA requires:
1. **Result type**: Handle errors without exceptions
2. **Pipeline composition**: Chain operations with .Then()
3. **Type safety**: Invalid states unrepresentable
4. **Performance**: Production-grade latency
5. **Deployment simplicity**: Single binary, cross-platform

Need to choose language and programming paradigm.

## Decision

**Use Go with functional programming patterns via IBM's fp-go library.**

```go
import (
    "github.com/IBM/fp-go/either"
    "github.com/IBM/fp-go/option"
)

// Result[T] pattern using Either
type Result[T any] = either.Either[error, T]

// Pipeline composition
type Pipeline[In, Out any] interface {
    Run(ctx context.Context, input In) Result[Out]
    Then[Next any](next Pipeline[Out, Next]) Pipeline[In, Next]
}

// Usage
result := pipeline.
    Then(Ingest).
    Then(Query).
    Then(Verify).
    Run(ctx, input)
```

### Why Go + Functional

| Requirement | Go Solution |
|-------------|-------------|
| Result type | fp-go Either[error, T] |
| Composition | Generic interfaces + methods |
| Type safety | Go generics (1.18+) |
| Performance | Native compilation, no GC pressure for value types |
| Deployment | Single static binary |
| Ecosystem | Anthropic SDK, OpenAI SDK, Cobra CLI |

## Consequences

### Positive
- **Single binary deployment**: No runtime dependencies
- **Strong type system**: Generics enable type-safe pipelines
- **Performance**: Native code, no VM overhead
- **Ecosystem**: Excellent LLM SDKs (anthropic-sdk-go, openai-go)
- **Concurrency**: Goroutines for parallel operations
- **Maintainability**: Explicit error handling, no exceptions

### Negative
- **Functional patterns unfamiliar**: Not idiomatic Go
- **Generic syntax**: Can be verbose
- **No method generics**: Workarounds needed for some patterns
- **fp-go dependency**: External library (well-maintained by IBM)

### Neutral
- Testing with rapid (property-based) instead of just testify
- Need to document fp patterns for Go developers

## Alternatives Considered

### Alternative 1: Rust
- **Pros**: Native Result type, strong safety guarantees, no GC
- **Cons**: Steeper learning curve, smaller LLM SDK ecosystem, longer build times
- **Why rejected**: Ecosystem maturity for LLM integration; team familiarity

### Alternative 2: TypeScript
- **Pros**: Familiar, Effect-TS for functional patterns, fast iteration
- **Cons**: Requires Node.js runtime, deployment complexity, performance
- **Why rejected**: Enterprise deployment constraints, latency requirements

### Alternative 3: Python with Pydantic
- **Pros**: Best LLM ecosystem, rapid prototyping
- **Cons**: Runtime types only, performance, deployment complexity
- **Why rejected**: Type safety is critical; performance concerns

### Alternative 4: Scala with Cats Effect
- **Pros**: Best functional patterns, Effect system, JVM ecosystem
- **Cons**: Build complexity, JVM overhead, smaller DevOps adoption
- **Why rejected**: Deployment complexity, team familiarity

## References

- fp-go library: https://github.com/IBM/fp-go
- Go generics: https://go.dev/doc/tutorial/generics
- Anthropic Go SDK: https://github.com/anthropics/anthropic-sdk-go

## Constitution Compliance

- [x] Article I: Verification as First-Class - Result type enables verification
- [x] Article II: Composition Over Configuration - Pipeline.Then() composition
- [x] Article III: Provider Agnosticism - Interface-based abstraction
- [x] Article IV: Human Ownership - Go is readable; fp patterns documented
- [x] Article V: Type Safety - Generics + fp-go Either
- [x] Article VI: Categorical Correctness - fp-go follows functor/monad laws
- [x] Article VII: No Mocks in MVP - Go has good testing support
- [x] Article VIII: Graceful Degradation - Either forces error handling
- [x] Article IX: Observable by Default - OpenTelemetry Go support
