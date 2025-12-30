# VERA Project Configuration

**Project**: VERA - Verifiable Evidence-grounded Reasoning Architecture
**Type**: Categorical Verification System (Transcends RAG)
**Language**: Go with fp-go for functional programming
**Status**: Planning Phase (Pre-Implementation)

---

## Quick Links

- **Meta-Prompt**: `vera-plan-meta-prompt.md` - The foundation specification document
- **Foundation**: `docs/VERA-RAG-FOUNDATION.md` - Conceptual basis from OIS-CC2.0
- **Specs**: `specs/` - MVP and Production specifications (to be generated)
- **Research**: `research/` - Parallel research outputs (to be generated)

---

## Critical Principles

### 1. Spec Before Code

> The SPEC is the foundation and must be air-tight. It is MORE IMPORTANT than the code itself.

**Process**:
1. Complete meta-prompt execution
2. Generate all specs
3. Human review and approval
4. ONLY THEN begin implementation

### 2. Context7 Documentation Protocol

**Never assume library behavior. Always verify.**

Before implementing ANY file in `pkg/` or `cmd/`:
1. Spin up Context7 subagent for library documentation
2. Extract to `docs/context7/{library}.md`
3. Use comonadic extract pattern (focused core from broader context)
4. ONLY THEN proceed to implementation

Required queries:
- `pkg/core/`: fp-go
- `pkg/llm/anthropic/`: anthropic-sdk-go
- `pkg/llm/openai/`: openai-go
- `pkg/pipeline/`: Go generics
- `cmd/vera/`: cobra CLI

### 3. Categorical Correctness

Every implementation must satisfy:
- **Functor laws**: fmap(id) = id, fmap(g ∘ f) = fmap(g) ∘ fmap(f)
- **Natural transformation laws**: η distributes correctly
- **Composition associativity**: (f → g) → h = f → (g → h)

### 4. LLM Provider Agnosticism

Core business logic must NEVER contain LLM-specific code:

```go
// The ONLY LLM dependency
type LLMProvider interface {
    Complete(ctx context.Context, prompt Prompt) Result[Response]
    Embed(ctx context.Context, text string) Result[Embedding]
    Stream(ctx context.Context, prompt Prompt) <-chan StreamChunk
}
```

---

## Constitution (9 Immutable Articles)

See `vera-plan-meta-prompt.md` Section 2 for details:

1. **Verification as First-Class** - Every claim must be verifiable
2. **Composition Over Configuration** - Behavior from composition, not config
3. **Provider Agnosticism** - No LLM logic in business logic
4. **Human Ownership** - Any file understood in <10 min
5. **Type Safety** - Invalid states unrepresentable
6. **Categorical Correctness** - Laws verified through tests
7. **No Mocks in MVP** - Real capability demonstration
8. **Graceful Degradation** - Explicit failure handling
9. **Observable by Default** - Traces, logs, metrics

---

## Operator Reference

| Operator | Name | Type | Go Pattern |
|----------|------|------|------------|
| `→` | Sequential | Kleisli | `pipeline.Then(next)` |
| `||` | Parallel | Concurrent | `sync.WaitGroup` / goroutines |
| `×` | Product | Cartesian | `struct { A; B }` |
| `⊗` | Tensor | Monoidal | Type-level composition |
| `IF` | Conditional | Either | `Result[T].Match()` |
| `UNTIL` | Recursive | Fixed point | Quality-gated loop |
| `η` | Verify | Natural transformation | Middleware pattern |

---

## Directory Structure

```
VERA/
├── vera-plan-meta-prompt.md  # Foundation specification
├── CLAUDE.md                  # This file
├── .specify/
│   └── decisions/             # ADRs
├── specs/
│   ├── MVP-SPEC.md           # Minimal viable specification
│   └── PRODUCTION-SPEC.md    # Full production specification
├── research/
│   ├── spec-kit-analysis.md
│   ├── 12-factor-analysis.md
│   ├── go-functional.md
│   ├── context-engineering.md
│   ├── verification-architectures.md
│   └── go-project-patterns.md
├── docs/
│   ├── VERA-RAG-FOUNDATION.md
│   └── context7/              # Library documentation extracts
├── cmd/
│   ├── vera/                  # CLI entry point
│   └── vera-server/           # API server (production)
├── pkg/
│   ├── core/                  # Result, Pipeline, base types
│   ├── llm/                   # LLM provider abstraction
│   │   ├── anthropic/
│   │   ├── openai/
│   │   └── ollama/
│   ├── ingest/                # Document processing
│   ├── verify/                # Verification engine
│   ├── pipeline/              # Pipeline composition
│   └── api/                   # REST API handlers
├── internal/
│   ├── config/                # Configuration
│   ├── storage/               # Persistence
│   └── observability/         # Tracing, metrics
└── tests/
    ├── laws/                  # Categorical law tests
    ├── integration/           # Integration tests
    └── benchmarks/            # Performance tests
```

---

## Go Patterns

### Result[T] (Either Pattern)

```go
type Result[T any] struct {
    value T
    err   error
    ok    bool
}

func Ok[T any](value T) Result[T]
func Err[T any](err error) Result[T]
func (r Result[T]) Map[U any](f func(T) U) Result[U]
func (r Result[T]) FlatMap[U any](f func(T) Result[U]) Result[U]
```

### Pipeline Composition

```go
type Pipeline[In, Out any] interface {
    Run(ctx context.Context, input In) Result[Out]
    Then[Next any](next Pipeline[Out, Next]) Pipeline[In, Next]
}
```

### Verification as Middleware

```go
type Verifier func(Pipeline) Pipeline

func WithGroundingVerification(threshold float64) Verifier {
    return func(p Pipeline) Pipeline {
        return &verifiedPipeline{inner: p, threshold: threshold}
    }
}

// Usage
pipeline := NewPipeline().
    Then(Ingest).
    Then(Query).
    Apply(WithGroundingVerification(0.8)).  // η inserted here
    Then(Respond)
```

---

## Phase Tracking

### Current Phase: 0 - Planning

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | IN PROGRESS | Meta-prompt creation |
| 1 | PENDING | Parallel research (6 streams) |
| 2 | PENDING | Research synthesis |
| 3 | PENDING | MVP specification |
| 4 | PENDING | Production specification |
| 5 | PENDING | Build planning |
| 6 | PENDING | **HUMAN GATE** (approval required) |
| 7 | PENDING | Implementation |
| 8 | PENDING | Testing + verification |
| 9 | PENDING | Documentation |
| 10 | PENDING | Ownership transfer |

---

## Commands

```bash
# Execute meta-prompt
/meta @tier:L6 @quality:0.95 @mode:iterative "Execute VERA vera-plan-meta-prompt.md"

# Recursive refinement
/rmp @max_iterations:5 @convergence:0.95 "Refine each spec section until quality met"

# Context7 documentation lookup
/ctx7 fp-go either option
/ctx7 anthropic-sdk-go messages streaming
/ctx7 cobra cli-patterns

# OIS workflow planning
/ois-plan "VERA verification engine" --complexity L4
```

---

## Quality Gates

| Phase | Gate | Threshold |
|-------|------|-----------|
| Research | Individual quality | ≥ 0.85 |
| Synthesis | MERCURIO | ≥ 8.5/10 |
| Architecture | MARS | ≥ 92% |
| Specs | Human Review | Explicit approval |
| Build | Test pass | 100% |
| Build | Coverage | ≥ 80% (MVP), ≥ 90% (Prod) |
| Ownership | Understanding | < 10 min per file |

---

## OIS-CC2.0 Integration

VERA uses OIS for orchestration and CC2.0 for categorical structures:

| Phase | CC2.0 Function | OIS Agents |
|-------|---------------|------------|
| Research | OBSERVE (Comonad) | KnowledgeGatherer × 6 |
| Synthesis | REASON (Functor) | ConnectionFinder |
| Architecture | REASON (Functor) | ArchitectureGenerator |
| Specs | CREATE (Function) | PromptGenerator |
| Review | VERIFY (Applicative) | MERCURIO, MARS |
| Build | CREATE (Function) | PatternApplicator |
| Test | VERIFY (Applicative) | TypeRefiner |

---

## Notes

- **No implementation until specs approved**
- **All code must have Context7 documentation grounding**
- **Run categorical law tests on EVERY change**
- **Human ownership is the end goal** - clarity over cleverness

---

*Last Updated*: 2025-12-29
*Phase*: Planning
*Status*: Meta-prompt complete, research pending
