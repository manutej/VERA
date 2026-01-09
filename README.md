# VERA - Verifiable Evidence-grounded Reasoning Architecture

A categorical verification system that transcends traditional RAG by making verification a first-class citizen through natural transformations.

## Status

**Status**: Active Development - Core architecture complete (M1-M3 milestones done)

See [`docs/SPEC-COMPLIANCE-REPORT.md`](docs/SPEC-COMPLIANCE-REPORT.md) for current implementation status.

## Getting Started

### Prerequisites

- Go 1.21 or later
- An Anthropic API key (for LLM integration)

### Installation

```bash
git clone https://github.com/manutej/VERA.git
cd VERA
go build ./...
```

### Quick Start

```bash
# Run the test suite to verify installation
go test ./...

# Check verification laws pass
go test ./tests/laws/...
```

### Project Structure

```
VERA/
├── pkg/           # Core packages (pipeline, verification, providers)
├── internal/      # Internal utilities (config, storage, observability)
├── cmd/           # CLI entry points
├── specs/         # Formal specifications
├── docs/          # Documentation and guides
└── tests/         # Test suites including categorical law tests
```

## Core Concept

VERA treats verification as a **natural transformation (eta)** insertable at any point in the pipeline:

```
Pipeline[A,B] --eta--> Pipeline[A, Verified[B]]
```

This is fundamentally different from RAG which generates first, verifies later (or not at all).

## Documentation

| Document | Purpose |
|----------|---------|
| [`specs/MVP-SPEC.md`](specs/MVP-SPEC.md) | MVP specification |
| [`specs/PRODUCTION-SPEC.md`](specs/PRODUCTION-SPEC.md) | Production specification |
| [`specs/BUILD-PLAN.md`](specs/BUILD-PLAN.md) | Implementation roadmap |
| [`specs/synthesis.md`](specs/synthesis.md) | Research synthesis |
| [`docs/VERA-RAG-FOUNDATION.md`](docs/VERA-RAG-FOUNDATION.md) | Conceptual foundation |

## Constitution (9 Immutable Articles)

1. **Verification as First-Class** - Every claim must be verifiable
2. **Composition Over Configuration** - Behavior from composition, not config
3. **Provider Agnosticism** - No LLM logic in business logic
4. **Human Ownership** - Any file understood in <10 min
5. **Type Safety** - Invalid states unrepresentable
6. **Categorical Correctness** - Laws verified through tests
7. **No Mocks in MVP** - Real capability demonstration
8. **Graceful Degradation** - Explicit failure handling
9. **Observable by Default** - Traces, logs, metrics

## Technology Stack

- **Language**: Go with generics
- **FP Library**: IBM fp-go (Result[T], Either, Option)
- **CLI**: Cobra
- **LLM**: Anthropic SDK (MVP), multi-provider (Production)
- **Observability**: OpenTelemetry

## License

MIT

---

*Built with categorical foundations and functional programming principles.*
