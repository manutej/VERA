# VERA: Verifiable Evidence-Grounded Reasoning Assistant

## Overview

VERA is a categorical verification framework that transcends traditional RAG by enforcing grounding quality through natural transformations.

## Key Features

- **Evidence Grounding**: Every claim must trace to source documents
- **Categorical Verification**: Uses category theory for compositional reasoning
- **Type-Safe Architecture**: Result[T] monad eliminates runtime errors
- **Provider Agnostic**: Swap LLMs with single line change

## Architecture

### Layer 1: Foundation (core)
- Result[T] monad for error handling
- VERAError with categorical error kinds
- Type-safe functional primitives

### Layer 2: Providers
- Anthropic Claude (completions)
- Ollama embeddings (nomic-embed-text)
- Provider-agnostic interfaces

### Layer 3: Ingestion
- PDF parsing (ledongthuc/pdf)
- Markdown parsing (goldmark)
- Format detection and validation

## Performance Goals

| Operation | Target | Actual |
|-----------|--------|--------|
| Parse 10 files | < 1s | TBD |
| Embed 100 chunks | < 5s | TBD |
| Ground verification | < 200ms | TBD |

## Usage

```go
// Parse document
parser := NewMarkdownParser()
doc := parser.Parse(ctx, "README.md")

// Generate embeddings
provider := NewOllamaEmbeddingProvider()
embeddings := provider.Embed(ctx, []string{doc.Content})

// Verify grounding
score := CalculateGroundingScore(claim, evidence)
```

## Constitutional Principles

1. **Type Safety** (Article V): Invalid states unrepresentable
2. **Provider Agnosticism** (Article III): No vendor lock-in
3. **Observable by Default** (Article IX): Track all metrics
4. **Graceful Degradation** (Article VIII): Never fail silently

## Status

- ✅ M1 Foundation (89.5% coverage)
- ✅ M2 Providers (12/12 tests passing)
- 🚧 M3 Ingestion (in progress)
- ⏳ M4 Verification (pending)
- ⏳ M5 Query Interface (pending)

---

*VERA: Where categorical precision meets practical verification*
