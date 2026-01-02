# ADR-0024: Embedding Provider Selection

**Status**: Accepted
**Date**: 2025-12-30
**Context**: VERA Categorical Verification System
**Supersedes**: ADR-0015 (Embedding Provider - Draft)

## Context

VERA requires text embeddings for:
1. Document chunk indexing
2. Query embedding for retrieval
3. Semantic similarity for grounding verification

The original recommendation was OpenAI `text-embedding-3-small`. After comprehensive research on 2024-2025 embedding advances, we need to select a provider strategy that balances:
- Quality (MTEB retrieval scores)
- Provider independence (Constitution Article III)
- Latency (< 800ms batch for 10 files)
- Cost efficiency
- Future migration path

## Decision

### Primary Strategy: Provider-Agnostic Interface with Tiered Implementation

**MVP (Weeks 1-2)**: `nomic-ai/nomic-embed-text-v1.5` via Ollama
**Production (Month 2+)**: Same model via ONNX Runtime for performance
**Fallback**: OpenAI `text-embedding-3-small` for API simplicity

### Rationale

| Criterion | nomic-embed-text-v1.5 | OpenAI text-embedding-3-small | BGE-M3 |
|-----------|----------------------|------------------------------|--------|
| MTEB Score | 62.28 | ~62* | 64.5 |
| Matryoshka Support | **Yes (native)** | Yes (via dims param) | Partial |
| License | **Apache 2.0** | Proprietary | MIT |
| Self-Hosted | **Yes** | No | Yes |
| Go Integration | ONNX/Ollama | HTTP API | HTTP/ONNX |
| Memory | **200MB** | N/A | 1.2GB |
| Latency (batch 10) | **~60ms (ONNX)** | 120-300ms (API) | ~150ms |
| Provider Lock-in | **None** | OpenAI dependency | None |

*OpenAI scores approximated from third-party benchmarks

### Why nomic-embed-text-v1.5?

1. **Provider Independence** (Constitution Article III): No API dependency
2. **Matryoshka Support**: Reduce to 512 dims with 99.5% quality retention
3. **Apache 2.0 License**: Full commercial use, no restrictions
4. **8192 Token Context**: Handles long document chunks
5. **Instruction-Tuned**: Better asymmetric retrieval (query vs document prefixes)
6. **Smallest Memory**: 200MB fits embedded CLI scenarios
7. **Fastest Latency**: ~60ms ONNX, ~180ms Ollama (well under 800ms target)

### Interface Design

```go
// pkg/embedding/provider.go

type EmbeddingProvider interface {
    Embed(ctx context.Context, texts []string) Result[[]Embedding]
    EmbedWithDimension(ctx context.Context, texts []string, dim int) Result[[]Embedding]
    Dimension() int
    SupportsMatryoshka() bool
    Close() error
}

type Embedding struct {
    Vector []float32 `json:"vector"`
    Tokens int       `json:"tokens"`
}
```

### Configuration

```yaml
# MVP Configuration
embedding:
  provider: ollama
  model: nomic-embed-text
  dimension: 768  # Full, or 512 with Matryoshka
  batch_size: 50
  timeout: 5s
  task_prefix: "search_document"
  query_prefix: "search_query"

# Production Configuration (ONNX)
embedding:
  provider: onnx
  model_path: ./models/nomic-embed-text-v1.5.onnx
  dimension: 768
  batch_size: 100
```

### Dimension Strategy (Matryoshka)

| Dimension | Quality Retention | Storage | Use Case |
|-----------|------------------|---------|----------|
| 768 (full) | 100% | 100% | Grounding verification (high precision) |
| 512 | 99.5% | 67% | **Default retrieval** |
| 256 | 98.0% | 33% | Coarse retrieval pass |

**Recommendation**: Use 512 dims for retrieval, 768 for verification grounding scores.

## Consequences

### Positive

- **No vendor lock-in**: Can swap providers without code changes
- **Cost reduction**: Self-hosted eliminates per-token API costs
- **Latency control**: No network round-trips for embedding
- **Privacy**: Documents never leave local environment
- **Matryoshka flexibility**: Optimize storage/speed without quality loss

### Negative

- **Setup complexity**: Requires Ollama installation for MVP
- **ONNX complexity**: Production requires ONNX runtime integration in Go
- **Model updates**: Must manually update when better models release

### Neutral

- Similar quality to OpenAI (~62% MTEB) - both competitive
- Learning curve for team on self-hosted embeddings
- OpenAI fallback available if self-hosting problematic

## Migration Path

```
MVP (Week 1-4)              Production (Month 2+)
─────────────────────       ────────────────────────
Ollama (nomic-v1.5)    →    ONNX Runtime (nomic-v1.5)
       │                           │
       │                    If quality critical:
       │                           │
       │                           ▼
       │                    mxbai-embed-large (ONNX)
       │                           │
       │                    If multi-lingual:
       │                           │
       │                           ▼
       │                    BGE-M3 (FastEmbed service)
       │
       │  Fallback (API simplicity):
       └──────────────────────────────┐
                                      ▼
                              OpenAI text-embedding-3-small
```

## Alternatives Considered

### Alternative 1: OpenAI text-embedding-3-small (Original Recommendation)

- **Pros**: Simple API, proven, good quality (62.3% MTEB), Matryoshka support
- **Cons**: API dependency, per-token cost, network latency, vendor lock-in
- **Why not selected**: Violates Constitution Article III (Provider Agnosticism)

### Alternative 2: BAAI/bge-m3

- **Pros**: Highest quality (64.5% MTEB), multi-modal (dense + sparse), 100+ languages
- **Cons**: Larger memory (1.2GB), more complex integration, slower
- **Why not selected**: Over-engineered for MVP; consider for production multi-lingual

### Alternative 3: mxbai-embed-large-v1

- **Pros**: Strong quality (64.68% MTEB), Matryoshka support, Apache 2.0
- **Cons**: Larger memory (600MB), no instruction tuning
- **Why not selected**: nomic-v1.5 better quality/size ratio for MVP; consider for production

### Alternative 4: HuggingFace Inference API

- **Pros**: Access to any HF model, simple HTTP
- **Cons**: API dependency, rate limits, less control
- **Why not selected**: Same vendor lock-in concerns as OpenAI

## References

- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- nomic-embed-text-v1.5: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- Matryoshka Blog: https://huggingface.co/blog/matryoshka
- Ollama Embeddings: https://ollama.com/blog/embedding-models
- ONNX Runtime Go: https://github.com/yalue/onnxruntime_go

## Constitution Compliance

- [x] Article I: Verification as First-Class - Embeddings enable grounding verification
- [x] Article II: Composition Over Configuration - Provider interface composable
- [x] Article III: Provider Agnosticism - **Primary driver of this decision**
- [x] Article IV: Human Ownership - Simple interface, < 10 min understanding
- [x] Article V: Type Safety - Result[T] for all embedding operations
- [x] Article VI: Categorical Correctness - Interface follows functor laws
- [x] Article VII: No Mocks in MVP - Real model, real embeddings
- [x] Article VIII: Graceful Degradation - Fallback to OpenAI if needed
- [x] Article IX: Observable by Default - Embedding latency/dimension traced
