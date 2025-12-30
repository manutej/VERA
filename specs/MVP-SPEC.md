# VERA MVP Specification v1.0

**Version**: 1.0.0
**Status**: Draft
**Date**: 2025-12-29
**Classification**: SPECIFICATION DOCUMENT
**Prerequisite**: synthesis.md (0.89 quality score)

---

## 1. Overview

### 1.1 Purpose

VERA (Verifiable Evidence-grounded Reasoning Architecture) is a categorical verification system that TRANSCENDS traditional RAG through composable, type-safe verification pipelines.

**Goal Statement**: Demonstrate that verification can be modeled as a natural transformation (eta) insertable at ANY point in a document processing pipeline, producing formal grounding scores with citations.

### 1.2 Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Verification Accuracy | Grounding score correlation with human judgment | >= 0.85 |
| Pipeline Composition | Law tests passing (associativity, identity) | 100% |
| Test Coverage | Line coverage | >= 80% |
| Latency | P99 query response time | < 5 seconds |
| Human Understanding | Time to understand any file | < 10 minutes |

### 1.3 Timeline

**Duration**: 2 weeks (10 working days)

| Milestone | Days | Deliverable |
|-----------|------|-------------|
| M1: Foundation | 1-3 | Core types, Result[T], Pipeline interface |
| M2: LLM Abstraction | 4-5 | Provider interface, Anthropic implementation |
| M3: Verification Engine | 6-8 | Grounding score, citation extraction |
| M4: Pipeline Composition | 9-10 | Operators, middleware, integration |
| M5: CLI Demo | 11-12 | Working vera CLI |
| M6: Polish | 13-14 | Documentation, ownership transfer |

### 1.4 MVP Boundaries

#### In Scope

| Component | Scope |
|-----------|-------|
| LLM Provider | Anthropic Claude only |
| Document Type | PDF only |
| Verification Points | eta_1 (retrieval), eta_3 (output grounding) |
| Interface | CLI only |
| Storage | In-memory only (no persistence) |
| Users | Single-user, single-session |
| Embedding | Anthropic Voyage or OpenAI text-embedding-3-small |
| NLI | External API (DeBERTa via Hugging Face Inference) |

#### Out of Scope

- Multiple LLM providers (OpenAI, Ollama)
- REST API / server mode
- Persistent storage (PostgreSQL, pgvector)
- Streaming responses
- Multi-document type support (Markdown, HTML, DOCX)
- Custom verification policies
- Multi-user / multi-tenant
- Docker / Kubernetes deployment

---

## 2. Functional Requirements

### FR-001: Document Ingestion

**Priority**: P0
**Status**: Draft
**ADR Reference**: ADR-001

**Given** a valid PDF file path
**When** the user executes `vera ingest <path>`
**Then** the system MUST:
1. Parse the PDF and extract text content
2. Chunk the document into semantic segments (target: 512 tokens)
3. Generate embeddings for each chunk
4. Store chunks and embeddings in memory
5. Return a document ID and chunk count

**And** the system MUST emit OpenTelemetry traces for each operation

**Acceptance Criteria**:
- [ ] AC-001.1: PDF with < 100 pages ingests in < 30 seconds
- [ ] AC-001.2: Chunks are between 256-1024 tokens (configurable)
- [ ] AC-001.3: Invalid PDF returns `Result{err: ErrInvalidPDF}` (no panic)
- [ ] AC-001.4: Empty PDF returns `Result{err: ErrEmptyDocument}`
- [ ] AC-001.5: Ingestion emits span with `vera.ingest` name

---

### FR-002: Query with Verification

**Priority**: P0
**Status**: Draft
**ADR Reference**: ADR-001

**Given** ingested documents and a natural language query
**When** the user executes `vera query "<question>"`
**Then** the system MUST:
1. Generate query embedding
2. Retrieve relevant chunks using hybrid search (vector + BM25)
3. Apply eta_1: Verify retrieval quality (coverage score)
4. If coverage < 0.80, iterate retrieval (UNTIL pattern, max 3 hops)
5. Generate response using retrieved context
6. Apply eta_3: Verify output grounding against sources
7. Return response with grounding score and citations

**Acceptance Criteria**:
- [ ] AC-002.1: Query returns within 5 seconds (P99)
- [ ] AC-002.2: Response includes grounding score in [0.0, 1.0]
- [ ] AC-002.3: Response includes >= 1 citation for each claim
- [ ] AC-002.4: Low coverage triggers multi-hop retrieval automatically
- [ ] AC-002.5: Query with no relevant documents returns score < 0.70 with warning

---

### FR-003: Citation Display

**Priority**: P0
**Status**: Draft
**ADR Reference**: ADR-001

**Given** a verified response with citations
**When** the response is displayed to the user
**Then** the system MUST:
1. Display the response text
2. Display each citation with: source document, page number, text span
3. Display the grounding score for each citation
4. Display the aggregate grounding score

**Acceptance Criteria**:
- [ ] AC-003.1: Each citation includes document name and page number
- [ ] AC-003.2: Citation text spans are < 500 characters
- [ ] AC-003.3: Citations are sorted by relevance (highest grounding first)
- [ ] AC-003.4: Aggregate score displayed with interpretation (Grounded/Partial/Ungrounded)

---

### FR-004: Grounding Score Calculation

**Priority**: P0
**Status**: Draft
**ADR Reference**: ADR-001, ADR-007

**Given** a response and source documents
**When** grounding verification is triggered
**Then** the system MUST:
1. Extract atomic facts from the response
2. For each fact, calculate grounding score against sources
3. Weight facts by importance (position, specificity)
4. Aggregate into final grounding score

**Grounding Score Formula**:
```
G(response, sources) = SUM(w_i * verify(f_i, sources)) / SUM(w_i)

Where:
- f_i = atomic fact extracted from response
- w_i = importance weight
- verify(f, S) in [0,1] = hybrid embedding + NLI score
```

**Thresholds**:
| Score | Interpretation | Action |
|-------|---------------|--------|
| >= 0.85 | Fully Grounded | Approve |
| 0.70-0.84 | Partially Grounded | Approve with warning |
| < 0.70 | Ungrounded | Escalate / reject |

**Acceptance Criteria**:
- [ ] AC-004.1: Atomic fact extraction yields >= 1 fact per sentence
- [ ] AC-004.2: Grounding score is reproducible (same input = same score)
- [ ] AC-004.3: Score of 1.0 only when ALL facts are grounded
- [ ] AC-004.4: Score of 0.0 only when NO facts are grounded

---

### FR-005: Pipeline Composition

**Priority**: P0
**Status**: Draft
**ADR Reference**: ADR-001, ADR-002

**Given** pipeline stages implementing Pipeline[In, Out]
**When** pipelines are composed using operators
**Then** the system MUST satisfy categorical laws

**Operators**:
| Operator | Symbol | Go Method | Law |
|----------|--------|-----------|-----|
| Sequential | -> | `p1.Then(p2)` | Associative |
| Parallel | \|\| | `Parallel(p1, p2)` | Commutative |
| Conditional | IF | `Branch(pred, t, f)` | Either semantics |
| Iterative | UNTIL | `Until(cond, max, step)` | Fixed point |
| Verify | eta | `Apply(Verifier)` | Natural transformation |

**Law Tests**:
```go
// Associativity: (f.Then(g)).Then(h) == f.Then(g.Then(h))
// Identity: f.Then(Id) == f == Id.Then(f)
// Verification distributes: verify(f.Then(g)) == verify(f).Then(verify(g))
```

**Acceptance Criteria**:
- [ ] AC-005.1: Associativity law passes 1000 property-based tests
- [ ] AC-005.2: Identity law passes 1000 property-based tests
- [ ] AC-005.3: Composition of 5+ stages works correctly
- [ ] AC-005.4: Pipeline errors propagate without panic

---

### FR-006: UNTIL Retrieval Pattern

**Priority**: P1
**Status**: Draft
**ADR Reference**: ADR-009

**Given** a query and coverage threshold
**When** initial retrieval coverage < threshold
**Then** the system MUST:
1. Expand query using LLM
2. Retrieve additional chunks (excluding already retrieved)
3. Recalculate coverage
4. Repeat until coverage >= threshold OR max hops reached

**Configuration**:
- Coverage threshold: 0.80 (configurable)
- Max hops: 3 (configurable)
- Hybrid search weights: vector 0.5, BM25 0.5

**Acceptance Criteria**:
- [ ] AC-006.1: Multi-hop improves coverage in >= 80% of cases
- [ ] AC-006.2: Max hops limit is always respected
- [ ] AC-006.3: Each hop excludes previously retrieved chunks
- [ ] AC-006.4: Coverage calculation is deterministic

---

### FR-007: Error Handling

**Priority**: P0
**Status**: Draft
**ADR Reference**: ADR-002

**Given** any VERA operation
**When** an error occurs
**Then** the system MUST:
1. Return `Result[T]{err: VERAError}` (never panic)
2. Include error context (operation, input, cause)
3. Log error with structured fields
4. Emit error span to OpenTelemetry

**Error Types**:
```go
type ErrorKind string
const (
    ErrKindValidation   ErrorKind = "validation"
    ErrKindRetrieval    ErrorKind = "retrieval"
    ErrKindVerification ErrorKind = "verification"
    ErrKindProvider     ErrorKind = "provider"
    ErrKindInternal     ErrorKind = "internal"
)
```

**Acceptance Criteria**:
- [ ] AC-007.1: Zero panics in all test scenarios
- [ ] AC-007.2: All errors include operation context
- [ ] AC-007.3: Error logs use structured slog format
- [ ] AC-007.4: Error traces include error.type attribute

---

### FR-008: Observability

**Priority**: P1
**Status**: Draft
**ADR Reference**: ADR-005

**Given** any VERA operation
**When** the operation executes
**Then** the system MUST emit:
1. OpenTelemetry trace span with operation name
2. Structured log entry with context
3. Duration metric

**Trace Attributes**:
| Attribute | Description |
|-----------|-------------|
| `vera.operation` | Operation name |
| `vera.document_id` | Document being processed |
| `vera.grounding_score` | Verification score |
| `vera.phase` | Current pipeline phase |

**Acceptance Criteria**:
- [ ] AC-008.1: All operations emit at least one span
- [ ] AC-008.2: Spans include duration and status
- [ ] AC-008.3: Logs use slog with JSON output option
- [ ] AC-008.4: Traces can be exported to Jaeger/stdout

---

## 3. Core Types (Go Signatures)

### 3.1 Result[T] - Either Pattern

```go
// pkg/core/result.go

import E "github.com/IBM/fp-go/either"

// Result[T] wraps Either[error, T] for error handling
type Result[T any] = E.Either[error, T]

// Ok creates a successful Result
func Ok[T any](value T) Result[T] {
    return E.Right[error](value)
}

// Err creates a failed Result
func Err[T any](err error) Result[T] {
    return E.Left[T](err)
}

// Map applies f to the value if successful
func Map[T, U any](r Result[T], f func(T) U) Result[U] {
    return E.Map[error](f)(r)
}

// FlatMap chains Result-returning functions (Kleisli composition)
func FlatMap[T, U any](r Result[T], f func(T) Result[U]) Result[U] {
    return E.Chain(f)(r)
}

// Match pattern matches on Result
func Match[T, U any](r Result[T], onErr func(error) U, onOk func(T) U) U {
    return E.Fold(onErr, onOk)(r)
}
```

### 3.2 Pipeline[In, Out] - Composable Processing

```go
// pkg/core/pipeline.go

// Pipeline represents a composable processing stage
type Pipeline[In, Out any] interface {
    // Run executes the pipeline with the given input
    Run(ctx context.Context, input In) Result[Out]
}

// ComposedPipeline chains two pipelines
type ComposedPipeline[In, Mid, Out any] struct {
    first  Pipeline[In, Mid]
    second Pipeline[Mid, Out]
}

func (c *ComposedPipeline[In, Mid, Out]) Run(ctx context.Context, input In) Result[Out] {
    mid := c.first.Run(ctx, input)
    return FlatMap(mid, func(m Mid) Result[Out] {
        return c.second.Run(ctx, m)
    })
}

// Then composes two pipelines sequentially (-> operator)
func Then[In, Mid, Out any](first Pipeline[In, Mid], second Pipeline[Mid, Out]) Pipeline[In, Out] {
    return &ComposedPipeline[In, Mid, Out]{first: first, second: second}
}
```

### 3.3 Verification[T] - Grounding Metadata

```go
// pkg/verify/verification.go

// Verification wraps a value with grounding metadata
type Verification[T any] struct {
    Value          T           `json:"value"`
    GroundingScore float64     `json:"grounding_score"` // [0.0, 1.0]
    Citations      []Citation  `json:"citations"`
    Phase          VerifyPhase `json:"phase"`
}

// Citation links a claim to a source
type Citation struct {
    ClaimText    string  `json:"claim_text"`
    SourceID     string  `json:"source_id"`
    SourceText   string  `json:"source_text"`
    PageNumber   int     `json:"page_number"`
    Score        float64 `json:"score"`
}

// VerifyPhase indicates where verification occurred
type VerifyPhase string
const (
    VerifyPhaseRetrieval VerifyPhase = "retrieval"  // eta_1
    VerifyPhaseGeneration VerifyPhase = "generation" // eta_2
    VerifyPhaseGrounding VerifyPhase = "grounding"   // eta_3
)

// IsGrounded returns true if score >= threshold
func (v Verification[T]) IsGrounded(threshold float64) bool {
    return v.GroundingScore >= threshold
}
```

### 3.4 VeraState - Session State

```go
// pkg/core/state.go

// VeraState holds the state of a VERA session
type VeraState struct {
    SessionID         string         `json:"session_id"`
    Phase             VeraPhase      `json:"phase"`
    Query             Query          `json:"query"`
    RetrievedDocs     []Document     `json:"retrieved_docs"`
    RetrievalHops     int            `json:"retrieval_hops"`
    CoverageScore     float64        `json:"coverage_score"`
    Response          string         `json:"response"`
    GroundingScore    float64        `json:"grounding_score"`
    Citations         []Citation     `json:"citations"`
    Checkpoints       []Checkpoint   `json:"checkpoints"`
}

// VeraPhase represents pipeline execution phases
type VeraPhase string
const (
    PhaseInit        VeraPhase = "init"
    PhaseIngest      VeraPhase = "ingest"
    PhaseQuery       VeraPhase = "query"
    PhaseRetrieve    VeraPhase = "retrieve"
    PhaseVerifyR     VeraPhase = "verify_retrieval"  // eta_1
    PhaseGenerate    VeraPhase = "generate"
    PhaseVerifyG     VeraPhase = "verify_grounding"  // eta_3
    PhaseComplete    VeraPhase = "complete"
    PhaseFailed      VeraPhase = "failed"
)

// Checkpoint stores state at a point in time
type Checkpoint struct {
    Phase     VeraPhase `json:"phase"`
    Timestamp time.Time `json:"timestamp"`
    StateHash string    `json:"state_hash"`
}
```

### 3.5 Error Types

```go
// pkg/core/errors.go

// VERAError provides structured error context
type VERAError struct {
    Kind    ErrorKind      `json:"kind"`
    Op      string         `json:"op"`
    Err     error          `json:"err"`
    Context map[string]any `json:"context,omitempty"`
}

func (e *VERAError) Error() string {
    return fmt.Sprintf("[%s] %s: %v", e.Kind, e.Op, e.Err)
}

func (e *VERAError) Unwrap() error {
    return e.Err
}

// ErrorKind categorizes errors
type ErrorKind string
const (
    ErrKindValidation   ErrorKind = "validation"
    ErrKindRetrieval    ErrorKind = "retrieval"
    ErrKindVerification ErrorKind = "verification"
    ErrKindProvider     ErrorKind = "provider"
    ErrKindInternal     ErrorKind = "internal"
)

// Sentinel errors
var (
    ErrEmptyDocument         = errors.New("document is empty")
    ErrInvalidPDF            = errors.New("invalid PDF format")
    ErrVerificationFailed    = errors.New("verification failed")
    ErrInsufficientEvidence  = errors.New("insufficient evidence for grounding")
    ErrGroundingBelowThreshold = errors.New("grounding score below threshold")
    ErrProviderUnavailable   = errors.New("LLM provider unavailable")
    ErrRateLimited           = errors.New("rate limited by provider")
    ErrContextTooLong        = errors.New("context exceeds model limit")
)
```

---

## 4. Pipeline Operators

### 4.1 Sequential Composition (->)

**Type Signature**: `Pipeline[A, B] -> Pipeline[B, C] -> Pipeline[A, C]`

**Go Implementation**:
```go
func Then[A, B, C any](first Pipeline[A, B], second Pipeline[B, C]) Pipeline[A, C]
```

**Semantics**:
- Output of first feeds input of second
- Error in first short-circuits (second never runs)
- Traces maintain parent-child relationship

**Laws**:
- Associativity: `(f.Then(g)).Then(h) == f.Then(g.Then(h))`
- Identity: `f.Then(Id) == f == Id.Then(f)`

### 4.2 Parallel Composition (||)

**Type Signature**: `Pipeline[A, B] -> Pipeline[A, C] -> Pipeline[A, (B, C)]`

**Go Implementation**:
```go
func Parallel[A, B, C any](p1 Pipeline[A, B], p2 Pipeline[A, C]) Pipeline[A, Tuple[B, C]]
```

**Semantics**:
- Both pipelines receive same input
- Execute concurrently via goroutines
- Wait for both to complete
- If either fails, result is error (first error returned)

**Laws**:
- Commutativity: `Parallel(f, g) == Parallel(g, f)` (up to tuple ordering)

### 4.3 Conditional (IF)

**Type Signature**: `(A -> bool) -> Pipeline[A, B] -> Pipeline[A, B] -> Pipeline[A, B]`

**Go Implementation**:
```go
func Branch[A, B any](
    predicate func(A) bool,
    ifTrue Pipeline[A, B],
    ifFalse Pipeline[A, B],
) Pipeline[A, B]
```

**Semantics**:
- Predicate evaluated on input
- Only one branch executes
- Result type is union of branch types

### 4.4 Iterative (UNTIL)

**Type Signature**: `(B -> bool) -> int -> Pipeline[A, B] -> Pipeline[A, B]`

**Go Implementation**:
```go
func Until[A, B any](
    condition func(B) bool,
    maxIterations int,
    step Pipeline[A, B],
    refine func(A, B) A,
) Pipeline[A, B]
```

**Semantics**:
- Execute step, check condition
- If condition true, return result
- If false and iterations < max, refine input and repeat
- If max reached, return last result with warning

### 4.5 Verification (eta)

**Type Signature**: `Pipeline[A, B] -> Pipeline[A, Verification[B]]`

**Go Implementation**:
```go
type Verifier[A, B any] func(Pipeline[A, B]) Pipeline[A, Verification[B]]

func WithGroundingVerification(threshold float64) Verifier[Context, Response]
func WithRetrievalVerification(coverageThreshold float64) Verifier[Query, Documents]
```

**Semantics**:
- Wraps pipeline output with verification metadata
- Calculates grounding/coverage score
- Extracts citations
- Does NOT block on failure (returns low score)

**Natural Transformation Property**:
- `verify(f.Then(g))` is equivalent to `verify(f).Then(verify(g))` when applicable

---

## 5. Verification Engine

### 5.1 eta_1: Retrieval Verification

**Input**: Query, Retrieved Documents
**Output**: Verification[Documents] with coverage score

**Algorithm**:
```
1. Parse query into key concepts
2. For each concept, check if any document chunk covers it
3. Coverage = (covered concepts) / (total concepts)
4. If coverage < threshold, return with flag for more retrieval
```

**Implementation Location**: `pkg/verify/retrieval.go`

### 5.2 eta_3: Output Grounding

**Input**: Response text, Source documents
**Output**: Verification[Response] with grounding score and citations

**Algorithm**:
```
1. Extract atomic facts from response (LLM call)
2. For each fact:
   a. Embed fact
   b. Find top-k similar source chunks (cosine similarity > 0.6)
   c. Run NLI verification on each candidate
   d. Score = max(NLI entailment scores)
3. Weight facts by importance:
   - Position weight: earlier = higher
   - Specificity weight: specific claims > general statements
   - Query relevance: directly answers query > tangential
4. Aggregate: G = sum(w_i * score_i) / sum(w_i)
5. Extract citations for facts with score > 0.7
```

**Implementation Location**: `pkg/verify/grounding.go`

### 5.3 NLI Integration

**MVP Approach**: Use Hugging Face Inference API with DeBERTa-v3-large-MNLI

**Input**: (premise: source text, hypothesis: atomic fact)
**Output**: {entailment: float, neutral: float, contradiction: float}

**Grounding Score** = entailment score (threshold: 0.7 for citation)

**Implementation Location**: `pkg/verify/nli.go`

### 5.4 Threshold Configuration

```go
// pkg/verify/config.go

type VerificationConfig struct {
    // Grounding thresholds
    GroundingFullThreshold    float64 `default:"0.85"`
    GroundingPartialThreshold float64 `default:"0.70"`

    // Retrieval thresholds
    RetrievalCoverageThreshold float64 `default:"0.80"`
    MaxRetrievalHops           int     `default:"3"`

    // Citation thresholds
    CitationScoreThreshold     float64 `default:"0.70"`
    SimilarityThreshold        float64 `default:"0.60"`

    // Atomic fact extraction
    MaxFactsPerResponse        int     `default:"20"`
}
```

---

## 6. LLM Provider Interface

### 6.1 Core Interface

```go
// pkg/llm/provider.go

// LLMProvider is the ONLY LLM dependency in VERA core business logic
type LLMProvider interface {
    // Complete generates a response for a prompt
    Complete(ctx context.Context, prompt Prompt) Result[Response]

    // Embed generates embeddings for text
    Embed(ctx context.Context, text string) Result[Embedding]

    // Stream generates a streaming response (optional for MVP)
    Stream(ctx context.Context, prompt Prompt) <-chan StreamChunk

    // Info returns provider metadata
    Info() ProviderInfo
}

// Prompt represents input to the LLM
type Prompt struct {
    System   string         `json:"system"`
    Messages []Message      `json:"messages"`
    Options  PromptOptions  `json:"options"`
}

// Message represents a conversation turn
type Message struct {
    Role    MessageRole `json:"role"`
    Content string      `json:"content"`
}

type MessageRole string
const (
    RoleUser      MessageRole = "user"
    RoleAssistant MessageRole = "assistant"
)

// PromptOptions configures generation
type PromptOptions struct {
    MaxTokens   int     `json:"max_tokens"`
    Temperature float64 `json:"temperature"`
    TopP        float64 `json:"top_p"`
    Stop        []string `json:"stop,omitempty"`
}

// Response represents LLM output
type Response struct {
    Content    string      `json:"content"`
    Usage      TokenUsage  `json:"usage"`
    Model      string      `json:"model"`
    StopReason StopReason  `json:"stop_reason"`
}

// TokenUsage tracks token consumption
type TokenUsage struct {
    InputTokens  int `json:"input_tokens"`
    OutputTokens int `json:"output_tokens"`
    TotalTokens  int `json:"total_tokens"`
}

// StopReason indicates why generation stopped
type StopReason string
const (
    StopReasonEndTurn   StopReason = "end_turn"
    StopReasonMaxTokens StopReason = "max_tokens"
    StopReasonStop      StopReason = "stop"
)

// Embedding is a vector representation
type Embedding struct {
    Vector    []float32 `json:"vector"`
    Model     string    `json:"model"`
    Dimension int       `json:"dimension"`
}

// StreamChunk represents a streaming response chunk
type StreamChunk struct {
    Delta string
    Done  bool
    Err   error
}

// ProviderInfo describes the provider
type ProviderInfo struct {
    Name           string `json:"name"`
    Model          string `json:"model"`
    EmbeddingModel string `json:"embedding_model"`
    MaxTokens      int    `json:"max_tokens"`
}
```

### 6.2 Anthropic MVP Implementation

```go
// pkg/llm/anthropic/provider.go

import "github.com/anthropics/anthropic-sdk-go"

type AnthropicProvider struct {
    client *anthropic.Client
    model  string
    embeddingProvider EmbeddingProvider // Separate for embeddings
}

type Config struct {
    APIKey         string
    Model          string // default: claude-sonnet-4-20250514
    MaxTokens      int    // default: 4096
    EmbeddingModel string // default: voyage-code-2 or text-embedding-3-small
}

func New(cfg Config) (*AnthropicProvider, error)

func (p *AnthropicProvider) Complete(ctx context.Context, prompt Prompt) Result[Response]
func (p *AnthropicProvider) Embed(ctx context.Context, text string) Result[Embedding]
func (p *AnthropicProvider) Stream(ctx context.Context, prompt Prompt) <-chan StreamChunk
func (p *AnthropicProvider) Info() ProviderInfo
```

**Note**: Anthropic does not have native embeddings. MVP MUST use:
- Option A: OpenAI text-embedding-3-small via separate client
- Option B: Voyage AI embeddings (Anthropic partner)

**ADR Required**: ADR-0015 to decide embedding strategy for MVP.

---

## 7. State Machine

### 7.1 Phase Transitions

```
                              +-----------+
                              |   init    |
                              +-----+-----+
                                    |
                              +-----v-----+
                              |  ingest   |
                              +-----+-----+
                                    |
                              +-----v-----+
                              |   query   |
                              +-----+-----+
                                    |
                              +-----v-----+
                              | retrieve  |
                              +-----+-----+
                                    |
                        +-----------v-----------+
                        |   verify_retrieval    | (eta_1)
                        +-----------+-----------+
                                    |
            coverage < 0.8 && hops < 3  |  coverage >= 0.8 || hops >= 3
                        +---------------+---------------+
                        |                               |
                        v                               v
                   +---------+                  +-------v-------+
                   | retrieve| <--(refine)      |   generate    |
                   +---------+                  +-------+-------+
                                                        |
                                          +-------------v-------------+
                                          |    verify_grounding       | (eta_3)
                                          +-------------+-------------+
                                                        |
                          grounding < 0.70              |  grounding >= 0.70
                        +-------------------------------+---------------+
                        |                                               |
                        v                                               v
                   +---------+                                  +-------v-------+
                   | failed  | (with low score, not error)      |   complete    |
                   +---------+                                  +---------------+
```

### 7.2 State Persistence (MVP)

**MVP**: In-memory only via `VeraState` struct.

```go
// internal/storage/memory.go

type MemoryStore struct {
    documents map[string]*Document
    chunks    map[string][]*Chunk
    embeddings map[string][]float32
    sessions  map[string]*VeraState
    mu        sync.RWMutex
}

func NewMemoryStore() *MemoryStore
func (s *MemoryStore) SaveDocument(doc *Document) error
func (s *MemoryStore) GetDocument(id string) (*Document, error)
func (s *MemoryStore) SaveChunks(docID string, chunks []*Chunk) error
func (s *MemoryStore) Search(query []float32, k int) ([]*Chunk, error)
func (s *MemoryStore) SaveSession(session *VeraState) error
func (s *MemoryStore) GetSession(id string) (*VeraState, error)
```

---

## 8. Observability

### 8.1 OpenTelemetry Tracing

```go
// internal/observability/tracer.go

import "go.opentelemetry.io/otel"

var tracer = otel.Tracer("vera")

// Span names follow semantic conventions
const (
    SpanIngest           = "vera.ingest"
    SpanQuery            = "vera.query"
    SpanRetrieve         = "vera.retrieve"
    SpanVerifyRetrieval  = "vera.verify.retrieval"
    SpanGenerate         = "vera.generate"
    SpanVerifyGrounding  = "vera.verify.grounding"
    SpanEmbed            = "vera.embed"
    SpanNLI              = "vera.nli"
)

// Attributes
const (
    AttrDocumentID     = "vera.document.id"
    AttrQueryText      = "vera.query.text"
    AttrGroundingScore = "vera.grounding.score"
    AttrCoverageScore  = "vera.coverage.score"
    AttrPhase          = "vera.phase"
    AttrChunkCount     = "vera.chunk.count"
    AttrHopNumber      = "vera.hop.number"
)
```

### 8.2 Structured Logging

```go
// internal/observability/logger.go

import "log/slog"

func NewLogger(level slog.Level, json bool) *slog.Logger

// Log patterns
slog.Info("document ingested",
    "document_id", docID,
    "chunk_count", len(chunks),
    "duration_ms", duration.Milliseconds(),
)

slog.Warn("low grounding score",
    "session_id", sessionID,
    "score", score,
    "threshold", threshold,
)

slog.Error("verification failed",
    "operation", "verify_grounding",
    "error", err,
    "context", ctx,
)
```

### 8.3 Metrics (Deferred to Production)

MVP exports traces and logs only. Metrics (Prometheus) deferred to production spec.

---

## 9. CLI Interface

### 9.1 Commands

```bash
# Document ingestion
vera ingest <path>            # Ingest a PDF document
  --chunk-size <int>          # Target chunk size in tokens (default: 512)
  --overlap <int>             # Chunk overlap in tokens (default: 50)

# Query with verification
vera query "<question>"       # Query ingested documents
  --threshold <float>         # Grounding threshold (default: 0.80)
  --max-hops <int>            # Max retrieval hops (default: 3)
  --verbose                   # Show detailed citations

# Document listing
vera list                     # List ingested documents
  --details                   # Show chunk counts

# Configuration
vera config show              # Show current configuration
vera config set <key> <value> # Set configuration value

# System
vera version                  # Show version information
vera help                     # Show help
```

### 9.2 Cobra Structure

```
cmd/vera/
├── main.go                   # Entry point
├── cmd/
│   ├── root.go               # Root command, global flags
│   ├── ingest.go             # vera ingest
│   ├── query.go              # vera query
│   ├── list.go               # vera list
│   ├── config.go             # vera config
│   └── version.go            # vera version
```

### 9.3 Output Format

**Query Response**:
```
Response:
  [Generated response text here...]

Grounding Score: 0.87 (GROUNDED)

Citations:
  1. [0.92] Document: contract.pdf, Page 4
     "The payment terms specify net-30 days..."

  2. [0.85] Document: contract.pdf, Page 7
     "Force majeure clauses include..."

Retrieval: 2 hops, 12 chunks, coverage 0.84
Duration: 3.2s
```

**Error Output**:
```
Error: [verification] verify_grounding: insufficient evidence for grounding
  Document: contract.pdf
  Query: "What is the penalty for early termination?"
  Grounding Score: 0.45 (UNGROUNDED)

  Suggestion: The query may require information not present in the document.
```

---

## 10. Quality Gates

### 10.1 MERCURIO Review

**Target**: >= 8.5/10 across Mental, Physical, Spiritual planes

| Plane | Focus | Threshold |
|-------|-------|-----------|
| Mental | Architectural coherence, type safety | >= 8.5 |
| Physical | Performance, implementation feasibility | >= 8.5 |
| Spiritual | Alignment with VERA philosophy, categorical correctness | >= 8.5 |

### 10.2 MARS Architecture Review

**Target**: >= 92% confidence

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Constitution Compliance | 30% | All 9 articles satisfied |
| Type Safety | 20% | Invalid states unrepresentable |
| Composability | 20% | Pipeline operators compose correctly |
| Error Handling | 15% | All paths return Result[T] |
| Observability | 15% | Traces, logs comprehensive |

### 10.3 Test Coverage

**Target**: >= 80% line coverage

| Package | Min Coverage | Focus |
|---------|--------------|-------|
| `pkg/core` | 90% | Result, Pipeline, errors |
| `pkg/llm` | 80% | Provider interface (real API tests) |
| `pkg/verify` | 85% | Grounding, citation extraction |
| `pkg/pipeline` | 90% | Operators, middleware |
| `cmd/vera` | 70% | CLI commands |

### 10.4 Law Tests

**Target**: 100% pass rate with 1000 iterations each

| Law | Test File | Property |
|-----|-----------|----------|
| Associativity | `tests/laws/associativity_test.go` | `(f.Then(g)).Then(h) == f.Then(g.Then(h))` |
| Left Identity | `tests/laws/identity_test.go` | `Id.Then(f) == f` |
| Right Identity | `tests/laws/identity_test.go` | `f.Then(Id) == f` |
| Functor Composition | `tests/laws/functor_test.go` | `Map(f.g) == Map(f).Map(g)` |

---

## 11. Open Questions

The following require ADRs before implementation:

### Q1: Embedding Provider for MVP
**Options**:
- A) OpenAI text-embedding-3-small (separate API key, proven quality)
- B) Voyage AI (Anthropic partner, tighter integration)
- C) Local model (no API dependency, setup complexity)

**Recommendation**: OpenAI (A) for MVP simplicity
**ADR**: ADR-0015

### Q2: NLI Model Hosting
**Options**:
- A) Hugging Face Inference API (no infrastructure)
- B) Local DeBERTa (better latency, setup required)
- C) LLM-based NLI (use Claude, higher cost)

**Recommendation**: Hugging Face API (A) for MVP
**ADR**: ADR-0016

### Q3: PDF Parsing Library
**Options**:
- A) pdfcpu (pure Go, good for simple PDFs)
- B) unipdf (commercial, better quality)
- C) External service (PDF.co, higher quality)

**Recommendation**: pdfcpu (A) for MVP, upgrade path to unipdf
**ADR**: ADR-0017

### Q4: Chunk Size Strategy
**Options**:
- A) Fixed 512 tokens (simple, may split sentences)
- B) Semantic chunking (better quality, more complex)
- C) Sliding window with overlap (balance)

**Recommendation**: Sliding window (C) with 512 target, 50 overlap
**ADR**: ADR-0018

### Q5: Human Escalation Trigger
**Options**:
- A) Grounding < 0.70 always escalates
- B) Configurable threshold
- C) User-initiated only in MVP

**Recommendation**: Configurable (B) with default 0.70
**ADR**: ADR-0019

---

## Appendix A: ADR References

| ADR | Title | Status |
|-----|-------|--------|
| ADR-0001 | Use Go as Implementation Language | Proposed |
| ADR-0002 | fp-go for Functional Patterns | Proposed |
| ADR-0003 | LLM Provider Interface Design | Proposed |
| ADR-0004 | Cobra for CLI Framework | Pending |
| ADR-0005 | OpenTelemetry for Observability | Pending |
| ADR-0006 | Provider Agnosticism Architecture | Proposed |
| ADR-0007 | Multi-stage Verification (eta_1, eta_3) | Pending |
| ADR-0008 | Atomic Fact Grounding Method | Pending |
| ADR-0009 | Hybrid Retrieval with RRF | Pending |
| ADR-0010 | Context7 Documentation Protocol | Pending |
| ADR-0011 | MVP Verification Scope | Pending |
| ADR-0012 | MVP Single Provider (Anthropic) | Pending |
| ADR-0013 | MVP In-Memory Storage | Pending |
| ADR-0014 | No Mocks in MVP (Article VII) | Pending |
| ADR-0015 | Embedding Provider Selection | Pending |
| ADR-0016 | NLI Model Hosting Strategy | Pending |
| ADR-0017 | PDF Parsing Library | Pending |
| ADR-0018 | Chunk Size Strategy | Pending |
| ADR-0019 | Human Escalation Configuration | Pending |

---

## Appendix B: Acceptance Criteria Matrix

| FR | AC Count | P0 | P1 | P2 | Status |
|----|----------|----|----|----|----|
| FR-001 | 5 | 5 | 0 | 0 | Draft |
| FR-002 | 5 | 5 | 0 | 0 | Draft |
| FR-003 | 4 | 4 | 0 | 0 | Draft |
| FR-004 | 4 | 4 | 0 | 0 | Draft |
| FR-005 | 4 | 4 | 0 | 0 | Draft |
| FR-006 | 4 | 0 | 4 | 0 | Draft |
| FR-007 | 4 | 4 | 0 | 0 | Draft |
| FR-008 | 4 | 0 | 4 | 0 | Draft |
| **Total** | **34** | **26** | **8** | **0** | |

---

## Appendix C: Constitution Compliance Checklist

| Article | Description | Compliance | Evidence |
|---------|-------------|------------|----------|
| I | Verification as First-Class | YES | eta_1, eta_3 at core of design |
| II | Composition Over Configuration | YES | Pipeline operators, no config flags |
| III | Provider Agnosticism | YES | LLMProvider interface, Section 6 |
| IV | Human Ownership | YES | < 10 min per file, clear naming |
| V | Type Safety | YES | Result[T], Verification[T], typed phases |
| VI | Categorical Correctness | YES | Law tests required, Section 4 |
| VII | No Mocks in MVP | YES | Real API calls, integration tests |
| VIII | Graceful Degradation | YES | Result[T] everywhere, Section 3.5 |
| IX | Observable by Default | YES | OpenTelemetry, slog, Section 8 |

---

**Document Status**: Draft - Pending Review
**Next Action**: MERCURIO + MARS review before Human Gate
**Quality Target**: MERCURIO >= 8.5/10, MARS >= 92%

---

*Generated by: Specification-Driven Development Expert*
*Date: 2025-12-29*
*Input: synthesis.md (0.89 quality score)*
