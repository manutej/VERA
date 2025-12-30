# VERA Research Synthesis

**Phase 2 Deliverable**: Unified Research Findings
**Quality Score**: 0.89 (weighted average across all streams)
**Synthesized**: 2025-12-29
**For**: MVP-SPEC.md Generation (Phase 3)

---

## Executive Summary

This synthesis integrates findings from six parallel research streams to establish VERA's architectural foundation. The research confirms that VERA's categorical verification approach aligns with industry best practices (2024-2025) while providing novel guarantees through natural transformation verification.

### Stream Quality Scores

| Stream | Topic | Quality | Key Contribution |
|--------|-------|---------|------------------|
| A | Specification Methodology | 0.86 | ADR format, requirement traceability |
| B | 12-Factor Agents | 0.92 | Agent lifecycle, human-in-the-loop patterns |
| C | Go Functional Patterns | 0.89 | Result[T], pipeline composition, fp-go usage |
| D | Context Engineering | 0.88 | UNTIL retrieval, hybrid search, multi-hop reasoning |
| E | Verification Architectures | 0.92 | Grounding scores, NLI verification, citation extraction |
| F | Go Project Patterns | 0.88 | cmd/pkg/internal structure, Cobra CLI, OpenTelemetry |

**Aggregate Quality**: 0.89 (exceeds 0.85 threshold)

---

## I. Unified Architecture

### 1.1 Core Type System

From **Stream C** (Go Functional Patterns), VERA's type foundation:

```go
// Result[T] - Error handling as values (Either pattern)
type Result[T any] = E.Either[error, T]

// Pipeline[In, Out] - Composable processing stages
type Pipeline[In, Out] interface {
    Run(ctx context.Context, input In) Result[Out]
}

// Verification[T] - Result with grounding metadata
type Verification[T any] struct {
    Value          T
    GroundingScore float64
    Citations      []Citation
}
```

**Key Decision**: Use IBM's fp-go library for Result/Either implementation. It provides law-compliant Functor/Monad instances with zero reflection.

### 1.2 Pipeline Operators

From **Streams B, C, D**, VERA's composition operators:

| Operator | Symbol | Go Implementation | Purpose |
|----------|--------|-------------------|---------|
| Sequential | `->` | `pipeline.Compose(first, second)` | Kleisli composition |
| Parallel | `\|\|` | `pipeline.Parallel(a, b)` | Concurrent execution |
| Conditional | `IF` | `pipeline.Branch(predicate, ifTrue, ifFalse)` | Either branching |
| Iterative | `UNTIL` | `pipeline.Until(condition, maxIter, step)` | Quality-gated loops |
| Verify | `eta` | `pipeline.WithVerification(threshold)` | Natural transformation |

### 1.3 Verification Natural Transformation

From **Stream E** (Verification Architectures), the eta (verification) insertion points:

```
Query -> eta_0 -> Retrieve -> eta_1 -> Generate -> eta_2 -> Response -> eta_3 -> Output
         |                    |                    |                    |
    Query Valid         Retrieval Quality    Generation Valid    Grounding Score
```

**MVP Scope**: Implement eta_1 (retrieval verification) and eta_3 (output grounding verification).

---

## II. Verification Strategy

### 2.1 Grounding Score Methodology

From **Stream E**, selected approach: **Weighted Atomic Fact Verification**

```
G(response, sources) = SUM(w_i * verify(f_i, sources)) / SUM(w_i)

Where:
- f_i = atomic fact extracted from response
- w_i = importance weight (position, specificity, query relevance)
- verify(f, S) in [0,1] = hybrid embedding + NLI score
```

**MVP Thresholds**:
| Score | Interpretation | Action |
|-------|---------------|--------|
| >= 0.85 | Fully Grounded | Approve |
| 0.70-0.84 | Partially Grounded | Approve with warnings |
| < 0.70 | Ungrounded | Reject or escalate to human |

### 2.2 Citation Extraction

From **Stream E**, selected approach: **Hybrid Embedding + NLI**

1. Extract atomic claims from response (LLM-based)
2. Embed claims and source chunks
3. Filter candidates by cosine similarity (>0.6)
4. Verify top candidates with NLI model (DeBERTa-v3-large-MNLI)
5. Link claims to highest-scoring source spans

### 2.3 Retrieval Verification (UNTIL Pattern)

From **Streams B, D**, the quality-gated retrieval loop:

```go
func UNTILCoverage(threshold float64, maxHops int) Pipeline[Query, RetrievalResult] {
    return Until(
        func(r RetrievalResult) bool { return r.Coverage >= threshold },
        maxHops,
        HybridSearch(),
        func(q Query, r RetrievalResult) Query { return q.Exclude(r.IDs) },
    )
}
```

**MVP Configuration**:
- Coverage threshold: 0.80
- Max retrieval hops: 3
- Hybrid search: Vector (0.5) + BM25 (0.5) with RRF fusion

---

## III. Agent Architecture (12-Factor Alignment)

### 3.1 Factor Mapping to VERA

From **Stream B**, VERA satisfies all 12 factors:

| Factor | VERA Implementation |
|--------|---------------------|
| 1. NL to Tool Calls | Pipeline stages are typed tools |
| 2. Own Your Prompts | Embedded, versioned prompt templates |
| 3. Own Your Context | OBSERVE stage + UNTIL coverage |
| 4. Tools as Functions | Pipeline[In, Out] interface |
| 5. Unified State | VeraState struct with checkpoints |
| 6. Launch/Resume | Phase-based checkpointing |
| 7. Human as Tool | Phase 6 Human Gate approval |
| 8. Own Control Flow | Explicit operators (no hidden loops) |
| 9. Errors as Context | Result[T] carries error chain |
| 10. Small Agents | Modular pkg/ structure |
| 11. Trigger Anywhere | cmd/vera (CLI) + cmd/vera-server (API) |
| 12. Observable | OpenTelemetry tracing throughout |

### 3.2 State Management

From **Stream B**, VERA state structure:

```go
type VeraState struct {
    SessionID       string
    Phase           VeraPhase
    Query           Query
    RetrievedDocs   []Document
    VerificationChain []Verification
    GroundingScore  float64
    Checkpoints     []Checkpoint
}

type VeraPhase string
const (
    PhaseObserve   VeraPhase = "observe"
    PhaseReason    VeraPhase = "reason"
    PhaseRetrieve  VeraPhase = "retrieve"
    PhaseVerifyR   VeraPhase = "verify_retrieval"
    PhaseCreate    VeraPhase = "create"
    PhaseVerifyG   VeraPhase = "verify_grounding"
    PhaseComplete  VeraPhase = "complete"
)
```

---

## IV. Project Structure

### 4.1 Directory Layout

From **Stream F**, the recommended structure:

```
VERA/
├── cmd/
│   ├── vera/              # CLI entry point
│   │   ├── main.go
│   │   └── cmd/           # Cobra commands
│   └── vera-server/       # API server (production)
├── pkg/
│   ├── core/              # Result[T], Pipeline, errors
│   ├── llm/               # Provider interface + implementations
│   │   ├── provider.go
│   │   ├── anthropic/
│   │   ├── openai/
│   │   └── ollama/
│   ├── verify/            # Verification engine
│   │   ├── grounding.go
│   │   └── citation.go
│   └── pipeline/          # Composition operators
├── internal/
│   ├── config/
│   ├── storage/
│   └── observability/
└── tests/
    ├── laws/              # Categorical law tests
    └── integration/
```

### 4.2 Error Handling

From **Stream F**, VERA error types:

```go
type VERAError struct {
    Kind    ErrorKind      // validation, verification, retrieval, provider
    Op      string         // operation that failed
    Err     error          // underlying error
    Context map[string]any // additional context
}

// Sentinel errors for comparison
var (
    ErrVerificationFailed   = errors.New("verification failed")
    ErrInsufficientEvidence = errors.New("insufficient evidence")
    ErrGroundingBelowThreshold = errors.New("grounding below threshold")
)
```

---

## V. Specification Methodology

### 5.1 Requirement Format

From **Stream A**, requirements follow RFC 2119 + Gherkin:

```markdown
### FR-XXX: [Requirement Name]

**Priority**: P0 | P1 | P2
**Status**: draft | approved | implemented

**Given** [precondition]
**When** [action/trigger]
**Then** [expected outcome]

**Acceptance Criteria**:
- [ ] AC-1: [Testable criterion]
- [ ] AC-2: [Testable criterion]

**ADR References**: [ADR-XXXX]
```

### 5.2 ADR Format

From **Stream A**, MADR-based ADRs:

```markdown
# ADR-XXXX: [Decision Title]

**Status**: proposed | accepted | deprecated
**Date**: YYYY-MM-DD
**Deciders**: [list]

## Context and Problem Statement
[Why this decision is needed]

## Decision Drivers
* [Driver 1]
* [Driver 2]

## Considered Options
1. [Option 1]
2. [Option 2]

## Decision Outcome
**Chosen option**: "[Option X]", because [justification]

### Positive Consequences
* [Benefit]

### Negative Consequences
* [Tradeoff]
```

---

## VI. Key Technical Decisions (Requiring ADRs)

Based on synthesis across all streams, the following decisions require formal ADRs:

### 6.1 Language and Libraries

| Decision | Choice | Rationale | ADR |
|----------|--------|-----------|-----|
| Language | Go 1.21+ | Performance, generics, deployment | ADR-0001 |
| FP Library | IBM fp-go | Law-compliant, no reflection | ADR-0002 |
| Result Type | fp-go Either | Native composition support | ADR-0003 |
| CLI Framework | Cobra | Industry standard, completion | ADR-0004 |
| Observability | OpenTelemetry | Vendor-neutral, comprehensive | ADR-0005 |

### 6.2 Architecture Decisions

| Decision | Choice | Rationale | ADR |
|----------|--------|-----------|-----|
| LLM Abstraction | Provider interface | Article III compliance | ADR-0006 |
| Verification Timing | Multi-stage (eta_1, eta_3) | Balance latency/accuracy | ADR-0007 |
| Grounding Method | Atomic + NLI | Fine-grained, accurate | ADR-0008 |
| Retrieval Strategy | Hybrid + RRF | Industry best practice | ADR-0009 |
| Context7 Protocol | Required for all deps | Grounded implementation | ADR-0010 |

### 6.3 MVP Scope Decisions

| Decision | Choice | Rationale | ADR |
|----------|--------|-----------|-----|
| MVP Verifications | eta_1 + eta_3 | Core value proposition | ADR-0011 |
| MVP Providers | Anthropic only | Simplify initial scope | ADR-0012 |
| MVP Storage | In-memory | No persistence complexity | ADR-0013 |
| MVP No Mocks | Real API calls | Article VII compliance | ADR-0014 |

---

## VII. Quality Gates

### 7.1 Research Phase (Complete)

| Gate | Threshold | Achieved |
|------|-----------|----------|
| Individual stream quality | >= 0.85 | All passed |
| Coverage of VERA requirements | 100% | Yes |
| Actionable patterns | Yes | All streams provide code |

### 7.2 Specification Phase (Next)

| Gate | Threshold |
|------|-----------|
| Requirements completeness | 100% mandatory sections |
| Requirements precision | Zero ambiguous language |
| ADR coverage | All decisions documented |
| MERCURIO review | >= 8.5/10 |
| MARS architecture review | >= 92% |

### 7.3 Implementation Phase (After Approval)

| Gate | Threshold |
|------|-----------|
| Categorical law tests | 100% pass |
| Test coverage | >= 80% |
| Human understanding | < 10 min per file |
| Context7 documentation | All dependencies |

---

## VIII. Cross-Cutting Concerns

### 8.1 Observability (All Streams)

Every pipeline stage must emit:
- **Traces**: OpenTelemetry spans with attributes
- **Metrics**: Duration, success rate, grounding scores
- **Logs**: Structured slog with claim/source context

### 8.2 Human Ownership (Constitution Article IV)

From all streams, the <10 minute rule applies:
- Single responsibility per file
- Clear naming conventions
- Documentation in code
- Maximum file size: ~500 lines

### 8.3 No Mocks in MVP (Constitution Article VII)

From **Stream B**, the MVP must demonstrate real capability:
- Real LLM API calls (Anthropic Claude)
- Real document processing
- Real grounding verification
- Integration tests, not unit mocks

---

## IX. Implementation Priority

Based on synthesis, the recommended implementation order for MVP:

### Phase 1: Core Foundation (Week 1, Days 1-3)
1. pkg/core/result.go - Result[T] type
2. pkg/core/pipeline.go - Pipeline[In, Out] interface
3. pkg/core/errors.go - VERA error types
4. tests/laws/ - Functor, Monad, Associativity tests

### Phase 2: LLM Abstraction (Week 1, Days 4-5)
1. pkg/llm/provider.go - Provider interface
2. pkg/llm/anthropic/client.go - Anthropic implementation
3. Context7 extraction for anthropic-sdk-go

### Phase 3: Verification Engine (Week 2, Days 1-3)
1. pkg/verify/grounding.go - Grounding score calculation
2. pkg/verify/citation.go - Citation extraction
3. pkg/verify/nli.go - NLI verification integration

### Phase 4: Pipeline Composition (Week 2, Days 4-5)
1. pkg/pipeline/compose.go - Then, Parallel, Until operators
2. pkg/pipeline/middleware.go - WithVerification transformer
3. Integration test: Full pipeline execution

### Phase 5: CLI and Demo (Week 3)
1. cmd/vera/ - Cobra CLI
2. End-to-end verification demo
3. Documentation and ownership transfer

---

## X. Unresolved Questions for MVP-SPEC

1. **Embedding Model**: Use OpenAI text-embedding-3-small or local alternative?
2. **NLI Model Hosting**: Cloud API or local DeBERTa inference?
3. **Document Format Support**: PDF only or also Markdown/HTML?
4. **Grounding Threshold**: Start at 0.80 or 0.85?
5. **Human Gate Trigger**: When grounding < 0.70 or configurable?

These questions should be resolved in MVP-SPEC.md with ADRs.

---

## XI. Synthesis Conclusion

The six research streams converge on a coherent architecture for VERA:

**Core Insight**: The industry has moved from "retrieve and hope" to "retrieve, verify, iterate" - exactly what VERA's categorical verification provides.

**Differentiator**: VERA's natural transformation (eta) as a first-class composable element distinguishes it from ad-hoc verification bolted onto existing RAG systems.

**Feasibility**: All proposed patterns have working Go implementations from the research. The MVP timeline (2 weeks) is achievable with focused scope.

**Risk Mitigation**: The 12-Factor Agents alignment ensures VERA can evolve from MVP to production without architectural rewrites.

---

**Document Status**: Ready for Phase 3 (MVP-SPEC Generation)
**Next Action**: Generate MVP-SPEC.md using this synthesis as input
**Human Gate**: Synthesis review before specification begins

---

*Synthesis by: MERCURIO THE SYNTHESIZER*
*Date: 2025-12-29*
*Quality: 0.89 (exceeds 0.85 threshold)*
