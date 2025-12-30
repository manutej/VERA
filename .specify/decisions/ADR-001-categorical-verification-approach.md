# ADR-001: Categorical Verification as Core Architecture

**Status**: Proposed
**Date**: 2025-12-29
**Context**: VERA Categorical Verification System - Foundational Decision

## Context

RAG (Retrieval-Augmented Generation) systems suffer from:
1. **Black-box retrieval**: No visibility into why documents were selected
2. **Unverified generation**: Outputs may not be grounded in retrieved documents
3. **Bolt-on verification**: Verification added as afterthought, not composable
4. **Single-pass limitation**: Fixed top-k retrieval, no multi-hop reasoning

Enterprises (legal, medical, financial) cannot trust RAG for high-stakes decisions because there is no formal guarantee of factual grounding.

## Decision

**Model verification as a natural transformation (η) that can be inserted at ANY point in the processing pipeline.**

```go
// Verification is a pipeline transformation, not a boolean flag
type Verifier func(Pipeline[In, Out]) Pipeline[In, Out]

// Can be inserted anywhere
pipeline := Ingest.
    Then(Query).
    Apply(WithGroundingVerification(0.8)).  // η inserted here
    Then(Synthesize).
    Apply(WithCitationVerification()).      // η₂ inserted here
    Then(Respond)
```

### Key Properties

1. **Natural Transformation**: Verification distributes over composition
   - `verify(f.Then(g)) = verify(f).Then(verify(g))` (when applicable)

2. **Composable**: Multiple verification policies can be stacked
   - `WithGrounding(0.8).Then(WithCitation()).Then(WithConfidence())`

3. **Type-Safe**: Verified responses have different type than unverified
   - `Response` vs `VerifiedResponse` (compile-time distinction)

4. **Observable**: Every verification step emits telemetry

## Consequences

### Positive
- **Formal guarantees**: Grounding can be proven, not just checked
- **Flexible insertion**: Verify at retrieval, generation, or both
- **Enterprise trust**: Auditable verification chain for compliance
- **Composability**: Build complex verification policies from simple ones
- **Type safety**: Compiler enforces verification requirements

### Negative
- **Complexity**: Category theory concepts may be unfamiliar to team
- **Performance overhead**: Extra verification passes add latency
- **Learning curve**: Engineers must understand pipeline composition
- **Testing complexity**: Need property-based tests for categorical laws

### Neutral
- Requires Result[T] monad pattern (standard in functional programming)
- Architecture is different from typical Go projects (more functional)

## Alternatives Considered

### Alternative 1: Post-processing Verification
- **Approach**: Generate first, verify after, filter/rewrite if failed
- **Pros**: Simpler, familiar pattern
- **Cons**: Cannot verify retrieval quality, wastes compute on bad generations
- **Why rejected**: Doesn't enable verification at arbitrary points

### Alternative 2: LLM Self-Verification
- **Approach**: Ask LLM to verify its own output
- **Pros**: No additional infrastructure
- **Cons**: LLMs can hallucinate verification (self-consistency != truth)
- **Why rejected**: Doesn't provide formal grounding guarantees

### Alternative 3: Ensemble Verification
- **Approach**: Multiple LLMs vote on correctness
- **Pros**: Reduces single-model bias
- **Cons**: Expensive, slow, doesn't verify against source documents
- **Why rejected**: Still no formal grounding to retrieved documents

## References

- VERA Foundation: `/docs/VERA-RAG-FOUNDATION.md`
- Natural Transformations: Category theory literature
- FactScore: Atomic fact verification methodology

## Constitution Compliance

- [x] Article I: Verification as First-Class - **This is the article**
- [x] Article II: Composition Over Configuration - Pipeline composition
- [x] Article III: Provider Agnosticism - Verification is LLM-agnostic
- [x] Article IV: Human Ownership - Clear mental model
- [x] Article V: Type Safety - VerifiedResponse type
- [x] Article VI: Categorical Correctness - Natural transformation laws
- [ ] Article VII: No Mocks in MVP - N/A for architecture decision
- [x] Article VIII: Graceful Degradation - Verification failure → Result.Err
- [x] Article IX: Observable by Default - Verification emits traces
