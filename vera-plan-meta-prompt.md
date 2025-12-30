# VERA Planning Meta-Prompt

**Version**: 2.0.0
**Created**: 2025-12-29
**Classification**: FOUNDATION DOCUMENT
**Purpose**: Generate air-tight specifications for VERA - Categorical Verification System

---

## CRITICAL STATEMENT

> **This specification is the FOUNDATION. It is MORE IMPORTANT than the code itself.**
>
> Code can be changed, scrapped, rebuilt. The spec is the source of truth.
> Everything downstream depends on this being AIR-TIGHT.
>
> This document will be under the HIGHEST DEGREE OF SCRUTINY.

---

## 1. WHAT IS VERA?

VERA (Verifiable Evidence-grounded Reasoning Architecture) is NOT a RAG system.
VERA TRANSCENDS RAG through categorical verification.

### What Traditional RAG Does
```
Query → Retrieve → Generate → Hope it's correct
```

### What VERA Does
```
Query → OBSERVE(context) → η₁(verify_query)
     → REASON(retrieval_plan)
     → UNTIL(coverage ≥ threshold,
         Retrieve || RetrieveMore → η₂(verify_retrieval)
       )
     → REASON(synthesis)
     → CREATE(response)
     → η₃(verify_grounding)
     → LEARN(update_memory)
     → Verified Response
```

Where:
- **η** (eta) is a natural transformation = verification insertable ANYWHERE
- **UNTIL** enables multi-hop reasoning (not single-pass)
- **LEARN** accumulates verified knowledge
- **||** enables parallel retrieval strategies

### Key Differentiators

| Traditional RAG | VERA |
|-----------------|------|
| Fixed top-k retrieval | Adaptive UNTIL coverage met |
| Single-pass generation | Multi-hop reasoning |
| Post-hoc citation | Verification at EVERY stage |
| Stateless queries | Memory integration (LEARN) |
| Hope it doesn't hallucinate | PROVE grounding categorically |

---

## 2. CONSTITUTION (Immutable Principles)

These 9 articles CANNOT be violated. Any spec that violates them is INVALID.

### Article I: Verification as First-Class
Every claim MUST be verifiable. No claim without citation. No output without grounding score.
```go
type Response interface {
    GroundingScore() float64  // MUST be present
    Citations() []Citation     // MUST be non-empty for claims
}
```

### Article II: Composition Over Configuration
System behavior emerges from composition of pure functions, not configuration files.
```go
// Good: Behavior from composition
pipeline := Ingest.Then(Query).Then(Verify).Then(Respond)

// Bad: Behavior from config
config.EnableVerification = true  // NO!
```

### Article III: Provider Agnosticism
No LLM provider logic in core business logic. All providers implement identical interface.
```go
// The ONLY LLM dependency is this interface
type LLMProvider interface {
    Complete(ctx context.Context, prompt Prompt) Result[Response]
    Embed(ctx context.Context, text string) Result[Embedding]
    Stream(ctx context.Context, prompt Prompt) <-chan StreamChunk
}
```

### Article IV: Human Ownership
Any engineer MUST understand any file in under 10 minutes.
- No clever tricks
- No hidden behavior
- No category theory jargon in public API
- Every function documented

### Article V: Type Safety
Compiler catches errors, not runtime. Invalid states are unrepresentable.
```go
// Good: Invalid states impossible
type VerifiedResponse struct {
    Content       string     // Required
    GroundingScore float64   // Required, in [0,1]
    Citations      []Citation // Required, len >= 1
}

// Bad: Invalid states possible
type Response struct {
    Content  string
    Verified bool        // Can be false with citations!
    Score    *float64    // Can be nil!
}
```

### Article VI: Categorical Correctness
Verified through tests, not trust:
- Sequential composition (→) is associative: `(f → g) → h = f → (g → h)`
- Parallel execution (||) is commutative: `f || g = g || f`
- Verification (η) is a natural transformation

### Article VII: No Mocks in MVP
MVP demonstrates REAL capability. Integration tests use real APIs (rate-limited).

### Article VIII: Graceful Degradation
Every failure mode is handled explicitly.
```go
// Every function returns Result[T], never panics
func Process(input Input) Result[Output]  // Good
func Process(input Input) Output           // Bad - can panic
```

### Article IX: Observable by Default
Every operation emits traces. Every decision is logged. Every metric is exposed.

---

## 3. OIS-CC2.0 INTEGRATION

VERA uses OIS (Ontological Implementation System) for task orchestration and CC2.0 for categorical structures.

### Phase-to-Function Mapping

| Phase | CC2.0 Function | Categorical Structure | OIS Agents |
|-------|---------------|----------------------|------------|
| Research | OBSERVE | Comonad (extract context) | KnowledgeGatherer × 6 |
| Synthesis | REASON | Functor (map insights) | ConnectionFinder |
| Architecture | REASON | Functor | ArchitectureGenerator |
| Spec Writing | CREATE | Function | PromptGenerator |
| Review | VERIFY | Applicative | DesignReflector, MERCURIO, MARS |
| Build | CREATE | Function | PatternApplicator × n |
| Test | VERIFY | Applicative | TypeRefiner |
| Ownership | LEARN | Profunctor | ExistentialReflector |

### Composition Operators

| Operator | Name | Type | Use Case |
|----------|------|------|----------|
| `→` | Sequential | Kleisli | Pipeline stages: A → B → C |
| `||` | Parallel | Concurrent execution | Independent research streams |
| `×` | Product | Cartesian | Combine outputs: A × B |
| `⊗` | Tensor | Monoidal (type-level) | Type composition |
| `IF` | Conditional | Either/Coproduct | Branch on result |
| `UNTIL` | Recursive | Fixed point | Iterate until quality met |

### Type Signatures for Phases

```
Phase 1 (Research):
  [Source₁ || Source₂ || ... || Source₆] → Observable<Research>⁶

Phase 2 (Synthesis):
  Observable<Research>⁶ → Reasoning<Synthesis>

Phase 3 (Architecture):
  Reasoning<Synthesis> → Generated<Architecture>

Phase 4-5 (Specs):
  Generated<Architecture> → [Generated<MVPSpec> || Generated<ProdSpec>]

Phase 6 (Human Gate):
  [Generated<MVPSpec> × Generated<ProdSpec>] → Verified<Specs>
  (STOP - requires human approval)

Phase 7 (Build):
  Verified<Specs> → Generated<Code>

Phase 8 (Test):
  Generated<Code> → Verified<Code>

Phase 9 (Ownership):
  Verified<Code> → Owned<System>
```

---

## 4. SPEC QUALITY FRAMEWORK

The spec must meet quality criteria across 5 dimensions.

### Dimension 1: Completeness (No Gaps)
- [ ] Every component has full specification
- [ ] Every interface has complete type signature
- [ ] Every error case has handling strategy
- [ ] Every dependency has version pinned
- [ ] Every assumption is documented

**Metric**: % of components with full specs (target: 100%)

### Dimension 2: Precision (No Ambiguity)
- [ ] Each requirement uses MUST/MUST NOT (not should/may)
- [ ] Each requirement has acceptance criteria
- [ ] Each acceptance criterion is testable
- [ ] Each test has expected outcome

**Metric**: % of requirements with testable criteria (target: 100%)

### Dimension 3: Consistency (No Contradictions)
- [ ] Types align across components
- [ ] Interfaces compose correctly
- [ ] Data flows match diagrams
- [ ] No circular dependencies

**Metric**: # of inconsistencies (target: 0)

### Dimension 4: Traceability (Every Decision Explained)
- [ ] ADR for each architectural choice
- [ ] Reference for each pattern used
- [ ] Link to research source
- [ ] Rationale for scope decisions

**Metric**: % of decisions with ADRs (target: 100%)

### Dimension 5: Reviewability (Human Can Verify)
- [ ] Clear structure with TOC
- [ ] Progressive disclosure (overview → details)
- [ ] Examples for complex concepts
- [ ] Diagrams for architecture

**Metric**: Time to understand (target: < 30 min for full spec)

### Anti-Patterns (NEVER DO THESE)

| BAD | GOOD |
|-----|------|
| "The system should handle errors appropriately" | "On API failure, return `Result[T]{err: ErrAPIFailure}` with retry count" |
| "Performance should be good" | "Query latency P99 < 3s for corpus ≤ 1000 docs" |
| "Uses best practices" | "Follows 12-factor agents: [list each + application]" |
| "Similar to LangChain" | "LLMProvider interface has: Complete, Embed, Stream. Returns Result[T]." |
| "The architecture is modular" | "Each package has single responsibility. pkg/llm handles ONLY LLM abstraction." |
| "Handles edge cases" | "Empty input → return Result{err: ErrEmptyInput}. Null → Result{err: ErrNullInput}" |

---

## 5. PHASE 1: PARALLEL RESEARCH

Launch 6 independent research agents. All run in parallel (||).

### Research Stream A: spec-kit Methodology
```yaml
agent: deep-researcher
source: https://github.com/github/spec-kit
output: /research/spec-kit-analysis.md
extract:
  - Template structure (headers, sections, formatting)
  - Quality gates GitHub uses
  - ADR (Architecture Decision Record) format
  - Review process and criteria
  - How specs connect to implementation
quality_gate: ≥ 0.85
```

### Research Stream B: 12-Factor Agents
```yaml
agent: deep-researcher
source: https://github.com/humanlayer/12-factor-agents
output: /research/12-factor-analysis.md
extract:
  - All 12 factors with definitions
  - Go implementation patterns for each
  - Agent lifecycle management
  - Human-in-the-loop patterns
  - State management approaches
  - Comparison to 12-factor apps
quality_gate: ≥ 0.85
```

### Research Stream C: Go Functional Patterns
```yaml
agent: deep-researcher
sources:
  - fp-go library (IBM)
  - Go generics patterns
  - Error handling as monads
  - Option/Result patterns in Go
output: /research/go-functional.md
extract:
  - Result[T] / Either patterns
  - Option[T] patterns
  - Pipeline composition in Go
  - Generic constraints
  - Performance considerations
  - Avoiding reflection
quality_gate: ≥ 0.85
```

### Research Stream D: Context Engineering (2024-2025)
```yaml
agent: deep-researcher
sources:
  - Latest papers on context engineering
  - Anthropic's context caching
  - OpenAI's retrieval improvements
  - Late chunking, contextual embeddings
  - Memory architectures
output: /research/context-engineering.md
extract:
  - State-of-art beyond naive RAG
  - Multi-hop reasoning patterns
  - Memory integration approaches
  - Context window optimization
  - Production patterns from leaders
quality_gate: ≥ 0.85
```

### Research Stream E: Verification Architectures
```yaml
agent: deep-researcher
sources:
  - Citation verification systems
  - Factual grounding research
  - Hallucination detection
  - Self-consistency checking
output: /research/verification-architectures.md
extract:
  - Grounding score methodologies
  - Citation extraction techniques
  - Verification at retrieval vs generation
  - Confidence calibration
  - Production verification systems
quality_gate: ≥ 0.85
```

### Research Stream F: Go Project Patterns
```yaml
agent: deep-researcher
sources:
  - Go project layout standards
  - CLI patterns (cobra, urfave)
  - Testing patterns (rapid, testify)
  - Observability (OpenTelemetry)
output: /research/go-project-patterns.md
extract:
  - Standard project layout
  - cmd/ vs pkg/ vs internal/
  - Error handling patterns
  - Testing best practices
  - Build and release patterns
quality_gate: ≥ 0.85
```

### Research Gate
**Condition**: All 6 streams complete with quality ≥ 0.85
**On Success**: Proceed to Phase 2
**On Failure**: Retry failed streams with refined prompts

---

## 5.5. CONTEXT7 DOCUMENTATION PROTOCOL (CRITICAL)

**PRINCIPLE: Never Assume, Always Verify**

> We can take NOTHING for granted. Every piece of core code MUST have documentation grounding.
> Before writing ANY implementation code, spin up a subagent to verify library behavior.

### Why This Matters

1. **Libraries change** - What worked in v1.2 may not work in v2.0
2. **Documentation is ground truth** - Code comments lie, docs don't
3. **Comonadic extract** - We extract focused context from broader documentation
4. **Audit trail** - Each code decision has documented basis

### Pattern: Context7 Documentation Subagent

For EVERY core code file, BEFORE implementation:

```yaml
pattern: context7-doc-extract
trigger: Before writing any file in pkg/ or cmd/
agent: context7-doc-reviewer
output: /docs/context7/{library}-{component}.md
```

### Required Documentation Queries

| Component | Libraries to Document | Context7 Query |
|-----------|----------------------|----------------|
| `pkg/core/result.go` | fp-go | `/ctx7 fp-go either option result error-handling` |
| `pkg/llm/anthropic/` | anthropic-go-sdk | `/ctx7 anthropic-sdk-go messages streaming tools` |
| `pkg/llm/openai/` | openai-go | `/ctx7 openai-go chat completions embeddings` |
| `pkg/ingest/pdf/` | pdfcpu, unipdf | `/ctx7 go-pdf-library text-extraction parsing` |
| `pkg/vector/` | pgvector-go | `/ctx7 pgvector-go similarity-search embeddings` |
| `pkg/pipeline/` | Go generics | `/ctx7 go-generics constraints type-parameters` |
| `pkg/verify/` | none (custom) | Research verification patterns |
| `cmd/vera/` | cobra, urfave/cli | `/ctx7 cobra cli-patterns commands flags` |

### Documentation Template

Each Context7 extract follows this structure:

```markdown
# Context7 Documentation: {Library}

**Source**: Context7 MCP
**Query**: `{exact query used}`
**Extracted**: {timestamp}
**For Component**: {target file path}

## Comonadic Extract (Core Focus)

### API We Need
{Minimal API surface we will use}

### Type Signatures
{Exact function signatures from documentation}

### Usage Patterns
{Documented patterns with examples}

### Edge Cases
{Documented edge cases and handling}

### Version Notes
{Version-specific information}

## Full Context (For Reference)

{Complete Context7 response for audit}

---
**Grounding Score**: {How well does our implementation match docs}
**Verified By**: Subagent {agent-id}
```

### Execution Pattern

```bash
# Before implementing pkg/llm/anthropic/client.go

# 1. Spin up subagent
Task: context7-doc-reviewer
Prompt: |
  Query Context7 for anthropic-sdk-go documentation.
  Focus on: Messages API, streaming, tool use, error handling.
  Extract to: /docs/context7/anthropic-sdk-client.md
  Use comonadic extract pattern (focused core from broader context).

# 2. Subagent queries Context7
/ctx7 anthropic-sdk-go messages streaming tool-use error-handling

# 3. Subagent writes extract
Write: /docs/context7/anthropic-sdk-client.md

# 4. ONLY THEN proceed to implementation
# Implementation MUST reference the documentation
```

### Integration with Build Phase

During Phase 7 (Build), EVERY task follows:

```
┌─────────────────────────────────────────────────────────┐
│  Task: Implement {component}                            │
├─────────────────────────────────────────────────────────┤
│  Step 1: Check for existing doc                         │
│    IF /docs/context7/{library}.md exists                │
│      → Proceed to implementation                        │
│    ELSE                                                 │
│      → Spawn Context7 subagent FIRST                   │
│      → Wait for documentation                          │
│      → THEN proceed                                    │
│                                                         │
│  Step 2: Write implementation                           │
│    → Reference documentation in comments               │
│    → Cite doc section for non-obvious patterns         │
│                                                         │
│  Step 3: Verify alignment                               │
│    → Implementation matches documented patterns        │
│    → Grounding score updated                           │
└─────────────────────────────────────────────────────────┘
```

### Parallel Documentation Queries

When starting build phase, pre-populate documentation in parallel:

```
Phase 7 Start:
  [Context7(fp-go) || Context7(anthropic) || Context7(cobra) ||
   Context7(pgvector) || Context7(otel) || Context7(go-generics)]

  → All docs extracted before first line of code
```

### Documentation Maintenance

- **On Library Update**: Re-run Context7 query, diff with previous
- **On Implementation Change**: Update grounding score
- **On Review**: Verify implementation matches docs

### Quality Gate

**All code files MUST have**:
1. Corresponding `/docs/context7/{library}.md` for external dependencies
2. Inline comments citing documentation section for non-obvious patterns
3. Grounding score ≥ 0.9 (implementation matches documentation)

**Failure Mode**:
- Code without documentation grounding → REJECTED
- Implementation deviating from docs without justification → REJECTED

---

## 6. PHASE 2: SYNTHESIS

Sequential phase - depends on all research completing.

### Step 2.1: Research Synthesis
```yaml
agent: mercurio-synthesizer
inputs: All 6 research outputs
output: /specs/synthesis.md
task: |
  Create unified synthesis:
  1. Common patterns across research
  2. Conflicts and resolutions
  3. Recommendations for VERA
  4. Gap analysis (what's not covered)
  5. Risk assessment
quality_gate: MERCURIO three-plane ≥ 8.5/10
```

### Step 2.2: Architecture Design
```yaml
agent: api-architect
inputs: /specs/synthesis.md
output: /specs/architecture.md
structure:
  - Component diagram (ASCII + explanation)
  - Interface definitions (Go code)
  - Data flow diagrams
  - Type definitions (all core types)
  - Composition model (how components connect)
  - Error taxonomy (all error types)
  - Deployment models (MVP vs Production)
quality_gate: MARS ≥ 92%
```

---

## 7. PHASE 3: MVP SPECIFICATION

### MVP Scope Definition

**In Scope**:
1. Single LLM provider (Anthropic Claude)
2. Single document type (PDF)
3. Single verification policy (citation grounding)
4. CLI interface only
5. In-memory vector store (no persistence)
6. Single-user, single-session
7. Demonstrate: Pipeline composition, Verification as η, Grounding scores

**Out of Scope**:
- Multiple LLM providers
- REST API
- Persistent storage
- Streaming
- Multi-document types
- Custom verification policies
- Multi-user/multi-tenant

### MVP Spec Template

```yaml
output: /specs/MVP-SPEC.md
structure:
  overview:
    - Goal statement (1 sentence)
    - Success criteria (measurable)
    - Timeline (2 weeks)

  functional_requirements:
    format: |
      ### FR-X: [Name]
      **Given** [precondition]
      **When** [action]
      **Then** [outcome]
      **And** [additional outcome]

      **Acceptance Criteria**:
      - [ ] Criterion 1 (testable)
      - [ ] Criterion 2 (testable)

    requirements:
      - FR-1: Document Ingestion
      - FR-2: Query with Verification
      - FR-3: Citation Display
      - FR-4: Grounding Score
      - FR-5: Pipeline Composition

  non_functional_requirements:
    - NFR-1: Performance (latency, memory)
    - NFR-2: Reliability (no panics)
    - NFR-3: Code Quality (coverage, lint)
    - NFR-4: Documentation (every public func)

  technical_design:
    - Core types (Go code)
    - Directory structure
    - Package dependencies
    - External dependencies (with versions)

  milestones:
    - M1: Foundation (days 1-3)
    - M2: Ingestion (days 4-6)
    - M3: Query (days 7-9)
    - M4: Verification (days 10-12)
    - M5: Polish (days 13-14)

  verification_checklist:
    - All FR tests pass
    - All NFR metrics met
    - Law tests pass
    - Coverage ≥ 80%
    - No linter warnings
    - README demonstrates working example
```

---

## 8. PHASE 4: PRODUCTION SPECIFICATION

### Production Scope (MVP + Everything Else)

**Additional Scope**:
1. Multiple LLM providers (Anthropic, OpenAI, Ollama, local)
2. Multiple document types (PDF, Web, Markdown, DOCX, HTML)
3. Composable verification policies
4. REST API + CLI + Library
5. Persistent storage (PostgreSQL + pgvector)
6. Multi-user, multi-tenant
7. Streaming responses
8. Observability (OpenTelemetry)
9. Auth, rate limiting, quotas
10. Docker, Kubernetes deployment

### Production Spec Template

```yaml
output: /specs/PRODUCTION-SPEC.md
additional_sections:
  api_specification:
    - OpenAPI 3.0 spec
    - Authentication (API keys, OAuth)
    - Rate limiting
    - Error responses

  multi_provider:
    - Provider interface (same as MVP)
    - Provider registry
    - Fallback strategy
    - Cost tracking

  persistence:
    - Schema design
    - Migration strategy
    - Backup/restore

  multi_tenant:
    - Tenant isolation
    - Resource quotas
    - Usage tracking

  observability:
    - Metrics (Prometheus)
    - Tracing (OpenTelemetry)
    - Logging (structured)
    - Alerting

  security:
    - Auth (API keys, OAuth)
    - Encryption (at rest, in transit)
    - Audit logging
    - Vulnerability scanning

  deployment:
    - Docker image
    - Kubernetes manifests
    - Helm chart
    - Single binary option

quality_gates:
  - Test coverage ≥ 90%
  - Property-based law tests
  - Performance benchmarks
  - Security checklist
  - API documentation complete
```

---

## 9. PHASE 5: BUILD PLANNING

```yaml
agent: project-orchestrator
output: /specs/BUILD-PLAN.md
structure:
  task_breakdown:
    - Each task < 4 hours
    - Clear input/output
    - Verification criteria

  dependency_graph:
    - What blocks what
    - Critical path
    - Parallel opportunities

  risk_assessment:
    - Per-task risks
    - Mitigation strategies

  time_estimates:
    - Conservative (assume problems)
    - Buffer included
```

---

## 10. PHASE 6: HUMAN GATE

**STOP. DO NOT PROCEED WITHOUT HUMAN APPROVAL.**

### Review Package

Deliver to human reviewer:
1. `/specs/synthesis.md` - Research synthesis
2. `/specs/architecture.md` - System architecture
3. `/specs/MVP-SPEC.md` - MVP specification
4. `/specs/PRODUCTION-SPEC.md` - Production specification
5. `/specs/BUILD-PLAN.md` - Implementation plan

### Review Criteria

The human reviewer evaluates:

| Criterion | Question | Pass Condition |
|-----------|----------|----------------|
| Constitution | Do specs honor all 9 articles? | Yes, with evidence |
| Completeness | Any missing components? | No gaps identified |
| Precision | Any ambiguous requirements? | All MUST/testable |
| Consistency | Any contradictions? | Zero found |
| Feasibility | Is timeline realistic? | Reviewer agrees |
| Quality | Would you approve this? | Explicit sign-off |

### Approval Template

```markdown
# VERA Spec Approval

**Reviewer**: [Name]
**Date**: [Date]
**Version**: [Spec version]

## Decision: [ ] APPROVED / [ ] REJECTED / [ ] APPROVED WITH CHANGES

## Changes Required (if any):
1. [Change 1]
2. [Change 2]

## Sign-off:
- [ ] I have reviewed all specifications
- [ ] I believe they are complete and correct
- [ ] I approve proceeding to implementation

Signature: _____________
```

### On Approval
- Specs become immutable (versioned)
- Implementation begins
- Changes require new review cycle

### On Rejection
- Feedback incorporated
- Specs revised
- Re-submitted for review

---

## 11. PHASES 7-10: POST-APPROVAL

### Phase 7: Build
```
TaskDecomposer → [Builder₁ || Builder₂ || ... || Builderₙ]

Each Builder:
  1. Read spec for component
  2. Write tests first (TDD)
  3. Implement to pass tests
  4. Run linter
  5. Document
  6. Mark task complete
```

### Phase 8: Verify
```
[UnitTests || LawTests || IntegrationTests || E2ETests]

Law Tests (categorical correctness):
  - Composition associativity
  - Identity laws
  - Natural transformation properties

All tests MUST pass before ownership transfer.
```

### Phase 9: Documentation
```
Generate:
  - README.md (quick start)
  - docs/architecture.md (how it works)
  - docs/api.md (API reference)
  - docs/ownership.md (for handoff)
```

### Phase 10: Ownership Transfer
```
Criteria:
  - Any file understood in < 10 min
  - All tests pass
  - All docs complete
  - Reviewer can make a change independently
```

---

## 12. COMPOSITION SUMMARY

### Full Flow Diagram

```
CONSTITUTION
     │
     ▼
PHASE 1: OBSERVE (Comonad)
┌────────────────────────────────────────────────────┐
│  Research₁ || Research₂ || Research₃ ||           │
│  Research₄ || Research₅ || Research₆              │
└────────────────────────────────────────────────────┘
     │
     ▼ (all must complete)
PHASE 2: REASON (Functor)
┌────────────────────────────────────────────────────┐
│  Synthesize(research₁ × research₂ × ... × research₆)│
│  → Architecture                                     │
└────────────────────────────────────────────────────┘
     │
     ▼
PHASE 3-4: CREATE (Function)
┌────────────────────────────────────────────────────┐
│  [MVP-Spec || Production-Spec] → Build-Plan        │
└────────────────────────────────────────────────────┘
     │
     ▼
PHASE 5: VERIFY (Applicative) - HUMAN GATE
┌────────────────────────────────────────────────────┐
│  [MERCURIO || MARS || HumanReview]                 │
│  ═══════════════════════════════════════════════   │
│  ▲▲▲ STOP - WAIT FOR HUMAN APPROVAL ▲▲▲          │
└────────────────────────────────────────────────────┘
     │ (only after approval)
     ▼
PHASE 6: CREATE (Function)
┌────────────────────────────────────────────────────┐
│  [Builder₁ || Builder₂ || ... || Builderₙ]        │
└────────────────────────────────────────────────────┘
     │
     ▼
PHASE 7: VERIFY (Applicative)
┌────────────────────────────────────────────────────┐
│  [UnitTests || LawTests || IntegrationTests]       │
└────────────────────────────────────────────────────┘
     │
     ▼
PHASE 8: LEARN (Profunctor)
┌────────────────────────────────────────────────────┐
│  OwnershipTransfer: Code × Docs → HumanOwner       │
└────────────────────────────────────────────────────┘
```

### Operators Used

- `→` : Sequential (output of A feeds B)
- `||` : Parallel (independent, run simultaneously)
- `×` : Product (combine multiple outputs)
- `η` : Verification (natural transformation)
- `UNTIL` : Recursive (iterate until quality)

### Quality Gates Summary

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

## 13. EXECUTION

### To Execute This Meta-Prompt

```bash
# From main context
/meta @tier:L6 @quality:0.95 @mode:iterative \
  "Execute VERA vera-plan-meta-prompt.md"

# With recursive refinement
/rmp @max_iterations:5 @convergence:0.95 \
  "Refine each spec section until quality met"

# With task wrapping
/ralph @wrap:true @trace:full \
  "Track each phase with explicit boundaries"
```

### Expected Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Research | 1-2 hours | 6 research docs |
| Synthesis | 30 min | synthesis.md |
| Architecture | 1 hour | architecture.md |
| MVP Spec | 1 hour | MVP-SPEC.md |
| Prod Spec | 1 hour | PRODUCTION-SPEC.md |
| Build Plan | 30 min | BUILD-PLAN.md |
| **Human Gate** | **Variable** | **Approval** |
| Build (MVP) | 2 weeks | Working system |
| Verify | 2 days | Test results |
| Ownership | 1 day | Documentation |

### Success Criteria

This meta-prompt succeeds when:
1. All specs pass quality framework (5 dimensions)
2. Human reviewer approves without major changes
3. MVP demonstrates categorical verification power
4. Code is owned by humans (understandable, maintainable)

---

## 14. APPENDIX: GO CATEGORICAL PATTERNS

### Result[T] (Either Pattern)
```go
type Result[T any] struct {
    value T
    err   error
    ok    bool
}

func Ok[T any](value T) Result[T] {
    return Result[T]{value: value, ok: true}
}

func Err[T any](err error) Result[T] {
    return Result[T]{err: err, ok: false}
}

func (r Result[T]) Map[U any](f func(T) U) Result[U] {
    if !r.ok {
        return Err[U](r.err)
    }
    return Ok(f(r.value))
}

func (r Result[T]) FlatMap[U any](f func(T) Result[U]) Result[U] {
    if !r.ok {
        return Err[U](r.err)
    }
    return f(r.value)
}
```

### Pipeline Composition
```go
type Pipeline[In, Out any] interface {
    Run(ctx context.Context, input In) Result[Out]
    Then[Next any](next Pipeline[Out, Next]) Pipeline[In, Next]
}

// Associativity test (law)
func TestAssociativity(t *testing.T) {
    rapid.Check(t, func(t *rapid.T) {
        f, g, h := randomPipelines(t)
        input := rapid.Int().Draw(t, "input")

        left := f.Then(g).Then(h).Run(ctx, input)
        right := f.Then(g.Then(h)).Run(ctx, input)

        assert.Equal(t, left, right)
    })
}
```

### Verification as Middleware
```go
type Verifier func(Pipeline) Pipeline

func WithGroundingVerification(threshold float64) Verifier {
    return func(p Pipeline) Pipeline {
        return &verifiedPipeline{
            inner:     p,
            threshold: threshold,
        }
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

*This meta-prompt is the foundation. Review before execution.*
*Air-tight specs → Quality code → Human ownership.*
