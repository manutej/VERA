# VERA Specification Gap Analysis

**Date**: 2025-12-29
**Analyzer**: MERCURIO (Mixture of Experts)
**Method**: Three-Plane Convergence Analysis
**Input**: MVP-SPEC.md v1.0.0, PRODUCTION-SPEC.md v1.0.0
**Stakeholder Concerns**: Multi-format, Multi-document, Wow Factor, Performance, Evaluation, ADR Traceability, Use Cases

---

## Executive Summary

**Aggregate MERCURIO Score**: 7.2/10 (BELOW TARGET)
**Target**: ≥ 9.0/10
**Status**: CRITICAL GAPS IDENTIFIED - SPECS REQUIRE REVISION

### Critical Findings

1. **Multi-format Support**: MISSING (mentioned as "out of scope" but stakeholder requires)
2. **Multi-document Scope**: VAGUE (no explicit 10-file limit specification)
3. **Wow Factor**: INSUFFICIENT (categorical correctness present, but differentiation unclear)
4. **Performance Methodology**: VAGUE (targets present, extraction methodology missing)
5. **Evaluation Framework**: ABSENT (no systematic evaluation beyond quality gates)
6. **ADR Traceability**: WEAK (ADRs listed but not integrated into requirements)
7. **Use Cases**: LACKING (brief examples, no compelling narratives)

---

## Three-Plane Analysis

### Mental Plane: Conceptual Integrity (6.5/10)

**Core Question**: Does the spec capture the "wow factor"? Is differentiation clear?

#### Strengths
- ✅ **Categorical foundation is sound**: Pipeline composition, natural transformations, typed verification
- ✅ **Type safety rigorous**: Result[T], Verification[T], invalid states unrepresentable
- ✅ **Provider agnosticism well-specified**: LLMProvider interface cleanly separates concerns

#### Critical Gaps

**GAP-M1: Differentiation from RAG is Buried** (CRITICAL)
- **Severity**: CRITICAL
- **Location**: MVP-SPEC.md Section 1.1
- **Issue**: The phrase "TRANSCENDS traditional RAG" appears once in overview but is never unpacked
- **Stakeholder Impact**: Readers won't understand the "wow factor" without deep categorical knowledge
- **Evidence**:
  - MVP Section 1.1 states goal but doesn't contrast with RAG
  - No explicit comparison table: "Traditional RAG vs VERA"
  - Categorical advantages buried in technical implementation details

**Recommendation**:
```markdown
Add Section 1.3: "Why VERA Transcends RAG"

| Feature | Traditional RAG | VERA |
|---------|----------------|------|
| Verification | Post-hoc (if at all) | Compositional (η at ANY point) |
| Grounding | Binary (cited/not cited) | Continuous score + NLI |
| Iterative Retrieval | Manual refinement | UNTIL operator (automatic) |
| Type Safety | Stringly-typed | Result[T], Verification[T] |
| Composability | Linear pipeline | Categorical operators (→, ||, IF, UNTIL, η) |
| Error Handling | Exceptions | Result monad (never panic) |
| Observability | Added after | Built-in (traces, metrics, checkpoints) |
| Provider Lock-in | Often vendor-specific | Agnostic interface |
```

**GAP-M2: Multi-Format Support Not Specified** (CRITICAL)
- **Severity**: CRITICAL
- **Location**: MVP-SPEC.md Section 1.4, Line 49
- **Issue**: "Document Type: PDF only" contradicts stakeholder requirement for Markdown + PDF
- **Stakeholder Impact**: Cannot evaluate multi-format proposals without spec
- **Evidence**:
  - MVP Line 49: "PDF only"
  - MVP Line 63: "Out of Scope: Multi-document type support (Markdown, HTML, DOCX)"
  - Production spec extends to multiple formats (Section 3) but no extraction methodology specified

**Recommendation**:
```markdown
MVP Section 2.9: Document Format Support

**In Scope (MVP)**:
- PDF (primary)
- Markdown (secondary)

**Interface**:
```go
type DocumentParser interface {
    Parse(ctx context.Context, data []byte) Result[ParsedDocument]
    SupportedFormats() []string
}

type ParsedDocument struct {
    Text       string
    Chunks     []TextChunk
    Metadata   DocumentMetadata
    Format     DocumentFormat
}

type DocumentFormat string
const (
    FormatPDF      DocumentFormat = "pdf"
    FormatMarkdown DocumentFormat = "markdown"
)
```

**FR-009: Multi-Format Document Parsing**
- AC-009.1: PDF files parsed with page numbers
- AC-009.2: Markdown files parsed with heading structure
- AC-009.3: Parser selection based on file extension
- AC-009.4: Unsupported formats return ErrUnsupportedFormat
```

**GAP-M3: Multi-Document Scope Vague** (HIGH)
- **Severity**: HIGH
- **Location**: MVP-SPEC.md Section 1.4, FR-002
- **Issue**: "ingested documents" (plural) mentioned but no explicit limit (e.g., 10 files)
- **Stakeholder Impact**: Cannot evaluate if system handles realistic multi-document scenarios
- **Evidence**:
  - FR-001 handles single document ingestion
  - FR-002 queries "ingested documents" (Line 99) but no multi-doc test scenario
  - No acceptance criteria for cross-document verification

**Recommendation**:
```markdown
FR-002 Enhancement: Multi-Document Query

**Given** 10 ingested documents across 2 formats (8 PDF + 2 Markdown)
**When** user queries "What evidence supports claim X?"
**Then** system MUST:
1. Retrieve chunks from ALL relevant documents
2. Calculate coverage across document set
3. Provide citations with document names
4. Aggregate grounding score across sources

**Acceptance Criteria**:
- AC-002.6: Queries spanning 10 documents complete within 10 seconds
- AC-002.7: Citations include document name + format + page/section
- AC-002.8: Cross-document grounding weighted by document relevance
```

**GAP-M4: Evaluation Framework Missing** (HIGH)
- **Severity**: HIGH
- **Location**: Entire spec
- **Issue**: Quality gates present (Section 10) but no systematic evaluation framework
- **Stakeholder Impact**: Cannot prove VERA improves over baselines
- **Evidence**:
  - Quality gates are pass/fail (coverage, law tests)
  - No accuracy benchmarks vs ground truth
  - No comparison to baseline RAG systems

**Recommendation**:
```markdown
Add Section 11: Evaluation Framework

**Benchmark Datasets**:
1. NQ-Open (Natural Questions)
2. HotpotQA (multi-hop reasoning)
3. Custom legal corpus (contracts + amendments)

**Metrics**:
| Metric | Measurement | Target |
|--------|-------------|--------|
| Answer Accuracy | EM, F1 vs ground truth | >= 75% |
| Grounding Precision | % verified claims actually grounded | >= 90% |
| Grounding Recall | % grounded claims verified | >= 85% |
| Citation Quality | Human judgment (1-5 scale) | >= 4.0 |
| False Positive Rate | % ungrounded flagged as grounded | <= 5% |
| Latency P99 | Query response time | < 5s |

**Baseline Comparisons**:
- Vanilla RAG (no verification)
- RAGChecker (post-hoc verification)
- FactScore (atomic fact grounding)
- VERA (compositional verification)
```

---

### Physical Plane: Practical Feasibility (7.0/10)

**Core Question**: Can we actually build this? What does execution require?

#### Strengths
- ✅ **Tech stack concrete**: Go, fp-go, Anthropic SDK, OpenTelemetry
- ✅ **Timeline realistic**: 2 weeks MVP, 8 weeks production
- ✅ **Resource breakdown present**: Milestones, task dependencies

#### Critical Gaps

**GAP-P1: Extraction Methodology for Markdown Missing** (HIGH)
- **Severity**: HIGH
- **Location**: Implementation details
- **Issue**: PDF parsing library specified (pdfcpu, ADR-0017) but Markdown unspecified
- **Stakeholder Impact**: Cannot assess feasibility without knowing parsing approach
- **Evidence**:
  - ADR-0017 proposes pdfcpu for PDF
  - Markdown mentioned as "out of scope" in MVP but required by stakeholder
  - No discussion of heading extraction, code block handling, etc.

**Recommendation**:
```markdown
ADR-0018: Markdown Parsing Strategy

**Decision**: Use goldmark (pure Go, CommonMark compliant)

**Implementation**:
```go
// pkg/ingest/markdown/parser.go

import "github.com/yuin/goldmark"

type MarkdownParser struct {
    parser goldmark.Markdown
}

func (p *MarkdownParser) Parse(ctx context.Context, data []byte) Result[ParsedDocument] {
    // Extract headings as section markers
    // Preserve code blocks as single chunks
    // Track heading hierarchy for metadata
}
```

**Chunking Strategy**:
- Respect heading boundaries (don't split mid-section)
- Code blocks treated as atomic units
- Links preserved with context
```

**GAP-P2: Performance Extraction Methodology Unclear** (MEDIUM)
- **Severity**: MEDIUM
- **Location**: MVP-SPEC.md Section 1.2, Line 26
- **Issue**: "P99 query response time < 5 seconds" specified but no breakdown of WHERE time is spent
- **Stakeholder Impact**: Cannot optimize without understanding bottlenecks
- **Evidence**:
  - Line 26: Target latency present
  - No breakdown: embedding (Xms) + retrieval (Yms) + LLM (Zms) + NLI (Wms)
  - No guidance on which component to optimize first

**Recommendation**:
```markdown
Section 1.2 Enhancement: Latency Budget Breakdown

| Component | P99 Target | Optimization Strategy |
|-----------|------------|----------------------|
| Document Embedding | 500ms | Batch processing, cache |
| Vector Search | 50ms | HNSW index, top-k limit |
| BM25 Retrieval | 30ms | Inverted index |
| Query Embedding | 100ms | Cache query embeddings |
| LLM Generation | 3000ms | Provider with streaming |
| NLI Verification | 1000ms | Batch verification, parallel |
| Total Budget | 4680ms | < 5s with 320ms buffer |

**If Exceeded**:
- First: Cache embeddings (eliminates repeat costs)
- Second: Reduce top-k (fewer chunks to verify)
- Third: Async NLI (stream while verifying)
```

**GAP-P3: ADR Traceability Weak** (MEDIUM)
- **Severity**: MEDIUM
- **Location**: Throughout spec
- **Issue**: ADRs referenced but not linked to specific requirements
- **Stakeholder Impact**: Hard to trace design decisions to requirements
- **Evidence**:
  - FR-001 references "ADR-001" but ADR not included in spec
  - Appendix A lists ADRs but no content
  - No decision context captured

**Recommendation**:
```markdown
Appendix A Enhancement: Include ADR Stubs

**ADR-0001: Use Go as Implementation Language**

**Status**: Proposed
**Date**: 2025-12-29
**Context**: Need type-safe, performant language with functional patterns
**Decision**: Go 1.23 with fp-go library
**Consequences**:
- ✅ Generics enable Result[T], Pipeline[In, Out]
- ✅ Goroutines enable parallel operators
- ✅ Ecosystem mature (Cobra, pgx, OTEL)
- ❌ No native HKT (use fp-go workarounds)
```

---

### Spiritual Plane: Ethical & Value Alignment (8.0/10)

**Core Question**: Is this right? What matters? Unintended consequences?

#### Strengths
- ✅ **Values-aligned**: Verification prevents hallucination harms
- ✅ **Trustworthiness central**: Grounding scores, citations, transparency
- ✅ **Graceful degradation**: Low scores return warnings, not errors

#### Critical Gaps

**GAP-S1: Use Cases Lack Compelling Narratives** (HIGH)
- **Severity**: HIGH
- **Location**: MVP-SPEC.md (minimal), PRODUCTION-SPEC.md Section 2 (brief)
- **Issue**: Examples generic ("query documents"), not emotionally resonant
- **Stakeholder Impact**: Cannot see VALUE without concrete scenarios
- **Evidence**:
  - Production spec has 5 decision types (Line 1073-1122) but shallow
  - No "day in the life" narrative showing impact
  - No stakeholder personas (lawyer, researcher, compliance officer)

**Recommendation**:
```markdown
Add Section 1.5: Stakeholder Scenarios

**Scenario 1: Legal Due Diligence**

**Persona**: Sarah, corporate lawyer reviewing acquisition contracts

**Problem**: Must verify 10 contracts (2000 pages) for liability clauses. Traditional search misses nuances. Associates take 40 hours.

**VERA Workflow**:
1. Ingest 10 PDFs (contracts + amendments)
2. Query: "What liability limitations exist for environmental damages?"
3. VERA retrieves 12 relevant clauses across 7 documents
4. Grounding score: 0.92 (GROUNDED)
5. Citations show exact page numbers + contract names
6. Sarah reviews in 2 hours (20x faster)
7. **Outcome**: Identifies hidden liability in Amendment #3 that associates missed

**Value**: Prevents $2M exposure, saves 38 hours

---

**Scenario 2: Regulatory Compliance Audit**

**Persona**: Miguel, compliance officer ensuring SOC 2 compliance

**Problem**: Must prove controls exist across 50 policy documents. Auditor requires evidence trail.

**VERA Workflow**:
1. Ingest 50 policy Markdown files
2. Query: "Which policies address data encryption at rest?"
3. VERA finds 8 relevant policies
4. Grounding score: 0.88 (GROUNDED)
5. Citations provide exact policy sections
6. Exports audit-ready report with verifiable citations
7. **Outcome**: Passes audit, demonstrates due diligence

**Value**: Avoids audit failure, demonstrates control rigor
```

**GAP-S2: Unintended Consequences Underspecified** (MEDIUM)
- **Severity**: MEDIUM
- **Location**: Error handling (Section 3.5) present but ethical risks absent
- **Issue**: What if system confidently returns wrong answer with high grounding score?
- **Stakeholder Impact**: Trust erosion if false positives occur
- **Evidence**:
  - Error handling technical (ErrInvalidPDF, etc.)
  - No discussion of false confidence
  - No human escalation triggers beyond grounding < 0.70

**Recommendation**:
```markdown
Add Section 3.6: Ethical Safeguards

**False Confidence Mitigation**:

**Risk**: High grounding score (0.90) but answer factually wrong
**Cause**: NLI model entailment error, or source documents contain misinformation

**Safeguards**:
1. **Contradiction Detection**: If NLI finds BOTH entailment AND contradiction, flag for review
2. **Uncertainty Quantification**: Report confidence intervals with grounding scores
3. **Human Escalation**: Grounding > 0.85 but contradiction score > 0.3 → escalate
4. **Audit Trail**: Log all verification decisions for post-hoc review
5. **Feedback Loop**: Collect human corrections to retrain NLI thresholds

**Example**:
```
Response: "The contract specifies net-60 payment terms."
Grounding Score: 0.92
Contradiction Score: 0.35 (one source says net-30)
Status: ESCALATED TO HUMAN REVIEW
Reason: High grounding but conflicting evidence detected
```
```

**GAP-S3: Long-Term Impact on Knowledge Work Unexplored** (LOW)
- **Severity**: LOW
- **Location**: Philosophical implications
- **Issue**: If VERA works, what happens to paralegals, research assistants?
- **Stakeholder Impact**: Adoption resistance without addressing workforce concerns
- **Evidence**:
  - Spec focuses on efficiency gains
  - No discussion of complementarity vs replacement
  - No guidance on human-AI collaboration patterns

**Recommendation**:
```markdown
Add Appendix D: Human-AI Collaboration Patterns

**Philosophy**: VERA augments, not replaces, human expertise

**Collaboration Models**:

1. **Human-in-the-Loop**: VERA retrieves, human validates
   - Use case: High-stakes legal decisions
   - Human role: Final judgment, contextual interpretation

2. **AI-Assisted Research**: VERA finds needles in haystacks
   - Use case: Literature review across 1000+ papers
   - Human role: Synthesis, novel connections

3. **Automated Triage**: VERA filters low-confidence queries
   - Use case: Customer support knowledge base
   - Human role: Handle escalations, update knowledge base

**Workforce Transition**:
- Junior roles shift from manual search to verification + synthesis
- Senior roles gain bandwidth for high-value strategic work
- New role: "Verification Engineer" - tuning policies, reviewing edge cases
```

---

## Gap Severity Summary

| Gap ID | Plane | Severity | Title | Spec Impact |
|--------|-------|----------|-------|-------------|
| GAP-M1 | Mental | CRITICAL | Differentiation from RAG buried | Readers miss "wow factor" |
| GAP-M2 | Mental | CRITICAL | Multi-format support not specified | Cannot build Markdown parser |
| GAP-M3 | Mental | HIGH | Multi-document scope vague | No 10-file test scenario |
| GAP-M4 | Mental | HIGH | Evaluation framework missing | Cannot prove superiority |
| GAP-P1 | Physical | HIGH | Markdown extraction methodology missing | Feasibility unclear |
| GAP-P2 | Physical | MEDIUM | Performance extraction unclear | Cannot optimize |
| GAP-P3 | Physical | MEDIUM | ADR traceability weak | Hard to trace decisions |
| GAP-S1 | Spiritual | HIGH | Use cases lack compelling narratives | Value unclear |
| GAP-S2 | Spiritual | MEDIUM | Unintended consequences underspecified | Trust risks |
| GAP-S3 | Spiritual | LOW | Long-term impact unexplored | Adoption resistance |

---

## Plane Convergence Analysis

### Where Planes Align (STRONG)
- ✅ **Type Safety**: Mental (categorical rigor) + Physical (Go generics) + Spiritual (prevent runtime failures)
- ✅ **Provider Agnosticism**: Mental (composition) + Physical (interface) + Spiritual (user choice)
- ✅ **Graceful Degradation**: Mental (Result monad) + Physical (no panics) + Spiritual (transparency)

### Where Planes Conflict (CREATIVE TENSION)

**Tension 1: Performance vs Accuracy**
- **Mental**: Multi-hop retrieval (UNTIL) improves coverage
- **Physical**: Each hop adds 500ms latency
- **Resolution**: Configurable max hops (default 3) with early stopping

**Tension 2: Multi-Format Support vs Simplicity**
- **Mental**: Multi-format demonstrates generality
- **Physical**: Each parser adds complexity
- **Spiritual**: Users need Markdown for internal docs
- **Resolution**: PDF + Markdown in MVP (proven libraries), others in production

**Tension 3: Evaluation Rigor vs Time-to-Market**
- **Mental**: Comprehensive benchmarks prove superiority
- **Physical**: Benchmarking adds 2 weeks to timeline
- **Spiritual**: Users trust systems with proven accuracy
- **Resolution**: Lightweight eval in MVP (NQ-Open subset), full suite in production

---

## Revised Specification Recommendations

### MVP-SPEC.md Additions

1. **Section 1.3**: "Why VERA Transcends RAG" (comparison table)
2. **Section 1.5**: Stakeholder Scenarios (2-3 compelling narratives)
3. **Section 2.9**: Multi-Format Document Parsing (FR-009)
4. **Section 3.6**: Ethical Safeguards (false confidence mitigation)
5. **Section 11**: Lightweight Evaluation Framework (NQ-Open subset)
6. **Appendix A**: ADR Stubs with context + decisions
7. **Appendix D**: Human-AI Collaboration Patterns

**Estimated Addition**: +1200 lines, +3 days specification work

### PRODUCTION-SPEC.md Enhancements

1. **Section 1.5**: Expand stakeholder scenarios to 5 detailed narratives
2. **Section 3**: Full extraction methodology for HTML, DOCX, etc.
3. **Section 11**: Comprehensive evaluation suite (3 benchmarks, 6 metrics)
4. **Section 15**: Long-term impact analysis + workforce transition guidance

**Estimated Addition**: +800 lines, +2 days specification work

---

## Aggregate Scores by Plane

| Plane | Current Score | Target | Gap | Priority Fixes |
|-------|--------------|--------|-----|----------------|
| **Mental** | 6.5/10 | ≥ 9.0 | -2.5 | GAP-M1 (differentiation), GAP-M2 (multi-format), GAP-M4 (eval) |
| **Physical** | 7.0/10 | ≥ 9.0 | -2.0 | GAP-P1 (Markdown extraction), GAP-P2 (performance breakdown) |
| **Spiritual** | 8.0/10 | ≥ 9.0 | -1.0 | GAP-S1 (compelling narratives), GAP-S2 (safeguards) |

**Weighted Aggregate**: (6.5 + 7.0 + 8.0) / 3 = **7.17/10** (BELOW TARGET)

---

## Confidence Scores per Expert Perspective

| Expert | Confidence | Rationale |
|--------|-----------|-----------|
| **Mental Plane** | 75% | Categorical foundation solid, but differentiation + evaluation gaps reduce confidence in specification completeness |
| **Physical Plane** | 80% | Tech stack proven, timeline realistic, but missing extraction details and performance methodology reduce implementation confidence |
| **Spiritual Plane** | 85% | Values clear, ethical intent present, but use cases and safeguards underspecified reduce stakeholder buy-in confidence |

**Aggregate Confidence**: 80% (MODERATE)
**Interpretation**: Specs are 80% ready for implementation, require 20% additional work to reach 95%+ confidence

---

## Recommended Action Plan

### Immediate (Before Implementation)

1. **Add GAP-M1**: Differentiation table in Section 1.3 (1 hour)
2. **Add GAP-M2**: Multi-format FR-009 + ADR-0018 (4 hours)
3. **Add GAP-S1**: 2 compelling stakeholder scenarios (3 hours)
4. **Add GAP-P2**: Latency budget breakdown (2 hours)

**Total**: 10 hours, raises aggregate score to 8.2/10

### Pre-Production (Before Production Spec Finalization)

5. **Add GAP-M4**: Evaluation framework (8 hours)
6. **Add GAP-P1**: Markdown extraction ADR (4 hours)
7. **Add GAP-S2**: Ethical safeguards section (4 hours)
8. **Enhance ADR traceability**: Expand Appendix A (6 hours)

**Total**: 22 hours, raises aggregate score to 9.1/10

### Post-MVP (Production Planning)

9. **Add GAP-S3**: Long-term impact analysis (4 hours)
10. **Expand scenarios**: 5 detailed narratives (6 hours)
11. **Full eval suite**: 3 benchmarks, comparison baselines (12 hours)

**Total**: 22 hours, raises aggregate score to 9.5/10

---

## Final Recommendations

### For Human Review Gate

**Decision Criteria**:
- ✅ **Proceed to Implementation**: If GAPs M1, M2, S1, P2 addressed (aggregate 8.2/10)
- ⚠️ **Conditional Approval**: If only M1, S1 addressed (aggregate 7.8/10), with M2 committed for Week 2
- ❌ **Reject**: If multi-format (M2) not addressed (stakeholder requirement unmet)

**Recommended Path**: Address M1, M2, S1, P2 (10 hours) → Achieve 8.2/10 → Proceed to MVP implementation

---

## Appendix: Stakeholder Concern Mapping

| Stakeholder Concern | Gap ID | Severity | Status |
|---------------------|--------|----------|--------|
| Multi-format support (Markdown + PDF) | GAP-M2 | CRITICAL | MISSING |
| Multi-document scope (10 files) | GAP-M3 | HIGH | VAGUE |
| Wow factor | GAP-M1 | CRITICAL | INSUFFICIENT |
| Performance clarity | GAP-P2 | MEDIUM | VAGUE |
| Evaluation framework | GAP-M4 | HIGH | ABSENT |
| ADR traceability | GAP-P3 | MEDIUM | WEAK |
| Concrete use cases | GAP-S1 | HIGH | LACKING |

**Stakeholder Satisfaction**: 2/7 concerns fully addressed (Type Safety, Provider Agnosticism)
**Recommended**: Address 5 remaining concerns before Human Gate

---

**Status**: ANALYSIS COMPLETE
**Next Action**: Prioritize immediate fixes (M1, M2, S1, P2) for 8.2/10 aggregate score
**Human Gate Decision**: CONDITIONAL APPROVAL pending 10-hour specification revision

---

*Generated by MERCURIO (Three-Plane Convergence Analysis)*
*Date: 2025-12-29*
*Input: MVP-SPEC.md (1190 lines), PRODUCTION-SPEC.md (2707 lines)*
*Method: Mental (conceptual) + Physical (practical) + Spiritual (ethical) analysis*
