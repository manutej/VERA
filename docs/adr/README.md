# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting key technical decisions in the VERA MVP project.

## What is an ADR?

An Architecture Decision Record (ADR) captures:
- **Context**: Why did we need to make this decision?
- **Decision**: What did we decide?
- **Alternatives**: What other options did we consider?
- **Consequences**: What are the positive/negative outcomes?
- **Compliance**: How does this align with VERA specifications?

ADRs provide a **historical record** of architectural decisions, ensuring that:
1. Future team members understand **why** decisions were made
2. Decisions can be **revisited** with full context
3. Compliance with specifications is **verifiable**
4. Trade-offs are **documented** and transparent

---

## Index

### M3: Ingestion

| ADR | Title | Status | Date | Key Decision |
|-----|-------|--------|------|--------------|
| [001](001-pdf-library-selection.md) | PDF Library Selection | ✅ ACCEPTED | 2024-12-31 | Use **ledongthuc/pdf** for text extraction (pdfcpu rejected - no text extraction support) |
| [002](002-markdown-parser-selection.md) | Markdown Parser Selection | ✅ ACCEPTED | 2024-12-31 | Use **goldmark** for CommonMark compliance (22% more memory efficient than blackfriday) |

### M4: Verification

| ADR | Title | Status | Date | Key Decision |
|-----|-------|--------|------|--------------|
| [003](003-chunking-algorithm.md) | Chunking Algorithm | ✅ ACCEPTED | 2024-12-31 | **Sentence-based chunking** with 3000 chars target, 20% overlap |
| [004](004-grounding-threshold.md) | Grounding Threshold | ✅ ACCEPTED | 2024-12-31 | **0.7 cosine similarity** threshold for grounding classification |

---

## Quick Reference

### ADR-001: PDF Library Selection
**Decision**: ledongthuc/pdf
**Why**: pdfcpu does NOT support text extraction (Issue #122)
**Impact**: Simple `GetPlainText()` API, zero dependencies
**Compliance**: Articles II, VII, VIII, IX

### ADR-002: Markdown Parser Selection
**Decision**: goldmark
**Why**: 100% CommonMark compliant, 22% more memory efficient
**Impact**: 0.28ms per document, clean AST walking
**Compliance**: Articles I, II, VII, VIII, IX

### ADR-003: Chunking Algorithm
**Decision**: Sentence-based, 3000 chars, 20% overlap
**Why**: Balance semantic coherence and retrieval accuracy
**Impact**: 4 chunks per 10K char document, 18% storage overhead
**Compliance**: Articles I, III, VIII, IX

### ADR-004: Grounding Threshold
**Decision**: 0.7 cosine similarity
**Why**: Optimal balance of precision (85%) and recall (80%)
**Impact**: Clear boundary between grounded/unsupported
**Compliance**: Articles I, III, VIII, IX

---

## ADR Statistics

**Total ADRs**: 4
**Status Breakdown**:
- ✅ ACCEPTED: 4
- 🚧 PROPOSED: 0
- ❌ REJECTED: 0
- 📦 SUPERSEDED: 0

**Coverage**:
- M1 (Foundation): 0 ADRs (core types, no major decisions)
- M2 (Providers): 0 ADRs (provider interfaces straightforward)
- M3 (Ingestion): 2 ADRs (PDF parser, Markdown parser)
- M4 (Verification): 2 ADRs (chunking algorithm, grounding threshold)

**Quality Score**: 0.95/1.0 average
- All ADRs include: Context, Decision, Alternatives, Consequences, Compliance
- All ADRs include: Code examples, benchmarks, references
- All ADRs include: Implementation status, tuning plans

---

## ADR Template

When creating a new ADR, use this template:

```markdown
# ADR-XXX: [Decision Title]

**Status**: 🚧 PROPOSED | ✅ ACCEPTED | ❌ REJECTED | 📦 SUPERSEDED

**Date**: YYYY-MM-DD

**Deciders**: [Who made this decision?]

**Technical Story**: [What milestone or feature does this support?]

---

## Context

[Why do we need to make this decision?]
[What are the requirements?]
[What are the constraints?]

---

## Decision

[What did we decide?]
[Why is this the best option?]

---

## Alternatives Considered

### Alternative 1: [Option Name]
**Rejected** - [Why?]

**Strengths**:
- [Pro 1]
- [Pro 2]

**Weaknesses**:
- ❌ [Con 1]
- ❌ [Con 2]

**Decision**: [Why eliminated?]

---

## Consequences

### Positive
1. ✅ [Benefit 1]
2. ✅ [Benefit 2]

### Negative
1. ⚠️ [Trade-off 1]
2. ⚠️ [Trade-off 2]

### Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| [Risk 1] | [Low/Med/High] | [Low/Med/High] | [How to mitigate?] |

---

## Compliance Verification

### Article X: [Requirement]
- ✅ **Requirement**: "[Quote from spec]"
- ✅ **Compliance**: [How does this decision comply?]
- ✅ **Evidence**: [Code, tests, or documentation proof]

---

## References

1. [External documentation]
2. [Research papers]
3. [VERA specification articles]

---

**ADR Quality Score**: X.XX/1.0
```

---

## Best Practices

1. **Write ADRs early**: Document decisions BEFORE implementation (when context is fresh)
2. **Keep them focused**: One decision per ADR (don't combine multiple topics)
3. **Include alternatives**: Show what you considered and why you rejected it
4. **Add code examples**: Make decisions concrete with implementation snippets
5. **Link to specs**: Verify compliance with VERA articles
6. **Update status**: Mark as SUPERSEDED if a new ADR replaces an old one

---

## When to Write an ADR?

Write an ADR when you make a decision that:
- ✅ **Is hard to reverse** (changing library, algorithm, architecture)
- ✅ **Has multiple viable options** (requires justification for choice)
- ✅ **Impacts team workflow** (affects how others will work)
- ✅ **Has non-obvious trade-offs** (needs documentation for future reference)

Don't write an ADR for:
- ❌ Trivial decisions (naming variables, code style)
- ❌ Obvious choices (using standard library over custom implementation)
- ❌ Experimental prototypes (unless they become production)

---

## Revision History

| Date | Change | Author |
|------|--------|--------|
| 2024-12-31 | Created ADRs 001-004 (M3, M4 decisions) | VERA Core Team |

---

**Status**: Phase 2 of refactoring plan complete ✅
**Next**: Phase 3 (CI/CD and benchmarks)
