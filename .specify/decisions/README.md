# Architecture Decision Records

This directory contains the Architecture Decision Records (ADRs) for VERA.

## What is an ADR?

An Architecture Decision Record (ADR) captures a significant architectural decision made along with its context and consequences. ADRs help future readers understand:

- **Why** a decision was made
- **What** alternatives were considered
- **When** the decision was made
- **Who** was involved in making it

## Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [ADR-0001](ADR-0001-use-go.md) | Use Go for Implementation | Proposed | 2025-12-29 |

## Status Definitions

- **Proposed**: Under consideration, open for discussion
- **Accepted**: Decision made, ready for implementation
- **Deprecated**: No longer applies, but kept for historical context
- **Superseded**: Replaced by a newer ADR (link to replacement)

## Required ADRs for VERA

Based on VERA's Constitution (9 Articles), the following ADRs are required:

| ADR | Decision | Constitution Article |
|-----|----------|---------------------|
| ADR-0001 | Language: Go | Art. IV (Human Ownership) |
| ADR-0002 | fp-go for Functional Patterns | Art. VI (Categorical Correctness) |
| ADR-0003 | Result[T] Error Handling | Art. VIII (Graceful Degradation) |
| ADR-0004 | LLMProvider Interface | Art. III (Provider Agnosticism) |
| ADR-0005 | Pipeline Composition | Art. II (Composition over Config) |
| ADR-0006 | Verification as Natural Transformation | Art. I (Verification First-Class) |
| ADR-0007 | No Mocks in MVP | Art. VII (Real Capability) |
| ADR-0008 | OpenTelemetry Observability | Art. IX (Observable by Default) |

## Creating New ADRs

1. Copy `template.md` to `ADR-XXXX-descriptive-title.md`
2. Fill in all sections
3. Submit PR for review
4. Update this index when merged

## File Naming Convention

```
ADR-XXXX-descriptive-title.md
```

- `XXXX`: Four-digit number, zero-padded
- `descriptive-title`: Lowercase, hyphen-separated

## References

- [MADR (Markdown ADR)](https://adr.github.io/madr/)
- [Original ADR Proposal by Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [VERA Research: spec-kit Analysis](../../research/spec-kit-analysis.md)
