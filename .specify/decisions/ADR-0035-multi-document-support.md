# ADR-0035: Multi-Document Support in MVP

**Status**: Accepted
**Date**: 2025-12-29
**Deciders**: Product Owner, Engineering Team
**Supersedes**: N/A
**Related**: ADR-001 (Go), ADR-003 (Result Type)

---

## Context and Problem Statement

The original MVP specification supported **single PDF file** ingestion and querying. Stakeholder feedback identified this as **insufficient for real-world use cases**:

1. **Legal domain**: Contracts involve multiple documents (MSA, SOW, amendments, policies)
2. **Research synthesis**: Literature reviews require 10+ papers
3. **Developer documentation**: API docs span multiple Markdown files
4. **Practicality**: Single-file verification is trivial, not impressive

**Problem**: How do we extend VERA to handle 10 documents while maintaining:
- Categorical verification correctness
- Sub-5s latency (MVP target)
- Clear source attribution
- Implementation within 2.5-week timeline

---

## Decision Drivers

* **Real-world applicability**: Single-file queries don't reflect actual usage patterns
* **Differentiation**: Multi-document cross-verification is VERA's strength vs simple RAG
* **Research backing**: `multi-doc-rag.md` shows proven patterns (chromem-go, hybrid search, RRF)
* **Performance feasibility**: Research shows <1s retrieval for 10 docs with proper architecture
* **Timeline constraint**: Must remain implementable in MVP (2.5 weeks)

---

## Considered Options

### Option 1: Keep Single-Document Scope
**Pros**:
- Simpler implementation
- Faster to ship
- Lower risk

**Cons**:
- ❌ Not realistic for target domains (legal, research)
- ❌ Fails "wow factor" test
- ❌ Doesn't demonstrate cross-document verification (VERA's unique value)
- ❌ Stakeholder feedback explicitly rejects this

### Option 2: Support 10 Documents (SELECTED)
**Pros**:
- ✅ Addresses legal domain (typical contract + amendments + policies = 5-10 docs)
- ✅ Enables cross-document citation and contradiction detection
- ✅ Research-backed patterns exist (chromem-go for < 100 docs)
- ✅ Feasible latency (<2s with hybrid search + parallel retrieval)
- ✅ Clear scope boundary (not unlimited scale)

**Cons**:
- Increased complexity in retrieval (multi-doc search)
- Citation attribution needs document metadata
- Cross-document grounding requires refinement
- ~3 days additional implementation time

### Option 3: Unlimited Document Scale
**Pros**:
- Maximum flexibility
- Production-ready architecture

**Cons**:
- ❌ Out of MVP scope (requires pgvector, HNSW indexing)
- ❌ Performance optimization dominates timeline
- ❌ Premature scaling (YAGNI violation)

---

## Decision Outcome

**Chosen option**: **Option 2 - Support 10 Documents**

### Rationale

From `multi-doc-rag.md` research:
- **chromem-go** handles 100+ documents in-memory (zero dependencies)
- **Hybrid search** (BM25 + vector + RRF) provides 15-20% precision improvement
- **Parallel retrieval** achieves <500ms for 10 documents
- **Citation injection** at ingestion enables accurate attribution

**Implementation Strategy**:
```go
type DocumentStore struct {
    docs      map[string]*Document  // doc_id -> full document
    vectorDB  *chromemgo.DB         // embedded vector store
    bm25Index *BM25Index            // keyword index
    maxDocs   int                   // MVP: 10
}
```

**Multi-Document Query Flow**:
1. User ingests 10 documents (PDF + Markdown mix)
2. Each chunk tagged with `document_id`, `document_name`, `page/line`
3. Query triggers hybrid search across ALL documents
4. RRF fusion combines BM25 + vector results
5. eta_1 verification checks cross-document coverage
6. Response generation with multi-source attribution
7. eta_3 grounding verification with document-specific citations

---

## Consequences

### Positive

1. **Real-world applicability**: VERA can handle actual legal/research workflows
2. **Differentiation**: Cross-document verification is unique selling point
3. **Wow factor**: "Detected contradiction between Contract.pdf page 3 and Policy.md line 45"
4. **Research-backed**: Proven patterns from production systems (Haystack, LlamaIndex)
5. **Performance**: <2s latency achievable with chromem-go + parallel search

### Negative

1. **Complexity**: Multi-doc retrieval more complex than single-doc
2. **Timeline**: +2-3 days implementation (within project buffer)
3. **Testing**: Need multi-doc test fixtures (legal contract sets, paper collections)
4. **Attribution**: Citations must include document name + page/line (more metadata)

### Neutral

1. **Scope boundary**: 10 docs is arbitrary but justified by research domain analysis
2. **Storage**: In-memory still viable for 10 docs (avg 100 pages * 10 = 1000 chunks)
3. **Future scaling**: Architecture supports upgrade to pgvector for 100+ docs (production)

---

## Implementation Notes

### MVP Requirements (FR-009)

**Given** 10 ingested documents (mixed PDF + Markdown)
**When** user queries `vera query "Does payment schedule comply with policy?"`
**Then** system:
1. Searches across ALL 10 documents (hybrid BM25 + vector)
2. Applies RRF fusion (k=60, as per research best practice)
3. Returns top-k chunks with document attribution
4. Verifies coverage across relevant documents (eta_1)
5. Generates response with multi-source citations
6. Verifies grounding score (eta_3) with per-document breakdown

**Acceptance Criteria**:
- AC-009.1: Supports 1-10 documents per session
- AC-009.2: Query across 10 docs completes in <5s (P99)
- AC-009.3: Citations include `{doc_name, page/line, score}`
- AC-009.4: Response synthesizes information across documents
- AC-009.5: Contradictions flagged when detected across sources

### Technical Components

**Document Metadata**:
```go
type DocumentMetadata struct {
    ID          string         // UUID
    Name        string         // "Contract_MSA.pdf"
    Format      DocumentFormat // pdf, markdown
    PageCount   int            // PDF: page count, MD: line count
    UploadedAt  time.Time
    Checksum    string         // SHA-256 for deduplication
}
```

**Citation Format**:
```go
type Citation struct {
    ClaimText    string  // "Payment terms specify Net-120"
    DocumentID   string  // UUID
    DocumentName string  // "SOW_2024.pdf"
    PageNumber   *int    // 3 (nil for Markdown)
    LineNumber   *int    // 45 (nil for PDF)
    SourceText   string  // "Payment: Net-120 days from invoice"
    Score        float64 // 0.92 (NLI entailment score)
}
```

**Retrieval Strategy**:
- **BM25**: Keyword search across all documents (Okapi BM25, k1=1.5, b=0.75)
- **Vector**: Dense retrieval (OpenAI text-embedding-3-small, 1536 dims)
- **RRF Fusion**: Reciprocal Rank Fusion with k=60 (research best practice)
- **Top-k**: Retrieve 50 chunks, rerank to 10 (cross-encoder optional for production)

**Performance Targets**:
- Ingestion: 10 documents in <60s (P99)
- Query: <5s end-to-end (MVP), <2s target (stretch)
- Retrieval: <500ms for 10 docs with parallel search
- Verification: <1s for grounding score calculation

---

## Validation

### From Research (`multi-doc-rag.md`)

**Quote**: "chromem-go handles 100+ documents in-memory with zero dependencies, achieving <1s retrieval latency with HNSW indexing."

**Benchmark**: Haystack two-stage retrieval (BM25 + vector → rerank) achieves:
- Precision@10: 0.85
- Recall@10: 0.78
- Latency: 600ms (10 docs)

**Citation**: LlamaIndex document-per-index pattern works well for <100 documents with proper metadata extraction.

### From Evaluation Research (`evaluation-frameworks.md`)

**Multi-Document Grounding**:
- Cross-document faithfulness: F_cross = min(F_doc_i) for i in cited_docs
- Citation graph analysis: Detect contradictions across sources
- Information integration: Measure synthesis quality across documents

**Metrics for VERA**:
- **Cross-document precision**: TP_citations / (TP_citations + FP_citations_wrong_doc)
- **Coverage**: Fraction of relevant documents cited
- **Contradiction detection**: Automatically flag conflicting claims

---

## Compliance with Constitution

| Article | Requirement | Compliance |
|---------|-------------|------------|
| I. Verification as First-Class | Every claim verifiable | ✅ Citations include document attribution |
| III. Provider Agnosticism | No LLM-specific logic | ✅ Works with any embedding provider |
| IV. Human Ownership | <10 min file understanding | ✅ DocumentStore is ~200 lines |
| V. Type Safety | Invalid states unrepresentable | ✅ Result[DocumentStore] with proper errors |
| VIII. Graceful Degradation | Handle failures | ✅ Partial results if some docs fail |
| IX. Observable by Default | Traces + metrics | ✅ Span per document, retrieval metrics |

---

## Related ADRs

- **ADR-001**: Go language choice enables chromem-go (zero-dependency vector DB)
- **ADR-003**: Result[T] type handles multi-doc retrieval errors gracefully
- **ADR-008**: Atomic + NLI grounding extends to cross-document verification
- **ADR-0036**: Evaluation framework includes multi-doc metrics

---

## References

1. Research: `VERA/research/multi-doc-rag.md` (2,100 lines)
2. Library: chromem-go (https://github.com/philippgille/chromem-go)
3. Paper: Reciprocal Rank Fusion (Cormack et al., 2009)
4. Benchmark: Haystack retrieval pipeline (https://haystack.deepset.ai/)
5. Framework: LlamaIndex multi-document patterns (https://docs.llamaindex.ai/)

---

**Status**: ✅ **ACCEPTED**
**Implementation Target**: Week 1-2 of MVP development
**Risk Level**: Low (proven patterns, conservative scope)
**Confidence**: 9.5/10
