# ADR-0036: Evaluation Framework as First-Class Component

**Status**: Accepted
**Date**: 2025-12-29
**Deciders**: Product Owner, Engineering Team
**Supersedes**: N/A
**Related**: ADR-008 (Grounding Method), ADR-0035 (Multi-Document Support)

---

## Context and Problem Statement

The original MVP and Production specifications **lacked an evaluation framework**. This is a **critical oversight** because:

1. **No proof VERA works**: Cannot demonstrate grounding verification accuracy
2. **No baseline comparison**: Cannot show improvement over traditional RAG
3. **No regression testing**: Cannot detect quality degradation over time
4. **No stakeholder confidence**: "How do we know it's better?" has no answer

**Stakeholder Feedback**: "Evaluation framework itself needs to be part of our whole integrated system rather than just an afterthought."

**Problem**: How do we design an evaluation framework that:
- Measures VERA's unique value proposition (verifiable grounding)
- Integrates with the categorical pipeline architecture
- Provides CI/CD-ready regression testing
- Aligns with production RAG evaluation best practices (RAGAS, TruLens)
- Is implementable within MVP timeline

---

## Decision Drivers

* **Critical missing piece**: Cannot ship without measurable quality
* **Stakeholder requirement**: Must be first-class, not bolted on
* **Research backing**: `evaluation-frameworks.md` provides proven metrics and patterns
* **Differentiation proof**: Must show VERA > traditional RAG with numbers
* **Categorical correctness**: Evaluation must verify functor laws, composition properties
* **Domain requirements**: Legal domain needs >0.95 citation accuracy

---

## Considered Options

### Option 1: Post-hoc Evaluation (Add Later)
**Pros**:
- Simpler initial implementation
- Focus on core features first

**Cons**:
- ❌ Violates stakeholder requirement ("not an afterthought")
- ❌ Cannot validate MVP quality before shipping
- ❌ No baseline comparison to justify VERA's approach
- ❌ Deferred evaluation often never happens

### Option 2: Minimal Metrics Only
**Pros**:
- Lightweight (just grounding score)
- Easy to implement

**Cons**:
- ❌ Insufficient for production systems (no precision/recall)
- ❌ Cannot compare to baselines
- ❌ No regression testing capability
- ❌ Research shows comprehensive metrics needed (RAGAS, TruLens)

### Option 3: Comprehensive Evaluation Framework (SELECTED)
**Pros**:
- ✅ First-class component with CLI (`vera eval`)
- ✅ Full metric suite (faithfulness, citation accuracy, performance)
- ✅ Baseline comparison (naive RAG, VERA)
- ✅ CI/CD integration for regression testing
- ✅ Research-backed (RAGAS faithfulness, TruLens RAG Triad)
- ✅ Categorical property testing (functor laws, composition)

**Cons**:
- Additional implementation time (~3-4 days)
- Requires test dataset construction
- More complex testing infrastructure

---

## Decision Outcome

**Chosen option**: **Option 3 - Comprehensive Evaluation Framework**

### Rationale

From `evaluation-frameworks.md` research:
- **RAGAS faithfulness**: F = |V| / |S| (verified claims / total claims) achieves 0.84-0.87 F1 for hallucination detection
- **TruLens RAG Triad**: Context relevance + groundedness + answer relevance = complete RAG evaluation
- **NLI-based grounding**: Faster than LLM-based, suitable for real-time verification
- **Multi-document metrics**: Cross-document faithfulness, citation graph analysis

**VERA's Unique Requirements**:
1. **Grounding precision**: % of generated claims that are grounded (target: ≥0.90)
2. **Citation accuracy**: % of citations with correct source attribution (target: ≥0.95)
3. **Categorical correctness**: Verify functor laws, natural transformation distribution
4. **Cross-document verification**: Multi-doc faithfulness, contradiction detection

---

## Architecture

### Evaluation Pipeline

```go
// pkg/eval/framework.go

type EvaluationFramework struct {
    pipeline     *Pipeline
    baseline     *BaselineRAG
    metrics      []Metric
    dataset      *EvaluationDataset
    reporter     *Reporter
}

// Core evaluation flow
func (e *EvaluationFramework) Evaluate(ctx context.Context) Result[EvaluationReport] {
    return Do(
        func() Result[[]QueryResult] {
            return e.runQueries(ctx)
        },
        func(results []QueryResult) Result[MetricScores] {
            return e.computeMetrics(ctx, results)
        },
        func(scores MetricScores) Result[BaselineComparison] {
            return e.compareBaseline(ctx, scores)
        },
        func(comparison BaselineComparison) Result[EvaluationReport] {
            return e.generateReport(ctx, comparison)
        },
    )
}
```

### Metric Suite

#### 1. Grounding Metrics (VERA-Specific)

```go
// Grounding Precision: % of generated claims that are grounded
type GroundingPrecision struct{}

func (m *GroundingPrecision) Compute(response Response, sources []Document) float64 {
    claims := extractClaims(response.Text)  // Atomic fact extraction
    groundedClaims := 0

    for _, claim := range claims {
        score := verifyWithNLI(claim, sources)  // DeBERTa NLI
        if score >= 0.70 {  // Entailment threshold
            groundedClaims++
        }
    }

    return float64(groundedClaims) / float64(len(claims))
}

// Target: >= 0.90 for VERA (vs 0.60-0.70 for naive RAG)
```

```go
// Citation Accuracy: % of citations with correct source attribution
type CitationAccuracy struct{}

func (c *CitationAccuracy) Compute(response Response, groundTruth GroundTruth) float64 {
    correctCitations := 0

    for _, citation := range response.Citations {
        if c.verifyCitation(citation, groundTruth) {
            correctCitations++
        }
    }

    return float64(correctCitations) / float64(len(response.Citations))
}

// Target: >= 0.95 for VERA (legal domain requirement)
```

#### 2. Faithfulness Metrics (RAGAS-based)

```go
// Faithfulness: F = |V| / |S| (RAGAS formula)
type Faithfulness struct{}

func (f *Faithfulness) Compute(question string, response string, contexts []string) float64 {
    statements := extractStatements(response)  // Sentence-level decomposition
    verifiedCount := 0

    for _, stmt := range statements {
        nliScore := f.nliModel.Verify(contexts, stmt)
        if nliScore.Entailment >= 0.70 {
            verifiedCount++
        }
    }

    return float64(verifiedCount) / float64(len(statements))
}

// RAGAS benchmark: 0.84-0.87 F1 for hallucination detection
```

#### 3. RAG Triad Metrics (TruLens)

```go
// Context Relevance: How relevant are retrieved chunks?
type ContextRelevance struct{}

func (c *ContextRelevance) Compute(query string, contexts []string) float64 {
    relevanceScores := make([]float64, len(contexts))
    for i, ctx := range contexts {
        relevanceScores[i] = c.llm.ScoreRelevance(query, ctx)  // LLM as judge
    }
    return average(relevanceScores)
}

// Target: >= 0.80 (indicates good retrieval quality)
```

```go
// Answer Relevance: Does response address the query?
type AnswerRelevance struct{}

func (a *AnswerRelevance) Compute(query string, response string) float64 {
    return a.llm.ScoreRelevance(query, response)
}

// Target: >= 0.85
```

#### 4. Categorical Correctness Metrics (VERA-Unique)

```go
// Functor Law: fmap(id) = id
func TestFunctorIdentity(t *testing.T, p Pipeline) {
    input := testInput()

    result1 := p.Map(identity).Run(input)
    result2 := p.Run(input)

    assert.Equal(t, result1, result2, "Functor identity law violated")
}

// Natural Transformation: η ∘ F = G ∘ η
func TestNaturalTransformationDistribution(t *testing.T) {
    pipeline := NewPipeline().Then(Retrieve).Then(Generate)
    verified := pipeline.WithVerification(0.80)

    // η should distribute correctly over composition
    result1 := pipeline.Then(WithVerification(0.80)).Run(input)
    result2 := WithVerification(0.80).Then(pipeline).Run(input)

    assert.GroundingScoresEqual(t, result1, result2)
}
```

### Evaluation Dataset

**MVP Dataset Structure**:
```go
type EvaluationDataset struct {
    Name        string
    Domain      string  // "legal", "research", "technical"
    Queries     []QueryCase
    Documents   []Document
    GroundTruth map[string]GroundTruth
}

type QueryCase struct {
    ID             string
    Query          string
    GroundedAnswer *string  // nil if unanswerable
    Citations      []Citation
    ExpectedScore  float64  // Expected grounding score
    Category       QueryCategory
}

type QueryCategory string
const (
    FactSingle       QueryCategory = "fact_single"      // Single fact retrieval
    FactMultiple     QueryCategory = "fact_multiple"    // Multiple facts
    Reasoning        QueryCategory = "reasoning"        // Requires inference
    Summary          QueryCategory = "summary"          // Document synthesis
    Unanswerable     QueryCategory = "unanswerable"     // No answer in sources
    Contradiction    QueryCategory = "contradiction"    // Conflicting sources
)
```

**MVP Dataset Composition**:
- **Legal Domain**: 50 queries across 5 contract documents
  - 20 fact_single (e.g., "What is the payment term?")
  - 15 fact_multiple (e.g., "What are all termination conditions?")
  - 10 reasoning (e.g., "Does X violate policy Y?")
  - 5 unanswerable (e.g., "What is the liability cap?" when not specified)

- **Research Domain**: 30 queries across 10 papers
  - 15 summary (e.g., "What causes LLM hallucinations?")
  - 10 contradiction (e.g., detect conflicting findings)
  - 5 cross-document synthesis

- **Technical Domain**: 20 queries across 8 Markdown docs
  - 15 fact_single (e.g., "How to authenticate?")
  - 5 reasoning (e.g., "Why is auth failing?")

**Total**: 100 queries, 23 documents, mixed categories

### Baseline Comparison

**Three Baselines**:

1. **Naive RAG**: Retrieve + generate (no verification)
2. **Simple Verification**: Retrieve + generate + post-hoc hallucination check
3. **VERA**: Full eta_1 + eta_3 categorical verification

**Comparison Metrics**:
```go
type BaselineComparison struct {
    VERA        MetricScores
    NaiveRAG    MetricScores
    SimpleVerif MetricScores
    Improvement map[string]float64  // % improvement over baselines
}

// Example output:
{
    "grounding_precision": {
        "VERA": 0.92,
        "NaiveRAG": 0.65,
        "Improvement": "+41.5%"
    },
    "citation_accuracy": {
        "VERA": 0.96,
        "NaiveRAG": 0.72,
        "Improvement": "+33.3%"
    },
    "hallucination_rate": {
        "VERA": 0.07,
        "NaiveRAG": 0.28,
        "Improvement": "-75.0%"
    }
}
```

---

## CLI Integration

### Evaluation Command

```bash
# Run full evaluation suite
vera eval --dataset data/legal-contracts.json --baseline naive-rag --output report.json

# Quick smoke test (10 queries)
vera eval --quick

# Specific metric only
vera eval --metric grounding_precision

# CI/CD regression test (fail if below threshold)
vera eval --regression --threshold 0.85
```

### Output Format

```json
{
  "dataset": "legal-contracts",
  "timestamp": "2025-12-29T10:30:00Z",
  "queries_evaluated": 50,
  "metrics": {
    "grounding_precision": 0.92,
    "grounding_recall": 0.84,
    "f1_score": 0.88,
    "citation_accuracy": 0.96,
    "faithfulness": 0.89,
    "hallucination_rate": 0.07,
    "context_relevance": 0.83,
    "answer_relevance": 0.87,
    "latency_p99_ms": 3200
  },
  "baseline_comparison": {
    "naive_rag": {
      "grounding_precision": 0.65,
      "citation_accuracy": 0.72,
      "hallucination_rate": 0.28
    },
    "improvement_over_baseline": {
      "grounding_precision": "+41.5%",
      "citation_accuracy": "+33.3%",
      "hallucination_rate": "-75.0%"
    }
  },
  "categorical_tests": {
    "functor_identity": "PASS",
    "functor_composition": "PASS",
    "natural_transformation_distribution": "PASS"
  },
  "per_category_breakdown": {
    "fact_single": {"precision": 0.95, "count": 20},
    "reasoning": {"precision": 0.88, "count": 10},
    "unanswerable": {"precision": 0.92, "count": 5}
  }
}
```

---

## Consequences

### Positive

1. **Measurable quality**: Can prove VERA works with numbers
2. **Baseline comparison**: Shows 40%+ improvement in grounding precision
3. **Regression testing**: CI/CD integration prevents quality degradation
4. **Stakeholder confidence**: Data-driven proof of value proposition
5. **Research-backed**: Metrics align with RAGAS, TruLens best practices
6. **Domain-specific**: Legal domain gets >0.95 citation accuracy
7. **Categorical validation**: Tests functor laws, natural transformations

### Negative

1. **Implementation time**: +3-4 days to MVP timeline (acceptable)
2. **Dataset construction**: Requires creating 100-query ground truth set
3. **Complexity**: Evaluation framework adds ~800 lines of code
4. **Dependency**: Requires NLI model (DeBERTa API or local inference)

### Neutral

1. **Baseline maintenance**: Need to keep naive RAG baseline updated
2. **Dataset evolution**: Ground truth may need updates as VERA improves
3. **Metric selection**: Starting with 8 core metrics, can expand in production

---

## Implementation Plan

### Week 1: Core Metrics
- Grounding precision/recall
- Citation accuracy
- Faithfulness (RAGAS-based)
- Basic CLI (`vera eval --quick`)

### Week 2: Dataset + Baselines
- Legal domain dataset (50 queries)
- Naive RAG baseline implementation
- Baseline comparison logic

### Week 3 (Stretch): Advanced Features
- Categorical property tests
- CI/CD integration
- Per-category breakdown
- Dashboard visualization

---

## Compliance with Constitution

| Article | Requirement | Compliance |
|---------|-------------|------------|
| I. Verification as First-Class | Every claim verifiable | ✅ Citation accuracy >= 0.95 |
| VI. Categorical Correctness | Laws verified | ✅ Functor/NatTransform tests |
| VIII. Graceful Degradation | Explicit failures | ✅ Unanswerable category detected |
| IX. Observable by Default | Metrics + traces | ✅ Full evaluation pipeline |

---

## Related ADRs

- **ADR-008**: Grounding method (atomic + NLI) validated by faithfulness metrics
- **ADR-0035**: Multi-document support requires cross-doc evaluation metrics
- **Future ADR-0037**: Production monitoring will extend evaluation to real-time

---

## References

1. Research: `VERA/research/evaluation-frameworks.md` (1,977 lines)
2. RAGAS: https://github.com/explodinggradients/ragas
3. TruLens: https://www.trulens.org/
4. Paper: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)
5. Framework: DeepEval (https://docs.confident-ai.com/)

---

**Status**: ✅ **ACCEPTED**
**Implementation Target**: Week 1-3 of MVP development (parallel with core features)
**Risk Level**: Low (proven frameworks, clear requirements)
**Confidence**: 9.8/10
