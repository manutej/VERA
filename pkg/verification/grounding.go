package verification

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/manu/vera/pkg/core"
)

// GroundingScore represents the evidence strength for a claim.
//
// Architecture:
//   - Immutable value object (Article V)
//   - Range: 0.0 (no evidence) to 1.0 (perfect grounding)
//   - Observable: includes top evidence chunks with scores
//
// Interpretation (from verification research):
//   - 0.9-1.0: Strong evidence (direct quote or paraphrase)
//   - 0.7-0.9: Good evidence (clear semantic match)
//   - 0.5-0.7: Weak evidence (tangential relationship)
//   - 0.0-0.5: No evidence (unrelated content)
//
// Usage:
//
//	score := GroundingScore{
//	    Score:     0.87,
//	    Claim:     "VERA uses categorical verification",
//	    Evidence:  []EvidenceChunk{{Text: "...", Score: 0.87}},
//	    Threshold: 0.7,
//	    IsGrounded: true,
//	}
type GroundingScore struct {
	// Score is the overall grounding confidence (0.0-1.0).
	Score float64

	// Claim is the text being verified.
	Claim string

	// Evidence contains the top-k supporting chunks with individual scores.
	Evidence []EvidenceChunk

	// Threshold is the minimum score for "grounded" classification.
	Threshold float64

	// IsGrounded indicates if Score >= Threshold.
	IsGrounded bool

	// Metadata stores additional verification details (optional).
	Metadata map[string]string
}

// EvidenceChunk represents a supporting chunk with similarity score.
type EvidenceChunk struct {
	// Text is the chunk content.
	Text string

	// Score is the cosine similarity to the claim (0.0-1.0).
	Score float64

	// Source is the document path.
	Source string

	// Offset is the character offset in the source document.
	Offset int
}

// GroundingConfig configures the verification process.
//
// Guidelines (from verification research):
//   - Top-k: 3-5 chunks (more = noisy, fewer = missing evidence)
//   - Threshold: 0.7 (balance precision/recall)
//   - Aggregation: Max or weighted average
type GroundingConfig struct {
	// TopK is the number of evidence chunks to retrieve.
	TopK int

	// Threshold is the minimum similarity for "grounded".
	Threshold float64

	// AggregationStrategy determines how to combine chunk scores.
	AggregationStrategy AggregationStrategy
}

// AggregationStrategy defines how to combine evidence scores.
type AggregationStrategy string

const (
	// AggregationMax takes the maximum similarity (single best evidence).
	AggregationMax AggregationStrategy = "max"

	// AggregationMean takes the average similarity (overall support).
	AggregationMean AggregationStrategy = "mean"

	// AggregationWeighted uses top-k weighted average (diminishing weights).
	AggregationWeighted AggregationStrategy = "weighted"
)

// DefaultGroundingConfig returns sensible defaults.
//
// Defaults:
//   - Top-3 evidence chunks
//   - 0.7 threshold (70% similarity required)
//   - Max aggregation (single strong evidence suffices)
func DefaultGroundingConfig() GroundingConfig {
	return GroundingConfig{
		TopK:                3,
		Threshold:           0.7,
		AggregationStrategy: AggregationMax,
	}
}

// GroundingCalculator computes evidence strength for claims.
//
// Architecture:
//   - Uses embedding similarity (cosine distance)
//   - Retrieves top-k chunks from evidence corpus
//   - Aggregates scores into final grounding confidence
//
// Process:
//   1. Embed claim text
//   2. Compute cosine similarity to all evidence chunks
//   3. Retrieve top-k most similar chunks
//   4. Aggregate scores (max, mean, or weighted)
//   5. Compare to threshold → IsGrounded
//
// Usage:
//
//	calculator := NewGroundingCalculator(DefaultGroundingConfig())
//	result := calculator.Calculate(ctx, claim, claimEmbedding, evidenceChunks)
//	if result.IsOk() {
//	    score := result.Unwrap()
//	    if score.IsGrounded {
//	        fmt.Printf("✅ Grounded with %.2f confidence\n", score.Score)
//	    }
//	}
type GroundingCalculator struct {
	config GroundingConfig
}

// NewGroundingCalculator creates a grounding score calculator.
func NewGroundingCalculator(config GroundingConfig) *GroundingCalculator {
	return &GroundingCalculator{
		config: config,
	}
}

// Calculate computes grounding score for a claim against evidence chunks.
//
// Parameters:
//   - ctx: Context for cancellation
//   - claim: Text being verified
//   - claimEmbedding: Vector representation of claim (must be L2-normalized)
//   - evidenceChunks: Corpus of evidence chunks with embeddings
//
// Returns:
//   - Ok(GroundingScore) with score, evidence, and grounding decision
//   - Err(VERAError) on failure:
//     - ErrorKindValidation: Empty claim, mismatched dimensions
//     - ErrorKindInternal: Calculation error
func (gc *GroundingCalculator) Calculate(
	ctx context.Context,
	claim string,
	claimEmbedding []float32,
	evidenceChunks []Chunk,
) core.Result[GroundingScore] {
	// Validate input
	if claim == "" {
		return core.Err[GroundingScore](
			core.NewError(core.ErrorKindValidation, "claim cannot be empty", nil),
		)
	}

	if len(claimEmbedding) == 0 {
		return core.Err[GroundingScore](
			core.NewError(core.ErrorKindValidation, "claim embedding cannot be empty", nil),
		)
	}

	if len(evidenceChunks) == 0 {
		// No evidence = score 0.0
		return core.Ok(GroundingScore{
			Score:      0.0,
			Claim:      claim,
			Evidence:   []EvidenceChunk{},
			Threshold:  gc.config.Threshold,
			IsGrounded: false,
			Metadata:   map[string]string{"reason": "no_evidence"},
		})
	}

	// Compute cosine similarity to all evidence chunks
	type similarity struct {
		chunk *Chunk
		score float64
	}

	similarities := make([]similarity, 0, len(evidenceChunks))
	claimDim := len(claimEmbedding)

	for i := range evidenceChunks {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return core.Err[GroundingScore](
				core.NewError(core.ErrorKindInternal, "grounding calculation cancelled", ctx.Err()),
			)
		default:
		}

		chunk := &evidenceChunks[i]

		// Validate embedding dimensions match
		if len(chunk.Embedding) != claimDim {
			return core.Err[GroundingScore](
				core.NewError(
					core.ErrorKindValidation,
					fmt.Sprintf("dimension mismatch: claim %d vs chunk %d", claimDim, len(chunk.Embedding)),
					nil,
				),
			)
		}

		// Compute cosine similarity (assumes L2-normalized embeddings)
		score := cosineSimilarity(claimEmbedding, chunk.Embedding)
		similarities = append(similarities, similarity{chunk: chunk, score: score})
	}

	// Sort by score descending
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].score > similarities[j].score
	})

	// Take top-k
	topK := gc.config.TopK
	if topK > len(similarities) {
		topK = len(similarities)
	}

	topSimilarities := similarities[:topK]

	// Build evidence chunks
	evidence := make([]EvidenceChunk, topK)
	scores := make([]float64, topK)
	for i, sim := range topSimilarities {
		evidence[i] = EvidenceChunk{
			Text:   sim.chunk.Text,
			Score:  sim.score,
			Source: sim.chunk.Source,
			Offset: sim.chunk.Offset,
		}
		scores[i] = sim.score
	}

	// Aggregate scores
	var finalScore float64
	switch gc.config.AggregationStrategy {
	case AggregationMax:
		finalScore = scores[0] // Already sorted descending

	case AggregationMean:
		sum := 0.0
		for _, s := range scores {
			sum += s
		}
		finalScore = sum / float64(len(scores))

	case AggregationWeighted:
		// Weighted average: 0.5, 0.3, 0.2 for top-3
		weights := []float64{0.5, 0.3, 0.2}
		weightedSum := 0.0
		for i, s := range scores {
			weight := 1.0 / float64(i+1) // Default: 1, 1/2, 1/3, ...
			if i < len(weights) {
				weight = weights[i]
			}
			weightedSum += s * weight
		}
		finalScore = weightedSum

	default:
		finalScore = scores[0] // Fallback to max
	}

	// Create grounding score
	groundingScore := GroundingScore{
		Score:      finalScore,
		Claim:      claim,
		Evidence:   evidence,
		Threshold:  gc.config.Threshold,
		IsGrounded: finalScore >= gc.config.Threshold,
		Metadata: map[string]string{
			"top_k":       fmt.Sprintf("%d", topK),
			"aggregation": string(gc.config.AggregationStrategy),
		},
	}

	return core.Ok(groundingScore)
}

// Config returns the grounding configuration.
func (gc *GroundingCalculator) Config() GroundingConfig {
	return gc.config
}

// cosineSimilarity computes the cosine similarity between two vectors.
//
// Formula: cos(θ) = (a · b) / (||a|| * ||b||)
//
// For L2-normalized vectors (||a|| = ||b|| = 1):
// cos(θ) = a · b (just dot product)
//
// Parameters:
//   - a, b: L2-normalized vectors (same length)
//
// Returns:
//   - Similarity score in [0, 1] (0 = orthogonal, 1 = identical)
//
// Note: Assumes vectors are already L2-normalized (as done by Ollama provider).
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := float64(0.0)
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
	}

	// Clamp to [0, 1] to handle floating-point errors
	// (normalized vectors should give [-1, 1], but we map to [0, 1])
	similarity := (dotProduct + 1.0) / 2.0
	return math.Max(0.0, math.Min(1.0, similarity))
}
