package core

// Verification[T] represents a verified computation result with grounding provenance.
//
// This type wraps any value T with:
//   - GroundingScore: Continuous [0,1] score indicating fact-based support
//   - Citations: Source references with character offsets and confidence
//   - VerificationLog: Audit trail of verification steps
//
// Example:
//
//	answer := "The Eiffel Tower is 300 meters tall"
//	verification := NewVerification(answer, 0.92, []Citation{
//	    {SourceID: "wiki.pdf", Text: "height: 300m", Confidence: 0.95},
//	})
//	verification.AddLog("NLI entailment score: 0.94")
type Verification[T any] struct {
	// Value is the verified computation result
	Value T

	// GroundingScore is the continuous [0,1] score indicating factual support.
	// Higher scores mean stronger evidence from source documents.
	// Interpretation:
	//   >= 0.85: GROUNDED (high confidence)
	//   0.70-0.85: PARTIALLY GROUNDED (medium confidence)
	//   < 0.70: UNGROUNDED (low confidence, requires human review)
	GroundingScore float64

	// Citations are source references supporting the value.
	// Each citation includes character offsets for precise provenance.
	Citations []Citation

	// VerificationLog is an audit trail of verification steps.
	// Example: ["Extracted 5 atomic facts", "NLI score: 0.92", "Coverage: 0.88"]
	VerificationLog []string
}

// Citation represents a source reference with provenance metadata.
type Citation struct {
	// SourceID identifies the source document (e.g., "contract_005.pdf")
	SourceID string

	// Text is the exact quoted text from the source
	Text string

	// CharOffset is the character position where this text starts in the source
	CharOffset int

	// CharLength is the length of the cited text in characters
	CharLength int

	// Confidence is the [0,1] score indicating citation relevance
	// Higher values mean stronger support for the claim
	Confidence float64
}

// NewVerification creates a Verification[T] with value, score, and citations.
func NewVerification[T any](value T, score float64, citations []Citation) Verification[T] {
	return Verification[T]{
		Value:           value,
		GroundingScore:  score,
		Citations:       citations,
		VerificationLog: make([]string, 0),
	}
}

// IsVerified returns true if grounding score meets the threshold.
//
// Example:
//
//	if verification.IsVerified(0.85) {
//	    // High confidence - use answer
//	} else {
//	    // Low confidence - escalate to human
//	}
func (v Verification[T]) IsVerified(threshold float64) bool {
	return v.GroundingScore >= threshold
}

// AddLog appends a verification step to the audit trail.
func (v *Verification[T]) AddLog(message string) {
	v.VerificationLog = append(v.VerificationLog, message)
}

// TopCitation returns the citation with highest confidence, or nil if none.
func (v Verification[T]) TopCitation() *Citation {
	if len(v.Citations) == 0 {
		return nil
	}

	topIdx := 0
	topConf := v.Citations[0].Confidence

	for i, citation := range v.Citations[1:] {
		if citation.Confidence > topConf {
			topConf = citation.Confidence
			topIdx = i + 1
		}
	}

	return &v.Citations[topIdx]
}
