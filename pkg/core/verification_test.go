package core_test

import (
	"testing"

	"github.com/manu/vera/pkg/core"
)

// TestNewVerification verifies Verification constructor.
func TestNewVerification(t *testing.T) {
	value := "test answer"
	score := 0.92
	citations := []core.Citation{
		{SourceID: "doc1.pdf", Text: "Section 1", CharOffset: 0, CharLength: 100, Confidence: 0.95},
		{SourceID: "doc2.pdf", Text: "Section 2", CharOffset: 50, CharLength: 80, Confidence: 0.88},
	}

	v := core.NewVerification(value, score, citations)

	if v.Value != value {
		t.Errorf("expected value %q, got %q", value, v.Value)
	}

	if v.GroundingScore != score {
		t.Errorf("expected score %.2f, got %.2f", score, v.GroundingScore)
	}

	if len(v.Citations) != 2 {
		t.Errorf("expected 2 citations, got %d", len(v.Citations))
	}

	if len(v.VerificationLog) != 0 {
		t.Errorf("expected empty log, got %d entries", len(v.VerificationLog))
	}
}

// TestIsVerified_AboveThreshold verifies IsVerified returns true when score ≥ threshold.
func TestIsVerified_AboveThreshold(t *testing.T) {
	v := core.NewVerification("answer", 0.92, nil)

	if !v.IsVerified(0.85) {
		t.Error("expected IsVerified(0.85) = true for score 0.92")
	}

	if !v.IsVerified(0.92) {
		t.Error("expected IsVerified(0.92) = true for score 0.92 (equal)")
	}
}

// TestIsVerified_BelowThreshold verifies IsVerified returns false when score < threshold.
func TestIsVerified_BelowThreshold(t *testing.T) {
	v := core.NewVerification("answer", 0.72, nil)

	if v.IsVerified(0.85) {
		t.Error("expected IsVerified(0.85) = false for score 0.72")
	}
}

// TestIsVerified_ZeroScore verifies IsVerified handles zero score.
func TestIsVerified_ZeroScore(t *testing.T) {
	v := core.NewVerification("answer", 0.0, nil)

	if v.IsVerified(0.85) {
		t.Error("expected IsVerified(0.85) = false for score 0.0")
	}

	if !v.IsVerified(0.0) {
		t.Error("expected IsVerified(0.0) = true for score 0.0 (equal)")
	}
}

// TestIsVerified_PerfectScore verifies IsVerified handles perfect score.
func TestIsVerified_PerfectScore(t *testing.T) {
	v := core.NewVerification("answer", 1.0, nil)

	if !v.IsVerified(0.85) {
		t.Error("expected IsVerified(0.85) = true for perfect score 1.0")
	}

	if !v.IsVerified(1.0) {
		t.Error("expected IsVerified(1.0) = true for perfect score 1.0")
	}
}

// TestAddLog verifies AddLog appends to verification log.
func TestAddLog(t *testing.T) {
	v := core.NewVerification("answer", 0.92, nil)

	if len(v.VerificationLog) != 0 {
		t.Fatalf("expected empty log initially, got %d entries", len(v.VerificationLog))
	}

	v.AddLog("NLI check passed")
	v.AddLog("Citation extracted")

	if len(v.VerificationLog) != 2 {
		t.Errorf("expected 2 log entries, got %d", len(v.VerificationLog))
	}

	if v.VerificationLog[0] != "NLI check passed" {
		t.Errorf("log[0] = %q, want %q", v.VerificationLog[0], "NLI check passed")
	}

	if v.VerificationLog[1] != "Citation extracted" {
		t.Errorf("log[1] = %q, want %q", v.VerificationLog[1], "Citation extracted")
	}
}

// TestTopCitation_NoCitations verifies TopCitation returns nil when no citations.
func TestTopCitation_NoCitations(t *testing.T) {
	v := core.NewVerification("answer", 0.92, nil)

	topCite := v.TopCitation()

	if topCite != nil {
		t.Errorf("expected nil for no citations, got %+v", topCite)
	}
}

// TestTopCitation_EmptyCitations verifies TopCitation returns nil for empty slice.
func TestTopCitation_EmptyCitations(t *testing.T) {
	v := core.NewVerification("answer", 0.92, []core.Citation{})

	topCite := v.TopCitation()

	if topCite != nil {
		t.Errorf("expected nil for empty citations, got %+v", topCite)
	}
}

// TestTopCitation_SingleCitation verifies TopCitation returns the only citation.
func TestTopCitation_SingleCitation(t *testing.T) {
	citations := []core.Citation{
		{SourceID: "doc1.pdf", Text: "Section 1", Confidence: 0.88},
	}
	v := core.NewVerification("answer", 0.92, citations)

	topCite := v.TopCitation()

	if topCite == nil {
		t.Fatal("expected citation, got nil")
	}

	if topCite.SourceID != "doc1.pdf" {
		t.Errorf("expected SourceID doc1.pdf, got %s", topCite.SourceID)
	}

	if topCite.Confidence != 0.88 {
		t.Errorf("expected Confidence 0.88, got %.2f", topCite.Confidence)
	}
}

// TestTopCitation_MultipleCitations verifies TopCitation returns highest confidence.
func TestTopCitation_MultipleCitations(t *testing.T) {
	citations := []core.Citation{
		{SourceID: "doc1.pdf", Text: "Section 1", Confidence: 0.88},
		{SourceID: "doc2.pdf", Text: "Section 2", Confidence: 0.95}, // Highest
		{SourceID: "doc3.pdf", Text: "Section 3", Confidence: 0.72},
		{SourceID: "doc4.pdf", Text: "Section 4", Confidence: 0.91},
	}
	v := core.NewVerification("answer", 0.92, citations)

	topCite := v.TopCitation()

	if topCite == nil {
		t.Fatal("expected citation, got nil")
	}

	if topCite.SourceID != "doc2.pdf" {
		t.Errorf("expected top citation from doc2.pdf, got %s", topCite.SourceID)
	}

	if topCite.Confidence != 0.95 {
		t.Errorf("expected top confidence 0.95, got %.2f", topCite.Confidence)
	}
}

// TestTopCitation_TieBreaking verifies TopCitation returns first when tied.
func TestTopCitation_TieBreaking(t *testing.T) {
	citations := []core.Citation{
		{SourceID: "doc1.pdf", Text: "Section 1", Confidence: 0.90},
		{SourceID: "doc2.pdf", Text: "Section 2", Confidence: 0.95}, // First with 0.95
		{SourceID: "doc3.pdf", Text: "Section 3", Confidence: 0.95}, // Second with 0.95
	}
	v := core.NewVerification("answer", 0.92, citations)

	topCite := v.TopCitation()

	if topCite == nil {
		t.Fatal("expected citation, got nil")
	}

	// Should return first citation with highest confidence
	if topCite.SourceID != "doc2.pdf" {
		t.Errorf("expected first citation with 0.95 (doc2.pdf), got %s", topCite.SourceID)
	}
}

// TestCitation_Complete verifies Citation struct holds all fields.
func TestCitation_Complete(t *testing.T) {
	cite := core.Citation{
		SourceID:   "contract_005.pdf",
		Text:       "Section 3.1 states that...",
		CharOffset: 1234,
		CharLength: 156,
		Confidence: 0.94,
	}

	if cite.SourceID != "contract_005.pdf" {
		t.Errorf("SourceID = %s, want contract_005.pdf", cite.SourceID)
	}

	if cite.Text != "Section 3.1 states that..." {
		t.Errorf("Text = %s, want 'Section 3.1 states that...'", cite.Text)
	}

	if cite.CharOffset != 1234 {
		t.Errorf("CharOffset = %d, want 1234", cite.CharOffset)
	}

	if cite.CharLength != 156 {
		t.Errorf("CharLength = %d, want 156", cite.CharLength)
	}

	if cite.Confidence != 0.94 {
		t.Errorf("Confidence = %.2f, want 0.94", cite.Confidence)
	}
}

// TestVerification_GenericType verifies Verification works with different types.
func TestVerification_GenericType(t *testing.T) {
	// String type
	vString := core.NewVerification("answer", 0.92, nil)
	if vString.Value != "answer" {
		t.Errorf("string Verification: expected 'answer', got %q", vString.Value)
	}

	// Int type
	vInt := core.NewVerification(42, 0.88, nil)
	if vInt.Value != 42 {
		t.Errorf("int Verification: expected 42, got %d", vInt.Value)
	}

	// Struct type
	type Doc struct {
		ID      string
		Content string
	}
	doc := Doc{ID: "doc1", Content: "text"}
	vDoc := core.NewVerification(doc, 0.95, nil)
	if vDoc.Value.ID != "doc1" {
		t.Errorf("struct Verification: expected ID 'doc1', got %q", vDoc.Value.ID)
	}
}
