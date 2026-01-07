package verification

import (
	"context"
	"strings"

	"github.com/manu/vera/pkg/core"
)

// Chunk represents a semantic unit of text with embedding.
//
// Architecture:
//   - Immutable value object (Article V: Type Safety)
//   - Self-contained: text + embedding + metadata
//   - Size-bounded: 500-1000 tokens for optimal retrieval
//
// Design Rationale:
//   - Chunks are the atomic unit of evidence grounding
//   - Size chosen for balance: small enough for precision, large enough for context
//   - Embeddings enable similarity search (cosine distance)
//
// Usage:
//
//	chunk := Chunk{
//	    Text:      "VERA verifies evidence through categorical verification...",
//	    Embedding: []float32{0.1, 0.2, ...},  // 512 dims
//	    Source:    "/path/to/document.pdf",
//	    Offset:    0,
//	    Length:    150,
//	}
type Chunk struct {
	// Text is the chunk content (plain text).
	Text string

	// Embedding is the vector representation (512 or 768 dims).
	// Normalized for cosine similarity (||v|| = 1).
	Embedding []float32

	// Source is the document path this chunk came from.
	Source string

	// Offset is the character offset in the source document.
	Offset int

	// Length is the number of characters in this chunk.
	Length int

	// Metadata stores additional properties (optional).
	Metadata map[string]string
}

// ChunkStrategy defines how documents are split into chunks.
type ChunkStrategy string

const (
	// StrategyFixed splits by fixed character count (simple, fast).
	StrategyFixed ChunkStrategy = "fixed"

	// StrategySentence splits at sentence boundaries (better semantic units).
	StrategySentence ChunkStrategy = "sentence"

	// StrategyParagraph splits at paragraph boundaries (preserves structure).
	StrategyParagraph ChunkStrategy = "paragraph"

	// StrategySemantic splits by semantic similarity (most sophisticated).
	// Uses embeddings to find natural breakpoints.
	StrategySemantic ChunkStrategy = "semantic"
)

// ChunkConfig configures the chunking process.
//
// Guidelines (from RAG research):
//   - Target 500-1000 tokens (~2000-4000 chars) for academic papers
//   - 20% overlap to avoid losing context at boundaries
//   - Sentence-based splitting preserves meaning
//
// Usage:
//
//	config := ChunkConfig{
//	    Strategy:   StrategySentence,
//	    TargetSize: 3000,  // ~750 tokens
//	    Overlap:    600,   // 20% overlap
//	}
type ChunkConfig struct {
	// Strategy determines how to split documents.
	Strategy ChunkStrategy

	// TargetSize is the desired chunk size in characters.
	// Actual chunks may be slightly larger/smaller to respect boundaries.
	TargetSize int

	// Overlap is the number of characters to overlap between chunks.
	// Prevents losing context at chunk boundaries.
	Overlap int

	// MinSize is the minimum chunk size (skip chunks smaller than this).
	MinSize int
}

// DefaultChunkConfig returns a sensible default configuration.
//
// Defaults:
//   - Sentence-based splitting (preserves semantic units)
//   - 3000 chars (~750 tokens, good for academic papers)
//   - 20% overlap (600 chars)
//   - Min 500 chars (avoid tiny chunks)
func DefaultChunkConfig() ChunkConfig {
	return ChunkConfig{
		Strategy:   StrategySentence,
		TargetSize: 3000,
		Overlap:    600,
		MinSize:    500,
	}
}

// Chunker splits documents into semantic units.
//
// Architecture:
//   - Strategy pattern: different chunking strategies
//   - Result[T] monad for error handling
//   - Observable: track chunk count, timing
//
// Implementations:
//   - FixedChunker: Split by character count (simple)
//   - SentenceChunker: Split at sentence boundaries (recommended)
//   - ParagraphChunker: Split at paragraphs (preserves structure)
//
// Usage:
//
//	chunker := NewSentenceChunker(DefaultChunkConfig())
//	result := chunker.Chunk(ctx, document.Content)
//	if result.IsOk() {
//	    chunks := result.Unwrap()
//	    fmt.Printf("Created %d chunks\n", len(chunks))
//	}
type Chunker interface {
	// Chunk splits text into semantic units.
	//
	// Parameters:
	//   - ctx: Context for cancellation
	//   - text: Document text to chunk
	//   - source: Document path for metadata
	//
	// Returns:
	//   - Ok([]Chunk) with text chunks (no embeddings yet)
	//   - Err(VERAError) on failure:
	//     - ErrorKindValidation: Empty text
	//     - ErrorKindInternal: Chunking algorithm failure
	Chunk(ctx context.Context, text, source string) core.Result[[]Chunk]

	// Name returns the chunker name for logging.
	Name() string

	// Config returns the chunking configuration.
	Config() ChunkConfig
}

// SentenceChunker splits text at sentence boundaries.
//
// Algorithm:
//   1. Split text into sentences (. ! ? followed by space/newline)
//   2. Accumulate sentences until reaching target size
//   3. Create chunk with overlap from previous chunk
//   4. Skip chunks smaller than MinSize
//
// Advantages:
//   - Preserves semantic units (complete sentences)
//   - No broken thoughts at chunk boundaries
//   - Works well for academic papers, documentation
//
// Limitations:
//   - Sentence detection heuristic (may split on abbreviations)
//   - Doesn't handle bullet points well
//   - Code blocks may be split awkwardly
//
// Usage:
//
//	chunker := NewSentenceChunker(DefaultChunkConfig())
//	chunks := chunker.Chunk(ctx, document.Content, document.Path)
type SentenceChunker struct {
	config ChunkConfig
}

// NewSentenceChunker creates a sentence-based chunker.
func NewSentenceChunker(config ChunkConfig) *SentenceChunker {
	return &SentenceChunker{
		config: config,
	}
}

// Chunk splits text at sentence boundaries with overlap.
func (c *SentenceChunker) Chunk(ctx context.Context, text, source string) core.Result[[]Chunk] {
	// Validate input
	if strings.TrimSpace(text) == "" {
		return core.Err[[]Chunk](
			core.NewError(core.ErrorKindValidation, "text cannot be empty", nil),
		)
	}

	// Split into sentences (simple heuristic)
	sentences := splitSentences(text)

	// Accumulate sentences into chunks
	var chunks []Chunk
	var currentChunk strings.Builder
	currentOffset := 0
	overlapText := ""

	for _, sentence := range sentences {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return core.Err[[]Chunk](
				core.NewError(core.ErrorKindInternal, "chunking cancelled", ctx.Err()),
			)
		default:
		}

		// Add sentence to current chunk
		if currentChunk.Len() > 0 {
			currentChunk.WriteByte(' ')
		}
		currentChunk.WriteString(sentence)

		// Check if chunk reached target size
		if currentChunk.Len() >= c.config.TargetSize {
			chunkText := currentChunk.String()

			// Create chunk (if large enough)
			if len(chunkText) >= c.config.MinSize {
				chunk := Chunk{
					Text:      chunkText,
					Embedding: nil, // Embeddings added later
					Source:    source,
					Offset:    currentOffset,
					Length:    len(chunkText),
					Metadata:  map[string]string{"strategy": "sentence"},
				}
				chunks = append(chunks, chunk)

				// Prepare overlap for next chunk
				if c.config.Overlap > 0 && len(chunkText) > c.config.Overlap {
					overlapText = chunkText[len(chunkText)-c.config.Overlap:]
				}
			}

			// Start new chunk with overlap
			currentOffset += len(chunkText) - len(overlapText)
			currentChunk.Reset()
			if overlapText != "" {
				currentChunk.WriteString(overlapText)
			}
		}
	}

	// Add final chunk
	if currentChunk.Len() >= c.config.MinSize {
		chunkText := currentChunk.String()
		chunk := Chunk{
			Text:      chunkText,
			Embedding: nil,
			Source:    source,
			Offset:    currentOffset,
			Length:    len(chunkText),
			Metadata:  map[string]string{"strategy": "sentence"},
		}
		chunks = append(chunks, chunk)
	}

	return core.Ok(chunks)
}

// Name returns the chunker name.
func (c *SentenceChunker) Name() string {
	return "sentence_chunker"
}

// Config returns the chunking configuration.
func (c *SentenceChunker) Config() ChunkConfig {
	return c.config
}

// splitSentences splits text into sentences using simple heuristics.
//
// Rules:
//   - Split on . ! ? followed by whitespace or end of string
//   - Keep punctuation with sentence
//   - Trim whitespace
//
// Limitations:
//   - May split on abbreviations (Dr. Mr. etc.)
//   - Doesn't handle quoted sentences well
//   - Simple but fast (good enough for MVP)
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		current.WriteRune(r)

		// Check for sentence ending
		if r == '.' || r == '!' || r == '?' {
			// Look ahead: whitespace or end of string
			if i+1 >= len(runes) || runes[i+1] == ' ' || runes[i+1] == '\n' {
				sentence := strings.TrimSpace(current.String())
				if sentence != "" {
					sentences = append(sentences, sentence)
				}
				current.Reset()
			}
		}
	}

	// Add remaining text
	if current.Len() > 0 {
		sentence := strings.TrimSpace(current.String())
		if sentence != "" {
			sentences = append(sentences, sentence)
		}
	}

	return sentences
}
