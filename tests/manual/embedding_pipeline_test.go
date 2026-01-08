package manual

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/manu/vera/pkg/core"
	"github.com/manu/vera/pkg/ingestion"
	"github.com/manu/vera/pkg/providers"
)

func TestEmbeddingPipeline(t *testing.T) {
	// Skip in CI or short mode (requires Ollama running)
	if testing.Short() {
		t.Skip("Skipping manual test - requires Ollama")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	t.Log("🧪 Starting embedding pipeline test...")

	// Step 1: Parse a Markdown document
	t.Log("Step 1/4: Parsing Markdown document...")
	parser := ingestion.NewMarkdownParser()
	parseResult := parser.Parse(ctx, "../../tests/fixtures/sample.md")

	if parseResult.IsErr() {
		t.Fatalf("❌ Parse failed: %v", parseResult.Error())
	}

	doc := parseResult.Unwrap()
	t.Logf("✅ Parsed document: %d words, %s format", doc.WordCount(), doc.Format)

	// Step 2: Generate embeddings for document content
	t.Log("Step 2/4: Generating embedding (this may take 5-7s for first run)...")
	embedder := providers.NewOllamaEmbeddingProvider("http://localhost:11434", "nomic-embed-text")

	request := providers.EmbeddingRequest{
		Texts:     []string{doc.Content},
		Normalize: true, // Request L2 normalization
	}

	start := time.Now()
	embedResult := embedder.Embed(ctx, request)
	latency := time.Since(start)

	if embedResult.IsErr() {
		t.Fatalf("❌ Embedding failed: %v\nHint: Is Ollama running? Try: ollama serve", embedResult.Error())
	}

	response := embedResult.Unwrap()
	t.Logf("✅ Generated embedding: %d dimensions, %dms latency", len(response.Embeddings[0]), latency.Milliseconds())

	// Step 3: Verify embedding quality
	t.Log("Step 3/4: Verifying embedding quality...")
	embedding := response.Embeddings[0]

	if len(embedding) != 768 {
		t.Errorf("❌ Expected 768 dimensions, got %d", len(embedding))
	}

	// Verify L2 norm is ~1.0 (normalized)
	var sumSquares float64
	for _, val := range embedding {
		sumSquares += float64(val) * float64(val)
	}
	norm := math.Sqrt(sumSquares)

	if norm < 0.99 || norm > 1.01 {
		t.Errorf("❌ Expected L2 norm ~1.0, got %.4f (not normalized)", norm)
	} else {
		t.Logf("✅ Embedding quality verified: L2 norm = %.4f (normalized)", norm)
	}

	// Step 4: Verify pipeline composition
	t.Log("Step 4/4: Testing pipeline composition...")

	// Parse pipeline
	parsePipeline := core.PipelineFunc[string, ingestion.Document](func(ctx context.Context, path string) core.Result[ingestion.Document] {
		return parser.Parse(ctx, path)
	})

	// Embed pipeline
	embedPipeline := core.PipelineFunc[ingestion.Document, []float32](func(ctx context.Context, doc ingestion.Document) core.Result[[]float32] {
		req := providers.EmbeddingRequest{Texts: []string{doc.Content}}
		result := embedder.Embed(ctx, req)
		return core.Map(result, func(resp providers.EmbeddingResponse) []float32 {
			return resp.Embeddings[0]
		})
	})

	// Compose pipelines with → operator (Sequence)
	composed := core.Sequence(parsePipeline, embedPipeline)

	finalResult := composed.Execute(ctx, "../../tests/fixtures/sample.md")
	if finalResult.IsErr() {
		t.Fatalf("❌ Composed pipeline failed: %v", finalResult.Error())
	}

	finalEmbedding := finalResult.Unwrap()
	t.Logf("✅ Pipeline composition works: %d-dim embedding from file path", len(finalEmbedding))

	t.Log("\n🎉 ALL TESTS PASSED - Embedding pipeline is working!")
}
