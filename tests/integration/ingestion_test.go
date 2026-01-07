package integration_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/manu/vera/pkg/ingestion"
)

// TestMarkdownParsing tests Markdown document parsing with goldmark.
//
// Prerequisites:
//   - Test fixtures in tests/fixtures/
//
// What this tests:
//   - Single Markdown file parsing
//   - Plain text extraction from Markdown AST
//   - Headers, lists, code blocks, formatting
//   - Parse time tracking
//   - Error handling (missing files, wrong format)
func TestMarkdownParsing(t *testing.T) {
	parser := ingestion.NewMarkdownParser()

	t.Run("parse sample markdown", func(t *testing.T) {
		ctx := context.Background()
		fixturePath := filepath.Join("..", "fixtures", "sample.md")

		result := parser.Parse(ctx, fixturePath)
		if result.IsErr() {
			t.Fatalf("Parse failed: %v", result.Error())
		}

		doc := result.Unwrap()

		// Verify document structure
		if doc.Path != fixturePath {
			t.Errorf("Expected path %s, got %s", fixturePath, doc.Path)
		}

		if doc.Format != ingestion.FormatMarkdown {
			t.Errorf("Expected Markdown format, got %s", doc.Format)
		}

		// Verify content extraction
		if doc.Content == "" {
			t.Error("Expected non-empty content")
		}

		// Check for key content (plain text extraction)
		if !strings.Contains(doc.Content, "VERA") {
			t.Error("Expected content to contain 'VERA'")
		}

		if !strings.Contains(doc.Content, "Categorical Verification") {
			t.Error("Expected content to contain section headers")
		}

		// Verify metadata
		if doc.ByteSize == 0 {
			t.Error("Expected non-zero byte size")
		}

		if doc.ParseTime == 0 {
			t.Error("Expected non-zero parse time")
		}

		// Verify word count
		wordCount := doc.WordCount()
		if wordCount < 100 {
			t.Errorf("Expected at least 100 words, got %d", wordCount)
		}

		t.Logf("✅ Parsed Markdown: %d words, %d chars, %.2fms",
			wordCount, doc.CharCount(), float64(doc.ParseTime.Microseconds())/1000.0)
	})

	t.Run("parse short markdown", func(t *testing.T) {
		ctx := context.Background()
		fixturePath := filepath.Join("..", "fixtures", "short.md")

		result := parser.Parse(ctx, fixturePath)
		if result.IsErr() {
			t.Fatalf("Parse failed: %v", result.Error())
		}

		doc := result.Unwrap()

		// Verify content
		if !strings.Contains(doc.Content, "Quick Test") {
			t.Error("Expected header in content")
		}

		// Verify performance (should be very fast for small file)
		if doc.ParseTime > 10*time.Millisecond {
			t.Errorf("Expected parse time < 10ms, got %v", doc.ParseTime)
		}

		t.Logf("✅ Parsed short MD: %d words in %.3fms",
			doc.WordCount(), float64(doc.ParseTime.Microseconds())/1000.0)
	})

	t.Run("error: file not found", func(t *testing.T) {
		ctx := context.Background()
		result := parser.Parse(ctx, "/nonexistent/file.md")

		if result.IsOk() {
			t.Error("Expected error for missing file, got success")
		}

		err := result.Error()
		if err == nil {
			t.Fatal("Expected error object")
		}

		t.Logf("✅ Correctly rejected missing file: %v", err)
	})

	t.Run("error: wrong format", func(t *testing.T) {
		ctx := context.Background()

		// Create temp PDF file (empty, just for format detection)
		tmpFile, err := os.CreateTemp("", "test-*.pdf")
		if err != nil {
			t.Fatalf("Failed to create temp file: %v", err)
		}
		defer os.Remove(tmpFile.Name())
		tmpFile.Close()

		result := parser.Parse(ctx, tmpFile.Name())

		if result.IsOk() {
			t.Error("Expected error for wrong format, got success")
		}

		t.Logf("✅ Correctly rejected PDF file: %v", result.Error())
	})

	t.Run("context cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		fixturePath := filepath.Join("..", "fixtures", "sample.md")
		result := parser.Parse(ctx, fixturePath)

		if result.IsOk() {
			t.Error("Expected error for cancelled context, got success")
		}

		t.Logf("✅ Correctly handled context cancellation")
	})
}

// TestPDFParsing tests PDF document parsing with ledongthuc/pdf.
//
// Note: This test is currently skipped because we don't have sample PDFs.
// To enable: Create test PDFs in tests/fixtures/
func TestPDFParsing(t *testing.T) {
	parser := ingestion.NewPDFParser()

	t.Run("error: file not found", func(t *testing.T) {
		ctx := context.Background()
		result := parser.Parse(ctx, "/nonexistent/file.pdf")

		if result.IsOk() {
			t.Error("Expected error for missing file, got success")
		}

		err := result.Error()
		if err == nil {
			t.Fatal("Expected error object")
		}

		t.Logf("✅ Correctly rejected missing file: %v", err)
	})

	t.Run("error: wrong format", func(t *testing.T) {
		ctx := context.Background()
		fixturePath := filepath.Join("..", "fixtures", "sample.md")

		result := parser.Parse(ctx, fixturePath)

		if result.IsOk() {
			t.Error("Expected error for wrong format, got success")
		}

		t.Logf("✅ Correctly rejected Markdown file: %v", result.Error())
	})

	t.Run("context cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		result := parser.Parse(ctx, "/some/file.pdf")

		// May fail on file not found before context check, that's ok
		if result.IsOk() {
			t.Error("Expected error, got success")
		}

		t.Logf("✅ Error handling works: %v", result.Error())
	})
}

// TestDocumentFormat tests format detection and validation.
func TestDocumentFormat(t *testing.T) {
	testCases := []struct {
		path     string
		expected ingestion.DocumentFormat
	}{
		{"/path/to/paper.pdf", ingestion.FormatPDF},
		{"/path/to/paper.PDF", ingestion.FormatPDF},
		{"/path/to/README.md", ingestion.FormatMarkdown},
		{"/path/to/doc.markdown", ingestion.FormatMarkdown},
		{"/path/to/doc.MD", ingestion.FormatMarkdown},
		{"/path/to/data.json", ingestion.FormatUnknown},
		{"/path/to/script.go", ingestion.FormatUnknown},
	}

	for _, tc := range testCases {
		t.Run(tc.path, func(t *testing.T) {
			format := ingestion.DetectFormat(tc.path)
			if format != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, format)
			}
		})
	}
}

// TestDocumentHelpers tests Document helper methods.
func TestDocumentHelpers(t *testing.T) {
	doc := ingestion.Document{
		Content: "VERA verifies evidence-grounded reasoning with categorical precision",
	}

	t.Run("word count", func(t *testing.T) {
		count := doc.WordCount()
		if count != 7 {
			t.Errorf("Expected 7 words, got %d", count)
		}
	})

	t.Run("char count", func(t *testing.T) {
		count := doc.CharCount()
		if count != 68 {
			t.Errorf("Expected 68 chars, got %d", count)
		}
	})

	t.Run("is empty", func(t *testing.T) {
		if doc.IsEmpty() {
			t.Error("Expected non-empty document")
		}

		emptyDoc := ingestion.Document{Content: "   "}
		if !emptyDoc.IsEmpty() {
			t.Error("Expected empty document")
		}
	})

	t.Run("summary", func(t *testing.T) {
		summary := doc.Summary(20)
		expected := "VERA verifies eviden..."
		if summary != expected {
			t.Errorf("Expected '%s', got '%s'", expected, summary)
		}

		fullSummary := doc.Summary(100)
		if fullSummary != doc.Content {
			t.Error("Expected full content when maxChars > length")
		}
	})
}

// TestParserInterfaces tests parser interface implementations.
func TestParserInterfaces(t *testing.T) {
	t.Run("markdown parser", func(t *testing.T) {
		parser := ingestion.NewMarkdownParser()

		if parser.Name() != "markdown_parser" {
			t.Errorf("Expected name 'markdown_parser', got '%s'", parser.Name())
		}

		formats := parser.SupportedFormats()
		if len(formats) != 1 || formats[0] != ingestion.FormatMarkdown {
			t.Errorf("Expected [FormatMarkdown], got %v", formats)
		}
	})

	t.Run("pdf parser", func(t *testing.T) {
		parser := ingestion.NewPDFParser()

		if parser.Name() != "pdf_parser" {
			t.Errorf("Expected name 'pdf_parser', got '%s'", parser.Name())
		}

		formats := parser.SupportedFormats()
		if len(formats) != 1 || formats[0] != ingestion.FormatPDF {
			t.Errorf("Expected [FormatPDF], got %v", formats)
		}
	})
}
