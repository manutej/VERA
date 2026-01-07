package ingestion

import (
	"context"
	"path/filepath"
	"strings"
	"time"

	"github.com/manu/vera/pkg/core"
)

// Document represents a parsed document with extracted text content.
//
// Architecture:
//   - Immutable value object (Article V: Type Safety)
//   - Format-agnostic (PDF, Markdown, etc.)
//   - Observable metadata (size, parsing time)
//
// Usage:
//
//	doc := Document{
//	    Path:      "/path/to/paper.pdf",
//	    Content:   "Extracted text...",
//	    Format:    FormatPDF,
//	    ByteSize:  1024,
//	    ParseTime: 150 * time.Millisecond,
//	}
type Document struct {
	// Path is the absolute file path to the source document.
	Path string

	// Content is the extracted plain text content.
	Content string

	// Format is the document format (PDF, Markdown, etc.).
	Format DocumentFormat

	// ByteSize is the original file size in bytes.
	ByteSize int64

	// ParseTime is the time taken to parse this document.
	ParseTime time.Duration

	// Metadata stores additional document properties (optional).
	Metadata map[string]string
}

// DocumentFormat represents supported document formats.
type DocumentFormat string

const (
	// FormatPDF represents Adobe PDF documents (.pdf)
	FormatPDF DocumentFormat = "pdf"

	// FormatMarkdown represents Markdown documents (.md, .markdown)
	FormatMarkdown DocumentFormat = "markdown"

	// FormatUnknown represents unsupported formats
	FormatUnknown DocumentFormat = "unknown"
)

// DocumentParser is the interface for format-specific document parsers.
//
// Architecture:
//   - Dependency Inversion (Article III: Provider Agnosticism)
//   - Type Safety (Article V: Result[T] monad)
//   - Observable (Article IX: parse time tracking)
//
// Implementations:
//   - PDFParser (ledongthuc/pdf)
//   - MarkdownParser (goldmark)
//
// Usage:
//
//	parser := NewPDFParser()
//	result := parser.Parse(ctx, "/path/to/paper.pdf")
//	if result.IsErr() {
//	    log.Fatal(result.Error())
//	}
//	doc := result.Unwrap()
//	fmt.Println(doc.Content)
type DocumentParser interface {
	// Parse extracts text content from a document file.
	//
	// Parameters:
	//   - ctx: Context for cancellation and timeouts
	//   - filePath: Absolute path to the document file
	//
	// Returns:
	//   - Ok(Document) on success with extracted text and metadata
	//   - Err(VERAError) on failure:
	//     - ErrorKindValidation: File doesn't exist, unsupported format
	//     - ErrorKindIO: File read error, permission denied
	//     - ErrorKindParsing: Malformed document, parsing failure
	Parse(ctx context.Context, filePath string) core.Result[Document]

	// SupportedFormats returns the document formats this parser can handle.
	SupportedFormats() []DocumentFormat

	// Name returns the parser name for logging and metrics.
	Name() string
}

// DetectFormat determines the document format from a file path.
//
// Detection Strategy:
//   - File extension (.pdf, .md, .markdown)
//   - Case-insensitive matching
//   - Returns FormatUnknown for unsupported extensions
//
// Usage:
//
//	format := DetectFormat("/path/to/paper.pdf")  // FormatPDF
//	format := DetectFormat("/path/to/README.md")  // FormatMarkdown
//	format := DetectFormat("/path/to/data.json")  // FormatUnknown
func DetectFormat(filePath string) DocumentFormat {
	ext := strings.ToLower(filepath.Ext(filePath))

	switch ext {
	case ".pdf":
		return FormatPDF
	case ".md", ".markdown":
		return FormatMarkdown
	default:
		return FormatUnknown
	}
}

// IsSupported checks if a document format is supported by VERA.
//
// Currently supported:
//   - PDF (via ledongthuc/pdf)
//   - Markdown (via goldmark)
//
// Usage:
//
//	if !IsSupported(FormatPDF) {
//	    return errors.New("unsupported format")
//	}
func IsSupported(format DocumentFormat) bool {
	return format == FormatPDF || format == FormatMarkdown
}

// WordCount estimates the number of words in document content.
//
// Algorithm:
//   - Split on whitespace
//   - Filter empty strings
//   - Simple but fast approximation
//
// Usage:
//
//	doc := Document{Content: "VERA verifies evidence-grounded reasoning"}
//	count := doc.WordCount()  // 4
func (d Document) WordCount() int {
	if d.Content == "" {
		return 0
	}

	words := strings.Fields(d.Content)
	return len(words)
}

// CharCount returns the number of characters in document content.
func (d Document) CharCount() int {
	return len(d.Content)
}

// IsEmpty checks if the document has no extracted content.
func (d Document) IsEmpty() bool {
	return strings.TrimSpace(d.Content) == ""
}

// Summary returns a truncated version of the document content.
//
// Parameters:
//   - maxChars: Maximum number of characters to return
//
// Returns:
//   - First maxChars of content with "..." suffix if truncated
//
// Usage:
//
//	summary := doc.Summary(100)  // First 100 chars + "..."
func (d Document) Summary(maxChars int) string {
	if len(d.Content) <= maxChars {
		return d.Content
	}
	return d.Content[:maxChars] + "..."
}
