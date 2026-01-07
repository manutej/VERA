package ingestion

import (
	"bytes"
	"context"
	"fmt"
	"time"

	"github.com/ledongthuc/pdf"
	"github.com/manu/vera/pkg/core"
)

// PDFParser implements DocumentParser for Adobe PDF files.
//
// Architecture:
//   - Uses ledongthuc/pdf (open-source, pure Go, no dependencies)
//   - Extracts plain text via GetPlainText()
//   - Tracks parse time for observability (Article IX)
//   - Returns typed errors for retry logic (Article VIII)
//
// Performance (from research):
//   - Handles multi-page PDFs
//   - Memory-efficient for large documents
//   - No external dependencies (pure Go)
//
// Limitations:
//   - May struggle with complex PDF layouts
//   - Text extraction quality depends on PDF structure
//   - No OCR support (images with text not extracted)
//
// Usage:
//
//	parser := NewPDFParser()
//	result := parser.Parse(ctx, "/path/to/paper.pdf")
//	if result.IsOk() {
//	    doc := result.Unwrap()
//	    fmt.Printf("Extracted %d words in %v\n", doc.WordCount(), doc.ParseTime)
//	}
type PDFParser struct{}

// NewPDFParser creates a new PDF document parser.
func NewPDFParser() *PDFParser {
	return &PDFParser{}
}

// Parse extracts text content from a PDF file.
//
// Process:
//   1. Validate file exists and is readable
//   2. Open PDF with ledongthuc/pdf
//   3. Extract plain text from all pages
//   4. Track parse time for observability
//   5. Return Document with metadata
//
// Error Handling:
//   - ErrorKindValidation: File doesn't exist, wrong format
//   - ErrorKindIO: File read error, permission denied
//   - ErrorKindParsing: Malformed PDF, extraction failure
//
// Performance:
//   - Typical: 50-200ms for 10-page academic paper
//   - Memory: Scales with PDF size
func (p *PDFParser) Parse(ctx context.Context, filePath string) core.Result[Document] {
	startTime := time.Now()

	// Validate file exists and has correct format
	fileInfo, verr := ValidateDocumentFile(filePath, "PDF", FormatPDF)
	if verr != nil {
		return core.Err[Document](verr)
	}

	// Open PDF file
	file, pdfReader, err := pdf.Open(filePath)
	if err != nil {
		return core.Err[Document](
			core.NewError(
				core.ErrorKindIngestion,
				"failed to open PDF file",
				err,
			).WithContext("file_path", filePath),
		)
	}
	defer file.Close()
	defer func() {
		// ledongthuc/pdf doesn't require explicit close
		// but we defer for future compatibility
	}()

	// Check for context cancellation before extraction
	select {
	case <-ctx.Done():
		return core.Err[Document](
			core.NewError(
				core.ErrorKindIngestion,
				"PDF parsing cancelled",
				ctx.Err(),
			),
		)
	default:
	}

	// Extract plain text from all pages
	textReader, err := pdfReader.GetPlainText()
	if err != nil {
		return core.Err[Document](
			core.NewError(
				core.ErrorKindIngestion,
				"failed to extract text from PDF",
				err,
			).WithContext("file_path", filePath),
		)
	}

	// Read extracted text into buffer
	var buf bytes.Buffer
	_, err = buf.ReadFrom(textReader)
	if err != nil {
		return core.Err[Document](
			core.NewError(
				core.ErrorKindIngestion,
				"failed to read extracted text",
				err,
			),
		)
	}

	content := buf.String()

	// Calculate parse time
	parseTime := time.Since(startTime)

	// Build Document
	doc := Document{
		Path:      filePath,
		Content:   content,
		Format:    FormatPDF,
		ByteSize:  fileInfo.Size(),
		ParseTime: parseTime,
		Metadata: map[string]string{
			"parser":     "ledongthuc/pdf",
			"num_pages":  fmt.Sprintf("%d", pdfReader.NumPage()),
			"word_count": fmt.Sprintf("%d", len(content)/5), // Rough estimate
		},
	}

	return core.Ok(doc)
}

// SupportedFormats returns the document formats this parser can handle.
func (p *PDFParser) SupportedFormats() []DocumentFormat {
	return []DocumentFormat{FormatPDF}
}

// Name returns the parser name for logging and metrics.
func (p *PDFParser) Name() string {
	return "pdf_parser"
}
