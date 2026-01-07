package ingestion

import (
	"fmt"
	"os"

	"github.com/manu/vera/pkg/core"
)

// ValidateFile checks if a file exists and is readable.
//
// Architecture:
//   - Single source of truth for file validation
//   - Consistent error messages across all parsers
//   - Type-safe error handling (returns VERAError, not error)
//
// Error Handling:
//   - ErrorKindValidation: File doesn't exist (user error)
//   - ErrorKindIngestion: File exists but unreadable (permission/system error)
//
// Usage:
//
//	fileInfo, verr := ValidateFile(filePath, "PDF")
//	if verr != nil {
//	    return core.Err[Document](verr)
//	}
//
// Why separate validation from parsing?
// - DRY: Identical logic in PDF/Markdown/future parsers
// - Testing: Test validation once, not per parser
// - Consistency: Same error messages for all file types
func ValidateFile(filePath, fileType string) (os.FileInfo, *core.VERAError) {
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		// File doesn't exist = user error (don't retry)
		if os.IsNotExist(err) {
			return nil, core.NewError(
				core.ErrorKindValidation,
				fmt.Sprintf("%s file not found: %s", fileType, filePath),
				err,
			)
		}

		// File exists but can't stat = system error (maybe retry after permission fix)
		return nil, core.NewError(
			core.ErrorKindIngestion,
			fmt.Sprintf("cannot stat %s file: %s", fileType, filePath),
			err,
		)
	}

	return fileInfo, nil
}

// ValidateFormat checks if a file has the expected document format.
//
// Architecture:
//   - Uses DetectFormat() for extension-based detection
//   - Returns enriched error with both expected and actual formats
//   - Enables clear user feedback ("expected PDF, got Markdown")
//
// Error Handling:
//   - ErrorKindValidation: Wrong format (user provided wrong file)
//   - Includes context: file_path, expected_format, detected_format
//
// Usage:
//
//	if verr := ValidateFormat(filePath, FormatPDF); verr != nil {
//	    return core.Err[Document](verr)
//	}
//
// Why check format before parsing?
// - Fail fast: Detect wrong files before expensive parsing
// - Clear errors: "Expected PDF, got .txt" vs "PDF parse failed"
// - Type safety: Ensures parser only receives valid file types
func ValidateFormat(filePath string, expected DocumentFormat) *core.VERAError {
	actual := DetectFormat(filePath)

	if actual != expected {
		return core.NewError(
			core.ErrorKindValidation,
			fmt.Sprintf("expected %s file, got %s", expected, actual),
			nil,
		).WithContext("file_path", filePath).
			WithContext("expected_format", string(expected)).
			WithContext("detected_format", string(actual))
	}

	return nil
}

// ValidateDocumentFile performs both file existence and format validation.
//
// This is a convenience function that combines ValidateFile and ValidateFormat
// into a single call, reducing boilerplate in parsers.
//
// Returns:
//   - fileInfo on success
//   - VERAError on failure (either file not found or wrong format)
//
// Usage:
//
//	fileInfo, verr := ValidateDocumentFile(filePath, "PDF", FormatPDF)
//	if verr != nil {
//	    return core.Err[Document](verr)
//	}
//
// Why combine validations?
// - Common pattern: Every parser validates both existence and format
// - Reduces parser boilerplate from ~30 lines to ~3 lines
// - Maintains same error handling (returns first error encountered)
func ValidateDocumentFile(filePath, fileType string, expectedFormat DocumentFormat) (os.FileInfo, *core.VERAError) {
	// First: Check file exists (fail fast if missing)
	fileInfo, verr := ValidateFile(filePath, fileType)
	if verr != nil {
		return nil, verr
	}

	// Second: Check format matches (fail fast if wrong type)
	if verr := ValidateFormat(filePath, expectedFormat); verr != nil {
		return nil, verr
	}

	return fileInfo, nil
}
