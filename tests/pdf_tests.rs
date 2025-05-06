// tests/pdf_tests.rs
use paper2code_rs::{
    pdf::{PdfExtractor, parser::PdfParser},
    AppResult,
};
use std::path::Path;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_pdf_extraction() -> AppResult<()> {
    // Create a test PDF file
    let temp_dir = TempDir::new()?;
    let test_pdf_path = temp_dir.path().join("test.pdf");
    
    // Copy a test PDF from test fixtures
    fs::copy(Path::new("tests/fixtures/sample.pdf"), &test_pdf_path).await?;
    
    // Create PDF extractor
    let extractor = PdfExtractor::new(1000);
    
    // Test text extraction
    let text_chunks = extractor.extract_text_from_file(&test_pdf_path).await?;
    
    // Verify we got some text
    assert!(!text_chunks.is_empty(), "Should extract text from PDF");
    
    // Test metadata extraction
    let parser = PdfParser::new();
    let metadata = parser.extract_metadata(&test_pdf_path).await?;
    
    // Verify metadata
    assert!(metadata.page_count > 0, "Should have at least one page");
    
    Ok(())
}

#[tokio::test]
async fn test_page_by_page_extraction() -> AppResult<()> {
    // Create a test PDF file
    let temp_dir = TempDir::new()?;
    let test_pdf_path = temp_dir.path().join("test.pdf");
    
    // Copy a test PDF from test fixtures
    fs::copy(Path::new("tests/fixtures/sample.pdf"), &test_pdf_path).await?;
    
    // Create PDF extractor
    let extractor = PdfExtractor::new(1000);
    
    // Get metadata to know page count
    let parser = PdfParser::new();
    let metadata = parser.extract_metadata(&test_pdf_path).await?;
    
    // Extract text page by page
    let mut all_text = Vec::new();
    for page_num in 1..=metadata.page_count {
        let page_text = extractor.extract_text_from_pages(&test_pdf_path, &[page_num]).await?;
        all_text.extend(page_text);
    }
    
    // Verify we got some text
    assert!(!all_text.is_empty(), "Should extract text from PDF page by page");
    
    Ok(())
}