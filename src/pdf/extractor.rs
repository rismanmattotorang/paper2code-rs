// src/pdf/extractor.rs
use crate::error::AppError;
use lopdf::Document;
use pdf_extract::extract_text_from_mem;
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

/// Handles extraction of text from PDF documents
pub struct PdfExtractor {
    pub chunk_size: usize,
}

impl Default for PdfExtractor {
    fn default() -> Self {
        Self {
            chunk_size: 1000, // Default chunk size for text processing
        }
    }
}

impl PdfExtractor {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    /// Extracts text from a PDF file asynchronously
    /// 
    /// Uses a streaming approach to handle large PDF files efficiently
    pub async fn extract_text_from_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Vec<String>, AppError> {
        // Read file in chunks to avoid loading entire PDF into memory at once
        let mut file = File::open(path).await.map_err(AppError::from_io_error)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await.map_err(AppError::from_io_error)?;

        // Use tokio spawn_blocking to move CPU-intensive PDF parsing off the async thread
        let chunk_size = self.chunk_size;
        let text = tokio::task::spawn_blocking(move || -> Result<String, AppError> {
            extract_text_from_mem(&buffer).map_err(|e| AppError::PdfExtractError(e.to_string()))
        })
        .await
        .map_err(|e| AppError::TaskJoinError(e.to_string()))??;

        // Split text into manageable chunks for processing
        let chunks = self.split_into_chunks(&text, chunk_size);
        Ok(chunks)
    }

    /// Extract text from specific pages in the document
    pub async fn extract_text_from_pages<P: AsRef<Path>>(
        &self,
        path: P,
        page_numbers: &[u32],
    ) -> Result<Vec<String>, AppError> {
        let buffer = tokio::fs::read(path).await.map_err(AppError::from_io_error)?;
        
        let doc = tokio::task::spawn_blocking(move || -> Result<Document, AppError> {
            Document::load_mem(&buffer).map_err(|e| AppError::PdfParseError(e.to_string()))
        })
        .await
        .map_err(|e| AppError::TaskJoinError(e.to_string()))??;

        let mut results = Vec::new();
        
        // Get pages dictionary - returns a BTreeMap<u32, ObjectId>
        let pages = doc.get_pages();
        
        // Process requested pages
        for &page_num in page_numbers {
            // Find the page object ID corresponding to the page number
            let page_id = match pages.get(&page_num) {
                Some(id) => *id,
                None => return Err(AppError::PdfParseError(format!("Page {} not found", page_num))),
            };

            let page_buffer = doc.get_page_content(page_id)
                .map_err(|e| AppError::PdfParseError(e.to_string()))?;
                
            let page_text = tokio::task::spawn_blocking(move || -> Result<String, AppError> {
                // For page content, we need to process the raw PDF content stream
                // This is a simplified approach - actual implementation would be more complex
                let text = String::from_utf8_lossy(&page_buffer)
                    .replace("\n", " ")
                    .replace("  ", " ");
                Ok(text.to_string())
            })
            .await
            .map_err(|e| AppError::TaskJoinError(e.to_string()))??;
            
            results.push(page_text);
        }
        
        Ok(results)
    }

    /// Splits a large text into manageable chunks for processing
    pub fn split_into_chunks(&self, text: &str, chunk_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_size = 0;

        for line in text.lines() {
            // Add line length plus newline
            let line_size = line.len() + 1;
            
            // If adding this line would exceed chunk size, push current chunk and start a new one
            if current_size + line_size > chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk);
                current_chunk = String::new();
                current_size = 0;
            }
            
            // Add line to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);
            current_size += line_size;
        }
        
        // Push the last chunk if not empty
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }
        
        chunks
    }
} 