// src/utils/helpers.rs
use crate::error::AppError;
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use std::time::Duration;
use crate::pdf::PdfExtractor;
use std::collections::HashMap;

/// Process a large PDF file in chunks
pub async fn process_large_pdf<P: AsRef<Path> + Clone, F, Fut>(
    path: P,
    chunk_size: usize,
    process_fn: F,
) -> Result<(), AppError>
where
    F: Fn(Vec<u8>) -> Fut,
    Fut: std::future::Future<Output = Result<(), AppError>>,
{
    let mut file = File::open(path.as_ref()).await.map_err(AppError::FileError)?;
    let mut buffer = vec![0; chunk_size];
    
    loop {
        let bytes_read = file.read(&mut buffer).await.map_err(AppError::FileError)?;
        if bytes_read == 0 {
            break;
        }
        
        let chunk = buffer[..bytes_read].to_vec();
        process_fn(chunk).await?;
    }
    
    Ok(())
}

/// Process text in chunks
pub async fn process_text_chunks<F, Fut>(
    text: &str,
    chunk_size: usize,
    process_fn: F,
) -> Result<(), AppError>
where
    F: Fn(&str) -> Fut,
    Fut: std::future::Future<Output = Result<(), AppError>>,
{
    let mut start = 0;
    let text_len = text.len();
    
    while start < text_len {
        let end = (start + chunk_size).min(text_len);
        let chunk = &text[start..end];
        
        process_fn(chunk).await?;
        start = end;
    }
    
    Ok(())
}

/// Retry an operation with exponential backoff
pub async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    max_retries: u32,
    initial_delay: Duration,
) -> Result<T, AppError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, AppError>>,
{
    let mut retries = 0;
    let mut delay = initial_delay;
    
    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if retries >= max_retries {
                    return Err(e);
                }
                
                tokio::time::sleep(delay).await;
                delay *= 2;
                retries += 1;
            }
        }
    }
}

/// Format duration for display
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();
    
    if secs > 0 {
        format!("{}.{:03}s", secs, millis)
    } else {
        format!("{}ms", millis)
    }
}

/// Format file size for display
pub fn format_file_size(size: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    
    if size >= GB {
        format!("{:.2} GB", size as f64 / GB as f64)
    } else if size >= MB {
        format!("{:.2} MB", size as f64 / MB as f64)
    } else if size >= KB {
        format!("{:.2} KB", size as f64 / KB as f64)
    } else {
        format!("{} B", size)
    }
}

/// Memory-efficient text processing for large content
pub fn chunk_large_text(text: &str, max_chunk_size: usize) -> Vec<String> {
    if text.len() <= max_chunk_size {
        return vec![text.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < text.len() {
        let mut end = start + max_chunk_size;
        if end >= text.len() {
            end = text.len();
        } else {
            // Try to find a paragraph break or newline
            let slice = &text[start..end];
            if let Some(pos) = slice.rfind("\n\n") {
                end = start + pos + 2;
            } else if let Some(pos) = slice.rfind('\n') {
                end = start + pos + 1;
            } else if let Some(pos) = slice.rfind(". ") {
                end = start + pos + 2;
            }
        }
        
        chunks.push(text[start..end].to_string());
        start = end;
    }
    
    chunks
}

/// Calculate approximate memory usage for a given text
pub fn estimate_memory_usage(text: &str) -> usize {
    // Basic estimate: each character is at least 1 byte
    // In Rust String, each char can be up to 4 bytes (UTF-8)
    // Plus some overhead for the String struct itself
    let _char_count = text.chars().count();
    let byte_count = text.len();
    
    // Estimate: about 2x the byte count for processing overhead
    byte_count * 2
}

/// Estimate if processing can be done in-memory
pub fn can_process_in_memory(file_size: u64, available_memory: u64, safety_factor: f64) -> bool {
    // Estimate memory needed with safety factor
    let estimated_memory = (file_size as f64 * safety_factor) as u64;
    estimated_memory < available_memory
}

// Enhanced PDF extractor with memory-efficient processing
// Add this to src/pdf/extractor.rs
impl PdfExtractor {
    /// Extract text from a large PDF with controlled memory usage
    pub async fn extract_text_memory_efficient<P: AsRef<Path>>(
        &self,
        path: P,
        available_memory: u64,
    ) -> Result<Vec<String>, AppError> {
        let file_size = tokio::fs::metadata(&path).await
            .map_err(AppError::FileError)?
            .len();
            
        // Determine if we need memory-efficient processing
        let safety_factor = 3.0; // PDF processing can use 3x the file size in memory
        let need_memory_efficient = !can_process_in_memory(file_size, available_memory, safety_factor);
        
        if need_memory_efficient {
            // Process page by page to reduce memory usage
            let mut all_chunks = Vec::new();
            
            let doc_metadata = crate::pdf::parser::PdfParser::new()
                .extract_metadata(&path).await?;
                
            // Process each page individually
            for page_num in 1..=doc_metadata.page_count {
                let page_text = self.extract_text_from_pages(&path, &[page_num]).await?;
                
                for text in page_text {
                    let chunks = self.split_into_chunks(&text, self.chunk_size);
                    all_chunks.extend(chunks);
                }
                
                // Optional: yield to allow other tasks to run
                tokio::task::yield_now().await;
            }
            
            Ok(all_chunks)
        } else {
            // Use normal extraction for smaller files
            self.extract_text_from_file(path).await
        }
    }
}

/// Process a large PDF file in chunks
/// This is a higher-level function that combines file IO with text processing
pub async fn process_pdf<P: AsRef<Path>, F, Fut, T>(
    path: P,
    extractor: &PdfExtractor,
    processor: F,
) -> Result<Vec<T>, AppError>
where
    F: Fn(String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<T, AppError>> + Send + 'static,
    T: Send + 'static,
{
    // Extract text from PDF
    let text_chunks = extractor.extract_text_from_file(&path).await?;
    
    // Process each chunk
    let mut results = Vec::new();
    for chunk in text_chunks {
        let result = processor(chunk).await?;
        results.push(result);
    }
    
    Ok(results)
}

/// Extension trait for PdfExtractor providing additional utility methods
#[allow(async_fn_in_trait)]
pub trait PdfExtractorExt {
    /// Extract text with timeout protection
    async fn extract_text_with_timeout<P: AsRef<Path>>(&self, path: P, timeout: Duration) -> Result<Vec<String>, AppError>;
    
    /// Extract text from pages with a specific format
    async fn extract_formatted_text<P: AsRef<Path>>(&self, path: P) -> Result<Vec<String>, AppError>;
}

impl PdfExtractorExt for PdfExtractor {
    async fn extract_text_with_timeout<P: AsRef<Path>>(&self, path: P, timeout: Duration) -> Result<Vec<String>, AppError> {
        tokio::time::timeout(
            timeout,
            self.extract_text_from_file(path)
        ).await.map_err(|_| AppError::TimeoutError(timeout.as_secs() as u64))?
    }
    
    async fn extract_formatted_text<P: AsRef<Path>>(&self, path: P) -> Result<Vec<String>, AppError> {
        let chunks = self.extract_text_from_file(path).await?;
        
        // Apply additional formatting to extracted text
        let formatted = chunks.into_iter()
            .map(|chunk| {
                // Remove excessive whitespace
                let formatted = chunk.lines()
                    .map(|line| line.trim())
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                formatted
            })
            .collect();
            
        Ok(formatted)
    }
}

pub fn calculate_text_stats(text: &str) -> HashMap<String, usize> {
    let mut stats = HashMap::new();
    
    // Basic stats
    let _char_count = text.chars().count();
    let word_count = text.split_whitespace().count();
    let line_count = text.lines().count();
    
    stats.insert("word_count".to_string(), word_count);
    stats.insert("line_count".to_string(), line_count);
    
    // More advanced stats
    let paragraph_count = text.split("\n\n").filter(|p| !p.trim().is_empty()).count();
    stats.insert("paragraph_count".to_string(), paragraph_count);
    
    stats
}