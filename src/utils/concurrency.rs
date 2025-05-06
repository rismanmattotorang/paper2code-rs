// src/utils/concurrency.rs
use crate::error::AppError;
use std::path::Path;
use std::sync::Arc;
use std::future::Future;
use tokio::sync::{Semaphore, SemaphorePermit};
use tracing::info;
use anyhow::Result;

/// Configuration for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub max_concurrent: usize,
    pub chunk_size: usize,
    pub adaptive: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_concurrent: num_cpus::get(),
            chunk_size: 1024 * 1024, // 1MB chunks
            adaptive: true,
        }
    }
}

/// Execute tasks in parallel with concurrency control
pub async fn parallel_execute<I, T, O, F, Fut>(
    items: I,
    config: ParallelConfig,
    task_fn: F,
) -> Result<Vec<O>, AppError>
where
    I: IntoIterator<Item = T> + Send + 'static,
    T: Send + 'static,
    O: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, AppError>> + Send + 'static,
{
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
    let mut handles = Vec::new();
    
    for item in items {
        let sem = semaphore.clone();
        let task = task_fn(item);
        
        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.map_err(|e| AppError::TaskJoinError(e.to_string()))?;
            task.await
        });
        
        handles.push(handle);
    }
    
    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.map_err(|e| AppError::TaskJoinError(e.to_string()))??);
    }
    
    Ok(results)
}

/// Resource-managed semaphore guard that ensures release
pub struct SemaphoreGuard<'a> {
    _permit: SemaphorePermit<'a>,
    resource_name: &'static str,
}

impl<'a> SemaphoreGuard<'a> {
    pub fn new(permit: SemaphorePermit<'a>, resource_name: &'static str) -> Self {
        Self {
            _permit: permit,
            resource_name,
        }
    }
}

impl<'a> Drop for SemaphoreGuard<'a> {
    fn drop(&mut self) {
        info!("Released resource: {}", self.resource_name);
    }
}

/// Enhanced PDF processing with parallel page extraction
pub mod pdf {
    use super::*;
    use crate::pdf::PdfExtractor;
    use std::path::Path;

    pub async fn extract_text_parallel<P: AsRef<Path> + Send + 'static>(
        extractor: &PdfExtractor,
        path: P,
    ) -> Result<Vec<String>, AppError> {
        let path_buf = path.as_ref().to_path_buf();
        let extractor = extractor.clone(); // Clone the extractor to avoid reference issues
        
        // Get page count from metadata or use a fixed number of pages
        let page_count = 10; // This should be determined dynamically in a real implementation
        
        // Create page ranges (simple consecutive pages for now)
        let page_ranges: Vec<(u32, u32)> = (1..=page_count)
            .map(|p| (p, p))
            .collect();
        
        // Process page ranges in parallel
        let results = parallel_execute(
            page_ranges,
            ParallelConfig::default(),
            move |(start, end)| {
                let path = path_buf.clone();
                let ext = extractor.clone();
                async move {
                    // Extract pages in the range
                    let pages: Vec<u32> = (start..=end).collect();
                    let text = ext.extract_text_from_pages(&path, &pages).await?;
                    // Join extracted text into a single string
                    Ok(text.join("\n"))
                }
            },
        ).await?;
        
        Ok(results)
    }
}

/// Parallel code block detection
pub mod text {
    use super::*;
    use crate::text::{CodeDetector, CodeBlock};

    pub async fn detect_code_blocks_parallel(
        detector: &CodeDetector,
        chunks: &[String],
    ) -> Result<Vec<CodeBlock>, AppError> {
        // Skip parallel processing for small number of chunks
        if chunks.len() <= 2 {
            let mut blocks = Vec::new();
            for chunk in chunks {
                blocks.extend(detector.detect_code_blocks(chunk));
            }
            return Ok(blocks);
        }
        
        // Configure parallel execution
        let config = ParallelConfig {
            max_concurrent: 4,
            chunk_size: 1024,
            adaptive: false,
        };
        
        // Clone the detector to avoid reference issues
        let detector_clone = detector.clone();
        
        // Process chunks in parallel - note we're using Vec<CodeBlock> as the output type
        let results = parallel_execute(
            chunks.to_vec(),
            config,
            move |chunk| {
                let det = detector_clone.clone();
                async move {
                    let blocks = det.detect_code_blocks(&chunk);
                    Ok(blocks)
                }
            }
        ).await?;
        
        // Flatten results
        let blocks = results.into_iter().flatten().collect();
        
        Ok(blocks)
    }
}

/// Extension to make PDF extractor cloneable for parallel processing
impl Clone for crate::pdf::PdfExtractor {
    fn clone(&self) -> Self {
        Self::new(self.chunk_size)
    }
}

/// Process text chunks in parallel
pub async fn process_chunks_parallel(
    chunks: Vec<String>,
    config: ParallelConfig,
    detector: &crate::text::CodeDetector,
) -> Result<Vec<crate::text::CodeBlock>, AppError> {
    // Clone the detector to avoid reference issues
    let detector_clone = detector.clone();
    
    // Process chunks in parallel - using Vec<crate::text::CodeBlock> as the output type O
    let results = parallel_execute(
        chunks,
        config,
        move |chunk| {
            let det = detector_clone.clone();
            async move {
                let blocks = det.detect_code_blocks(&chunk);
                Ok(blocks)
            }
        },
    ).await?;
    
    // Flatten results
    let blocks = results.into_iter().flatten().collect();
    
    Ok(blocks)
}

/// Extract text from PDF in parallel
pub async fn extract_text_parallel(
    path: &Path,
    config: ParallelConfig,
    extractor: &crate::pdf::PdfExtractor,
) -> Result<Vec<String>, AppError> {
    // Clone path and extractor to avoid reference issues
    let path_buf = path.to_path_buf();
    let extractor_clone = extractor.clone();
    
    // Get page count (hardcoded for now)
    let page_count = 10; // This should be determined dynamically
    
    // Create page ranges
    let page_ranges: Vec<(u32, u32)> = (1..=page_count)
        .map(|p| (p, p))
        .collect();
    
    // Process page ranges in parallel
    let results = parallel_execute(
        page_ranges,
        config,
        move |(start, end)| {
            let path_clone = path_buf.clone();
            let ext = extractor_clone.clone();
            async move {
                // Extract pages in the range
                let pages: Vec<u32> = (start..=end).collect();
                let text = ext.extract_text_from_pages(&path_clone, &pages).await?;
                Ok(text.join("\n"))
            }
        },
    ).await?;
    
    Ok(results)
}