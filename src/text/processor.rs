// src/text/processor.rs
use crate::error::AppError;
use crate::text::code_detector::{CodeBlock, CodeDetector};
use std::sync::Arc;

/// Processes text to detect and extract code blocks
pub struct TextProcessor {
    code_detector: Arc<CodeDetector>,
    chunk_size: usize,
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self {
            code_detector: Arc::new(CodeDetector::default()),
            chunk_size: 5000,
        }
    }
}

impl TextProcessor {
    pub fn new(code_detector: CodeDetector, chunk_size: usize) -> Self {
        Self {
            code_detector: Arc::new(code_detector),
            chunk_size,
        }
    }
    
    /// Process a single text chunk to extract code blocks
    pub fn process_chunk(&self, chunk: &str) -> Vec<CodeBlock> {
        self.code_detector.detect_code_blocks(chunk)
    }
    
    /// Process multiple text chunks in parallel
    pub fn process_chunks(&self, chunks: &[String]) -> Result<Vec<CodeBlock>, AppError> {
        use rayon::prelude::*;
        use std::sync::Mutex;
        
        // Use rayon for parallel processing on multiple threads
        let results = Mutex::new(Vec::new());
        
        chunks.par_iter().for_each(|chunk| {
            let blocks = self.code_detector.detect_code_blocks(chunk);
            
            // Add chunk index metadata to help with merging later if needed
            let blocks_with_metadata: Vec<CodeBlock> = blocks.into_iter()
                .enumerate()
                .map(|(i, block)| {
                    block.with_metadata("local_index", &i.to_string())
                })
                .collect();
            
            if !blocks_with_metadata.is_empty() {
                let mut results_guard = results.lock().unwrap();
                results_guard.extend(blocks_with_metadata);
            }
        });
        
        let collected_blocks = results.into_inner()
            .map_err(|e| AppError::TaskJoinError(e.to_string()))?;
        
        Ok(collected_blocks)
    }
    
    /// Split a long text into chunks for more efficient processing
    pub fn split_into_chunks(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_size = 0;
        
        for line in text.lines() {
            let line_size = line.len() + 1; // Include newline
            
            // If adding this line would exceed chunk size, start a new chunk
            if current_size + line_size > self.chunk_size && !current_chunk.is_empty() {
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