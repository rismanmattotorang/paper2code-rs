// src/utils/performance.rs
use std::time::{Duration, Instant};
use tracing::info;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use crate::error::AppError;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::future::Future;

/// Measures and logs the execution time of a function
pub async fn measure_time<F, Fut, T>(name: &str, f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let start = Instant::now();
    let result = f().await;
    let elapsed = start.elapsed();
    
    info!(
        target: "performance", 
        "Task '{}' completed in {:.2?}", 
        name, elapsed
    );
    
    result
}

/// Performance tracking counters
#[derive(Debug, Clone)]
pub struct PerformanceCounters {
    // Total bytes read from PDFs
    pdf_bytes_read: Arc<AtomicU64>,
    // Total number of PDF pages processed
    pdf_pages_processed: Arc<AtomicU64>,
    // Total number of code blocks detected
    code_blocks_detected: Arc<AtomicU64>,
    // Total number of API calls made
    api_calls: Arc<AtomicU64>,
    // Total tokens consumed
    tokens_consumed: Arc<AtomicU64>,
    // Total execution time
    execution_time_ms: Arc<AtomicU64>,
}

impl Default for PerformanceCounters {
    fn default() -> Self {
        Self {
            pdf_bytes_read: Arc::new(AtomicU64::new(0)),
            pdf_pages_processed: Arc::new(AtomicU64::new(0)),
            code_blocks_detected: Arc::new(AtomicU64::new(0)),
            api_calls: Arc::new(AtomicU64::new(0)),
            tokens_consumed: Arc::new(AtomicU64::new(0)),
            execution_time_ms: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl PerformanceCounters {
    pub fn add_pdf_bytes(&self, bytes: u64) {
        self.pdf_bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }
    
    pub fn add_pdf_pages(&self, pages: u64) {
        self.pdf_pages_processed.fetch_add(pages, Ordering::Relaxed);
    }
    
    pub fn add_code_blocks(&self, blocks: u64) {
        self.code_blocks_detected.fetch_add(blocks, Ordering::Relaxed);
    }
    
    pub fn add_api_call(&self) {
        self.api_calls.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn add_tokens(&self, tokens: u64) {
        self.tokens_consumed.fetch_add(tokens, Ordering::Relaxed);
    }
    
    pub fn add_execution_time(&self, duration: Duration) {
        self.execution_time_ms.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
    }
    
    pub fn pdf_bytes_read(&self) -> u64 {
        self.pdf_bytes_read.load(Ordering::Relaxed)
    }
    
    pub fn pdf_pages_processed(&self) -> u64 {
        self.pdf_pages_processed.load(Ordering::Relaxed)
    }
    
    pub fn code_blocks_detected(&self) -> u64 {
        self.code_blocks_detected.load(Ordering::Relaxed)
    }
    
    pub fn api_calls(&self) -> u64 {
        self.api_calls.load(Ordering::Relaxed)
    }
    
    pub fn tokens_consumed(&self) -> u64 {
        self.tokens_consumed.load(Ordering::Relaxed)
    }
    
    pub fn execution_time_ms(&self) -> u64 {
        self.execution_time_ms.load(Ordering::Relaxed)
    }
    
    pub fn report(&self) -> String {
        format!(
            "Performance Report:\n\
             - PDF bytes read: {} KB\n\
             - PDF pages processed: {}\n\
             - Code blocks detected: {}\n\
             - API calls made: {}\n\
             - Tokens consumed: {}\n\
             - Total execution time: {:.2} seconds",
            self.pdf_bytes_read() / 1024,
            self.pdf_pages_processed(),
            self.code_blocks_detected(),
            self.api_calls(),
            self.tokens_consumed(),
            self.execution_time_ms() as f64 / 1000.0
        )
    }
}

/// Cache implementation for API responses to avoid duplicate work
pub mod cache {
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use std::time::{Duration, SystemTime};
    
    /// Simple in-memory cache for API responses
    pub struct ResponseCache<K, V> {
        cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
        ttl: Duration,
    }
    
    struct CacheEntry<V> {
        value: V,
        expires_at: SystemTime,
    }
    
    impl<K, V> ResponseCache<K, V>
    where
        K: Eq + Hash + Clone,
        V: Clone,
    {
        pub fn new(ttl: Duration) -> Self {
            Self {
                cache: Arc::new(RwLock::new(HashMap::new())),
                ttl,
            }
        }
        
        pub async fn get(&self, key: &K) -> Option<V> {
            let cache = self.cache.read().await;
            
            if let Some(entry) = cache.get(key) {
                if entry.expires_at > SystemTime::now() {
                    return Some(entry.value.clone());
                }
            }
            
            None
        }
        
        pub async fn set(&self, key: K, value: V) {
            let mut cache = self.cache.write().await;
            
            let expires_at = SystemTime::now() + self.ttl;
            let entry = CacheEntry { value, expires_at };
            
            cache.insert(key, entry);
        }
        
        pub async fn clear(&self) {
            let mut cache = self.cache.write().await;
            cache.clear();
        }
        
        pub async fn remove_expired(&self) {
            let mut cache = self.cache.write().await;
            let now = SystemTime::now();
            
            cache.retain(|_, entry| entry.expires_at > now);
        }
    }
}

/// Optimize memory usage with a streaming JSON parser
pub mod streaming {
    use crate::error::AppError;
    use serde::de::DeserializeOwned;
    use bytes::Bytes;
    use futures::StreamExt;
    
    /// Process a JSON array in streaming fashion
    pub async fn process_json_stream<T, F, Fut>(
        mut stream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
        _processor: F,
    ) -> Result<Vec<T>, AppError>
    where
        T: DeserializeOwned,
        F: FnMut(T) -> Fut,
        Fut: std::future::Future<Output = Result<(), AppError>>,
    {
        let mut results = Vec::new();
        let mut buffer = Vec::new();
        
        // Collect all bytes first (not ideal but simpler than mixing async & sync)
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| AppError::ApiError(format!("Stream error: {}", e)))?;
            buffer.extend_from_slice(&chunk);
        }
        
        // Now use serde to deserialize the entire thing
        match serde_json::from_slice::<Vec<T>>(&buffer) {
            Ok(items) => {
                // Process items
                for item in items {
                    results.push(item);
                }
                Ok(results)
            },
            Err(e) => Err(AppError::SerializationError(e)),
        }
    }
}

/// Performance monitoring for PDF processing
pub mod pdf_perf {
    use super::PerformanceCounters;
    use std::sync::Arc;
    use std::path::Path;
    use tokio::fs;
    use crate::pdf::PdfExtractor;
    use crate::error::AppError;
    
    pub struct OptimizedPdfExtractor {
        inner: PdfExtractor,
        counters: Arc<PerformanceCounters>,
    }
    
    impl OptimizedPdfExtractor {
        pub fn new(inner: PdfExtractor, counters: Arc<PerformanceCounters>) -> Self {
            Self { inner, counters }
        }
        
        pub async fn extract_text_from_file<P: AsRef<Path>>(
            &self,
            path: P,
        ) -> Result<Vec<String>, AppError> {
            // Measure file size
            let metadata = fs::metadata(&path).await
                .map_err(AppError::FileError)?;
                
            let file_size = metadata.len();
            self.counters.add_pdf_bytes(file_size);
            
            // Extract text
            let start = std::time::Instant::now();
            let result = self.inner.extract_text_from_file(path).await;
            let elapsed = start.elapsed();
            
            self.counters.add_execution_time(elapsed);
            
            result
        }
    }
}

/// Batch processing for LLM API calls
pub mod llm_batching {
    use crate::error::AppError;
    use crate::llm::{ClaudeClient, PromptBuilder};
    use crate::llm::client::LlmClient;
    use crate::text::code_detector::CodeBlock;
    use futures::{stream, StreamExt};
    use super::PerformanceCounters;
    use std::sync::Arc;
    
    pub struct BatchProcessor {
        client: ClaudeClient,
        counters: Arc<PerformanceCounters>,
        batch_size: usize,
    }
    
    impl BatchProcessor {
        pub fn new(
            client: ClaudeClient, 
            counters: Arc<PerformanceCounters>,
            batch_size: usize,
        ) -> Self {
            Self { client, counters, batch_size }
        }
        
        pub async fn process_blocks(
            &self,
            blocks: &[CodeBlock], 
            prompt_template: &str,
        ) -> Result<Vec<(CodeBlock, String)>, AppError> {
            // Split blocks into batches
            let batches: Vec<Vec<CodeBlock>> = blocks
                .chunks(self.batch_size)
                .map(|chunk| chunk.to_vec())
                .collect();
                
            let total_batches = batches.len();
            tracing::info!("Processing {} code blocks in {} batches", blocks.len(), total_batches);
            
            // Process batches in sequence (to control rate limiting)
            let mut all_results = Vec::new();
            
            for (i, batch) in batches.into_iter().enumerate() {
                tracing::info!("Processing batch {}/{}", i + 1, total_batches);
                
                // Process items in batch concurrently
                let batch_start = std::time::Instant::now();
                
                let batch_results = stream::iter(batch)
                    .map(|block| {
                        let client = self.client.clone();
                        let prompt_template = prompt_template.to_string();
                        let counters = self.counters.clone();
                        
                        async move {
                            counters.add_api_call();
                            
                            let prompt = PromptBuilder::new()
                                .with_template(&prompt_template)
                                .with_code_block(&block.content)
                                .with_language(block.language.as_deref().unwrap_or("unknown"));
                                
                            let start = std::time::Instant::now();
                            let result = client.generate_code(&prompt).await;
                            let elapsed = start.elapsed();
                            
                            counters.add_execution_time(elapsed);
                            
                            match result {
                                Ok(code) => {
                                    // Estimate token count (rough estimate)
                                    let token_estimate = (block.content.len() + code.len()) / 4;
                                    counters.add_tokens(token_estimate as u64);
                                    
                                    Ok((block, code))
                                }
                                Err(e) => Err(e),
                            }
                        }
                    })
                    .buffer_unordered(self.batch_size)
                    .collect::<Vec<Result<(CodeBlock, String), AppError>>>()
                    .await;
                    
                let batch_elapsed = batch_start.elapsed();
                tracing::info!("Batch {}/{} completed in {:.2?}", i + 1, total_batches, batch_elapsed);
                
                // Process batch results
                for result in batch_results {
                    match result {
                        Ok(pair) => all_results.push(pair),
                        Err(e) => {
                            tracing::error!("Error processing code block: {}", e);
                            // Continue with other blocks
                        }
                    }
                }
                
                // Add a small delay between batches to avoid rate limiting
                if i < total_batches - 1 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            }
            
            Ok(all_results)
        }
    }
}

// Add the performance monitoring to the actual code block processing
pub mod code_block {
    use super::PerformanceCounters;
    use crate::text::code_detector::{CodeBlock, CodeDetector};
    use std::sync::Arc;
    
    pub struct MonitoredCodeDetector {
        inner: CodeDetector,
        counters: Arc<PerformanceCounters>,
    }
    
    impl MonitoredCodeDetector {
        pub fn new(inner: CodeDetector, counters: Arc<PerformanceCounters>) -> Self {
            Self { inner, counters }
        }
        
        pub fn detect_code_blocks(&self, text: &str) -> Vec<CodeBlock> {
            let start = std::time::Instant::now();
            let blocks = self.inner.detect_code_blocks(text);
            let elapsed = start.elapsed();
            
            self.counters.add_execution_time(elapsed);
            self.counters.add_code_blocks(blocks.len() as u64);
            
            blocks
        }
        
        pub async fn detect_code_blocks_parallel(&self, chunks: &[String]) -> Vec<CodeBlock> {
            let start = std::time::Instant::now();
            let blocks = self.inner.detect_code_blocks_parallel(chunks).await;
            let elapsed = start.elapsed();
            
            self.counters.add_execution_time(elapsed);
            self.counters.add_code_blocks(blocks.len() as u64);
            
            blocks
        }
    }
    
    impl Clone for MonitoredCodeDetector {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
                counters: self.counters.clone(),
            }
        }
    }
}

/// Track performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_duration: Duration,
    pub min_duration: Option<Duration>,
    pub max_duration: Option<Duration>,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_duration: Duration::from_secs(0),
            min_duration: None,
            max_duration: None,
        }
    }
    
    pub fn record_operation(&mut self, duration: Duration, success: bool) {
        self.total_operations += 1;
        if success {
            self.successful_operations += 1;
        } else {
            self.failed_operations += 1;
        }
        
        self.total_duration += duration;
        
        if let Some(min) = self.min_duration {
            if duration < min {
                self.min_duration = Some(duration);
            }
        } else {
            self.min_duration = Some(duration);
        }
        
        if let Some(max) = self.max_duration {
            if duration > max {
                self.max_duration = Some(duration);
            }
        } else {
            self.max_duration = Some(duration);
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            return 0.0;
        }
        self.successful_operations as f64 / self.total_operations as f64
    }
    
    pub fn average_duration(&self) -> Duration {
        if self.total_operations == 0 {
            Duration::from_secs(0)
        } else {
            let nanos = self.total_duration.as_nanos() as u64;
            Duration::from_nanos(nanos / self.total_operations)
        }
    }
}

/// Track memory usage
#[derive(Debug, Clone, Default)]
pub struct MemoryTracker {
    pub total_allocated: u64,
    pub peak_usage: u64,
    pub current_usage: u64,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            peak_usage: 0,
            current_usage: 0,
        }
    }
    
    pub fn allocate(&mut self, size: u64) {
        self.total_allocated += size;
        self.current_usage += size;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }
    
    pub fn deallocate(&mut self, size: u64) {
        if size > self.current_usage {
            self.current_usage = 0;
        } else {
            self.current_usage -= size;
        }
    }
    
    pub fn reset(&mut self) {
        self.total_allocated = 0;
        self.peak_usage = 0;
        self.current_usage = 0;
    }
}

/// Track operation timing
pub struct OperationTimer {
    start_time: Instant,
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl OperationTimer {
    pub fn new(metrics: Arc<RwLock<PerformanceMetrics>>) -> Self {
        Self {
            start_time: Instant::now(),
            metrics,
        }
    }
    
    pub async fn finish(self, success: bool) {
        let duration = self.start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        metrics.record_operation(duration, success);
    }
}

/// Track memory usage for an operation
pub struct MemoryTrackerGuard {
    size: u64,
    tracker: Arc<RwLock<MemoryTracker>>,
}

impl MemoryTrackerGuard {
    pub fn new(size: u64, tracker: Arc<RwLock<MemoryTracker>>) -> Self {
        // Allocate the memory first
        {
            let mut guard = tracker.blocking_write();
            guard.allocate(size);
        }
        
        // Return a new guard with the cloned tracker
        Self { 
            size, 
            tracker: tracker.clone()
        }
    }
    
    pub async fn finish(self) {
        let mut tracker = self.tracker.write().await;
        tracker.deallocate(self.size);
    }
}

/// Track performance for a specific operation type
pub struct OperationTracker {
    metrics: Arc<RwLock<HashMap<String, PerformanceMetrics>>>,
}

impl OperationTracker {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn track_operation<F, Fut, T>(
        &self,
        operation_type: &str,
        operation: F,
    ) -> Result<T, AppError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, AppError>>,
    {
        let start_time = Instant::now();
        let result = operation().await;
        let duration = start_time.elapsed();
        
        let mut metrics = self.metrics.write().await;
        let operation_metrics = metrics
            .entry(operation_type.to_string())
            .or_insert_with(PerformanceMetrics::new);
            
        operation_metrics.record_operation(duration, result.is_ok());
        
        result
    }
    
    pub async fn get_metrics(&self) -> HashMap<String, PerformanceMetrics> {
        self.metrics.read().await.clone()
    }
}

/// Performance statistics for efficient tracking
pub struct PerformanceStats {
    _total_operations: u64,
    _total_duration: Duration,
    _operation_counts: HashMap<String, u64>,
    _operation_durations: HashMap<String, Duration>,
}

/// Memory allocation with tracking
pub struct TrackedMemory {
    size: usize,
    tracker: Arc<RwLock<MemoryTracker>>,
    released: bool,
}

impl TrackedMemory {
    // Create a new TrackedMemory instance
    pub fn new(size: usize, tracker: Arc<RwLock<MemoryTracker>>) -> Result<Self, AppError> {
        // Allocate the memory
        let result = tracker.clone().try_write_owned();
        match result {
            Ok(mut guard) => {
                guard.allocate(size as u64);
                Ok(Self { 
                    size, 
                    tracker,
                    released: false,
                })
            },
            Err(_) => Err(AppError::ApiError("Failed to allocate memory".to_string())),
        }
    }
    
    // Release memory explicitly - this is safer than relying on Drop
    pub async fn release(&mut self) -> Result<(), AppError> {
        if self.released {
            return Ok(());
        }
        
        let mut guard = self.tracker.write().await;
        guard.deallocate(self.size as u64);
        self.released = true;
        Ok(())
    }
}

impl Drop for TrackedMemory {
    fn drop(&mut self) {
        if !self.released {
            // Try to release memory if not already released
            if let Ok(mut tracker) = self.tracker.try_write() {
                tracker.deallocate(self.size as u64);
            }
        }
    }
}

/// Allocate memory from a memory tracker
pub fn allocate(tracker: &Arc<RwLock<MemoryTracker>>, size: usize) -> Result<TrackedMemory, AppError> {
    TrackedMemory::new(size, tracker.clone())
}

/// LRU cache implementation for memory optimization
pub mod lru_cache {
    // Empty module for now
}

/// Memory-efficient compilation cache
pub mod compilation_cache {
    // Empty module for now
}