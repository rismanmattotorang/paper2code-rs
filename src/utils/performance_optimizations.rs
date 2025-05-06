// src/utils/performance_optimizations.rs
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, RwLock};
use tokio::task::JoinSet;
use tracing::{debug, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use rayon::prelude::*;
use lru::LruCache;

use crate::error::{AppError, Result};

/// Memory usage profile for the application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryProfile {
    /// Low memory usage for resource-constrained environments
    Low,
    
    /// Standard memory usage for typical environments
    Standard,
    
    /// High memory usage for high-performance environments
    High,
    
    /// Automatic memory usage based on available system resources
    Auto,
}

impl MemoryProfile {
    /// Get chunk size for text processing based on memory profile
    pub fn get_chunk_size(&self) -> usize {
        match self {
            Self::Low => 500,
            Self::Standard => 2000,
            Self::High => 8000,
            Self::Auto => {
                // Get system memory
                let sys_mem = sys_info::mem_info().map(|info| info.total).unwrap_or(8 * 1024 * 1024); // Default to 8GB
                
                // Scale chunk size based on available memory
                if sys_mem < 4 * 1024 * 1024 {
                    // Less than 4GB
                    500
                } else if sys_mem < 16 * 1024 * 1024 {
                    // 4GB to 16GB
                    2000
                } else {
                    // More than 16GB
                    8000
                }
            }
        }
    }
    
    /// Get maximum parallel tasks based on memory profile
    pub fn get_max_parallel_tasks(&self) -> usize {
        match self {
            Self::Low => 2,
            Self::Standard => 4,
            Self::High => 8,
            Self::Auto => {
                // Get number of CPU cores
                let num_cpus = num_cpus::get();
                
                // Get system memory
                let sys_mem = sys_info::mem_info().map(|info| info.total).unwrap_or(8 * 1024 * 1024); // Default to 8GB
                
                // Calculate max tasks based on both CPU and memory
                let cpu_based = num_cpus;
                let mem_based = (sys_mem / (1024 * 1024) / 2) as usize; // Rough estimate: 2GB per task
                
                cpu_based.min(mem_based).max(1)
            }
        }
    }
}

/// Cache entry with metadata
#[derive(Clone)]
struct CacheEntry<T> {
    value: T,
    _created_at: Instant,
    last_accessed: Instant,
    access_count: u32,
}

impl<T> CacheEntry<T> {
    fn new(value: T) -> Self {
        let now = Instant::now();
        Self {
            value,
            _created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }
    
    fn update_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Thread-safe LRU cache with metadata
pub struct MetadataCache<K, V> {
    cache: Arc<RwLock<LruCache<K, CacheEntry<V>>>>,
    max_age: Duration,
    max_access_count: u32,
}

impl<K: std::hash::Hash + Eq + Clone + Send + Sync + 'static, 
     V: Clone + Send + Sync + 'static> 
MetadataCache<K, V> {
    pub fn new(capacity: usize, max_age_secs: u64, max_access_count: u32) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(NonZeroUsize::new(capacity).unwrap()))),
            max_age: Duration::from_secs(max_age_secs),
            max_access_count,
        }
    }
    
    pub async fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().await;
        
        if let Some(entry) = cache.get_mut(key) {
            // Check if entry is expired
            if entry.last_accessed.elapsed() > self.max_age || 
               entry.access_count >= self.max_access_count {
                cache.pop(key);
                return None;
            }
            
            entry.update_access();
            Some(entry.value.clone())
        } else {
            None
        }
    }
    
    pub async fn insert(&self, key: K, value: V) {
        let mut cache = self.cache.write().await;
        cache.put(key, CacheEntry::new(value));
    }
    
    pub async fn remove(&self, key: &K) {
        let mut cache = self.cache.write().await;
        cache.pop(key);
    }
    
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

/// Rate limiter using token bucket algorithm
pub struct RateLimiter {
    tokens: Arc<RwLock<u32>>,
    max_tokens: u32,
    refill_rate: u32,
    last_refill: Arc<RwLock<Instant>>,
}

impl RateLimiter {
    pub fn new(max_tokens: u32, refill_rate: u32) -> Self {
        Self {
            tokens: Arc::new(RwLock::new(max_tokens)),
            max_tokens,
            refill_rate,
            last_refill: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    pub async fn acquire(&self) -> Result<RateLimitPermit> {
        let mut tokens = self.tokens.write().await;
        let mut last_refill = self.last_refill.write().await;
        
        // Refill tokens based on time elapsed
        let now = Instant::now();
        let elapsed = now.duration_since(*last_refill);
        let refill_amount = (elapsed.as_secs_f64() * self.refill_rate as f64) as u32;
        
        if refill_amount > 0 {
            *tokens = (*tokens + refill_amount).min(self.max_tokens);
            *last_refill = now;
        }
        
        if *tokens == 0 {
            return Err(AppError::ApiError("Rate limit exceeded".to_string()));
        }
        
        *tokens -= 1;
        Ok(RateLimitPermit {
            _limiter: self.clone(),
        })
    }
}

impl Clone for RateLimiter {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            max_tokens: self.max_tokens,
            refill_rate: self.refill_rate,
            last_refill: self.last_refill.clone(),
        }
    }
}

/// Permit for rate-limited operations
pub struct RateLimitPermit {
    _limiter: RateLimiter,
}

impl Drop for RateLimitPermit {
    fn drop(&mut self) {
        // Permit is automatically released when dropped
    }
}

/// Memory tracker for operations
pub struct MemoryTracker {
    used_memory: Arc<RwLock<u64>>,
    max_memory: u64,
}

impl MemoryTracker {
    pub fn new(max_memory: u64) -> Self {
        Self {
            used_memory: Arc::new(RwLock::new(0)),
            max_memory,
        }
    }
    
    pub async fn allocate(&self, size: u64) -> Result<MemoryPermit> {
        let mut used = self.used_memory.write().await;
        if *used + size > self.max_memory {
            return Err(AppError::ApiError("Memory limit exceeded".to_string()));
        }
        
        *used += size;
        Ok(MemoryPermit {
            size,
            tracker: self.clone(),
        })
    }
}

impl Clone for MemoryTracker {
    fn clone(&self) -> Self {
        Self {
            used_memory: self.used_memory.clone(),
            max_memory: self.max_memory,
        }
    }
}

/// Permit for memory allocation
pub struct MemoryPermit {
    size: u64,
    tracker: MemoryTracker,
}

impl Drop for MemoryPermit {
    fn drop(&mut self) {
        let mut used = self.tracker.used_memory.blocking_write();
        *used -= self.size;
    }
}

/// Adaptive concurrency controller
pub struct AdaptiveConcurrency {
    current_limit: Arc<RwLock<usize>>,
    min_limit: usize,
    max_limit: usize,
    step_size: usize,
    success_threshold: f64,
    window_size: usize,
    recent_results: Arc<RwLock<Vec<bool>>>,
}

impl AdaptiveConcurrency {
    pub fn new(
        initial_limit: usize,
        min_limit: usize,
        max_limit: usize,
        step_size: usize,
        success_threshold: f64,
        window_size: usize,
    ) -> Self {
        Self {
            current_limit: Arc::new(RwLock::new(initial_limit)),
            min_limit,
            max_limit,
            step_size,
            success_threshold,
            window_size,
            recent_results: Arc::new(RwLock::new(Vec::with_capacity(window_size))),
        }
    }
    
    pub async fn acquire(&self) -> Result<ConcurrencyPermit> {
        let limit = *self.current_limit.read().await;
        Ok(ConcurrencyPermit {
            _controller: self.clone(),
            _limit: limit,
        })
    }
    
    pub async fn record_result(&self, success: bool) {
        let mut results = self.recent_results.write().await;
        results.push(success);
        
        if results.len() > self.window_size {
            results.remove(0);
        }
        
        // Calculate success rate
        let success_rate = results.iter()
            .filter(|&&success| success)
            .count() as f64 / results.len() as f64;
        
        // Adjust limit based on success rate
        let mut limit = *self.current_limit.read().await;
        
        if success_rate >= self.success_threshold {
            // Increase limit if success rate is good
            limit = (limit + self.step_size).min(self.max_limit);
        } else {
            // Decrease limit if success rate is poor
            limit = limit.saturating_sub(self.step_size).max(self.min_limit);
        }
        
        *self.current_limit.write().await = limit;
    }
}

impl Clone for AdaptiveConcurrency {
    fn clone(&self) -> Self {
        Self {
            current_limit: Arc::clone(&self.current_limit),
            min_limit: self.min_limit,
            max_limit: self.max_limit,
            step_size: self.step_size,
            success_threshold: self.success_threshold,
            window_size: self.window_size,
            recent_results: Arc::clone(&self.recent_results),
        }
    }
}

/// Permit for adaptive concurrency
pub struct ConcurrencyPermit {
    _controller: AdaptiveConcurrency,
    _limit: usize,
}

impl Drop for ConcurrencyPermit {
    fn drop(&mut self) {
        // Permit is automatically released when dropped
    }
}

/// Enhanced parallel executor with adaptive batch sizing
pub struct AdaptiveParallelExecutor<T> {
    max_concurrent: usize,
    adaptive_batch_size: bool,
    items_per_task: usize,
    semaphore: Arc<Semaphore>,
    tasks: JoinSet<Result<T>>,
}

impl<T: Send + 'static> AdaptiveParallelExecutor<T> {
    /// Create a new adaptive parallel executor
    pub fn new(max_concurrent: usize, adaptive_batch_size: bool) -> Self {
        Self {
            max_concurrent,
            adaptive_batch_size,
            items_per_task: 1,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            tasks: JoinSet::new(),
        }
    }
    
    /// Set items per task (batch size)
    pub fn with_items_per_task(mut self, items_per_task: usize) -> Self {
        self.items_per_task = items_per_task.max(1);
        self
    }
    
    /// Add a task to the executor
    pub fn add_task<F, Fut>(&mut self, task: F) 
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        let semaphore = self.semaphore.clone();
        
        self.tasks.spawn(async move {
            // Acquire a permit
            let _permit = semaphore.acquire().await.map_err(|e| 
                AppError::TaskJoinError(format!("Failed to acquire semaphore: {}", e))
            )?;
            
            // Execute the task
            let start = Instant::now();
            let result = task().await;
            let elapsed = start.elapsed();
            
            debug!("Task completed in {:?}", elapsed);
            
            result
        });
    }
    
    /// Add a batch of items to process
    pub fn add_batch<I, F, Fut>(&mut self, items: I, task_fn: F)
    where
        I: IntoIterator<Item = T> + Send + 'static,
        T: Clone + Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send,
    {
        let items: Vec<_> = items.into_iter().collect();
        
        // Determine optimal batch size
        let batch_size = if self.adaptive_batch_size {
            // Adjust batch size based on number of items and concurrency
            let total_items = items.len();
            let ideal_batches = self.max_concurrent * 2; // Target 2x concurrency for good utilization
            
            if total_items <= ideal_batches {
                // One item per batch if we have few items
                1
            } else {
                // Otherwise, distribute items evenly
                (total_items / ideal_batches).max(1)
            }
        } else {
            self.items_per_task
        };
        
        // Create batches
        for chunk in items.chunks(batch_size) {
            let chunk = chunk.to_vec();
            let task_fn = task_fn.clone();
            
            self.add_task(move || async move {
                // Process entire batch and collect results
                let mut results = Vec::with_capacity(chunk.len());
                
                for item in chunk {
                    match task_fn(item).await {
                        Ok(result) => results.push(result),
                        Err(e) => return Err(e),
                    }
                }
                
                // Return the first result (assumes all results in batch are similar)
                results.into_iter().next().ok_or_else(|| 
                    AppError::Unknown("Batch processing returned no results".to_string())
                )
            });
        }
    }
    
    /// Wait for all tasks to complete
    pub async fn join_all(mut self) -> Result<Vec<T>> {
        let mut results = Vec::new();
        
        // Process each completed task
        while let Some(task_result) = self.tasks.join_next().await {
            match task_result {
                Ok(Ok(result)) => {
                    results.push(result);
                },
                Ok(Err(e)) => {
                    warn!("Task error: {}", e);
                    return Err(e);
                },
                Err(e) => {
                    return Err(AppError::TaskJoinError(format!("Failed to join task: {}", e)));
                }
            }
        }
        
        Ok(results)
    }
    
    /// Wait for the first successful task to complete
    pub async fn join_first(mut self) -> Result<T> {
        loop {
            match self.tasks.join_next().await {
                Some(Ok(Ok(result))) => {
                    // Found a successful result, cancel other tasks
                    self.tasks.abort_all();
                    return Ok(result);
                },
                Some(Ok(Err(e))) => {
                    warn!("Task error: {}", e);
                    // Continue with other tasks
                }
                Some(Err(e)) => {
                    warn!("Join error: {}", e);
                    // Continue with other tasks
                }
                None => {
                    // All tasks completed without success
                    return Err(AppError::Unknown("All tasks failed or were aborted".to_string()));
                }
            }
        }
    }
}

/// Optimized text processing for large documents
pub struct OptimizedTextProcessor {
    chunk_size: usize,
    _parallelism: usize,
}

impl OptimizedTextProcessor {
    /// Create a new optimized text processor
    pub fn new(memory_profile: MemoryProfile) -> Self {
        Self {
            chunk_size: memory_profile.get_chunk_size(),
            _parallelism: memory_profile.get_max_parallel_tasks(),
        }
    }
    
    /// Process a large text document in chunks
    pub fn process_text<F>(&self, text: &str, processor: F) -> Vec<String>
    where
        F: Fn(&str) -> Vec<String> + Sync,
    {
        // Split text into chunks
        let chunks = self.split_into_chunks(text);
        
        // Process chunks in parallel using rayon
        let results: Vec<Vec<String>> = chunks.par_iter()
            .with_max_len(1)
            .map(|chunk| processor(chunk))
            .collect();
        
        // Flatten results
        results.into_iter().flatten().collect()
    }
    
    /// Split text into chunks for parallel processing
    fn split_into_chunks(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::with_capacity(self.chunk_size);
        let mut current_size = 0;
        
        // Try to split at paragraph boundaries
        for paragraph in text.split("\n\n") {
            let paragraph_size = paragraph.len() + 2; // +2 for the "\n\n"
            
            if current_size + paragraph_size > self.chunk_size && !current_chunk.is_empty() {
                // Current chunk is full, push it and start a new one
                chunks.push(current_chunk);
                current_chunk = String::with_capacity(self.chunk_size);
                current_size = 0;
            }
            
            // Add paragraph to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(paragraph);
            current_size += paragraph_size;
        }
        
        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }
        
        // If we got no chunks (e.g., no paragraph breaks), fall back to line-based chunking
        if chunks.is_empty() {
            current_chunk = String::with_capacity(self.chunk_size);
            current_size = 0;
            
            for line in text.lines() {
                let line_size = line.len() + 1; // +1 for the newline
                
                if current_size + line_size > self.chunk_size && !current_chunk.is_empty() {
                    // Current chunk is full, push it and start a new one
                    chunks.push(current_chunk);
                    current_chunk = String::with_capacity(self.chunk_size);
                    current_size = 0;
                }
                
                // Add line to current chunk
                if !current_chunk.is_empty() {
                    current_chunk.push('\n');
                }
                current_chunk.push_str(line);
                current_size += line_size;
            }
            
            // Don't forget the last chunk
            if !current_chunk.is_empty() {
                chunks.push(current_chunk);
            }
        }
        
        // Final fallback: just split the text into equal-sized chunks
        if chunks.is_empty() {
            let mut i = 0;
            while i < text.len() {
                let end = (i + self.chunk_size).min(text.len());
                chunks.push(text[i..end].to_string());
                i = end;
            }
        }
        
        chunks
    }
}

/// Smart rate limiter for API calls
pub struct SmartRateLimiter {
    /// Maximum requests per minute
    max_rpm: u32,
    
    /// Tokens per minute (for token-based rate limits)
    tokens_per_minute: u32,
    
    /// Request history (timestamps)
    request_history: Arc<RwLock<VecDeque<Instant>>>,
    
    /// Token usage history (timestamp, tokens)
    token_history: Arc<RwLock<VecDeque<(Instant, u32)>>>,
    
    /// Semaphore for limiting concurrent requests
    semaphore: Arc<Semaphore>,
    
    /// Adaptive rate limiting
    adaptive: bool,
    
    /// Current adaptive limit factor (1.0 = normal, <1.0 = reduced)
    adaptive_factor: Arc<RwLock<f64>>,
    
    /// Last error timestamp
    last_error: Arc<RwLock<Option<Instant>>>,
}

impl SmartRateLimiter {
    /// Create a new smart rate limiter
    pub fn new(max_rpm: u32, max_concurrent: usize) -> Self {
        Self {
            max_rpm,
            tokens_per_minute: max_rpm * 1000, // Default token limit
            request_history: Arc::new(RwLock::new(VecDeque::with_capacity(max_rpm as usize))),
            token_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            adaptive: true,
            adaptive_factor: Arc::new(RwLock::new(1.0)),
            last_error: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Set tokens per minute limit
    pub fn with_tokens_per_minute(mut self, tokens_per_minute: u32) -> Self {
        self.tokens_per_minute = tokens_per_minute;
        self
    }
    
    /// Disable adaptive rate limiting
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }
    
    /// Acquire permission to make an API call
    pub async fn acquire(&self) -> Result<SmartRateLimitPermit> {
        // Non-recursive implementation to avoid the recursion in async fn issue
        let mut now = Instant::now();
        let max_wait = Duration::from_secs(300); // Max 5 minutes wait
        let start_time = now;
        
        // Try to get a permit with retries
        loop {
            // Acquire semaphore permit
            let permit = match self.semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(e) => return Err(AppError::ApiError(format!("Failed to acquire semaphore: {}", e))),
            };
            
            now = Instant::now();
            
            // Get rate limit history
            let mut history = self.request_history.write().await;
            
            // Remove old entries (older than 1 minute)
            while let Some(timestamp) = history.front() {
                if now.duration_since(*timestamp) > Duration::from_secs(60) {
                    history.pop_front();
                } else {
                    break;
                }
            }
            
            // Calculate effective rate limit based on adaptive factor
            let effective_max_rpm = if self.adaptive {
                let factor = *self.adaptive_factor.read().await;
                (self.max_rpm as f64 * factor) as u32
            } else {
                self.max_rpm
            };
            
            // Check if we're at the limit
            if history.len() >= effective_max_rpm as usize {
                // We're at the limit, check timing
                if let Some(oldest) = history.front() {
                    let elapsed = now.duration_since(*oldest);
                    let one_minute = Duration::from_secs(60);
                    
                    if elapsed < one_minute {
                        // Need to wait until oldest request is one minute old
                        let delay = one_minute - elapsed;
                        
                        // Release semaphore permit and history lock during delay
                        drop(permit);
                        drop(history);
                        
                        // Check if we've waited too long overall
                        if start_time.elapsed() > max_wait {
                            return Err(AppError::TimeoutError(300));
                        }
                        
                        debug!("Rate limit reached, waiting for {:?}", delay);
                        tokio::time::sleep(delay).await;
                        
                        // Try again in the next loop iteration
                        continue;
                    }
                }
            }
            
            // We're under the limit, add current request to history and return permit
            history.push_back(now);
            
            // Create permit
            return Ok(SmartRateLimitPermit {
                _semaphore_permit: permit,
                limiter: self.clone(),
                tokens: 0,
            });
        }
    }
    
    /// Record API call result
    pub async fn record_result(&self, success: bool, tokens: u32) {
        // Record token usage
        let now = Instant::now();
        
        if tokens > 0 {
            let mut token_history = self.token_history.write().await;
            
            // Remove old entries (older than 1 minute)
            while let Some((timestamp, _)) = token_history.front() {
                if now.duration_since(*timestamp) > Duration::from_secs(60) {
                    token_history.pop_front();
                } else {
                    break;
                }
            }
            
            // Add current usage
            token_history.push_back((now, tokens));
        }
        
        // Update adaptive factor based on errors
        if self.adaptive {
            let mut adaptive_factor = self.adaptive_factor.write().await;
            let mut last_error = self.last_error.write().await;
            
            if !success {
                // Record error
                *last_error = Some(now);
                
                // Reduce rate limit
                *adaptive_factor = (*adaptive_factor * 0.8).max(0.2);
                debug!("Reduced rate limit factor to {}", *adaptive_factor);
            } else if let Some(error_time) = *last_error {
                // Gradually recover if no errors for a while
                let since_error = now.duration_since(error_time);
                
                if since_error > Duration::from_secs(30) {
                    // No errors for 30+ seconds, start recovery
                    *adaptive_factor = (*adaptive_factor * 1.05).min(1.0);
                    
                    if *adaptive_factor >= 0.99 {
                        // Fully recovered
                        *adaptive_factor = 1.0;
                        *last_error = None;
                    }
                    
                    debug!("Increased rate limit factor to {}", *adaptive_factor);
                }
            }
        }
    }
    
    /// Check if token limit would be exceeded
    pub async fn check_token_limit(&self, tokens: u32) -> bool {
        let now = Instant::now();
        let token_history = self.token_history.read().await;
        
        // Sum tokens used in the last minute
        let mut total_tokens = tokens;
        
        for &(timestamp, used_tokens) in token_history.iter() {
            if now.duration_since(timestamp) <= Duration::from_secs(60) {
                total_tokens += used_tokens;
            }
        }
        
        // Calculate effective token limit based on adaptive factor
        let effective_token_limit = if self.adaptive {
            let factor = *self.adaptive_factor.read().await;
            (self.tokens_per_minute as f64 * factor) as u32
        } else {
            self.tokens_per_minute
        };
        
        total_tokens <= effective_token_limit
    }
    
    /// Wait until token limit allows the specified tokens
    pub async fn wait_for_token_capacity(&self, tokens: u32) -> Result<()> {
        let start = Instant::now();
        let max_wait = Duration::from_secs(300); // Max 5 minutes wait
        
        while !self.check_token_limit(tokens).await {
            if start.elapsed() > max_wait {
                return Err(AppError::TimeoutError(300));
            }
            
            // Wait a bit before checking again
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        Ok(())
    }
}

impl Clone for SmartRateLimiter {
    fn clone(&self) -> Self {
        Self {
            max_rpm: self.max_rpm,
            tokens_per_minute: self.tokens_per_minute,
            request_history: self.request_history.clone(),
            token_history: self.token_history.clone(),
            semaphore: self.semaphore.clone(),
            adaptive: self.adaptive,
            adaptive_factor: self.adaptive_factor.clone(),
            last_error: self.last_error.clone(),
        }
    }
}

/// Permit granted by the smart rate limiter
pub struct SmartRateLimitPermit {
    _semaphore_permit: tokio::sync::OwnedSemaphorePermit,
    limiter: SmartRateLimiter,
    tokens: u32,
}

impl SmartRateLimitPermit {
    /// Record token usage for this request
    pub fn record_tokens(&mut self, tokens: u32) {
        self.tokens = tokens;
    }
}

impl Drop for SmartRateLimitPermit {
    fn drop(&mut self) {
        // Create a task to record the result
        let limiter = self.limiter.clone();
        let tokens = self.tokens;
        
        tokio::spawn(async move {
            limiter.record_result(true, tokens).await;
        });
    }
}

/// Represents a cached compilation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedCompilation {
    code: String,
    created_at: i64, // Unix timestamp
}

/// Caching mechanism for compiled code
pub struct CompilationCache {
    cache: Arc<RwLock<HashMap<String, CachedCompilation>>>,
    max_size: usize,
}

impl CompilationCache {
    /// Create a new compilation cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::with_capacity(max_size))),
            max_size,
        }
    }
    
    /// Get a cached compilation
    pub async fn get(&self, key: &str) -> Option<CachedCompilation> {
        let cache = self.cache.read().await;
        cache.get(key).cloned()
    }
    
    /// Store a compilation in the cache
    pub async fn store(
        &self,
        key: String,
        _language: &str,
        source_code: &str,
        _compiled_code: Option<Vec<u8>>,
    ) -> Result<()> {
        let mut cache = self.cache.write().await;
        
        // Check if we need to evict entries
        if cache.len() >= self.max_size && !cache.contains_key(&key) {
            // Simple eviction: remove oldest entry
            if let Some(oldest) = cache.iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest);
            }
        }
        
        // Store the new entry
        cache.insert(key, CachedCompilation {
            code: source_code.to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
        });
        
        Ok(())
    }
    
    /// Store execution results for a cached compilation
    pub async fn store_execution_results(
        &self,
        key: &str,
        _input: &str,
        output: &str,
    ) -> Result<()> {
        let mut cache = self.cache.write().await;
        
        if let Some(entry) = cache.get_mut(key) {
            entry.code = output.to_string();
            Ok(())
        } else {
            Err(AppError::Unknown(format!("Compilation not found in cache: {}", key)))
        }
    }
    
    /// Save cache to disk
    pub async fn save_to_disk<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let cache = self.cache.read().await;
        
        let serialized = serde_json::to_string(&*cache)?;
        tokio::fs::write(path, serialized).await?;
        
        Ok(())
    }
    
    /// Load cache from disk
    pub async fn load_from_disk<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Ok(());
        }
        
        let serialized = tokio::fs::read_to_string(path).await?;
        let loaded: HashMap<String, CachedCompilation> = serde_json::from_str(&serialized)?;
        
        let mut cache = self.cache.write().await;
        *cache = loaded;
        
        Ok(())
    }
    
    /// Add a new entry to the cache
    pub async fn add(&self, key: &str, source_code: &str, _language: &str, _compiled_code: Option<Vec<u8>>) -> Result<()> {
        let mut cache = self.cache.write().await;
        
        // Check if we need to evict old entries
        if cache.len() >= self.max_size {
            // Simple eviction: remove oldest entry
            if let Some(oldest) = cache.iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest);
            }
        }
        
        // Store the new entry
        cache.insert(key.to_string(), CachedCompilation {
            code: source_code.to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
        });
        
        Ok(())
    }
    
    /// Store compilation output to cache
    pub async fn store_compilation(
        &self,
        key: &str,
        source_code: &str,
        _language: &str,
        _compiled_code: Option<Vec<u8>>,
    ) -> Result<()> {
        self.add(key, source_code, _language, _compiled_code).await
    }
    
    /// Add execution result
    pub async fn add_execution_result(
        &self,
        key: &str,
        _input: &str,
        output: &str,
    ) -> Result<()> {
        let mut cache = self.cache.write().await;
        
        if let Some(entry) = cache.get_mut(key) {
            entry.code = output.to_string();
            Ok(())
        } else {
            Err(AppError::Unknown(format!("Compilation not found in cache: {}", key)))
        }
    }
}