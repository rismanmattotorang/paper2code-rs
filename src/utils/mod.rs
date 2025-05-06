// src/utils/mod.rs

// Public submodules
pub mod concurrency;
pub mod error_recovery;
pub mod helpers;
pub mod performance;
pub mod performance_optimizations;
pub mod gamification;

// Re-export commonly used items
pub use concurrency::pdf::extract_text_parallel;
pub use error_recovery::pdf::extract_text_with_recovery;
pub use helpers::*;

// Selective re-exports instead of glob imports to avoid ambiguity
pub use performance::{measure_time, PerformanceCounters};
pub use performance_optimizations::{
    RateLimiter, 
    AdaptiveConcurrency,
    CompilationCache
};
pub use gamification::*; 