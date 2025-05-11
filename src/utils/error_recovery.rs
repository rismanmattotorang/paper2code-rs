// src/utils/error_recovery.rs
use crate::error::AppError;
use crate::llm::client::LlmClient;
use crate::llm::prompt::PromptBuilder;
use std::future::Future;
use std::time::Duration;
use std::path::Path;
use crate::pdf::PdfExtractor;

/// Defines different recovery strategies for operations that might fail
pub enum RecoveryStrategy {
    /// Retry a failed operation with exponential backoff
    RetryWithBackoff {
        max_retries: usize,
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
    
    /// Fallback to an alternative operation if the primary one fails
    Fallback,
    
    /// Skip the operation and continue with the next one
    Skip,
    
    /// Fail immediately without recovery
    Fail,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::RetryWithBackoff {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 10000,
        }
    }
}

/// Retry an operation with exponential backoff
pub async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    max_retries: u32,
    initial_delay: Duration,
) -> Result<T, AppError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, AppError>>,
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
                
                retries += 1;
                tokio::time::sleep(delay).await;
                delay = delay * 2;
            }
        }
    }
}

/// Execute an operation with the specified recovery strategy
pub async fn execute_with_recovery<F, Fut, T, FF, FFut>(
    operation: F,
    fallback: Option<FF>,
    strategy: RecoveryStrategy,
    _operation_name: &str,
) -> Result<T, AppError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, AppError>>,
    FF: Fn() -> FFut,
    FFut: Future<Output = Result<T, AppError>>,
{
    match strategy {
        RecoveryStrategy::RetryWithBackoff { max_retries, base_delay_ms, max_delay_ms: _ } => {
            retry_with_backoff(
                operation,
                max_retries as u32,
                Duration::from_millis(base_delay_ms)
            ).await
        }
        
        RecoveryStrategy::Fallback => {
            match operation().await {
                Ok(result) => Ok(result),
                Err(e) => {
                    if let Some(fallback_op) = fallback {
                        fallback_op().await
                    } else {
                        Err(e)
                    }
                }
            }
        }
        
        RecoveryStrategy::Skip => {
            match operation().await {
                Ok(result) => Ok(result),
                Err(_) => Err(AppError::Unknown(format!("Operation skipped: {}", _operation_name)))
            }
        }
        
        RecoveryStrategy::Fail => {
            operation().await
        }
    }
}

/// Specialized recovery for API calls with rate limiting awareness
pub async fn api_call_with_recovery<F, Fut, T>(
    operation: F,
    _operation_name: &str,
) -> Result<T, AppError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, AppError>>,
{
    const MAX_RETRIES: u32 = 3;
    const INITIAL_DELAY: Duration = Duration::from_secs(1);
    const _MAX_DELAY: Duration = Duration::from_secs(60);  // Max 1 minute

    retry_with_backoff(
        || async {
            match operation().await {
                Ok(result) => Ok(result),
                Err(e) => {
                    if e.to_string().contains("rate limit") {
                        Err(e)
                    } else {
                        Err(AppError::ApiError(format!("API call failed: {}", e)))
                    }
                }
            }
        },
        MAX_RETRIES,
        INITIAL_DELAY,
    ).await
}

/// Apply error recovery strategies for PDF processing
pub mod pdf {
    use super::*;

    pub async fn extract_text_with_recovery<P: AsRef<Path>>(
        extractor: &PdfExtractor,
        path: P,
    ) -> Result<Vec<String>, AppError> {
        const MAX_RETRIES: u32 = 3;
        const INITIAL_DELAY: Duration = Duration::from_secs(1);
        
        retry_with_backoff(
            || async { extractor.extract_text_from_file(&path).await },
            MAX_RETRIES,
            INITIAL_DELAY,
        ).await
    }
}

/// Apply error recovery strategies for LLM API calls
pub mod llm {
    use super::*;

    pub async fn generate_code_with_recovery(
        client: &dyn LlmClient,
        prompt: &PromptBuilder,
    ) -> Result<String, AppError> {
        const MAX_RETRIES: u32 = 3;
        const INITIAL_DELAY: Duration = Duration::from_secs(1);
        
        retry_with_backoff(
            || async { client.generate_code(prompt).await },
            MAX_RETRIES,
            INITIAL_DELAY,
        ).await
    }
}

/// Extract text from PDF with error recovery
pub async fn extract_text_with_recovery(
    extractor: &crate::pdf::PdfExtractor,
    path: &std::path::Path,
) -> Result<Vec<String>, AppError> {
    const MAX_RETRIES: u32 = 3;
    const INITIAL_DELAY: Duration = Duration::from_secs(1);
    
    retry_with_backoff(
        || async { extractor.extract_text_from_file(path).await },
        MAX_RETRIES,
        INITIAL_DELAY,
    ).await
}

/// Generate code with error recovery
pub async fn generate_code_with_recovery(
    client: &dyn LlmClient,
    prompt: &PromptBuilder,
) -> Result<String, AppError> {
    const MAX_RETRIES: u32 = 3;
    const INITIAL_DELAY: Duration = Duration::from_secs(1);
    
    retry_with_backoff(
        || async { client.generate_code(prompt).await },
        MAX_RETRIES,
        INITIAL_DELAY,
    ).await
}

/// Process text with error recovery
pub async fn process_text_with_recovery(
    processor: &crate::text::TextProcessor,
    text: &str,
) -> Result<Vec<crate::text::CodeBlock>, AppError> {
    const MAX_RETRIES: u32 = 3;
    const INITIAL_DELAY: Duration = Duration::from_secs(1);
    
    retry_with_backoff(
        || async { processor.process_chunks(&[text.to_string()]) },
        MAX_RETRIES,
        INITIAL_DELAY,
    ).await
}

/// Write file with error recovery
pub async fn write_file_with_recovery(
    path: &std::path::Path,
    content: &str,
) -> Result<(), AppError> {
    const MAX_RETRIES: u32 = 3;
    const INITIAL_DELAY: Duration = Duration::from_secs(1);
    
    retry_with_backoff(
        || async { tokio::fs::write(path, content).await.map_err(AppError::from_io_error) },
        MAX_RETRIES,
        INITIAL_DELAY,
    ).await
}

/// Read file with error recovery
pub async fn read_file_with_recovery(
    path: &std::path::Path,
) -> Result<String, AppError> {
    const MAX_RETRIES: u32 = 3;
    const INITIAL_DELAY: Duration = Duration::from_secs(1);
    
    retry_with_backoff(
        || async { tokio::fs::read_to_string(path).await.map_err(AppError::from_io_error) },
        MAX_RETRIES,
        INITIAL_DELAY,
    ).await
}