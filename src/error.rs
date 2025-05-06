// src/error.rs
use std::fmt;
use thiserror::Error;

/// Type alias for Result with AppError as the error type
pub type Result<T> = std::result::Result<T, AppError>;

/// Central error type for the Paper2Code application
#[derive(Error, Debug)]
pub enum AppError {
    #[error("I/O error: {0}")]
    FileError(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("PDF parsing error: {0}")]
    PdfParseError(String),
    
    #[error("PDF extraction error: {0}")]
    PdfExtractError(String),
    
    #[error("Regex error: {0}")]
    RegexError(String),
    
    #[error("API error: {0}")]
    ApiError(String),
    
    #[error("Task join error: {0}")]
    TaskJoinError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("HTTP error: {status} - {message}")]
    HttpError {
        status: u16,
        message: String,
    },
    
    #[error("Timeout error: operation took longer than {0} seconds")]
    TimeoutError(u64),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),
    
    #[error("Language detection failed: {0}")]
    LanguageDetectionError(String),
    
    #[error("Code generation failed: {0}")]
    CodeGenerationError(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

// Add context to errors for better debugging
#[derive(Debug)]
pub struct ErrorContext {
    file: &'static str,
    line: u32,
    context: String,
    error: AppError,
}

impl ErrorContext {
    pub fn new(file: &'static str, line: u32, context: impl Into<String>, error: AppError) -> Self {
        Self {
            file,
            line,
            context: context.into(),
            error,
        }
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Error at {}:{} - {}: {}",
            self.file, self.line, self.context, self.error
        )
    }
}

impl std::error::Error for ErrorContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// Macros for error handling
#[macro_export]
macro_rules! with_context {
    ($error:expr, $context:expr) => {
        $crate::error::ErrorContext::new(file!(), line!(), $context, $error)
    };
}

// Add a traceable error type
pub trait TracedError: std::error::Error {
    fn with_trace(&self, file: &'static str, line: u32, context: String) -> ErrorContext;
}

impl TracedError for AppError {
    fn with_trace(&self, file: &'static str, line: u32, context: String) -> ErrorContext {
        ErrorContext::new(file, line, context, self.clone())
    }
}

impl Clone for AppError {
    fn clone(&self) -> Self {
        match self {
            Self::FileError(e) => Self::FileError(std::io::Error::new(e.kind(), e.to_string())),
            Self::ConfigError(s) => Self::ConfigError(s.clone()),
            Self::PdfParseError(s) => Self::PdfParseError(s.clone()),
            Self::PdfExtractError(s) => Self::PdfExtractError(s.clone()),
            Self::RegexError(s) => Self::RegexError(s.clone()),
            Self::ApiError(s) => Self::ApiError(s.clone()),
            Self::TaskJoinError(s) => Self::TaskJoinError(s.clone()),
            Self::SerializationError(e) => Self::SerializationError(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))),
            Self::HttpError { status, message } => Self::HttpError { 
                status: *status, 
                message: message.clone(),
            },
            Self::TimeoutError(s) => Self::TimeoutError(*s),
            Self::RateLimitError(s) => Self::RateLimitError(s.clone()),
            Self::LanguageDetectionError(s) => Self::LanguageDetectionError(s.clone()),
            Self::CodeGenerationError(s) => Self::CodeGenerationError(s.clone()),
            Self::Unknown(s) => Self::Unknown(s.clone()),
        }
    }
}

// Extension traits for better error handling
pub trait ResultExt<T, E> {
    fn context(self, context: impl Into<String>) -> std::result::Result<T, ErrorContext>;
    fn with_context<F>(self, f: F) -> std::result::Result<T, ErrorContext>
    where
        F: FnOnce() -> String;
}

impl<T, E> ResultExt<T, E> for std::result::Result<T, E>
where
    E: Into<AppError>,
{
    fn context(self, context: impl Into<String>) -> std::result::Result<T, ErrorContext> {
        self.map_err(|e| {
            let error: AppError = e.into();
            ErrorContext::new(file!(), line!(), context.into(), error)
        })
    }

    fn with_context<F>(self, f: F) -> std::result::Result<T, ErrorContext>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let error: AppError = e.into();
            ErrorContext::new(file!(), line!(), f(), error)
        })
    }
}

impl AppError {
    /// Create a new FileError from a std::io::Error
    pub fn from_io_error(error: std::io::Error) -> Self {
        Self::FileError(error)
    }
}