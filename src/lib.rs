// src/lib.rs
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]

//! Paper2Code: Extract and generate code from research papers using multiple LLMs
//!
//! This library provides tools to extract, identify, and generate executable code
//! from research papers using various LLM providers.

// Public modules
pub mod cli;
pub mod config;
pub mod error;
pub mod llm;
pub mod pdf;
pub mod text;
pub mod code;
pub mod utils;

// Re-export commonly used types
pub use cli::{Cli, ExtractArgs, ConfigArgs, TestArgs};
pub use config::AppConfig;
pub use error::{AppError, Result as AppResult};
pub use llm::{
    client::{LlmClient, ClaudeClient, OpenAiClient, MultiLlmClient},
    strategy::{LlmStrategy, TaskType},
    prompt::PromptBuilder,
    DomainPromptLibrary,
};
pub use pdf::PdfExtractor;
pub use text::{TextProcessor, code_detector::{CodeBlock, CodeDetector}, domain_detector::{DomainDetector, ComputationalDomain}};
pub use code::{CodeGenerator, CodeWriter, DomainAwareCodeGenerator};