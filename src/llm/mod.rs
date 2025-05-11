// src/llm/mod.rs

// Public submodules
pub mod client;
pub mod prompt;
pub mod strategy;
pub mod domain_prompts;

pub use domain_prompts::DomainPromptLibrary;

// Re-export commonly used items
pub use client::{LlmClient, ClaudeClient, OpenAiClient, MultiLlmClient};
pub use prompt::PromptBuilder;
pub use strategy::{LlmStrategy, TaskType};