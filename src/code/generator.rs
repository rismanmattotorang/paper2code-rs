// src/code/generator.rs
use crate::error::AppError;
use crate::llm::{
    client::LlmClient,
    prompt::PromptBuilder,
    strategy::{LlmStrategy, TaskType},
};
use crate::text::code_detector::CodeBlock;
use std::path::Path;
use tokio::fs;
use tracing::{info, warn, debug};
use uuid::Uuid;

/// Handles generation of executable code from code blocks
pub struct CodeGenerator {
    client: Box<dyn LlmClient>,
    output_dir: String,
    strategy: LlmStrategy,
}

impl CodeGenerator {
    pub fn new(client: Box<dyn LlmClient>, output_dir: String, strategy: LlmStrategy) -> Self {
        Self {
            client,
            output_dir,
            strategy,
        }
    }
    
    /// Get a reference to the LLM client
    pub fn client(&self) -> &dyn LlmClient {
        self.client.as_ref()
    }
    
    /// Get a reference to the strategy
    pub fn strategy(&self) -> &LlmStrategy {
        &self.strategy
    }
    
    /// Generate executable code from detected code blocks
    pub async fn generate_from_blocks(
        &self,
        blocks: &[CodeBlock],
        prompt_template: &str,
        target_language: Option<&str>,
    ) -> Result<Vec<String>, AppError> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(&self.output_dir).await
            .map_err(AppError::from_io_error)?;
        
        // Process blocks in parallel with controlled concurrency
        let mut file_paths = Vec::new();
        
        for (i, block) in blocks.iter().enumerate() {
            debug!("Processing code block {}/{}", i + 1, blocks.len());
            
            // Get the block's language or use the target language
            let language = if let Some(lang) = &block.language {
                lang.as_str()
            } else if let Some(lang) = target_language {
                lang
            } else {
                "unknown"
            };
            
            // 1. First pass: Generate executable code
            let prompt = PromptBuilder::new()
                .with_template(prompt_template)
                .with_code_block(&block.content)
                .with_language(&language);
            
            let code = match self.client.generate_code(&prompt).await {
                Ok(code) => code,
                Err(e) => {
                    warn!("Failed to generate code for block {}: {}", i, e);
                    continue;
                }
            };
            
            // 2. Second pass: Fix any bugs in the generated code
            let bug_fixing_template = "You are a skilled programmer tasked with fixing bugs in code extracted from a research paper.\n\
                  The following code was generated from a research paper code snippet, but may contain bugs or issues:\n\n\
                  ```{{LANGUAGE}}\n{{CODE}}\n```\n\n\
                  Please fix any bugs or issues in this code while preserving the original functionality.\n\
                  Ensure it's properly structured, handles edge cases, and follows best practices.\n\
                  Return only the fixed, executable code.";
                  
            let mut bug_fixing_builder = PromptBuilder::new();
            bug_fixing_builder = bug_fixing_builder.with_template(bug_fixing_template);
            bug_fixing_builder = bug_fixing_builder.with_code_block(&code);
            bug_fixing_builder = bug_fixing_builder.with_language(language);
            
            let improved_code = match self.client.generate_code(&bug_fixing_builder).await {
                Ok(code) => code,
                Err(e) => {
                    warn!("Failed to improve code for block {}: {}", i, e);
                    code // Use original code if improvement fails
                }
            };
            
            // 3. Save generated code to file
            match self.save_code(block, &improved_code, i).await {
                Ok(file_path) => {
                    file_paths.push(file_path);
                },
                Err(e) => {
                    warn!("Failed to save code for block {}: {}", i, e);
                }
            }
        }
        
        Ok(file_paths)
    }
    
    /// Generate documentation for code
    pub async fn generate_documentation(
        &self,
        code: &str,
        language: &str,
    ) -> Result<String, AppError> {
        let doc_template = "You are a skilled technical writer tasked with documenting code from a research paper.\n\
              The following code was extracted from a research paper:\n\n\
              ```{{LANGUAGE}}\n{{CODE}}\n```\n\n\
              Please add comprehensive documentation to this code following these guidelines:\n\
              1. Add a detailed header comment explaining the purpose and context\n\
              2. Document each function with clear descriptions of parameters and return values\n\
              3. Explain complex algorithms or data structures\n\
              4. Add inline comments for non-obvious code sections\n\
              5. Follow documentation best practices for {{LANGUAGE}}\n\
              6. Do not change the functionality of the code\n\n\
              Return only the documented code.";
        
        let mut prompt_builder = PromptBuilder::new();
        prompt_builder = prompt_builder.with_template(doc_template);
        prompt_builder = prompt_builder.with_code_block(code);
        prompt_builder = prompt_builder.with_language(language);
        
        self.client.generate_code(&prompt_builder).await
    }
    
    /// Save generated code to file
    async fn save_code(&self, block: &CodeBlock, code: &str, index: usize) -> Result<String, AppError> {
        // Generate a unique filename with appropriate extension
        let language = block.language.as_deref().unwrap_or("txt");
        let extension = get_file_extension(language);
        
        let uuid = Uuid::new_v4();
        let filename = format!("code_{}_{}.{}", language, uuid.to_string().split('-').next().unwrap(), extension);
        let file_path = Path::new(&self.output_dir).join(&filename);
        
        // Write code to file
        fs::write(&file_path, code).await
            .map_err(AppError::from_io_error)?;
            
        info!("Saved code block {} to {}", index, file_path.display());
        
        Ok(file_path.to_string_lossy().to_string())
    }
}

/// Get file extension for a language
fn get_file_extension(language: &str) -> &str {
    match language.to_lowercase().as_str() {
        "python" | "py" => "py",
        "rust" | "rs" => "rs",
        "java" => "java",
        "cpp" | "c++" => "cpp",
        "c" => "c",
        "javascript" | "js" => "js",
        "typescript" | "ts" => "ts",
        "go" => "go",
        "ruby" | "rb" => "rb",
        "php" => "php",
        "shell" | "bash" | "sh" => "sh",
        "r" => "r",
        "matlab" | "m" => "m",
        "swift" => "swift",
        "kotlin" | "kt" => "kt",
        "scala" => "scala",
        "perl" | "pl" => "pl",
        "haskell" | "hs" => "hs",
        "julia" | "jl" => "jl",
        "lua" => "lua",
        _ => "txt",
    }
}

// Extension trait for LlmClient
#[async_trait::async_trait]
pub trait LlmClientExt {
    /// Generate code for a specific task
    async fn generate_code_for_task(
        &self,
        prompt: &str,
        task_type: TaskType,
        language: Option<&str>,
    ) -> Result<String, AppError>;
}

#[async_trait::async_trait]
impl LlmClientExt for Box<dyn LlmClient> {
    async fn generate_code_for_task(
        &self,
        prompt: &str,
        task_type: TaskType,
        language: Option<&str>,
    ) -> Result<String, AppError> {
        // Configure token limit based on task type
        let _max_tokens = match task_type {
            TaskType::CodeGeneration => 4000,
            TaskType::CodeImprovement => 4000,
            TaskType::BugFixing => 3000,
            TaskType::Documentation => 5000,
            _ => 2000,
        };
        
        // Create a prompt builder with the given text
        let prompt_builder = PromptBuilder::new()
            .with_template(prompt)
            .with_language(language.unwrap_or("unknown"));
            
        // Use the standard generate_code method
        self.generate_code(&prompt_builder).await
    }
} 