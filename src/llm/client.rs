// src/llm/client.rs
use crate::error::AppError;
use crate::llm::prompt::PromptBuilder;
use crate::llm::strategy::{LlmClientPreference, LlmStrategy, TaskType};
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tracing::{info, warn, debug};

/// Trait for LLM clients
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// Generate code from a prompt
    async fn generate_code(&self, prompt: &PromptBuilder) -> Result<String, AppError>;
    
    /// Get the name of this LLM client
    fn name(&self) -> &str;
    
    /// Clone the client
    fn box_clone(&self) -> Box<dyn LlmClient>;
}

impl Clone for Box<dyn LlmClient> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// Configuration for OpenAI GPT
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_seconds: u64,
    pub max_concurrent_requests: usize,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "gpt-4-turbo".to_string(),
            max_tokens: 4096,
            temperature: 0.2,
            timeout_seconds: 120,
            max_concurrent_requests: 3,
        }
    }
}

/// Configuration for Claude API
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClaudeConfig {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_seconds: u64,
    pub max_concurrent_requests: usize,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 4096,
            temperature: 0.2,
            timeout_seconds: 120,
            max_concurrent_requests: 3,
        }
    }
}

/// OpenAI API Client
pub struct OpenAiClient {
    config: OpenAiConfig,
    http_client: Client,
    semaphore: Arc<Semaphore>,
}

impl OpenAiClient {
    pub fn new(config: OpenAiConfig) -> Result<Self, AppError> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            "Authorization",
            header::HeaderValue::from_str(&format!("Bearer {}", config.api_key))
                .map_err(|e| AppError::ApiError(format!("Invalid API key: {}", e)))?,
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let http_client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .default_headers(headers)
            .build()
            .map_err(|e| AppError::ApiError(format!("Failed to build HTTP client: {}", e)))?;

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

        Ok(Self {
            config,
            http_client,
            semaphore,
        })
    }
}

#[async_trait::async_trait]
impl LlmClient for OpenAiClient {
    async fn generate_code(&self, prompt: &PromptBuilder) -> Result<String, AppError> {
        // Acquire a permit from the semaphore to limit concurrent requests
        let _permit = self.semaphore.clone().acquire_owned().await
            .map_err(|e| AppError::ApiError(format!("Failed to acquire semaphore: {}", e)))?;
            
        info!("Generating code with OpenAI...");
        
        // Construct the request JSON
        let request_body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert programmer who helps extract and improve code from research papers. \
                               Your task is to convert code snippets from papers into fully working, executable code. \
                               Only output the code itself, with appropriate comments and documentation."
                },
                {
                    "role": "user",
                    "content": prompt.build()
                }
            ]
        });
        
        debug!("Sending request to OpenAI: {}", prompt.build());
        
        // Send request to OpenAI API
        let response = self.http_client
            .post("https://api.openai.com/v1/chat/completions")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AppError::ApiError(format!("OpenAI API request failed: {}", e)))?;
            
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await
                .unwrap_or_else(|_| String::from("Failed to read error response"));
                
            return Err(AppError::HttpError {
                status: status.as_u16(),
                message: format!("OpenAI API error: {}", error_text),
            });
        }
        
        // Parse the response
        let response_json: Value = response.json().await
            .map_err(|e| AppError::ApiError(format!("Failed to parse OpenAI response: {}", e)))?;
            
        // Extract the generated text
        let generated_text = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| AppError::ApiError("Invalid response format from OpenAI".to_string()))?
            .trim()
            .to_string();
            
        info!("Successfully generated code with OpenAI");
        
        Ok(generated_text)
    }
    
    fn name(&self) -> &str {
        "OpenAI"
    }
    
    fn box_clone(&self) -> Box<dyn LlmClient> {
        Box::new(Self {
            config: self.config.clone(),
            http_client: self.http_client.clone(),
            semaphore: self.semaphore.clone(),
        })
    }
}

/// Claude API Client
#[derive(Clone)]
pub struct ClaudeClient {
    config: ClaudeConfig,
    http_client: Client,
    semaphore: Arc<Semaphore>,
}

impl ClaudeClient {
    pub fn new(config: ClaudeConfig) -> Result<Self, AppError> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            header::HeaderValue::from_str(&config.api_key)
                .map_err(|e| AppError::ApiError(format!("Invalid API key: {}", e)))?,
        );
        headers.insert(
            "anthropic-version",
            header::HeaderValue::from_static("2023-06-01"),
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let http_client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .default_headers(headers)
            .build()
            .map_err(|e| AppError::ApiError(format!("Failed to build HTTP client: {}", e)))?;

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

        Ok(Self {
            config,
            http_client,
            semaphore,
        })
    }
}

#[async_trait::async_trait]
impl LlmClient for ClaudeClient {
    async fn generate_code(&self, prompt: &PromptBuilder) -> Result<String, AppError> {
        // Acquire a permit from the semaphore to limit concurrent requests
        let _permit = self.semaphore.clone().acquire_owned().await
            .map_err(|e| AppError::ApiError(format!("Failed to acquire semaphore: {}", e)))?;
            
        info!("Generating code with Claude...");
        
        // Construct the request JSON directly
        let request_body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": "You are an expert programmer who helps extract and improve code from research papers. \
                      Your task is to convert code snippets from papers into fully working, executable code. \
                      Only output the code itself, with appropriate comments and documentation.",
            "messages": [
                {
                    "role": "user",
                    "content": prompt.build()
                }
            ]
        });
        
        debug!("Sending request to Claude: {}", prompt.build());

        // Send the request directly
        let response = self.http_client.post("https://api.anthropic.com/v1/messages")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AppError::ApiError(format!("Failed to send request to Claude API: {}", e)))?;
        
        // Check for errors
        if !response.status().is_success() {
            let error_text = response.text().await
                .map_err(|e| AppError::ApiError(format!("Failed to read error response: {}", e)))?;
            return Err(AppError::ApiError(format!("Claude API error: {}", error_text)));
        }
        
        // Parse the response
        let response_json: Value = response.json()
            .await
            .map_err(|e| AppError::ApiError(format!("Failed to parse Claude API response: {}", e)))?;
        
        info!("Received response from Claude");
        
        // Extract text content
        let content = &response_json["content"];
        let text = if content.is_array() {
            // Find text content in the array
            content.as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(|item| {
                    if item["type"] == "text" {
                        Some(item["text"].as_str().unwrap_or(""))
                    } else {
                        None
                    }
                })
                .collect::<Vec<&str>>()
                .join("")
        } else {
            // Fallback if structure is different
            response_json["content"].as_str()
                .unwrap_or("")
                .to_string()
        };
        
        Ok(text)
    }
    
    fn name(&self) -> &str {
        "Claude"
    }
    
    fn box_clone(&self) -> Box<dyn LlmClient> {
        Box::new(Self {
            config: self.config.clone(),
            http_client: self.http_client.clone(),
            semaphore: self.semaphore.clone(),
        })
    }
}

/// Client that manages multiple LLM providers
pub struct MultiLlmClient {
    openai: Option<Box<dyn LlmClient>>,
    claude: Option<Box<dyn LlmClient>>,
    strategy: LlmStrategy,
}

impl MultiLlmClient {
    pub fn new(
        openai: Option<Box<dyn LlmClient>>,
        claude: Option<Box<dyn LlmClient>>,
        strategy: LlmStrategy,
    ) -> Self {
        Self {
            openai,
            claude,
            strategy,
        }
    }
    
    /// Create a new client with default strategy
    pub fn with_default_strategy(
        openai: Option<Box<dyn LlmClient>>,
        claude: Option<Box<dyn LlmClient>>,
    ) -> Self {
        Self::new(openai, claude, LlmStrategy::default_strategy())
    }
    
    /// Set strategy
    pub fn with_strategy(mut self, strategy: LlmStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    
    /// Generate code using the appropriate LLM based on context
    pub async fn generate_code_for_task(
        &self,
        prompt: &PromptBuilder,
        task_type: TaskType,
        language: Option<&str>,
    ) -> Result<String, AppError> {
        // Use the strategy's get_client_for_task method instead of reimplementing it
        let preference = self.strategy.get_client_for_task(&task_type, language);
        
        // Route to appropriate implementation based on preference
        match preference {
            LlmClientPreference::UseOpenAi => self.generate_with_openai(prompt).await,
            LlmClientPreference::UseClaude => self.generate_with_claude(prompt).await,
            
            LlmClientPreference::PreferOpenAi => {
                if self.openai.is_some() {
                    self.generate_with_openai(prompt).await
                } else if self.claude.is_some() {
                    info!("OpenAI preferred but not available, falling back to Claude");
                    self.generate_with_claude(prompt).await
                } else {
                    Err(AppError::ApiError("No LLM clients configured".to_string()))
                }
            },
            
            LlmClientPreference::PreferClaude => {
                if self.claude.is_some() {
                    self.generate_with_claude(prompt).await
                } else if self.openai.is_some() {
                    info!("Claude preferred but not available, falling back to OpenAI");
                    self.generate_with_openai(prompt).await
                } else {
                    Err(AppError::ApiError("No LLM clients configured".to_string()))
                }
            },
            
            LlmClientPreference::OpenAiFirst => {
                self.generate_with_refinement(prompt, true).await
            },
            
            LlmClientPreference::ClaudeFirst => {
                self.generate_with_refinement(prompt, false).await
            },
            
            LlmClientPreference::Both => {
                self.generate_with_both(prompt).await
            },
        }
    }
    
    /// Generate code with OpenAI
    async fn generate_with_openai(&self, prompt: &PromptBuilder) -> Result<String, AppError> {
        if let Some(client) = &self.openai {
            client.generate_code(prompt).await
        } else {
            Err(AppError::ApiError("OpenAI client not configured".to_string()))
        }
    }
    
    /// Generate code with Claude
    async fn generate_with_claude(&self, prompt: &PromptBuilder) -> Result<String, AppError> {
        if let Some(client) = &self.claude {
            client.generate_code(prompt).await
        } else {
            Err(AppError::ApiError("Claude client not configured".to_string()))
        }
    }
    
    /// Generate with one LLM, then refine with the other
    async fn generate_with_refinement(
        &self, 
        prompt: &PromptBuilder,
        openai_first: bool,
    ) -> Result<String, AppError> {
        // First generation with the primary LLM
        let first_result = if openai_first {
            self.generate_with_openai(prompt).await
        } else {
            self.generate_with_claude(prompt).await
        };
        
        match first_result {
            Ok(code) => {
                // Now refine with the secondary LLM
                let refinement_prompt = PromptBuilder::new()
                    .with_template("You are an expert programmer. Please review and improve the following code:\n\n```\n{{CODE}}\n```\n\nProvide a refined version of this code, ensuring it's correctly structured, has proper error handling, and follows best practices.")
                    .with_replacement("{{CODE}}", &code);
                
                let second_result = if openai_first {
                    // OpenAI first, Claude second
                    if let Some(client) = &self.claude {
                        client.generate_code(&refinement_prompt).await
                    } else {
                        return Ok(code); // Just return first result if second LLM not available
                    }
                } else {
                    // Claude first, OpenAI second
                    if let Some(client) = &self.openai {
                        client.generate_code(&refinement_prompt).await
                    } else {
                        return Ok(code); // Just return first result if second LLM not available
                    }
                };
                
                // Return refined code if successful, otherwise return the first result
                match second_result {
                    Ok(refined) => Ok(refined),
                    Err(e) => {
                        warn!("Refinement failed: {}, returning initial code", e);
                        Ok(code)
                    }
                }
            },
            Err(e) => Err(e),
        }
    }
    
    /// Generate with both LLMs and merge results
    async fn generate_with_both(&self, prompt: &PromptBuilder) -> Result<String, AppError> {
        // Check if both clients are available
        if self.openai.is_none() && self.claude.is_none() {
            return Err(AppError::ApiError("No LLM clients configured".to_string()));
        }
        
        // If only one client is available, just use that
        if self.openai.is_none() {
            return self.generate_with_claude(prompt).await;
        }
        if self.claude.is_none() {
            return self.generate_with_openai(prompt).await;
        }
        
        // Generate with both in parallel
        let openai_future = self.generate_with_openai(prompt);
        let claude_future = self.generate_with_claude(prompt);
        
        let (openai_result, claude_result) = futures::join!(openai_future, claude_future);
        
        // Process results
        match (openai_result, claude_result) {
            (Ok(openai_code), Ok(claude_code)) => {
                // Both succeeded, create a merged version
                let merge_prompt = PromptBuilder::new()
                    .with_template("You have two versions of the same code. Please analyze both and create a single optimal version that combines the best elements of each:\n\nVersion 1 (OpenAI):\n```\n{{CODE1}}\n```\n\nVersion 2 (Claude):\n```\n{{CODE2}}\n```\n\nProvide a single optimized implementation.")
                    .with_replacement("{{CODE1}}", &openai_code)
                    .with_replacement("{{CODE2}}", &claude_code);
                
                // Use Claude for merging if available, otherwise OpenAI
                if let Some(client) = &self.claude {
                    match client.generate_code(&merge_prompt).await {
                        Ok(merged) => Ok(merged),
                        Err(e) => {
                            warn!("Failed to merge results: {}, using Claude's code", e);
                            Ok(claude_code)
                        }
                    }
                } else {
                    // This shouldn't happen due to checks above, but just in case
                    Ok(claude_code)
                }
            },
            (Ok(openai_code), Err(_)) => {
                // Only OpenAI succeeded
                Ok(openai_code)
            },
            (Err(_), Ok(claude_code)) => {
                // Only Claude succeeded
                Ok(claude_code)
            },
            (Err(e1), Err(_)) => {
                // Both failed
                Err(e1)
            }
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for MultiLlmClient {
    async fn generate_code(&self, prompt: &PromptBuilder) -> Result<String, AppError> {
        self.generate_code_for_task(prompt, TaskType::CodeGeneration, None).await
    }
    
    fn name(&self) -> &str {
        "Multi-LLM"
    }
    
    fn box_clone(&self) -> Box<dyn LlmClient> {
        Box::new(Self {
            openai: self.openai.clone(),
            claude: self.claude.clone(),
            strategy: self.strategy.clone(),
        })
    }
}