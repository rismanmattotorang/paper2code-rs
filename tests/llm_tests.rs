// tests/llm_tests.rs
use paper2code_rs::{
    llm::{
        client::{ClaudeClient, LlmClient, OpenAiClient},
        prompt::PromptBuilder,
    },
    config::AppConfig,
    error::AppError,
};
use anyhow::Result;
use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use super::*;
    
    // Skip this test by default since it requires API keys
    #[tokio::test]
    #[ignore]
    async fn test_claude_client_connection() {
        let config_path = PathBuf::from("config.toml");
        let config = AppConfig::load(&config_path).unwrap();
        
        if let Some(claude_config) = &config.claude {
            let client = ClaudeClient::new(claude_config.clone());
            assert!(client.is_ok(), "Failed to create Claude client");
            
            let client = client.unwrap();
            let prompt_builder = PromptBuilder::new()
                .with_template("Write a simple hello world function in Python.");
            
            let result = client.generate_code(&prompt_builder).await;
            
            assert!(result.is_ok(), "Failed to generate text from Claude: {:?}", result.err());
            let response = result.unwrap();
            assert!(!response.is_empty(), "Empty response from Claude");
            assert!(response.contains("def") || response.contains("print"), 
                   "Response doesn't contain expected Python code: {}", response);
        } else {
            println!("Skipping Claude test - no configuration found");
        }
    }
    
    // Skip this test by default since it requires API keys
    #[tokio::test]
    #[ignore]
    async fn test_openai_client_connection() {
        let config_path = PathBuf::from("config.toml");
        let config = AppConfig::load(&config_path).unwrap();
        
        if let Some(openai_config) = &config.openai {
            let client = OpenAiClient::new(openai_config.clone());
            assert!(client.is_ok(), "Failed to create OpenAI client");
            
            let client = client.unwrap();
            let prompt_builder = PromptBuilder::new()
                .with_template("Write a simple hello world function in Python.");
                
            let result = client.generate_code(&prompt_builder).await;
            
            assert!(result.is_ok(), "Failed to generate text from OpenAI: {:?}", result.err());
            let response = result.unwrap();
            assert!(!response.is_empty(), "Empty response from OpenAI");
            assert!(response.contains("def") || response.contains("print"), 
                   "Response doesn't contain expected Python code: {}", response);
        } else {
            println!("Skipping OpenAI test - no configuration found");
        }
    }
    
    #[test]
    fn test_prompt_builder() {
        let template = "Hello {{NAME}}";
        let builder = PromptBuilder::new()
            .with_template(template)
            .with_replacement("{{NAME}}", "World");
        
        let prompt = builder.build();
        assert_eq!(prompt, "Hello World");
    }
    
    #[tokio::test]
    async fn test_prompt_builder_complex() -> Result<()> {
        // Create a prompt builder
        let prompt = PromptBuilder::new()
            .with_template("Convert this {{LANGUAGE}} code: {{CODE}}")
            .with_language("python")
            .with_code_block("def hello(): print('Hello')");
        
        // Build the prompt
        let final_prompt = prompt.build();
        
        // Verify replacements
        assert!(final_prompt.contains("python"), "Should replace {{LANGUAGE}}");
        assert!(final_prompt.contains("def hello(): print('Hello')"), "Should replace {{CODE}}");
        
        Ok(())
    }
}
