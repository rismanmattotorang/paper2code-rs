// src/config.rs
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use crate::error::AppError;
use crate::llm::client::{OpenAiConfig, ClaudeConfig};
use crate::llm::strategy::LlmStrategy;

#[derive(Debug, Deserialize, Serialize)]
pub struct AppConfig {
    pub openai: Option<OpenAiConfig>,
    pub claude: Option<ClaudeConfig>,
    pub llm_strategy: LlmStrategyConfig,
    pub pdf: PdfConfig,
    pub code_detection: CodeDetectionConfig,
    pub output: OutputConfig,
    pub prompt: PromptConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LlmStrategyConfig {
    pub strategy_type: String,
    pub code_detection_preference: Option<String>,
    pub code_improvement_preference: Option<String>,
    pub code_generation_preference: Option<String>,
    pub documentation_preference: Option<String>,
    pub bug_fixing_preference: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PdfConfig {
    pub chunk_size: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CodeDetectionConfig {
    pub min_confidence: f64,
    pub min_lines: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OutputConfig {
    pub path: String,
    pub overwrite: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PromptConfig {
    pub extraction_template: String,
    pub detection_template: String,
    pub documentation_template: String,
    pub bug_fixing_template: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            openai: Some(OpenAiConfig::default()),
            claude: Some(ClaudeConfig::default()),
            llm_strategy: LlmStrategyConfig {
                strategy_type: "adaptive".to_string(),
                code_detection_preference: Some("prefer_claude".to_string()),
                code_improvement_preference: Some("prefer_openai".to_string()),
                code_generation_preference: Some("prefer_openai".to_string()),
                documentation_preference: Some("prefer_claude".to_string()),
                bug_fixing_preference: Some("prefer_openai".to_string()),
            },
            pdf: PdfConfig {
                chunk_size: 1000,
            },
            code_detection: CodeDetectionConfig {
                min_confidence: 0.7,
                min_lines: 3,
            },
            output: OutputConfig {
                path: "output".to_string(),
                overwrite: false,
            },
            prompt: PromptConfig {
                extraction_template: crate::llm::prompt::PromptBuilder::default_code_extraction_template(),
                detection_template: crate::llm::prompt::PromptBuilder::code_detection_template(),
                documentation_template: crate::llm::prompt::PromptBuilder::documentation_template(),
                bug_fixing_template: crate::llm::prompt::PromptBuilder::bug_fixing_template(),
            },
        }
    }
}

impl AppConfig {
    /// Load configuration from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, AppError> {
        // If file doesn't exist, return default config
        if !path.as_ref().exists() {
            return Ok(Self::default());
        }
        
        let config_str = fs::read_to_string(path)
            .map_err(|e| AppError::ConfigError(format!("Failed to read config file: {}", e)))?;
            
        let config = toml::from_str(&config_str)
            .map_err(|e| AppError::ConfigError(format!("Failed to parse config: {}", e)))?;
            
        Ok(config)
    }
    
    /// Get LLM strategy from config
    pub fn get_llm_strategy(&self) -> LlmStrategy {
        match self.llm_strategy.strategy_type.as_str() {
            "openai_only" => LlmStrategy::OpenAiOnly,
            "claude_only" => LlmStrategy::ClaudeOnly,
            "openai_first" => LlmStrategy::OpenAiFirstClaudeSecond,
            "claude_first" => LlmStrategy::ClaudeFirstOpenAiSecond,
            "compare_and_merge" => LlmStrategy::CompareAndMerge,
            "adaptive" => {
                // Convert string preferences to enum values
                use crate::llm::strategy::AdaptivePreference;
                
                let code_detection = self.string_to_preference(
                    self.llm_strategy.code_detection_preference.as_deref(),
                    AdaptivePreference::prefer_claude(),
                );
                
                let code_improvement = self.string_to_preference(
                    self.llm_strategy.code_improvement_preference.as_deref(),
                    AdaptivePreference::prefer_open_ai(),
                );
                
                let code_generation = self.string_to_preference(
                    self.llm_strategy.code_generation_preference.as_deref(),
                    AdaptivePreference::prefer_open_ai(),
                );
                
                let documentation = self.string_to_preference(
                    self.llm_strategy.documentation_preference.as_deref(),
                    AdaptivePreference::prefer_claude(),
                );
                
                let bug_fixing = self.string_to_preference(
                    self.llm_strategy.bug_fixing_preference.as_deref(),
                    AdaptivePreference::prefer_open_ai(),
                );
                
                LlmStrategy::Adaptive {
                    code_detection,
                    code_improvement,
                    code_generation,
                    documentation,
                    bug_fixing,
                    performance_optimization: AdaptivePreference::prefer_open_ai(),
                    safety_enhancement: AdaptivePreference::prefer_claude(),
                    test_generation: AdaptivePreference::prefer_open_ai(),
                }
            },
            _ => LlmStrategy::default_strategy(),
        }
    }
    
    /// Convert string preference to enum
    fn string_to_preference(
        &self,
        pref: Option<&str>,
        default: crate::llm::strategy::AdaptivePreference,
    ) -> crate::llm::strategy::AdaptivePreference {
        use crate::llm::strategy::AdaptivePreference;
        
        match pref {
            Some("prefer_openai") => AdaptivePreference::prefer_open_ai(),
            Some("prefer_claude") => AdaptivePreference::prefer_claude(),
            Some("use_openai") => AdaptivePreference::use_open_ai(),
            Some("use_claude") => AdaptivePreference::use_claude(),
            _ => default,
        }
    }
}