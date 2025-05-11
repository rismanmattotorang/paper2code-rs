// src/llm/strategy.rs
use std::fmt;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::{AppError, Result};
use crate::llm::client::LlmClient;
use crate::llm::prompt::PromptBuilder;
use crate::text::ComputationalDomain;

/// Defines different types of LLM tasks in the code extraction pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Initial code block detection from extracted text
    CodeDetection,
    
    /// Improving detected code blocks (fixing indentation, etc.)
    CodeImprovement,
    
    /// Converting code snippets to executable code
    CodeGeneration,
    
    /// Domain-specific code generation and optimization
    DomainSpecificGeneration(ComputationalDomain),
    
    /// Adding documentation to generated code
    Documentation,
    
    /// Fixing bugs in generated code
    BugFixing,
    
    /// Analyzing and optimizing code performance
    PerformanceOptimization,
    
    /// Enhancing code safety and error handling
    SafetyEnhancement,
    
    /// Adding tests to verify code correctness
    TestGeneration,
    
    /// Detecting the computational domain of a paper
    DomainDetection,
}

impl fmt::Display for TaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CodeDetection => write!(f, "Code Detection"),
            Self::CodeImprovement => write!(f, "Code Improvement"),
            Self::CodeGeneration => write!(f, "Code Generation"),
            Self::DomainSpecificGeneration(domain) => write!(f, "Domain-Specific Generation ({})", domain),
            Self::Documentation => write!(f, "Documentation"),
            Self::BugFixing => write!(f, "Bug Fixing"),
            Self::PerformanceOptimization => write!(f, "Performance Optimization"),
            Self::SafetyEnhancement => write!(f, "Safety Enhancement"),
            Self::TestGeneration => write!(f, "Test Generation"),
            Self::DomainDetection => write!(f, "Domain Detection"),
        }
    }
}

/// Strategy for selecting LLM based on task type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmStrategy {
    /// Always use OpenAI
    OpenAiOnly,
    /// Always use Claude
    ClaudeOnly,
    /// Try OpenAI first, fall back to Claude
    OpenAiFirstClaudeSecond,
    /// Try Claude first, fall back to OpenAI
    ClaudeFirstOpenAiSecond,
    /// Compare results from both and merge
    CompareAndMerge,
    /// Adaptive strategy based on task type
    Adaptive {
        code_detection: AdaptivePreference,
        code_improvement: AdaptivePreference,
        code_generation: AdaptivePreference,
        documentation: AdaptivePreference,
        bug_fixing: AdaptivePreference,
        performance_optimization: AdaptivePreference,
        safety_enhancement: AdaptivePreference,
        test_generation: AdaptivePreference,
    },
}

/// Preference for adaptive strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePreference {
    /// Weight for OpenAI (0.0 to 1.0)
    pub openai_weight: f64,
    /// Weight for Claude (0.0 to 1.0)
    pub claude_weight: f64,
}

impl Default for AdaptivePreference {
    fn default() -> Self {
        Self {
            openai_weight: 0.5,
            claude_weight: 0.5,
}
    }
}

impl AdaptivePreference {
    /// Create a preference that favors OpenAI
    pub fn prefer_open_ai() -> Self {
        Self {
            openai_weight: 0.7,
            claude_weight: 0.3,
        }
    }
    
    /// Create a preference that favors Claude
    pub fn prefer_claude() -> Self {
        Self {
            openai_weight: 0.3,
            claude_weight: 0.7,
        }
    }
    
    /// Create a preference that only uses OpenAI
    pub fn use_open_ai() -> Self {
        Self {
            openai_weight: 1.0,
            claude_weight: 0.0,
        }
    }
    
    /// Create a preference that only uses Claude
    pub fn use_claude() -> Self {
        Self {
            openai_weight: 0.0,
            claude_weight: 1.0,
        }
    }
}

/// Enum indicating which LLM client(s) to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmClientPreference {
    /// Use only OpenAI
    UseOpenAi,
    
    /// Use only Claude
    UseClaude,
    
    /// Prefer OpenAI, but fall back to Claude if needed
    PreferOpenAi,
    
    /// Prefer Claude, but fall back to OpenAI if needed
    PreferClaude,
    
    /// Use OpenAI first, then Claude for refinement
    OpenAiFirst,
    
    /// Use Claude first, then OpenAI for refinement
    ClaudeFirst,
    
    /// Use both and combine results
    Both,
}

impl LlmStrategy {
    /// Get the preferred LLM for a task
    pub fn get_preferred_llm(&self, task: &TaskType) -> &'static str {
        match self {
            LlmStrategy::OpenAiOnly => "openai",
            LlmStrategy::ClaudeOnly => "claude",
            LlmStrategy::OpenAiFirstClaudeSecond => "openai",
            LlmStrategy::ClaudeFirstOpenAiSecond => "claude",
            LlmStrategy::CompareAndMerge => "both",
            LlmStrategy::Adaptive {
            code_detection,
            code_improvement,
            code_generation,
            documentation,
            bug_fixing,
            performance_optimization,
            safety_enhancement,
            test_generation,
            } => {
            let preference = match task {
                TaskType::CodeDetection => code_detection,
                TaskType::CodeImprovement => code_improvement,
                TaskType::CodeGeneration => code_generation,
                TaskType::DomainSpecificGeneration(_) => code_generation, // Use code generation settings for domain-specific tasks
                TaskType::Documentation => documentation,
                TaskType::BugFixing => bug_fixing,
                TaskType::PerformanceOptimization => performance_optimization,
                TaskType::SafetyEnhancement => safety_enhancement,
                TaskType::TestGeneration => test_generation,
                TaskType::DomainDetection => code_detection, // Use code detection settings for domain detection
            };
            
                if preference.openai_weight > preference.claude_weight {
                    "openai"
                } else if preference.claude_weight > preference.openai_weight {
                    "claude"
                } else {
                    "both"
                }
            }
        }
    }
    
    /// Provides the default strategy
    pub fn default_strategy() -> Self {
        Self::Adaptive {
            code_detection: AdaptivePreference::prefer_claude(),
            code_improvement: AdaptivePreference::prefer_open_ai(),
            code_generation: AdaptivePreference::prefer_open_ai(),
            documentation: AdaptivePreference::prefer_claude(),
            bug_fixing: AdaptivePreference::prefer_open_ai(),
            performance_optimization: AdaptivePreference::prefer_open_ai(),
            safety_enhancement: AdaptivePreference::prefer_claude(),
            test_generation: AdaptivePreference::prefer_open_ai(),
        }
    }
    
    /// Get the fallback LLM for a task
    pub fn get_fallback_llm(&self, _task: &TaskType) -> Option<&'static str> {
        match self {
            LlmStrategy::OpenAiOnly | LlmStrategy::ClaudeOnly => None,
            LlmStrategy::OpenAiFirstClaudeSecond => Some("claude"),
            LlmStrategy::ClaudeFirstOpenAiSecond => Some("openai"),
            LlmStrategy::CompareAndMerge => None,
            LlmStrategy::Adaptive { .. } => None,
    }
}

    /// Check if both LLMs should be used
    pub fn should_use_both(&self, task: &TaskType) -> bool {
        match self {
            LlmStrategy::CompareAndMerge => true,
            LlmStrategy::Adaptive {
                code_detection,
                code_improvement,
                code_generation,
                documentation,
                bug_fixing,
                performance_optimization,
                safety_enhancement,
                test_generation,
            } => {
                let preference = match task {
                    TaskType::CodeDetection => code_detection,
                    TaskType::CodeImprovement => code_improvement,
                    TaskType::CodeGeneration => code_generation,
                    TaskType::DomainSpecificGeneration(_) => code_generation, // Use code generation settings
                    TaskType::Documentation => documentation,
                    TaskType::BugFixing => bug_fixing,
                    TaskType::PerformanceOptimization => performance_optimization,
                    TaskType::SafetyEnhancement => safety_enhancement,
                    TaskType::TestGeneration => test_generation,
                    TaskType::DomainDetection => code_detection, // Use code detection settings
                };
                
                (preference.openai_weight - preference.claude_weight).abs() < 0.1
            }
            _ => false,
        }
    }

    /// Get the client to use for a specific task and language
    pub fn get_client_for_task(&self, task_type: &TaskType, language: Option<&str>) -> LlmClientPreference {
        match self {
            LlmStrategy::OpenAiOnly => LlmClientPreference::UseOpenAi,
            LlmStrategy::ClaudeOnly => LlmClientPreference::UseClaude,
            LlmStrategy::OpenAiFirstClaudeSecond => LlmClientPreference::PreferOpenAi,
            LlmStrategy::ClaudeFirstOpenAiSecond => LlmClientPreference::PreferClaude,
            LlmStrategy::CompareAndMerge => LlmClientPreference::Both,
            LlmStrategy::Adaptive {
                code_detection,
                code_improvement,
                code_generation,
                documentation,
                bug_fixing,
                performance_optimization,
                safety_enhancement,
                test_generation,
            } => {
                let preference = match task_type {
                    TaskType::CodeDetection => code_detection,
                    TaskType::CodeImprovement => code_improvement,
                    TaskType::CodeGeneration => code_generation,
                    TaskType::DomainSpecificGeneration(domain) => {
                        // We need a static reference to return, so we'll use code_generation as the base
                        // but adjust our client selection logic based on the domain
                        match *domain {
                            ComputationalDomain::DeepLearning | 
                            ComputationalDomain::Transformers | 
                            ComputationalDomain::ClassicalML => {
                                // Claude may be better at ML domains, but return the existing preference
                                // and we'll adjust weights in language selection
                                code_generation
                            },
                            ComputationalDomain::QuantumComputing => {
                                // OpenAI may be better at quantum computing
                                code_generation
                            },
                            _ => code_generation, // Default to standard code generation preference
                        }
                    },
                    TaskType::Documentation => documentation,
                    TaskType::BugFixing => bug_fixing,
                    TaskType::PerformanceOptimization => performance_optimization,
                    TaskType::SafetyEnhancement => safety_enhancement,
                    TaskType::TestGeneration => test_generation,
                    TaskType::DomainDetection => code_detection, // Use code detection preference
                };
                
                // Consider language preferences (could be expanded)
                let lang_factor = match language {
                    Some("rust") => 0.1, // OpenAI slightly better at Rust
                    Some("python") => -0.05, // Claude slightly better at Python
                    _ => 0.0,
                };
                
                let adjusted_openai = (preference.openai_weight + lang_factor).clamp(0.0, 1.0);
                let adjusted_claude = preference.claude_weight;
                
                if (adjusted_openai - adjusted_claude).abs() < 0.1 {
                    LlmClientPreference::Both
                } else if adjusted_openai > adjusted_claude {
                    LlmClientPreference::PreferOpenAi
                } else {
                    LlmClientPreference::PreferClaude
                }
            }
        }
    }
}

/// Performance metrics for LLM selection
#[derive(Debug, Default)]
struct PerformanceMetrics {
    openai_success: u32,
    openai_failure: u32,
    claude_success: u32,
    claude_failure: u32,
}

/// Dynamic learning strategy
pub struct DynamicLearningStrategy {
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    exploration_rate: Arc<RwLock<f64>>,
    min_exploration_rate: f64,
    _max_exploration_rate: f64,
    exploration_decay: f64,
}

impl DynamicLearningStrategy {
    pub fn new(
        initial_exploration_rate: f64,
        min_exploration_rate: f64,
        max_exploration_rate: f64,
        exploration_decay: f64,
    ) -> Self {
        Self {
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            exploration_rate: Arc::new(RwLock::new(initial_exploration_rate)),
            min_exploration_rate,
            _max_exploration_rate: max_exploration_rate,
            exploration_decay,
        }
    }
    
    pub async fn record_result(&self, used_openai: bool, success: bool) {
        let mut metrics = self.performance_metrics.write().await;
        
        if used_openai {
            if success {
                metrics.openai_success += 1;
            } else {
                metrics.openai_failure += 1;
            }
        } else if success {
            metrics.claude_success += 1;
        } else {
            metrics.claude_failure += 1;
        }
    
        // Update exploration rate
        let mut rate = self.exploration_rate.write().await;
        *rate = (*rate * self.exploration_decay).max(self.min_exploration_rate);
    }
    
    pub async fn should_explore(&self) -> bool {
        let rate = *self.exploration_rate.read().await;
        rand::random::<f64>() < rate
    }
    
    pub async fn get_preference(&self) -> AdaptivePreference {
        let metrics = self.performance_metrics.read().await;
        
        let openai_success_rate = if metrics.openai_success + metrics.openai_failure > 0 {
            metrics.openai_success as f64 / (metrics.openai_success + metrics.openai_failure) as f64
        } else {
            0.5
        };
        
        let claude_success_rate = if metrics.claude_success + metrics.claude_failure > 0 {
            metrics.claude_success as f64 / (metrics.claude_success + metrics.claude_failure) as f64
                } else {
            0.5
                };
        
        let total = openai_success_rate + claude_success_rate;
        let openai_weight = if total > 0.0 {
            openai_success_rate / total
                } else {
            0.5
        };
        
        AdaptivePreference {
            openai_weight,
            claude_weight: 1.0 - openai_weight,
        }
    }
}

/// Enhanced Multi-LLM client with advanced strategies and feedback
pub struct EnhancedMultiLlmClient {
    openai: Option<Box<dyn LlmClient>>,
    claude: Option<Box<dyn LlmClient>>,
    strategy: LlmStrategy,
    // Performance tracking
    stats: Option<Arc<DynamicLearningStrategy>>,
}

impl EnhancedMultiLlmClient {
    /// Create a new enhanced multi-LLM client
    pub fn new(
        openai: Option<Box<dyn LlmClient>>,
        claude: Option<Box<dyn LlmClient>>,
        strategy: LlmStrategy,
    ) -> Self {
        // Extract DynamicLearningStrategy if present
        let stats = if let LlmStrategy::Adaptive { .. } = &strategy {
            Some(Arc::new(DynamicLearningStrategy::new(0.5, 0.1, 0.9, 0.95)))
        } else {
            None
        };
        
        Self {
            openai,
            claude,
            strategy,
            stats,
        }
    }
    
    /// Generate code for a specific task with performance tracking
    pub async fn generate_code_for_task(
        &self,
        prompt: &PromptBuilder,
        task_type: TaskType,
        language: Option<&str>,
    ) -> Result<String> {
        let preference = self.strategy.get_preferred_llm(&task_type);
        let lang = language.unwrap_or("unknown");
        
        debug!("Using strategy {:?} for task {:?} in language {}", preference, task_type, lang);
        
        let start_time = Instant::now();
        let result = match preference {
            "openai" => {
                self.generate_with_openai(prompt).await
            },
            
            "claude" => {
                self.generate_with_claude(prompt).await
            },
            
            "both" => {
                self.generate_with_both(prompt, task_type).await
            },
            
            _ => {
                // Default to openai if no clear preference
                debug!("No clear preference, defaulting to openai");
                    self.generate_with_openai(prompt).await
                }
        };
        
        let _elapsed = start_time.elapsed();
        
        // Record metrics if using DynamicLearningStrategy
        if let Some(stats) = &self.stats {
            let success = result.is_ok();
            let client_name = preference;
            
            stats.record_result(client_name == "openai", success).await;
        }
        
        result
    }
    
    /// Generate with OpenAI
    async fn generate_with_openai(&self, prompt: &PromptBuilder) -> Result<String> {
        if let Some(client) = &self.openai {
            client.generate_code(prompt).await
        } else {
            Err(AppError::ApiError("OpenAI client not configured".to_string()))
        }
    }
    
    /// Generate with Claude
    async fn generate_with_claude(&self, prompt: &PromptBuilder) -> Result<String> {
        if let Some(client) = &self.claude {
            client.generate_code(prompt).await
        } else {
            Err(AppError::ApiError("Claude client not configured".to_string()))
        }
    }
    
    /// Generate with both LLMs and merge results
    async fn generate_with_both(&self, prompt: &PromptBuilder, task_type: TaskType) -> Result<String> {
        // Check if both clients are available
        if self.openai.is_none() || self.claude.is_none() {
            return if self.openai.is_some() {
                info!("Only OpenAI available for dual generation");
                self.generate_with_openai(prompt).await
            } else if self.claude.is_some() {
                info!("Only Claude available for dual generation");
                self.generate_with_claude(prompt).await
            } else {
                Err(AppError::ApiError("No LLM clients configured".to_string()))
            };
        }
        
        // Generate code with both LLMs in parallel
        let (openai_result, claude_result) = tokio::join!(
            self.generate_with_openai(prompt),
            self.generate_with_claude(prompt)
        );
        
        // Handle results
        match (openai_result, claude_result) {
            (Ok(openai_code), Ok(claude_code)) => {
                // Both succeeded, generate a merged version
                self.merge_code_outputs(prompt, &openai_code, &claude_code, task_type).await
            },
            (Ok(openai_code), Err(e)) => {
                warn!("Claude generation failed: {}", e);
                Ok(openai_code)
            },
            (Err(e), Ok(claude_code)) => {
                warn!("OpenAI generation failed: {}", e);
                Ok(claude_code)
            },
            (Err(e1), Err(e2)) => {
                Err(AppError::ApiError(format!(
                    "Both LLMs failed: OpenAI: {}, Claude: {}",
                    e1, e2
                )))
            }
        }
    }
    
    /// Merge code outputs from different LLMs
    async fn merge_code_outputs(
        &self,
        _prompt: &PromptBuilder,
        openai_code: &str,
        claude_code: &str,
        task_type: TaskType,
    ) -> Result<String> {
        // Create a task-specific merge prompt
        let merge_prompt = match task_type {
            TaskType::CodeGeneration => PromptBuilder::new()
                .with_template(
                    "You have two different implementations of code extracted from a research paper.\n\
                     Merge these implementations, taking the best parts of each to create a single, \
                     optimized, well-documented, and fully executable solution.\n\n\
                     Original code snippet from the paper:\n\
                     ```{{LANGUAGE}}\n{{ORIGINAL_CODE}}\n```\n\n\
                     OpenAI implementation:\n\
                     ```{{LANGUAGE}}\n{{OPENAI_CODE}}\n```\n\n\
                     Claude implementation:\n\
                     ```{{LANGUAGE}}\n{{CLAUDE_CODE}}\n```\n\n\
                     Return only the merged, improved code."
                )
                .with_replacement("{{ORIGINAL_CODE}}", openai_code)
                .with_replacement("{{OPENAI_CODE}}", claude_code)
                .with_replacement("{{CLAUDE_CODE}}", claude_code)
                .with_replacement("{{LANGUAGE}}", "rust"),
            _ => {
                // Default merge prompt for other task types
                PromptBuilder::new()
                .with_template(
                        "You have two different implementations of code extracted from a research paper.\n\
                     Merge these implementations, taking the best parts of each to create a single, \
                         optimized, well-documented, and fully executable solution.\n\n\
                     Original code snippet from the paper:\n\
                     ```{{LANGUAGE}}\n{{ORIGINAL_CODE}}\n```\n\n\
                     OpenAI implementation:\n\
                     ```{{LANGUAGE}}\n{{OPENAI_CODE}}\n```\n\n\
                     Claude implementation:\n\
                     ```{{LANGUAGE}}\n{{CLAUDE_CODE}}\n```\n\n\
                     Return only the merged, improved code."
                )
                    .with_replacement("{{ORIGINAL_CODE}}", openai_code)
                    .with_replacement("{{OPENAI_CODE}}", claude_code)
                .with_replacement("{{CLAUDE_CODE}}", claude_code)
                    .with_replacement("{{LANGUAGE}}", "rust")
            }
        };
        
        let result = self.generate_with_openai(&merge_prompt).await?;
        
        Ok(result)
    }
}