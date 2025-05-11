// src/code/domain_aware_generator.rs
use tracing::{debug, info};

use crate::error::Result;
use crate::text::{ComputationalDomain, DomainDetector};
use crate::llm::{
    client::LlmClient,
    DomainPromptLibrary,
    prompt::PromptBuilder,
    strategy::{LlmStrategy, TaskType},
};

/// A domain-aware code generator that specializes code generation based on
/// detected computational domains in research papers
pub struct DomainAwareCodeGenerator {
    llm_client: Box<dyn LlmClient>,
    domain_detector: DomainDetector,
    domain_prompts: DomainPromptLibrary,
    #[allow(dead_code)]
    strategy: LlmStrategy,
}

impl DomainAwareCodeGenerator {
    /// Create a new domain-aware code generator
    pub fn new(llm_client: Box<dyn LlmClient>, strategy: LlmStrategy) -> Self {
        Self {
            llm_client,
            domain_detector: DomainDetector::default(),
            domain_prompts: DomainPromptLibrary::default(),
            strategy,
        }
    }
    
    /// Detect the computational domain from paper text
    pub fn detect_domain(&self, text: &str) -> ComputationalDomain {
        self.domain_detector.detect_domain(text)
    }
    
    /// Generate domain-specific code from research paper text
    pub async fn generate_domain_specific_code(
        &self,
        paper_text: &str,
        code_snippets: &[String],
        language: &str,
    ) -> Result<String> {
        // Detect computational domain
        let domain = self.detect_domain(paper_text);
        info!("Detected computational domain: {}", domain);
        
        // Get domain-specific prompt
        let domain_prompt = self.domain_prompts.get_domain_prompt(&domain, language);
        
        // Build the prompt for code generation
        let mut prompt_builder = PromptBuilder::new()
            .with_template(
                "You are an expert programmer specializing in developing code for research papers.
                
Paper Abstract/Excerpt:
```
{{PAPER_TEXT}}
```

Code Snippets from the Paper:
```
{{CODE_SNIPPETS}}
```

{{DOMAIN_SPECIFIC_INSTRUCTIONS}}

Create comprehensive, executable {{LANGUAGE}} code that implements the algorithms and methods described in this research paper.
Follow these guidelines:
1. The code should be well-structured and production-quality
2. Include comprehensive documentation and comments explaining the algorithms
3. Implement robust error handling and validation
4. Organize code into modular components with clean interfaces
5. Use appropriate design patterns for the domain
6. Include example usage to demonstrate the implementation

Return ONLY the implementation code without explanations outside code comments."
            )
            .with_replacement("{{PAPER_TEXT}}", &paper_text[..paper_text.len().min(2000)]) // Limit to 2000 chars for efficiency
            .with_replacement("{{CODE_SNIPPETS}}", &code_snippets.join("\n\n"))
            .with_replacement("{{DOMAIN_SPECIFIC_INSTRUCTIONS}}", &domain_prompt)
            .with_replacement("{{LANGUAGE}}", language);
        
        // Add sample code templates if available
        if let Some(template) = self.domain_prompts.get_code_template(&domain, language) {
            prompt_builder = prompt_builder.with_replacement(
                "{{SAMPLE_TEMPLATE}}",
                &format!("Here's a starting template you can adapt:\n```{}\n{}\n```", language, template)
            );
        } else {
            prompt_builder = prompt_builder.with_replacement("{{SAMPLE_TEMPLATE}}", "");
        }
        
        // Use domain-specific task type for generation
        let task_type = TaskType::DomainSpecificGeneration(domain);
        debug!("Using task type: {:?}", task_type);
        
        // Generate code using the detected domain for specialized generation
        info!("Generating domain-specific code for domain: {}", domain);
        self.llm_client.generate_code(&prompt_builder).await
    }
    
    /// Enhance generated code with domain-specific optimizations
    pub async fn optimize_for_domain(
        &self,
        domain: ComputationalDomain,
        code: &str,
        language: &str,
    ) -> Result<String> {
        let optimization_prompt = match domain {
            ComputationalDomain::NumericalComputing => {
                "Optimize this numerical computing code for performance and accuracy:
                1. Use vectorized operations where possible
                2. Improve numerical stability
                3. Add boundary condition checks
                4. Optimize memory usage for large datasets"
            },
            ComputationalDomain::DeepLearning => {
                "Optimize this deep learning code:
                1. Add gradient clipping
                2. Implement early stopping
                3. Add learning rate scheduling
                4. Optimize memory usage for GPU training
                5. Add mixed precision training option"
            },
            ComputationalDomain::QuantumComputing => {
                "Optimize this quantum computing code:
                1. Reduce circuit depth where possible
                2. Implement error mitigation techniques
                3. Add measurement error handling
                4. Optimize for specific quantum hardware"
            },
            _ => {
                "Optimize this code for production use:
                1. Improve error handling
                2. Add performance optimizations
                3. Ensure proper resource management
                4. Add comprehensive documentation"
            }
        };
        
        let prompt = PromptBuilder::new()
            .with_template(
                "You are an expert in {{DOMAIN}} optimization. Enhance the following {{LANGUAGE}} code for production use.
                
{{OPTIMIZATION_INSTRUCTIONS}}

Original Code:
```{{LANGUAGE}}
{{CODE}}
```

Return ONLY the optimized version of the code with no explanations outside of code comments."
            )
            .with_replacement("{{DOMAIN}}", &domain.to_string())
            .with_replacement("{{LANGUAGE}}", language)
            .with_replacement("{{OPTIMIZATION_INSTRUCTIONS}}", optimization_prompt)
            .with_replacement("{{CODE}}", code);
        
        // Optimize code for the specific domain
        info!("Optimizing code for domain: {}", domain);
        self.llm_client.generate_code(&prompt).await
    }
    
    /// Generate test cases for domain-specific code
    pub async fn generate_domain_tests(
        &self,
        domain: ComputationalDomain,
        code: &str,
        language: &str,
    ) -> Result<String> {
        let test_prompt = match domain {
            ComputationalDomain::NumericalComputing => {
                "Generate comprehensive tests for this numerical computing code:
                1. Test with known analytical solutions
                2. Verify convergence rates
                3. Test boundary conditions
                4. Verify conservation properties
                5. Test with extreme values"
            },
            ComputationalDomain::DeepLearning => {
                "Generate comprehensive tests for this deep learning code:
                1. Test forward and backward passes
                2. Verify gradient calculations
                3. Test with dummy data
                4. Validate model saving/loading
                5. Test on edge cases"
            },
            ComputationalDomain::QuantumComputing => {
                "Generate comprehensive tests for this quantum computing code:
                1. Test with simple known cases
                2. Verify circuit construction
                3. Test measurement statistics
                4. Compare with classical simulations for small cases
                5. Test error handling"
            },
            _ => {
                "Generate comprehensive tests for this code:
                1. Unit tests for core functions
                2. Integration tests for modules
                3. Edge case handling
                4. Performance benchmarks
                5. Validation tests"
            }
        };
        
        let prompt = PromptBuilder::new()
            .with_template(
                "You are an expert in testing {{DOMAIN}} code. Create comprehensive tests for the following {{LANGUAGE}} code.
                
{{TEST_INSTRUCTIONS}}

Code to Test:
```{{LANGUAGE}}
{{CODE}}
```

Return ONLY the test code suitable for this implementation, using appropriate testing frameworks for {{LANGUAGE}}."
            )
            .with_replacement("{{DOMAIN}}", &domain.to_string())
            .with_replacement("{{LANGUAGE}}", language)
            .with_replacement("{{TEST_INSTRUCTIONS}}", test_prompt)
            .with_replacement("{{CODE}}", code);
        
        // Generate domain-specific tests
        info!("Generating tests for domain: {}", domain);
        self.llm_client.generate_code(&prompt).await
    }
}
