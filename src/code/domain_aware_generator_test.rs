// src/code/domain_aware_generator_test.rs
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use mockall::predicate::*;
    use mockall::mock;
    
    use crate::code::DomainAwareCodeGenerator;
    use crate::llm::{
        client::LlmClient,
        prompt::PromptBuilder,
        strategy::LlmStrategy,
    };
    use crate::text::ComputationalDomain;
    use crate::error::Result;

    // Create a mock LLM client for testing
    mock! {
        pub LlmClientMock {}
        impl Clone for LlmClientMock {
            fn clone(&self) -> Self;
        }
        #[async_trait::async_trait]
        impl LlmClient for LlmClientMock {
            async fn generate_code(&self, prompt: &PromptBuilder) -> Result<String>;
            fn box_clone(&self) -> Box<dyn LlmClient>;
        }
    }

    #[tokio::test]
    async fn test_domain_detection() {
        let mut mock_client = MockLlmClientMock::new();
        mock_client.expect_box_clone()
            .returning(|| Box::new(MockLlmClientMock::new()));
        
        let strategy = LlmStrategy::default_strategy();
        let generator = DomainAwareCodeGenerator::new(
            Box::new(mock_client),
            strategy,
        );
        
        // Test numerical computing domain detection
        let numerical_text = "This paper presents a finite element method for solving partial differential equations.";
        let domain = generator.detect_domain(numerical_text);
        
        assert_eq!(domain, ComputationalDomain::NumericalComputing);
    }

    #[tokio::test]
    async fn test_code_generation() {
        let mut mock_client = MockLlmClientMock::new();
        
        // Setup mock to return a specific response for any prompt
        mock_client.expect_generate_code()
            .returning(|_| Ok("def example_function():\n    return 'Hello World'".to_string()));
        
        mock_client.expect_box_clone()
            .returning(|| {
                let mut mock = MockLlmClientMock::new();
                mock.expect_generate_code()
                    .returning(|_| Ok("def example_function():\n    return 'Hello World'".to_string()));
                Box::new(mock)
            });
        
        let strategy = LlmStrategy::default_strategy();
        let generator = DomainAwareCodeGenerator::new(
            Box::new(mock_client),
            strategy,
        );
        
        // Test code generation for a specific domain
        let paper_text = "This paper presents a deep learning approach to image classification.";
        let code_snippets = vec![
            "def train_model(data):\n    # Training logic".to_string()
        ];
        
        let result = generator.generate_domain_specific_code(
            paper_text,
            &code_snippets,
            "python",
        ).await;
        
        assert!(result.is_ok());
        if let Ok(code) = result {
            assert!(code.contains("example_function"));
        }
    }

    #[tokio::test]
    async fn test_domain_optimization() {
        let mut mock_client = MockLlmClientMock::new();
        
        // Setup mock to return a specific response
        mock_client.expect_generate_code()
            .returning(|_| Ok("def optimized_function():\n    # Optimized implementation".to_string()));
        
        mock_client.expect_box_clone()
            .returning(|| {
                let mut mock = MockLlmClientMock::new();
                mock.expect_generate_code()
                    .returning(|_| Ok("def optimized_function():\n    # Optimized implementation".to_string()));
                Box::new(mock)
            });
        
        let strategy = LlmStrategy::default_strategy();
        let generator = DomainAwareCodeGenerator::new(
            Box::new(mock_client),
            strategy,
        );
        
        // Test code optimization for a specific domain
        let original_code = "def simple_function():\n    # Basic implementation";
        
        let result = generator.optimize_for_domain(
            ComputationalDomain::DeepLearning,
            original_code,
            "python",
        ).await;
        
        assert!(result.is_ok());
        if let Ok(code) = result {
            assert!(code.contains("optimized_function"));
        }
    }

    #[tokio::test]
    async fn test_domain_test_generation() {
        let mut mock_client = MockLlmClientMock::new();
        
        // Setup mock to return a specific response
        mock_client.expect_generate_code()
            .returning(|_| Ok("def test_example_function():\n    assert example_function() == 'Hello World'".to_string()));
        
        mock_client.expect_box_clone()
            .returning(|| {
                let mut mock = MockLlmClientMock::new();
                mock.expect_generate_code()
                    .returning(|_| Ok("def test_example_function():\n    assert example_function() == 'Hello World'".to_string()));
                Box::new(mock)
            });
        
        let strategy = LlmStrategy::default_strategy();
        let generator = DomainAwareCodeGenerator::new(
            Box::new(mock_client),
            strategy,
        );
        
        // Test test generation for a specific domain
        let original_code = "def example_function():\n    return 'Hello World'";
        
        let result = generator.generate_domain_tests(
            ComputationalDomain::QuantumComputing,
            original_code,
            "python",
        ).await;
        
        assert!(result.is_ok());
        if let Ok(code) = result {
            assert!(code.contains("test_example_function"));
        }
    }
}
