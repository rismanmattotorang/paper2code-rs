// src/llm/domain_prompts_test.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::DomainPromptLibrary;
    use crate::text::ComputationalDomain;

    #[test]
    fn test_prompt_templates_exist() {
        let library = DomainPromptLibrary::default();
        
        // Test that templates exist for important domains
        assert!(library.get_prompt_template(&ComputationalDomain::NumericalComputing).is_some());
        assert!(library.get_prompt_template(&ComputationalDomain::DeepLearning).is_some());
        assert!(library.get_prompt_template(&ComputationalDomain::QuantumComputing).is_some());
        assert!(library.get_prompt_template(&ComputationalDomain::Transformers).is_some());
    }

    #[test]
    fn test_template_language_replacement() {
        let library = DomainPromptLibrary::default();
        
        // Test Python replacement
        let python_prompt = library.get_domain_prompt(&ComputationalDomain::NumericalComputing, "python");
        assert!(python_prompt.contains("python"));
        assert!(!python_prompt.contains("{language}"));
        
        // Test Rust replacement
        let rust_prompt = library.get_domain_prompt(&ComputationalDomain::DeepLearning, "rust");
        assert!(rust_prompt.contains("rust"));
        assert!(!rust_prompt.contains("{language}"));
    }

    #[test]
    fn test_code_templates_exist() {
        let library = DomainPromptLibrary::default();
        
        // Check if code templates exist for Python and Rust in Deep Learning domain
        let dl_python_template = library.get_code_template(&ComputationalDomain::DeepLearning, "python");
        assert!(dl_python_template.is_some());
        
        let dl_rust_template = library.get_code_template(&ComputationalDomain::DeepLearning, "rust");
        assert!(dl_rust_template.is_some());
    }

    #[test]
    fn test_code_templates_content() {
        let library = DomainPromptLibrary::default();
        
        // Check if deep learning templates contain key components
        if let Some(dl_python_template) = library.get_code_template(&ComputationalDomain::DeepLearning, "python") {
            assert!(dl_python_template.contains("class CustomModel"));
            assert!(dl_python_template.contains("def forward"));
            assert!(dl_python_template.contains("train_model"));
        }
        
        if let Some(dl_rust_template) = library.get_code_template(&ComputationalDomain::DeepLearning, "rust") {
            assert!(dl_rust_template.contains("struct CustomModel"));
            assert!(dl_rust_template.contains("impl Module"));
            assert!(dl_rust_template.contains("fn forward"));
        }
    }

    #[test]
    fn test_quantum_computing_templates() {
        let library = DomainPromptLibrary::default();
        
        // Check if quantum computing templates contain key components
        if let Some(qc_python_template) = library.get_code_template(&ComputationalDomain::QuantumComputing, "python") {
            assert!(qc_python_template.contains("QuantumCircuit"));
            assert!(qc_python_template.contains("simulate_circuit"));
        }
    }

    #[test]
    fn test_fallback_domain_prompt() {
        let library = DomainPromptLibrary::default();
        
        // Test the fallback prompt for a language with no domain-specific template
        let fallback_prompt = library.get_domain_prompt(&ComputationalDomain::General, "golang");
        assert!(fallback_prompt.contains("golang"));
        assert!(fallback_prompt.contains("Generate production-quality"));
    }
}
