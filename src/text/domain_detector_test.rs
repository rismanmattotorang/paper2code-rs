// src/text/domain_detector_test.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{ComputationalDomain, DomainDetector};

    #[test]
    fn test_numerical_computing_detection() {
        let detector = DomainDetector::default();
        
        let numerical_text = "
            This paper introduces a novel numerical method for solving partial differential equations with 
            improved stability. We present a finite element approach that handles complex boundary conditions 
            and demonstrates better convergence properties than existing methods. Our algorithm employs 
            matrix decomposition techniques and efficient sparse matrix operations to achieve computational efficiency.
            We validate our method using numerical experiments on several benchmark problems.
        ";
        
        let domain = detector.detect_domain(numerical_text);
        assert_eq!(domain, ComputationalDomain::NumericalComputing);
    }

    #[test]
    fn test_deep_learning_detection() {
        let detector = DomainDetector::default();
        
        let dl_text = "
            We propose a novel deep learning architecture that combines convolutional layers with 
            attention mechanisms. Our model achieves state-of-the-art performance on image classification tasks.
            The training procedure involves gradient descent optimization with learning rate scheduling and 
            dropout regularization. We implement batch normalization between layers to improve training stability.
        ";
        
        let domain = detector.detect_domain(dl_text);
        assert_eq!(domain, ComputationalDomain::DeepLearning);
    }

    #[test]
    fn test_quantum_computing_detection() {
        let detector = DomainDetector::default();
        
        let quantum_text = "
            This paper presents a new quantum algorithm for factoring large integers using fewer qubits than
            Shor's algorithm. We implement our algorithm using a combination of quantum gates, including
            Hadamard, CNOT, and controlled phase operations. Our approach demonstrates quantum advantage
            for specific problem instances and is suitable for implementation on NISQ devices.
        ";
        
        let domain = detector.detect_domain(quantum_text);
        assert_eq!(domain, ComputationalDomain::QuantumComputing);
    }

    #[test]
    fn test_transformers_detection() {
        let detector = DomainDetector::default();
        
        let transformer_text = "
            We introduce a new transformer architecture with improved attention mechanisms for natural language
            processing tasks. Our model uses multi-head self-attention with a novel positional encoding scheme.
            The architecture includes layer normalization and residual connections for better gradient flow.
            Fine-tuning protocols are described for adapting the pre-trained model to downstream tasks.
        ";
        
        let domain = detector.detect_domain(transformer_text);
        assert_eq!(domain, ComputationalDomain::Transformers);
    }

    #[test]
    fn test_multiple_domain_signals() {
        let detector = DomainDetector::default();
        
        // Text with signals from both deep learning and transformers domains
        let mixed_text = "
            Our approach combines deep neural networks with transformer architectures to create
            a hybrid model that excels at sequential data processing. We implement attention mechanisms
            and convolutional layers in a novel arrangement. The training procedure uses backpropagation
            with adaptive learning rates and gradient accumulation for handling large batch sizes.
        ";
        
        // The detector should identify one of the domains based on keyword frequency
        // This is a flexible test that accepts either result
        let domain = detector.detect_domain(mixed_text);
        assert!(domain == ComputationalDomain::DeepLearning || domain == ComputationalDomain::Transformers);
    }

    #[test]
    fn test_general_computing_fallback() {
        let detector = DomainDetector::default();
        
        let general_text = "
            This paper discusses algorithmic approaches to data processing. Various methods are
            compared for efficiency and correctness. Implementation details are provided along
            with performance evaluations.
        ";
        
        let domain = detector.detect_domain(general_text);
        assert_eq!(domain, ComputationalDomain::General);
    }

    #[test]
    fn test_domain_preferred_languages() {
        let numerical_domain = ComputationalDomain::NumericalComputing;
        let langs = numerical_domain.preferred_languages();
        assert!(langs.contains(&"python"));
        assert!(langs.contains(&"c++"));
        assert!(langs.contains(&"rust"));
        
        let quantum_domain = ComputationalDomain::QuantumComputing;
        let langs = quantum_domain.preferred_languages();
        assert!(langs.contains(&"python"));
    }

    #[test]
    fn test_domain_preferred_frameworks() {
        let dl_domain = ComputationalDomain::DeepLearning;
        let frameworks = dl_domain.preferred_frameworks();
        assert!(frameworks.contains(&"pytorch"));
        
        let transformer_domain = ComputationalDomain::Transformers;
        let frameworks = transformer_domain.preferred_frameworks();
        assert!(frameworks.contains(&"transformers"));
    }
}
