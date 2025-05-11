// src/text/domain_detector.rs
use std::collections::HashMap;
use tracing::{debug, info};
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Computational domains that can be detected in research papers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputationalDomain {
    /// Numerical computing and simulations
    NumericalComputing,
    
    /// Chip design and hardware optimization
    ChipDesign,
    
    /// Bioinformatics and functional genomics
    Bioinformatics,
    
    /// Quantum computing algorithms
    QuantumComputing,
    
    /// Digital twin simulations (discrete event, state flow, multi-agent)
    DigitalTwin,
    
    /// Classical machine learning models
    ClassicalML,
    
    /// Deep learning models and neural networks
    DeepLearning,
    
    /// Transformer-based generative models
    Transformers,
    
    /// Computational physics
    ComputationalPhysics,
    
    /// Computational biology
    ComputationalBiology,
    
    /// Computational finance and quantitative finance
    ComputationalFinance,
    
    /// Supply chain algorithms and optimization
    SupplyChain,
    
    /// Logistics and distribution algorithms
    Logistics,
    
    /// General or undetected domain
    General,
}

impl std::fmt::Display for ComputationalDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumericalComputing => write!(f, "Numerical Computing"),
            Self::ChipDesign => write!(f, "Chip Design and Optimization"),
            Self::Bioinformatics => write!(f, "Bioinformatics and Functional Genomics"),
            Self::QuantumComputing => write!(f, "Quantum Computing"),
            Self::DigitalTwin => write!(f, "Digital Twin Simulations"),
            Self::ClassicalML => write!(f, "Classical Machine Learning"),
            Self::DeepLearning => write!(f, "Deep Learning"),
            Self::Transformers => write!(f, "Transformer-based Models"),
            Self::ComputationalPhysics => write!(f, "Computational Physics"),
            Self::ComputationalBiology => write!(f, "Computational Biology"),
            Self::ComputationalFinance => write!(f, "Computational Finance"),
            Self::SupplyChain => write!(f, "Supply Chain Algorithms"),
            Self::Logistics => write!(f, "Logistics and Distribution"),
            Self::General => write!(f, "General Computing"),
        }
    }
}

impl ComputationalDomain {
    /// Get the preferred programming languages for this domain
    pub fn preferred_languages(&self) -> Vec<&'static str> {
        match self {
            Self::NumericalComputing => vec!["python", "c++", "rust"],
            Self::ChipDesign => vec!["c", "c++", "rust"],
            Self::Bioinformatics => vec!["nextflow", "python", "rust"],
            Self::QuantumComputing => vec!["python", "rust"],
            Self::DigitalTwin => vec!["python", "rust", "c++"],
            Self::ClassicalML => vec!["python", "rust"],
            Self::DeepLearning => vec!["python", "rust"],
            Self::Transformers => vec!["python", "rust"],
            Self::ComputationalPhysics => vec!["c++", "python", "rust"],
            Self::ComputationalBiology => vec!["c++", "python", "rust"],
            Self::ComputationalFinance => vec!["c++", "python", "rust"],
            Self::SupplyChain => vec!["c++", "python", "rust"],
            Self::Logistics => vec!["c++", "python", "rust"],
            Self::General => vec!["python", "rust", "javascript"],
        }
    }
    
    /// Get the preferred frameworks/libraries for this domain
    pub fn preferred_frameworks(&self) -> Vec<&'static str> {
        match self {
            Self::NumericalComputing => vec!["numpy", "scipy", "ndarray"],
            Self::ChipDesign => vec!["verilator", "chisel", "verilog", "systemverilog"],
            Self::Bioinformatics => vec!["nextflow", "biopython", "rust-bio"],
            Self::QuantumComputing => vec!["qiskit", "cirq", "qutip", "pennylane"],
            Self::DigitalTwin => vec!["simpy", "mesa", "anylogic", "rust-sim"],
            Self::ClassicalML => vec!["scikit-learn", "linfa", "xgboost"],
            Self::DeepLearning => vec!["pytorch", "pytorch-lightning", "tch-rs"],
            Self::Transformers => vec!["transformers", "langchain", "tokenizers"],
            Self::ComputationalPhysics => vec!["numpy", "scipy", "lammps", "opensim"],
            Self::ComputationalBiology => vec!["biopython", "rust-bio", "bioconductor"],
            Self::ComputationalFinance => vec!["quantlib", "numpy-financial", "pandas"],
            Self::SupplyChain => vec!["pyomo", "ortools", "simpy"],
            Self::Logistics => vec!["ortools", "pyomo", "networkx"],
            Self::General => vec!["pandas", "numpy", "ndarray"],
        }
    }
    
    /// Get specialized code patterns or templates appropriate for this domain
    pub fn code_templates(&self) -> HashMap<&'static str, &'static str> {
        let mut templates = HashMap::new();
        match self {
            Self::NumericalComputing => {
                templates.insert("python", "import numpy as np\nimport scipy as sp\n\ndef solve_numerical_problem(data):\n    # Implementation\n    pass");
                templates.insert("rust", "use ndarray::prelude::*;\n\nfn solve_numerical_problem(data: &Array2<f64>) -> Array1<f64> {\n    // Implementation\n    Array1::zeros(data.nrows())\n}");
            },
            Self::ChipDesign => {
                templates.insert("c++", "#include <vector>\n#include <algorithm>\n\nclass ChipOptimizer {\npublic:\n    // Implementation\n};");
                templates.insert("rust", "pub struct ChipOptimizer {\n    // Fields\n}\n\nimpl ChipOptimizer {\n    // Methods\n}");
            },
            Self::QuantumComputing => {
                templates.insert("python", "import qiskit\nfrom qiskit import QuantumCircuit\n\ndef create_quantum_circuit():\n    # Implementation\n    qc = QuantumCircuit(2, 2)\n    return qc");
                templates.insert("rust", "use qoqo::prelude::*;\n\nfn create_quantum_circuit() -> QuantumCircuit {\n    // Implementation\n    let mut circuit = QuantumCircuit::new();\n    circuit\n}");
            },
            Self::DeepLearning => {
                templates.insert("python", "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\nclass NeuralNetwork(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # Define layers\n\n    def forward(self, x):\n        # Forward pass\n        return x");
                templates.insert("rust", "use tch::{nn, Device, Tensor};\n\npub struct NeuralNetwork {\n    network: nn::Sequential,\n}\n\nimpl NeuralNetwork {\n    pub fn new() -> Self {\n        // Implementation\n        Self { network: nn::seq() }\n    }\n\n    pub fn forward(&self, x: &Tensor) -> Tensor {\n        // Forward pass\n        self.network.forward(x)\n    }\n}");
            },
            _ => {}
        }
        templates
    }
}

/// Domain detector to identify computational domains in research papers
pub struct DomainDetector {
    domain_keywords: HashMap<ComputationalDomain, Vec<String>>,
    regex_patterns: HashMap<ComputationalDomain, Vec<Regex>>,
}

impl Default for DomainDetector {
    fn default() -> Self {
        let mut detector = Self::new();
        detector.initialize_domain_keywords();
        detector.initialize_domain_patterns();
        detector
    }
}

impl DomainDetector {
    /// Create a new domain detector
    pub fn new() -> Self {
        Self {
            domain_keywords: HashMap::new(),
            regex_patterns: HashMap::new(),
        }
    }
    
    /// Initialize domain-specific keywords for detection
    fn initialize_domain_keywords(&mut self) {
        // Numerical Computing
        self.domain_keywords.insert(
            ComputationalDomain::NumericalComputing,
            vec![
                "numerical method".to_string(),
                "numerical analysis".to_string(),
                "numerical simulation".to_string(),
                "finite element".to_string(),
                "finite difference".to_string(),
                "differential equation".to_string(),
                "numerical integration".to_string(),
                "discretization".to_string(),
                "approximation theory".to_string(),
            ],
        );
        
        // Chip Design
        self.domain_keywords.insert(
            ComputationalDomain::ChipDesign,
            vec![
                "chip design".to_string(),
                "integrated circuit".to_string(),
                "vlsi".to_string(),
                "semiconductor".to_string(),
                "hardware description".to_string(),
                "verilog".to_string(),
                "vhdl".to_string(),
                "asic".to_string(),
                "fpga".to_string(),
                "rtl".to_string(),
                "register transfer".to_string(),
                "logic synthesis".to_string(),
                "physical design".to_string(),
                "place and route".to_string(),
                "timing analysis".to_string(),
            ],
        );
        
        // Bioinformatics
        self.domain_keywords.insert(
            ComputationalDomain::Bioinformatics,
            vec![
                "bioinformatics".to_string(),
                "genomics".to_string(),
                "sequence alignment".to_string(),
                "dna sequence".to_string(),
                "rna".to_string(),
                "genome assembly".to_string(),
                "phylogenetic".to_string(),
                "protein structure".to_string(),
                "gene expression".to_string(),
                "next generation sequencing".to_string(),
                "ngs".to_string(),
                "transcriptomics".to_string(),
                "metagenomics".to_string(),
                "variant calling".to_string(),
                "genome annotation".to_string(),
            ],
        );
        
        // Quantum Computing
        self.domain_keywords.insert(
            ComputationalDomain::QuantumComputing,
            vec![
                "quantum computing".to_string(),
                "quantum algorithm".to_string(),
                "quantum gate".to_string(),
                "qubit".to_string(),
                "quantum circuit".to_string(),
                "quantum logic".to_string(),
                "quantum supremacy".to_string(),
                "quantum entanglement".to_string(),
                "quantum superposition".to_string(),
                "shor's algorithm".to_string(),
                "grover's algorithm".to_string(),
                "quantum annealing".to_string(),
                "quantum error correction".to_string(),
                "quantum machine learning".to_string(),
                "variational quantum".to_string(),
            ],
        );
        
        // Digital Twin
        self.domain_keywords.insert(
            ComputationalDomain::DigitalTwin,
            vec![
                "digital twin".to_string(),
                "discrete event simulation".to_string(),
                "agent-based model".to_string(),
                "multi-agent".to_string(),
                "state flow".to_string(),
                "system dynamics".to_string(),
                "process simulation".to_string(),
                "simulation model".to_string(),
                "discrete event".to_string(),
                "stateflow".to_string(),
                "state machine".to_string(),
                "behavioral model".to_string(),
                "event-driven simulation".to_string(),
                "petri net".to_string(),
                "monte carlo simulation".to_string(),
            ],
        );
        
        // Classical ML
        self.domain_keywords.insert(
            ComputationalDomain::ClassicalML,
            vec![
                "machine learning".to_string(),
                "supervised learning".to_string(),
                "unsupervised learning".to_string(),
                "classification algorithm".to_string(),
                "regression analysis".to_string(),
                "decision tree".to_string(),
                "random forest".to_string(),
                "support vector machine".to_string(),
                "svm".to_string(),
                "k-means".to_string(),
                "principal component analysis".to_string(),
                "pca".to_string(),
                "naive bayes".to_string(),
                "boosting algorithm".to_string(),
                "gradient boosting".to_string(),
                "xgboost".to_string(),
            ],
        );
        
        // Deep Learning
        self.domain_keywords.insert(
            ComputationalDomain::DeepLearning,
            vec![
                "deep learning".to_string(),
                "neural network".to_string(),
                "cnn".to_string(),
                "convolutional neural".to_string(),
                "rnn".to_string(),
                "recurrent neural".to_string(),
                "lstm".to_string(),
                "gru".to_string(),
                "backpropagation".to_string(),
                "gradient descent".to_string(),
                "deep neural".to_string(),
                "activation function".to_string(),
                "autoencoder".to_string(),
                "deep belief".to_string(),
                "batch normalization".to_string(),
                "dropout layer".to_string(),
            ],
        );
        
        // Transformers
        self.domain_keywords.insert(
            ComputationalDomain::Transformers,
            vec![
                "transformer model".to_string(),
                "attention mechanism".to_string(),
                "self-attention".to_string(),
                "bert".to_string(),
                "gpt".to_string(),
                "language model".to_string(),
                "generative model".to_string(),
                "token embedding".to_string(),
                "encoder-decoder".to_string(),
                "multi-head attention".to_string(),
                "fine-tuning".to_string(),
                "pre-trained model".to_string(),
                "sequence-to-sequence".to_string(),
                "large language model".to_string(),
                "llm".to_string(),
            ],
        );
        
        // Computational Physics
        self.domain_keywords.insert(
            ComputationalDomain::ComputationalPhysics,
            vec![
                "computational physics".to_string(),
                "molecular dynamics".to_string(),
                "monte carlo simulation".to_string(),
                "physics simulation".to_string(),
                "particle physics".to_string(),
                "fluid dynamics".to_string(),
                "computational fluid".to_string(),
                "cfd".to_string(),
                "quantum mechanics".to_string(),
                "lattice qcd".to_string(),
                "molecular modeling".to_string(),
                "electrodynamics".to_string(),
                "plasma physics".to_string(),
                "n-body simulation".to_string(),
                "statistical mechanics".to_string(),
            ],
        );
        
        // Computational Biology
        self.domain_keywords.insert(
            ComputationalDomain::ComputationalBiology,
            vec![
                "computational biology".to_string(),
                "systems biology".to_string(),
                "biological simulation".to_string(),
                "protein folding".to_string(),
                "molecular modeling".to_string(),
                "population dynamics".to_string(),
                "signaling pathway".to_string(),
                "ecological model".to_string(),
                "evolutionary algorithm".to_string(),
                "genetic algorithm".to_string(),
                "biochemical network".to_string(),
                "metabolic network".to_string(),
                "cellular automaton".to_string(),
                "reaction-diffusion".to_string(),
                "biomedical simulation".to_string(),
            ],
        );
        
        // Computational Finance
        self.domain_keywords.insert(
            ComputationalDomain::ComputationalFinance,
            vec![
                "computational finance".to_string(),
                "quantitative finance".to_string(),
                "financial modeling".to_string(),
                "option pricing".to_string(),
                "derivatives pricing".to_string(),
                "black-scholes".to_string(),
                "risk analysis".to_string(),
                "portfolio optimization".to_string(),
                "monte carlo finance".to_string(),
                "volatility model".to_string(),
                "algorithmic trading".to_string(),
                "high-frequency trading".to_string(),
                "stochastic process".to_string(),
                "interest rate model".to_string(),
                "financial time series".to_string(),
            ],
        );
        
        // Supply Chain
        self.domain_keywords.insert(
            ComputationalDomain::SupplyChain,
            vec![
                "supply chain".to_string(),
                "inventory management".to_string(),
                "warehouse optimization".to_string(),
                "demand forecasting".to_string(),
                "facility location".to_string(),
                "production planning".to_string(),
                "supply network".to_string(),
                "supply chain optimization".to_string(),
                "material requirement".to_string(),
                "mrp".to_string(),
                "just-in-time".to_string(),
                "bullwhip effect".to_string(),
                "inventory control".to_string(),
                "supplier selection".to_string(),
                "procurement optimization".to_string(),
            ],
        );
        
        // Logistics
        self.domain_keywords.insert(
            ComputationalDomain::Logistics,
            vec![
                "logistics optimization".to_string(),
                "transportation problem".to_string(),
                "vehicle routing".to_string(),
                "vrp".to_string(),
                "traveling salesman".to_string(),
                "tsp".to_string(),
                "distribution network".to_string(),
                "last mile delivery".to_string(),
                "facility location".to_string(),
                "route optimization".to_string(),
                "logistics network".to_string(),
                "fleet management".to_string(),
                "distribution center".to_string(),
                "transshipment problem".to_string(),
                "cross-docking".to_string(),
            ],
        );
    }
    
    /// Initialize domain-specific regex patterns for detection
    fn initialize_domain_patterns(&mut self) {
        // Numerical Computing
        let numerical_patterns = vec![
            Regex::new(r"(?i)\b(numerical method|finite element|finite difference|numerical integration|ode solver|pde solver)\b").unwrap(),
            Regex::new(r"(?i)\b(discretization|approximation theory|numerical analysis|numerical algorithm)\b").unwrap(),
            Regex::new(r"(?i)\b(eigenvalue|eigenvector|matrix decomposition|linear system|sparse matrix)\b").unwrap(),
        ];
        self.regex_patterns.insert(ComputationalDomain::NumericalComputing, numerical_patterns);
        
        // Deep Learning
        let dl_patterns = vec![
            Regex::new(r"(?i)\b(neural network|deep learning|CNN|RNN|LSTM|transformer|attention mechanism)\b").unwrap(),
            Regex::new(r"(?i)\b(backpropagation|gradient descent|activation function|regularization|batch normalization)\b").unwrap(),
            Regex::new(r"(?i)\b(epoch|training loop|inference|feature extraction|embedding|fine-tuning)\b").unwrap(),
        ];
        self.regex_patterns.insert(ComputationalDomain::DeepLearning, dl_patterns);
        
        // Add patterns for other domains as needed...
    }
    
    /// Detect the computational domain from paper text
    pub fn detect_domain(&self, text: &str) -> ComputationalDomain {
        debug!("Detecting computational domain from text");
        
        let mut domain_scores: HashMap<ComputationalDomain, usize> = HashMap::new();
        
        // Check for keyword matches
        for (domain, keywords) in &self.domain_keywords {
            let mut domain_score = 0;
            for keyword in keywords {
                let keyword_count = text.to_lowercase().matches(&keyword.to_lowercase()).count();
                domain_score += keyword_count;
            }
            domain_scores.insert(*domain, domain_score);
        }
        
        // Check for regex pattern matches
        for (domain, patterns) in &self.regex_patterns {
            let current_score = domain_scores.entry(*domain).or_insert(0);
            for pattern in patterns {
                let matches = pattern.find_iter(text).count();
                *current_score += matches * 2; // Weight regex matches higher
            }
        }
        
        // Find domain with highest score
        let mut best_domain = ComputationalDomain::General;
        let mut best_score = 0;
        
        for (domain, score) in domain_scores {
            if score > best_score {
                best_domain = domain;
                best_score = score;
            }
        }
        
        info!("Detected computational domain: {} (score: {})", best_domain, best_score);
        
        // Only return a specific domain if the score is above a threshold
        if best_score >= 3 {
            best_domain
        } else {
            ComputationalDomain::General
        }
    }
    
    /// Get code generation suggestions for a specific domain
    pub fn get_domain_prompt_suggestions(&self, domain: ComputationalDomain) -> String {
        let languages = domain.preferred_languages().join(", ");
        let frameworks = domain.preferred_frameworks().join(", ");
        
        format!(
            "This paper appears to be in the domain of {}.\n\
            Preferred programming languages: {}.\n\
            Recommended frameworks/libraries: {}.\n\
            Generate production-quality code that follows best practices for this domain.",
            domain, languages, frameworks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_numerical_computing_detection() {
        let detector = DomainDetector::default();
        let text = "In this paper, we present a novel numerical method for solving complex differential equations using finite element analysis.";
        let domain = detector.detect_domain(text);
        assert_eq!(domain, ComputationalDomain::NumericalComputing);
    }
    
    #[test]
    fn test_deep_learning_detection() {
        let detector = DomainDetector::default();
        let text = "We propose a new convolutional neural network architecture for image recognition with improved gradient descent techniques.";
        let domain = detector.detect_domain(text);
        assert_eq!(domain, ComputationalDomain::DeepLearning);
    }
    
    #[test]
    fn test_quantum_computing_detection() {
        let detector = DomainDetector::default();
        let text = "This paper presents a quantum algorithm for factorization using qubits and quantum gates.";
        let domain = detector.detect_domain(text);
        assert_eq!(domain, ComputationalDomain::QuantumComputing);
    }
    
    #[test]
    fn test_transformers_detection() {
        let detector = DomainDetector::default();
        let text = "We introduce a new transformer model with self-attention mechanisms for NLP tasks.";
        let domain = detector.detect_domain(text);
        assert_eq!(domain, ComputationalDomain::Transformers);
    }
    
    #[test]
    fn test_general_domain_for_insufficient_context() {
        let detector = DomainDetector::default();
        let text = "This is a general text without specific computational domain references.";
        let domain = detector.detect_domain(text);
        assert_eq!(domain, ComputationalDomain::General);
    }
    
    #[test]
    fn test_domain_preferred_languages() {
        assert!(ComputationalDomain::NumericalComputing.preferred_languages().contains(&"python"));
        assert!(ComputationalDomain::DeepLearning.preferred_languages().contains(&"python"));
        assert!(ComputationalDomain::QuantumComputing.preferred_languages().contains(&"python"));
    }
}
