// src/llm/domain_prompts.rs
use std::collections::HashMap;
use crate::text::domain_detector::ComputationalDomain;

/// Provides domain-specific prompt templates and examples for high-quality code generation
pub struct DomainPromptLibrary {
    prompt_templates: HashMap<ComputationalDomain, String>,
    sample_code_templates: HashMap<ComputationalDomain, HashMap<String, String>>,
}

impl Default for DomainPromptLibrary {
    fn default() -> Self {
        let mut library = Self::new();
        library.initialize_prompt_templates();
        library.initialize_code_templates();
        library
    }
}

impl DomainPromptLibrary {
    /// Create a new domain prompt library
    pub fn new() -> Self {
        Self {
            prompt_templates: HashMap::new(),
            sample_code_templates: HashMap::new(),
        }
    }
    
    /// Initialize domain-specific prompt templates
    fn initialize_prompt_templates(&mut self) {
        // Numerical Computing
        self.prompt_templates.insert(
            ComputationalDomain::NumericalComputing,
            r#"Generate production-quality code in {language} that implements the numerical computing algorithms described in the paper.
Focus on:
1. Numerical stability and precision - avoid catastrophic cancellation and use appropriate precision
2. Efficient matrix/vector operations - use optimized BLAS/LAPACK implementations where possible
3. Proper handling of boundary conditions - ensure all edge cases are properly validated
4. Thorough error checking - validate inputs and handle numerical exceptions (NaN, Inf, etc.)
5. Clear documentation of mathematical formulas - include LaTeX-style descriptions in comments
6. Appropriate use of vectorized operations - avoid unnecessary loops for elementwise operations
7. Memory management - minimize allocation in inner loops and manage large arrays efficiently
8. Parallelization opportunities - identify and implement parallel computations where beneficial

Recommended libraries for Python: 
- NumPy for core array operations and vectorization
- SciPy for specialized mathematical functions and algorithms
- JAX for automatic differentiation and accelerated computations
- Numba for JIT compilation of performance-critical sections

Recommended libraries for C++: 
- Eigen for template-based matrix operations
- Armadillo for LAPACK integration
- BLAS/LAPACK for optimized linear algebra
- Intel MKL for hardware-accelerated operations

Recommended libraries for Rust: 
- ndarray for n-dimensional arrays with efficient operations
- nalgebra for computer graphics and computer physics applications
- blas-src for BLAS integration
- matrixmultiply for optimized matrix multiplication

Best Practices for Numerical Computing:
1. Use double precision (64-bit) floating-point as default unless space constraints require otherwise
2. Implement unit tests with known analytical solutions to verify accuracy
3. Check for algorithm convergence with appropriate tolerances
4. Validate conservation properties where applicable
5. Provide comprehensive documentation of mathematical methods and their implementations
6. Include citations to relevant algorithms and papers

Ensure the code is optimized for both correctness and performance."#.to_string(),
        );
        
        // Chip Design
        self.prompt_templates.insert(
            ComputationalDomain::ChipDesign,
            r#"Generate production-quality hardware description code in {language} based on the paper.
Focus on:
1. Clear module interfaces and port definitions
2. Proper clock domain handling
3. Efficient resource utilization
4. Timing considerations and constraints
5. Power optimization techniques
6. Testability and verification
7. Parameterized designs for flexibility

For C/C++: Consider using SystemC or High-Level Synthesis approaches
For Rust: Consider using appropriate HDL crates like rust_hdl

Ensure the code is synthesizable and follows industry best practices."#.to_string(),
        );
        
        // Bioinformatics
        self.prompt_templates.insert(
            ComputationalDomain::Bioinformatics,
            r#"Generate production-quality bioinformatics code in {language} based on the paper.
Focus on:
1. Efficient sequence processing algorithms
2. Memory-efficient data structures for genomic data
3. Proper handling of FASTQ/FASTA/SAM/BAM formats
4. Parallel processing of genomic data where appropriate
5. Statistical rigor in analysis methods
6. Clear documentation of biological assumptions
7. Reproducible workflow structure

For Nextflow: Create a complete workflow with input validation and proper channels
For Python: Use Biopython, scikit-bio, pandas for data handling
For Rust: Use bio-rust or similar bioinformatics crates

Ensure the code follows best practices for reproducible computational biology research."#.to_string(),
        );
        
        // Quantum Computing
        self.prompt_templates.insert(
            ComputationalDomain::QuantumComputing,
            r#"Generate production-quality quantum computing code in {language} based on the paper.
Focus on:

1. Quantum Circuit Design:
   - Proper quantum circuit construction with appropriate gate sequences
   - Optimal circuit depth minimization techniques
   - Appropriate qubit layout and connectivity constraints
   - Gate-level optimization and circuit simplification
   - Custom gate definitions where needed

2. Qubit Resource Management:
   - Efficient qubit allocation and routing strategies
   - Ancilla qubit management and uncomputation
   - Consideration of hardware-specific qubit topologies
   - Qubit reuse and recycling patterns

3. Error Mitigation and Robustness:
   - Quantum error correction codes where applicable
   - Error mitigation techniques (ZNE, probabilistic error cancellation)
   - Noise-aware circuit compilation
   - Measurement error mitigation
   - Readout error correction techniques

4. Quantum-Classical Interface:
   - Clear separation between classical and quantum components
   - Efficient parameter passing to variational circuits
   - Classical optimization loops for variational algorithms
   - Post-processing of quantum measurements

5. Hardware vs. Simulation Considerations:
   - Simulator-specific optimizations for testing
   - Hardware-specific constraints and optimizations
   - Transpilation strategies for target hardware
   - Cloud API integration for real quantum hardware

6. Performance Optimization:
   - Parameter optimization methods for variational circuits
   - Gradient calculation techniques (parameter shift rule, etc.)
   - Batching of circuit evaluations where possible
   - Shot allocation strategies for noisy hardware

Recommended libraries for Python:
- Qiskit (IBM) for full-stack quantum computing
- Cirq (Google) for circuit construction and simulation
- PennyLane for quantum machine learning and differentiable programming
- QuTiP for open quantum systems simulation
- Amazon Braket for cross-platform quantum computing
- TensorFlow Quantum or Torch Quantum for quantum ML integration

Recommended libraries for Rust:
- qoqo for quantum circuit construction
- roqoqo for classical processing tied to qoqo
- qip for quantum information processing primitives
- qwtools for quantum walk simulations

Best Practices for Quantum Computing Implementation:
1. Use parameterized circuits for flexibility
2. Include visualization functions for circuits and results
3. Implement classical simulation for small problem instances as validation
4. Add noise models for realistic hardware simulation
5. Include benchmark comparisons with classical algorithms where applicable
6. Ensure reproducibility through fixed random seeds
7. Provide detailed documentation on quantum concepts and mathematical background
8. Include citations to relevant quantum algorithms and techniques
9. Implement progressive circuit complexity for debugging
10. Add comprehensive testing of both classical and quantum components

Ensure the code is well-documented with quantum mechanical concepts explained clearly, making it accessible to both quantum experts and those new to quantum computing."#.to_string(),
        );
        
        // Deep Learning
        self.prompt_templates.insert(
            ComputationalDomain::DeepLearning,
            r#"Generate production-quality deep learning code in {language} based on the paper.
Focus on:
1. Clean model architecture definition:
   - Modular design with clear component separation
   - Proper weight initialization strategies
   - Configurable architecture parameters
   - Docstrings for each layer and its purpose

2. Efficient data loading and preprocessing pipelines:
   - Dataset classes with appropriate transformations
   - Data augmentation techniques relevant to the domain
   - Efficient batch processing and prefetching
   - On-the-fly preprocessing where appropriate

3. Training infrastructure:
   - Proper training, validation, and testing splits
   - Well-designed training loop with progress tracking
   - Gradient accumulation for large models
   - Mixed-precision training where appropriate
   - Distributed training setup for multi-GPU/TPU

4. Experiment management:
   - Hyperparameter tracking and organization
   - Checkpointing and model serialization
   - Experiment logging (metrics, gradients, weights)
   - Reproducibility guarantees (setting seeds, etc.)

5. Evaluation and visualization:
   - Comprehensive evaluation metrics
   - Confusion matrices and error analysis
   - Visualization of model predictions
   - Interpretability methods where appropriate

6. Resource optimization:
   - Memory-efficient implementations
   - GPU utilization monitoring
   - Profiling and bottleneck identification
   - Inference optimization techniques

Recommended libraries for Python: 
- PyTorch for tensor operations and automatic differentiation
- PyTorch Lightning for streamlined training loops
- torchvision for computer vision models and datasets
- transformers for NLP models
- Ray or Optuna for hyperparameter tuning
- Weights & Biases or TensorBoard for experiment tracking
- NumPy and Pandas for data manipulation
- Matplotlib and Plotly for visualization

Recommended libraries for Rust: 
- tch-rs for PyTorch bindings
- burn for native Rust deep learning
- ndarray for numeric computing
- tokenizers for NLP preprocessing

Best Practices for Deep Learning Implementation:
1. Implement model components as separate modules for reusability
2. Use config files or structured configuration objects for experiments
3. Implement early stopping with patience for training efficiency
4. Add gradient clipping to prevent exploding gradients
5. Use learning rate schedulers for better convergence
6. Include model interpretability tools for black-box models
7. Implement data validation to catch issues early
8. Add unit tests for critical components
9. Document model assumptions and limitations
10. Include paper citations and implementation notes

Create a well-structured implementation that follows these deep learning best practices and is ready for both experimentation and production deployment."#.to_string(),
        );
        
        // Transformers
        self.prompt_templates.insert(
            ComputationalDomain::Transformers,
            r#"Generate production-quality transformer model code in {language} based on the paper.
Focus on:

1. Core Architecture Components:
   - Attention mechanism implementation (self-attention, multi-head, etc.)
   - Position-wise feed-forward networks
   - Tokenization strategies and embedding layers
   - Positional encoding (absolute, relative, rotary, ALiBi)
   - Layer normalization placement (pre/post norm) and implementation
   - Residual connections and gradient flow considerations
   - Activation functions (GELU, SwiGLU, etc.)
   - Output projection and classification heads

2. Training Infrastructure:
   - Efficient batch processing and sequence packing
   - Masking implementation (causal, padding, etc.)
   - Loss functions with appropriate label smoothing
   - Optimization techniques (AdamW, Lion, etc.)
   - Learning rate schedules with warmup
   - Gradient accumulation for large models
   - Gradient checkpointing for memory efficiency

3. Memory and Compute Optimization:
   - Efficient attention implementations (flash attention, etc.)
   - KV caching for inference
   - Mixed precision training (fp16/bf16)
   - Quantization techniques (int8, int4, etc.)
   - Parallelism strategies (tensor, pipeline, sequence)
   - Offloading techniques for limited hardware
   - Pruning and sparsity considerations

4. Advanced Techniques (where applicable):
   - Parameter-efficient fine-tuning (LoRA, adapters, etc.)
   - Retrieval augmentation (RAG) components
   - Context window extension methods
   - Prompt engineering and templating
   - Inference-time optimizations (speculative decoding, etc.)
   - Streaming implementation for real-time generation

5. Evaluation and Safety:
   - Evaluation metrics appropriate for the task
   - Validation techniques for generated outputs
   - Safety guardrails and content filtering
   - Bias mitigation strategies
   - Documentation of model limitations and intended use

Recommended libraries for Python:
- Hugging Face Transformers for pre-trained models and utilities
- PyTorch or TensorFlow for tensor operations
- Accelerate for distributed training and device placement
- PEFT for parameter-efficient fine-tuning
- DeepSpeed or FSDP for distributed training
- LangChain or LlamaIndex for application integration
- Tokenizers for custom tokenization
- Datasets for data processing pipelines

Recommended libraries for Rust:
- tch-rs for PyTorch bindings
- candle for native Rust transformer implementations
- tokenizers-rs for fast tokenization
- ort for ONNX Runtime integration

Best Practices for Transformer Implementation:
1. Use modular architecture with configurable components
2. Implement checkpointing with state dict saving/loading
3. Use efficient tensor operations and avoid redundant computations
4. Add comprehensive logging of training metrics
5. Implement early stopping based on validation metrics
6. Include memory profiling and optimization
7. Document attention patterns and model architecture clearly
8. Provide sample prompt templates and usage examples
9. Include model cards with performance and limitation details
10. Implement proper error handling for OOM and other failures

Ensure the code follows best practices for large language model development, is organized for maintainability, and includes appropriate documentation of design choices made based on the research paper."#.to_string(),
        );
        
        // Add templates for other domains...
        
        // Computational Finance
        self.prompt_templates.insert(
            ComputationalDomain::ComputationalFinance,
            r#"Generate production-quality financial algorithms code in {language} based on the paper.
Focus on:
1. Numerical stability for pricing models
2. Proper risk factor modeling
3. Efficient Monte Carlo simulation techniques
4. Performance optimization for large portfolios
5. Proper handling of market data
6. Accurate implementation of financial models
7. Clear documentation of mathematical assumptions

For Python: Use numpy-financial, pandas, scipy, QuantLib-Python
For C++: Use QuantLib, Boost
For Rust: Use appropriate financial crates

Ensure the code is both accurate and performant for financial applications."#.to_string(),
        );
    }
    
    /// Initialize domain-specific code templates
    fn initialize_code_templates(&mut self) {
        // Deep Learning domain code templates
        let mut dl_templates = HashMap::new();
        
        // Python PyTorch template
        dl_templates.insert("python".to_string(), r#"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class CustomModel(nn.Module):
    """
    Model implementation based on the paper: [PAPER_TITLE]
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define model layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.layers(x)
    
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """
    Train the model and return training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return history

def main():
    # Example usage
    # TODO: Implement specific model architecture from the paper
    model = CustomModel(input_dim=10, hidden_dim=64, output_dim=1)
    print(model)
    
    # TODO: Load and prepare dataset
    
if __name__ == "__main__":
    main()
"#.to_string());

        // Rust deep learning template
        dl_templates.insert("rust".to_string(), r#"
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
use std::collections::HashMap;
use std::error::Error;

// Model implementation based on the paper: [PAPER_TITLE]
pub struct CustomModel {
    layers: nn::Sequential,
}

impl CustomModel {
    pub fn new(vs: &nn::Path, input_dim: i64, hidden_dim: i64, output_dim: i64) -> Self {
        let seq = nn::seq()
            .add(nn::linear(vs / "layer1", input_dim, hidden_dim, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer2", hidden_dim, hidden_dim, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer3", hidden_dim, output_dim, Default::default()));
        
        Self { layers: seq }
    }
}

impl Module for CustomModel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.layers.forward(xs)
    }
}

pub fn train_model(
    model: &mut CustomModel,
    train_data: &(Tensor, Tensor),
    val_data: &(Tensor, Tensor),
    num_epochs: i64,
    learning_rate: f64,
    batch_size: i64,
    device: Device,
) -> Result<HashMap<String, Vec<f64>>, Box<dyn Error>> {
    let vs = nn::VarStore::new(device);
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;
    
    let (train_x, train_y) = train_data;
    let (val_x, val_y) = val_data;
    
    let n_samples = train_x.size()[0];
    let mut history: HashMap<String, Vec<f64>> = HashMap::new();
    history.insert("train_loss".to_string(), Vec::new());
    history.insert("val_loss".to_string(), Vec::new());
    
    for epoch in 0..num_epochs {
        // Training phase
        let mut train_loss = 0.0;
        for batch_idx in 0..(n_samples - 1) / batch_size + 1 {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n_samples);
            
            let batch_x = train_x.narrow(0, start, end - start);
            let batch_y = train_y.narrow(0, start, end - start);
            
            let prediction = model.forward(&batch_x);
            let loss = prediction.mse_loss(&batch_y, tch::Reduction::Mean);
            
            opt.backward_step(&loss);
            train_loss += f64::from(&loss);
        }
        train_loss /= (n_samples as f64) / (batch_size as f64);
        
        // Validation phase
        let prediction = model.forward(val_x);
        let val_loss = f64::from(&prediction.mse_loss(val_y, tch::Reduction::Mean));
        
        // Record metrics
        history.get_mut("train_loss").unwrap().push(train_loss);
        history.get_mut("val_loss").unwrap().push(val_loss);
        
        println!("Epoch {}/{}, Train Loss: {:.4}, Val Loss: {:.4}", 
                 epoch + 1, num_epochs, train_loss, val_loss);
    }
    
    Ok(history)
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::cuda_if_available();
    
    // Initialize model
    let vs = nn::VarStore::new(device);
    let model = CustomModel::new(&vs.root(), 10, 64, 1);
    
    // TODO: Load and prepare dataset
    
    println!("Model initialized successfully");
    Ok(())
}
"#.to_string());

        self.sample_code_templates.insert(ComputationalDomain::DeepLearning, dl_templates);
        
        // Quantum Computing domain code templates
        let mut qc_templates = HashMap::new();
        
        // Python Qiskit template
        qc_templates.insert("python".to_string(), r#"
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

def create_quantum_circuit(num_qubits: int, num_clbits: int) -> QuantumCircuit:
    """
    Create a quantum circuit based on the algorithm described in the paper.
    """
    qc = QuantumCircuit(num_qubits, num_clbits)
    
    # Initialize qubit(s)
    qc.h(0)  # Apply Hadamard gate to the first qubit
    
    # TODO: Implement specific quantum operations from the paper
    # Example: qc.cx(0, 1)  # CNOT gate
    
    # Measurement
    qc.measure([0], [0])
    
    return qc

def simulate_circuit(circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
    """
    Simulate the quantum circuit and return measurement results
    """
    simulator = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, simulator)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts(circuit)
    return counts

def visualize_results(counts: Dict[str, int]) -> None:
    """
    Visualize the results of the quantum computation
    """
    plot_histogram(counts)
    plt.show()

def run_on_real_quantum_computer(circuit: QuantumCircuit, 
                                backend_name: str = 'ibmq_qasm_simulator',
                                shots: int = 1024) -> Dict[str, int]:
    """
    Run the circuit on a real quantum computer or cloud simulator
    Requires IBMQ account configuration
    """
    # Load IBMQ account
    # IBMQ.load_account()  # Uncomment and configure with your IBMQ token
    
    # Get provider and backend
    # provider = IBMQ.get_provider(hub='ibm-q')
    # backend = provider.get_backend(backend_name)
    
    # For now, use simulator
    backend = Aer.get_backend('qasm_simulator')
    
    # Run circuit
    job = backend.run(transpile(circuit, backend), shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    
    return counts

def main():
    # Create quantum circuit
    num_qubits = 2
    num_clbits = 2
    circuit = create_quantum_circuit(num_qubits, num_clbits)
    
    # Display the circuit
    print(circuit)
    
    # Simulate the circuit
    counts = simulate_circuit(circuit)
    print("Simulation results:", counts)
    
    # Visualize the results
    visualize_results(counts)
    
    # TODO: Implement specific algorithm from the paper

if __name__ == "__main__":
    main()
"#.to_string());

        self.sample_code_templates.insert(ComputationalDomain::QuantumComputing, qc_templates);
        
        // Add more domain-specific templates...
    }
    
    /// Get a prompt template for a specific domain
    pub fn get_prompt_template(&self, domain: &ComputationalDomain) -> Option<&String> {
        self.prompt_templates.get(domain)
    }
    
    /// Get code templates for a specific domain and language
    pub fn get_code_template(&self, domain: &ComputationalDomain, language: &str) -> Option<&String> {
        self.sample_code_templates
            .get(domain)
            .and_then(|templates| templates.get(language))
    }
    
    /// Get a domain-specific prompt with language filled in
    pub fn get_domain_prompt(&self, domain: &ComputationalDomain, language: &str) -> String {
        if let Some(template) = self.get_prompt_template(domain) {
            template.replace("{language}", language)
        } else {
            // Fallback prompt if no domain-specific template exists
            format!(
                "Generate production-quality {} code based on the research paper, focusing on best practices for implementation.",
                language
            )
        }
    }
}
