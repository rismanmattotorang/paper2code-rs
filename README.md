# Paper2Code-rs - User Guide

A high-performance Rust command-line tool to extract code snippets from research papers and convert them into fully executable code using advanced LLMs (Large Language Models). This tool allows researchers and developers to quickly implement algorithms and techniques described in academic papers.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [LLM Strategies](#llm-strategies)
- [Troubleshooting](#troubleshooting)

## Features

- **Advanced PDF Processing**: Extract text and code from research papers with robust handling of various PDF formats
- **Multi-LLM Integration**: Leverage the combined capabilities of OpenAI GPT and Anthropic Claude models for more accurate code extraction and generation
- **Intelligent Code Detection**: Automatically identify code blocks in paper text with sophisticated pattern recognition
- **Computational Domain Detection**: Automatically identify the research paper's computational domain to optimize code generation
- **Domain-Specific Code Generation**: Generate optimized, production-quality code for specialized domains:
  - Numerical Computing
  - Chip Design and Optimization
  - Bioinformatics and Functional Genomics
  - Quantum Computing Algorithms
  - Digital Twin Simulations
  - Classical Machine Learning Models
  - Deep Learning Models
  - Transformer-based Generative Models
  - Computational Physics, Biology, and Finance
  - Supply Chain and Logistics Algorithms
- **Multiple LLM Strategies**:
  - Use single models (OpenAI only, Claude only)
  - Sequential fallback approaches (try one model first, fall back to another)
  - Compare and merge results from multiple models for higher quality
  - Adaptive strategy selection based on domain expertise and task complexity
- **Code Enhancement**: Convert pseudocode and code snippets into fully executable programs with proper syntax, imports, and error handling
- **Automatic Language Detection**: Intelligently determine the programming language when not explicitly specified
- **High Performance**:
  - Parallel processing of text chunks
  - Asynchronous LLM API calls
  - Efficient memory usage for large documents
- **Configurable Workflow**: Customize every aspect of the extraction and generation process
- **Robust Error Handling**: Graceful degradation when components fail with detailed logging

## Installation

### Prerequisites

- Rust toolchain (1.70.0 or newer)
- API keys for at least one of:
  - OpenAI API (required for OpenAI models)
  - Anthropic Claude API (required for Claude models)
- PDF research papers to process

### System Requirements

- **OS**: macOS, Linux, or Windows
- **Memory**: Minimum 4GB RAM recommended
- **Storage**: At least 100MB for installation plus space for generated code
- **Processor**: Multi-core CPU recommended for parallel processing
- **Internet**: Required for LLM API calls

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/rismanmattotorang/paper2code-rs.git
   cd paper2code-rs
   ```

2. Build the project in release mode for optimal performance:
   ```bash
   cargo build --release
   ```

3. Install the binary (optional):
   ```bash
   cargo install --path .
   ```

4. Or run directly from the project directory:
   ```bash
   cargo run --release -- [COMMAND] [OPTIONS]
   ```

### Environment Variables (Alternative to config file)

You can also use environment variables instead of a config file:

```bash
# Required for OpenAI functionality
export OPENAI_API_KEY="your_openai_key_here"

# Required for Claude functionality
export CLAUDE_API_KEY="your_claude_key_here"

# Optional configuration
export PAPER2CODE_OUTPUT_DIR="./generated_code"
```

## Quick Start

1. Generate a configuration file:
   ```bash
   paper2code-rs config --generate
   ```

2. Edit the configuration file (`config.toml`) to add your API keys.

3. Test your setup:
   ```bash
   paper2code-rs test
   ```

4. Extract code from a PDF:
   ```bash
   paper2code-rs extract --input research_paper.pdf
   ```

## Commands

### Extract

Extract code from PDF research papers:

```bash
paper2code-rs extract --input <PDF_FILE(S)> [OPTIONS]
```

Options:
- `--input`, `-i`: One or more PDF files to process (required)
- `--output`, `-o`: Output directory for generated code (default: "output")
- `--force`, `-f`: Force overwrite of existing files
- `--strategy`: LLM strategy to use ("openai_only", "claude_only", "openai_first", "claude_first", "compare_and_merge", "adaptive")
- `--language`: Target programming language for code generation (auto-detect if not specified)

### Config

Manage configuration:

```bash
paper2code-rs config [OPTIONS]
```

Options:
- `--generate`, `-g`: Generate a new configuration file
- `--output`, `-o`: Path to save the configuration file (default: "config.toml")
- `--force`, `-f`: Force overwrite of existing configuration file

### Test

Test LLM connectivity:

```bash
paper2code-rs test [OPTIONS]
```

Options:
- `--openai`: Test only OpenAI API
- `--claude`: Test only Claude API
- `--prompt`: Custom prompt for testing (default: "Write a simple 'Hello, World!' program.")

Global Options:
- `--config`, `-c`: Path to configuration file (default: "config.toml")
- `--verbose`, `-v`: Enable verbose output

## Configuration

The configuration file (`config.toml`) contains the following sections:

### OpenAI Configuration

```toml
[openai]
api_key = "your_openai_api_key_here"
model = "gpt-4-turbo"
max_tokens = 4096
temperature = 0.2
timeout_seconds = 120
max_concurrent_requests = 3
```

### Claude Configuration

```toml
[claude]
api_key = "your_claude_api_key_here"
model = "claude-3-opus-20240229"
max_tokens = 4096
temperature = 0.2
timeout_seconds = 120
max_concurrent_requests = 3
```

### LLM Strategy Configuration

```toml
[llm_strategy]
strategy_type = "adaptive"
code_detection_preference = "prefer_claude"
code_improvement_preference = "prefer_openai"
code_generation_preference = "prefer_openai"
documentation_preference = "prefer_claude"
bug_fixing_preference = "prefer_openai"
```

### PDF Processing Configuration

```toml
[pdf]
chunk_size = 1000
```

### Code Detection Configuration

```toml
[code_detection]
min_confidence = 0.7
min_lines = 3
```

### Output Configuration

```toml
[output]
path = "output"
overwrite = false
```

### Prompt Templates Configuration

```toml
[prompt]
extraction_template = """
You are an expert programmer tasked with extracting code from research papers.
Please analyze the following text and extract all code snippets.
The text may contain code blocks, pseudocode, or algorithm descriptions.
Return ONLY the extracted code blocks in a clear format.

Text:
{{TEXT}}
"""
# Other templates omitted for brevity
```

## Usage Examples

### Basic Extraction

Extract code from a single research paper with automatic settings:

```bash
paper2code-rs extract -i research_paper.pdf
```

### Processing Multiple Papers

Extract code from multiple research papers in one command:

```bash
paper2code-rs extract -i paper1.pdf paper2.pdf paper3.pdf -o ./extracted_code
```

### Targeting a Specific Language

Force code generation in a specific programming language:

```bash
paper2code-rs extract -i algorithm_paper.pdf --language python
```

### Using Different LLM Strategies

Experiment with different LLM strategies for better results:

```bash
# Use OpenAI models only
paper2code-rs extract -i paper.pdf --strategy openai_only

# Use Claude models only
paper2code-rs extract -i paper.pdf --strategy claude_only

# Try OpenAI first, fall back to Claude
paper2code-rs extract -i paper.pdf --strategy openai_first

# Compare and merge results from both models
paper2code-rs extract -i paper.pdf --strategy compare_and_merge
```

### Batch Processing with Custom Configuration

Process multiple papers with a custom configuration file:

```bash
paper2code-rs extract -i ./papers/*.pdf -o ./code -c ./custom_config.toml
```

### Process a Single PDF

```bash
paper2code-rs extract --input research_paper.pdf
```

### Process Multiple PDFs

```bash
paper2code-rs extract --input paper1.pdf paper2.pdf paper3.pdf
```

### Specify Output Directory

```bash
paper2code-rs extract --input paper.pdf --output ./my_code
```

### Force Overwrite Existing Files

```bash
paper2code-rs extract --input paper.pdf --force
```

### Use Specific LLM Strategy

```bash
paper2code-rs extract --input paper.pdf --strategy claude_only
```

### Target Specific Programming Language

```bash
paper2code-rs extract --input paper.pdf --language python
```

### Enable Verbose Output

```bash
paper2code-rs extract --input paper.pdf --verbose
```

## LLM Strategies

Paper2Code supports several strategies for using LLMs:

1. **Adaptive** (default): Intelligently selects the best LLM for each task type based on configured preferences
2. **OpenAI Only**: Use only OpenAI models for all tasks
3. **Claude Only**: Use only Claude models for all tasks
4. **OpenAI First, Claude Second**: Try OpenAI first, fall back to Claude if unsuccessful
5. **Claude First, OpenAI Second**: Try Claude first, fall back to OpenAI if unsuccessful
6. **Compare and Merge**: Use both models and merge the results for optimal output

Each task type (code detection, improvement, generation, documentation, etc.) can have different LLM preferences configured in the adaptive strategy. The system now intelligently adjusts LLM selection based on the detected computational domain of your research paper.

## Computational Domains

Paper2Code now automatically detects and optimizes code generation for various computational domains:

| Domain | Description | Preferred Languages | Key Frameworks |
|--------|-------------|---------------------|----------------|
| Numerical Computing | Algorithms for numerical methods and simulations | Python, C++, Rust | NumPy, SciPy, ndarray |
| Chip Design | Hardware design and optimization | C/C++, Rust | Verilog, SystemVerilog, Chisel |
| Bioinformatics | Genomic data processing and analysis | Python, Rust, Nextflow | Biopython, rust-bio |
| Quantum Computing | Quantum algorithms and circuits | Python, Rust | Qiskit, Cirq, PennyLane |
| Digital Twins | Simulation of physical systems | Python, Rust, C++ | SimPy, Mesa, AnyLogic |
| Classical ML | Traditional machine learning algorithms | Python, Rust | scikit-learn, XGBoost, Linfa |
| Deep Learning | Neural network architectures and training | Python, Rust | PyTorch, TensorFlow, tch-rs |
| Transformers | Large language models and attention mechanisms | Python, Rust | Hugging Face, LangChain |
| Computational Physics | Physics simulations and modeling | C++, Python, Rust | NumPy, SciPy, LAMMPS |
| Computational Finance | Financial modeling and analysis | C++, Python, Rust | QuantLib, numpy-financial |

When a domain is detected, Paper2Code selects the appropriate LLM, templates, and frameworks to generate the highest quality code for that specific domain.

## Troubleshooting

### Common Issues

#### API Authentication Errors

If you encounter API authentication errors:
- Verify your API keys are correct in the configuration file
- Check that you have sufficient credits/quota with the API provider
- Ensure you have internet connectivity

#### PDF Extraction Issues

If text extraction from PDFs fails:
- Try using a different PDF (some PDFs may have security features that prevent extraction)
- Check if the PDF contains actual text and not just images
- Ensure the PDF is not corrupted

#### Output Generation Issues

If code generation fails:
- Try using a different LLM strategy
- Check the verbose logs for specific error messages
- Ensure the extracted text contains valid code snippets

#### Performance Issues

If processing is slow:
- Consider adjusting the chunk size in the configuration
- Reduce the maximum concurrent requests if you're experiencing rate limiting
- Use a more powerful machine for processing large PDFs

For more help or to report issues, please visit the GitHub repository at [https://github.com/yourusername/paper2code-rs](https://github.com/yourusername/paper2code-rs). 