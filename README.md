# Paper2Code-rs - User Guide

A Rust command-line tool to extract code snippets from research papers and convert them into fully executable code using advanced LLMs (Large Language Models).

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

- Extract code from PDF research papers using optical character recognition (OCR)
- Leverage multiple LLMs (OpenAI GPT and Claude) for higher quality code extraction
- Convert code snippets to executable programs with proper syntax
- Intelligent language detection when not explicitly specified
- Customizable prompt templates for different extraction tasks
- Configure different strategies for each extraction phase
- High performance through parallel processing and asynchronous operations

## Installation

### Prerequisites

- Rust toolchain (1.70.0 or newer)
- API keys for at least one of:
  - OpenAI API
  - Anthropic Claude API

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paper2code-rs.git
   cd paper2code-rs
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Install the binary:
   ```bash
   cargo install --path .
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

Each task type (code detection, improvement, generation, documentation, etc.) can have different LLM preferences configured in the adaptive strategy.

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