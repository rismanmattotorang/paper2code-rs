# Paper2Code Tutorial

This tutorial will guide you through using paper2code-rs to extract, analyze, and generate executable code from research papers.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Working with PDF Files](#working-with-pdf-files)
- [Using the Python Client](#using-the-python-client)
- [Using the TypeScript Client](#using-the-typescript-client)
- [LLM Configuration](#llm-configuration)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Installation

### From Source

1. Ensure you have Rust installed. If not, install it with [rustup](https://rustup.rs/):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paper2code-rs.git
   cd paper2code-rs
   ```

3. Build the project:
   ```bash
   cargo build --release
   ```

4. Install the binary (optional):
   ```bash
   cargo install --path .
   ```

### From Binary (if available)

Download the latest release for your platform from the releases page and place it in your PATH.

## Basic Usage

### Checking Installation

Verify your installation is working:

```bash
paper2code-rs --version
```

### Configuration

Generate a default configuration file:

```bash
paper2code-rs config init
```

This creates a `config.toml` file that you can modify with your LLM API keys and preferences.

### Extracting Code from Text

Extract code snippets from text:

```bash
echo "Here's a code sample: \`\`\`python\ndef hello():\n    print('Hello world')\n\`\`\`" | paper2code-rs extract --from-text
```

## Working with PDF Files

### Extracting Code from PDFs

Extract code from a research paper:

```bash
paper2code-rs extract path/to/paper.pdf
```

Specify output directory:

```bash
paper2code-rs extract path/to/paper.pdf --output ./extracted_code
```

Extract from specific pages:

```bash
paper2code-rs extract path/to/paper.pdf --pages 10-15
```

### Generating Executable Code

Generate runnable code from the extracted snippets:

```bash
paper2code-rs generate --input extracted_snippets.txt --language python
```

The tool will use LLMs to convert the code snippets into fully executable code with proper imports, error handling, and documentation.

## Using the Python Client

For Python programmers, paper2code-rs provides a convenient Python client.

### Installation

```bash
cd examples/python_client
pip install -r requirements.txt
```

### Basic Usage

```python
from paper2code_client import Paper2CodeClient

# Initialize the client
client = Paper2CodeClient()

# Extract code from a PDF file
result = client.extract_code_from_pdf("path/to/paper.pdf")
print(f"Found {len(result['code_blocks'])} code blocks")

# Process each code block
for i, block in enumerate(result['code_blocks']):
    print(f"Block {i+1}, Language: {block.get('language', 'unknown')}")
    print(f"Content:\n{block['content']}")
```

### Generating Executable Code

```python
# Extract code blocks from PDF
result = client.extract_code_from_pdf("path/to/paper.pdf")

# Generate executable code
if result['code_blocks']:
    code_snippets = [block['content'] for block in result['code_blocks']]
    generated = client.generate_code(
        code_snippets, 
        language="python", 
        output_dir="output"
    )
    print("Generated files:", generated.get('files', []))
```

## Using the TypeScript Client

For JavaScript/TypeScript programmers, there's a TypeScript client available.

### Installation

```bash
cd examples/typescript_client
npm install
```

### Basic Usage

```typescript
import { Paper2CodeClient } from './paper2code';

async function main() {
  // Initialize the client
  const client = new Paper2CodeClient();
  
  // Extract code from a PDF file
  const result = await client.extractCodeFromPdf('path/to/paper.pdf');
  console.log(`Found ${result.code_blocks.length} code blocks`);
  
  // Process each code block
  result.code_blocks.forEach((block, i) => {
    console.log(`Block ${i+1}, Language: ${block.language || 'unknown'}`);
    console.log(`Content:\n${block.content}`);
  });
}

main().catch(console.error);
```

### Generating Executable Code

```typescript
import { Paper2CodeClient } from './paper2code';

async function main() {
  const client = new Paper2CodeClient();
  
  // Extract code from PDF
  const result = await client.extractCodeFromPdf('path/to/paper.pdf');
  
  // Generate executable code
  if (result.code_blocks.length > 0) {
    const snippets = result.code_blocks.map(block => block.content);
    const generated = await client.generateCode(snippets, { 
      language: 'python',
      outputDir: 'output' 
    });
    console.log('Generated files:', generated.files || []);
  }
}

main().catch(console.error);
```

## LLM Configuration

paper2code-rs supports multiple LLM providers to help convert code snippets into executable code.

### Setting API Keys

Set your API keys for LLM providers:

```bash
# For Claude
paper2code-rs config set claude.api_key your_api_key_here

# For OpenAI
paper2code-rs config set openai.api_key your_api_key_here
```

### Testing LLM Connection

Test your LLM connection:

```bash
paper2code-rs test llm --provider claude
paper2code-rs test llm --provider openai
```

### Selecting a Default Provider

Set your default LLM provider:

```bash
paper2code-rs config set default_provider claude
```

## Advanced Features

### Custom Prompt Templates

paper2code-rs uses prompt templates when interacting with LLMs. You can customize these:

1. Export the default template:
   ```bash
   paper2code-rs template export --type code_generation > my_template.txt
   ```

2. Edit the template file.

3. Use your custom template:
   ```bash
   paper2code-rs generate --input snippets.txt --template my_template.txt
   ```

### Batch Processing

Process multiple PDF files in batch:

```bash
paper2code-rs extract path/to/papers/*.pdf --output extracted_code
```

### Language-Specific Generation

Generate code for a specific language:

```bash
paper2code-rs generate --input snippets.txt --language python
paper2code-rs generate --input snippets.txt --language javascript
paper2code-rs generate --input snippets.txt --language rust
```

## Troubleshooting

### PDF Extraction Issues

If you're having trouble extracting code from PDFs:

1. Check if the PDF is searchable (has embedded text). Some scanned PDFs may not contain selectable text.
2. Try extracting from specific pages where you know code exists:
   ```bash
   paper2code-rs extract paper.pdf --pages 10-15
   ```

### LLM API Issues

If you encounter issues with LLM APIs:

1. Verify your API key is correct and has not expired
2. Check your internet connection
3. Try with a different provider:
   ```bash
   paper2code-rs test llm --provider openai
   ```

### Installation Problems

If you have issues installing or building:

1. Ensure you have the latest Rust version: `rustup update`
2. Install required development libraries:
   - Ubuntu/Debian: `apt-get install build-essential pkg-config libssl-dev`
   - macOS: `xcode-select --install`
   - Windows: Install Visual Studio build tools

### Client Library Issues

#### Python Client

If the Python client is not working:

1. Ensure the paper2code-rs binary is in your PATH
2. Check Python version (3.7+ recommended)
3. Try running with verbose output:
   ```python
   result = client._run_command(["--verbose", "extract", "paper.pdf"])
   ```

#### TypeScript Client

If the TypeScript client is not working:

1. Make sure you have Node.js installed (16+ recommended)
2. Rebuild the TypeScript code: `npm run build`
3. Check for missing dependencies: `npm install`

## More Resources

- [Official Documentation](https://github.com/yourusername/paper2code-rs)
- [Example Code Repository](https://github.com/yourusername/paper2code-examples)
- [Report Issues](https://github.com/yourusername/paper2code-rs/issues) 