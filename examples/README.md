# Paper2Code Clients

This directory contains client libraries for the `paper2code-rs` tool in various programming languages.

## Available Clients

### Python Client

A Python client for interacting with the `paper2code-rs` CLI tool.

#### Setup

```bash
cd python_client
pip install -r requirements.txt
```

#### Usage

```python
from paper2code_client import Paper2CodeClient

# Initialize the client
client = Paper2CodeClient()

# Extract code from a PDF file
result = client.extract_code_from_pdf("path/to/paper.pdf")
print(f"Found {len(result['code_blocks'])} code blocks")

# Extract code from text
text = "Here's some code: ```python\ndef hello():\n    print('Hello')\n```"
result = client.extract_code_from_text(text)
print(f"Found {len(result['code_blocks'])} code blocks")

# Generate executable code
if result['code_blocks']:
    snippets = [block['content'] for block in result['code_blocks']]
    generated = client.generate_code(snippets, language="python", output_dir="output")
    print("Generated code:", generated)
```

See the `example.py` file for a complete example.

### TypeScript Client

A TypeScript client for interacting with the `paper2code-rs` CLI tool.

#### Setup

```bash
cd typescript_client
npm install
```

#### Usage

```typescript
import { Paper2CodeClient } from './paper2code';

async function main() {
  // Initialize the client
  const client = new Paper2CodeClient();
  
  // Extract code from a PDF file
  const result = await client.extractCodeFromPdf('path/to/paper.pdf');
  console.log(`Found ${result.code_blocks.length} code blocks`);
  
  // Extract code from text
  const text = "Here's some code: ```python\ndef hello():\n    print('Hello')\n```";
  const textResult = await client.extractCodeFromText(text);
  console.log(`Found ${textResult.code_blocks.length} code blocks`);
  
  // Generate executable code
  if (textResult.code_blocks.length > 0) {
    const snippets = textResult.code_blocks.map(block => block.content);
    const generated = await client.generateCode(snippets, { 
      language: 'python',
      outputDir: 'output' 
    });
    console.log('Generated code:', generated);
  }
}

main().catch(console.error);
```

See the `src/example.ts` file for a complete example.

## Requirements

Both clients require the `paper2code-rs` binary to be installed and available in your PATH.

## Building the TypeScript Client

```bash
cd typescript_client
npm run build
```

This will compile the TypeScript code to JavaScript in the `dist` directory.

## Running the Examples

### Python

```bash
cd python_client
python example.py
```

### TypeScript

```bash
cd typescript_client
npm start
``` 