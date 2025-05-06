# Paper2Code TypeScript Client

A TypeScript client for interacting with the `paper2code-rs` CLI tool.

## Setup

1. Make sure you have Node.js installed (version 16+ recommended)
2. Install the dependencies:

```bash
npm install
```

## Build

To compile the TypeScript code to JavaScript:

```bash
npm run build
```

This will generate the compiled code in the `dist` directory.

## Running the Example

To run the example script:

```bash
npm start
```

This will execute the `src/example.ts` file using ts-node.

## Usage in Your Own Project

1. Copy the `src/paper2code.ts` file to your project
2. Import and use the client:

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

## Requirements

- The `paper2code-rs` binary must be installed and available on your PATH
- Node.js version 16 or higher is recommended

## Troubleshooting

If you get errors about missing Node.js type definitions, you can add the following to your `tsconfig.json`:

```json
{
  "compilerOptions": {
    // ... other options
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true
  }
}
```

Or install the Node.js type definitions:

```bash
npm install --save-dev @types/node
``` 