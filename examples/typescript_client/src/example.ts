/**
 * Example script demonstrating usage of the Paper2Code TypeScript client
 */

import { Paper2CodeClient } from './paper2code';

// Sample Python code for testing
const SAMPLE_CODE = `
Here's a research algorithm from a paper:

\`\`\`python
def gradient_descent(f, df, x0, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    # Initialize at the starting point
    x = x0
    # Keep track of function values for plotting
    values = [f(x)]
    
    for i in range(max_iterations):
        # Compute gradient
        gradient = df(x)
        # Take a step in the negative gradient direction
        x_new = x - learning_rate * gradient
        
        # Check for convergence
        if abs(f(x_new) - f(x)) < tolerance:
            break
            
        # Update x
        x = x_new
        values.append(f(x))
    
    return x, values
\`\`\`

The algorithm implements gradient descent for optimization.
`;

async function main() {
  try {
    // Initialize the client
    const client = new Paper2CodeClient();
    await client.checkBinary();
    console.log('Paper2Code Client initialized successfully.');
    
    // Extract code from text
    console.log('\n1. Extracting code from text...');
    const result = await client.extractCodeFromText(SAMPLE_CODE);
    console.log(`Extracted ${result.code_blocks.length} code blocks.`);
    
    // Print the first code block
    if (result.code_blocks.length > 0) {
      const block = result.code_blocks[0];
      console.log(`\nLanguage: ${block.language || 'unknown'}`);
      console.log('Code snippet:');
      console.log('------------');
      console.log(block.content);
      console.log('------------');
    }
    
    // Generate executable code
    console.log('\n2. Generating executable code...');
    if (result.code_blocks.length > 0) {
      const snippets = result.code_blocks.map(block => block.content);
      
      // Generate code and save to output directory
      const outputDir = 'generated_code';
      const genResult = await client.generateCode(
        snippets, 
        { 
          language: 'python', 
          outputDir 
        }
      );
      
      console.log(`Code generation result: ${JSON.stringify(genResult, null, 2)}`);
      console.log(`Generated code saved to ${outputDir}/`);
    }
    
    // Show the current configuration
    console.log('\n3. Getting current configuration...');
    const config = await client.getConfig();
    
    // Remove sensitive information for display
    const sanitizedConfig = { ...config };
    if (sanitizedConfig.claude?.api_key) {
      sanitizedConfig.claude.api_key = '***********';
    }
    if (sanitizedConfig.openai?.api_key) {
      sanitizedConfig.openai.api_key = '***********';
    }
    
    console.log(`Current configuration: ${JSON.stringify(sanitizedConfig, null, 2)}`);
    
  } catch (error: unknown) {
    if (error instanceof Error) {
      console.error(`Error: ${error.message}`);
    } else {
      console.error(`Unknown error occurred: ${String(error)}`);
    }
  }
}

// Run the main function
main(); 