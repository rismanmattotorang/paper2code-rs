#!/usr/bin/env python3
"""
Example script demonstrating usage of the Paper2Code Python client
"""

import json
import os
from paper2code_client import Paper2CodeClient

# Sample Python code for testing
SAMPLE_CODE = """
Here's a research algorithm from a paper:

```python
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
```

The algorithm implements gradient descent for optimization.
"""

def main():
    # Initialize the client
    client = Paper2CodeClient()
    print("Paper2Code Client initialized successfully.")
    
    # Extract code from text
    print("\n1. Extracting code from text...")
    result = client.extract_code_from_text(SAMPLE_CODE)
    print(f"Extracted {len(result.get('code_blocks', []))} code blocks.")
    
    # Print the first code block
    if result.get('code_blocks', []):
        block = result['code_blocks'][0]
        print(f"\nLanguage: {block.get('language', 'unknown')}")
        print("Code snippet:")
        print("------------")
        print(block.get('content', ''))
        print("------------")
    
    # Generate executable code
    print("\n2. Generating executable code...")
    if result.get('code_blocks', []):
        snippets = [block.get('content', '') for block in result.get('code_blocks', [])]
        
        # Create output directory if it doesn't exist
        output_dir = "generated_code"
        os.makedirs(output_dir, exist_ok=True)
        
        gen_result = client.generate_code(
            snippets, 
            language="python", 
            output_dir=output_dir
        )
        
        print(f"Code generation result: {json.dumps(gen_result, indent=2)}")
        print(f"Generated code saved to {output_dir}/")
    
    # Show the current configuration
    print("\n3. Getting current configuration...")
    config = client.get_config()
    
    # Remove sensitive information for display
    if 'claude' in config and 'api_key' in config['claude']:
        config['claude']['api_key'] = '***********'
    if 'openai' in config and 'api_key' in config['openai']:
        config['openai']['api_key'] = '***********'
    
    print(f"Current configuration: {json.dumps(config, indent=2)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}") 