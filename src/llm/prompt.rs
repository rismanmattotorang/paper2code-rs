// src/llm/prompt.rs
use std::collections::HashMap;

/// Builder for creating prompts for LLM APIs
#[derive(Debug, Clone)]
pub struct PromptBuilder {
    template: String,
    replacements: HashMap<String, String>,
    code_block: String,
    language: String,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self {
            template: String::new(),
            replacements: HashMap::new(),
            code_block: String::new(),
            language: "unknown".to_string(),
        }
    }
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the prompt template with placeholders
    pub fn with_template(mut self, template: &str) -> Self {
        self.template = template.to_string();
        self
    }

    /// Add the code block extracted from the paper
    pub fn with_code_block(mut self, code: &str) -> Self {
        self.code_block = code.to_string();
        self.replacements.insert("{{CODE}}".to_string(), code.to_string());
        self
    }

    /// Add the detected programming language
    pub fn with_language(mut self, language: &str) -> Self {
        self.language = language.to_string();
        self.replacements.insert("{{LANGUAGE}}".to_string(), language.to_string());
        self
    }

    /// Add a custom replacement
    pub fn with_replacement(mut self, key: &str, value: &str) -> Self {
        self.replacements.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Get the code block
    pub fn get_code_block(&self) -> &str {
        &self.code_block
    }
    
    /// Get the language
    pub fn get_language(&self) -> &str {
        &self.language
    }

    /// Build the final prompt string
    pub fn build(&self) -> String {
        let mut prompt = self.template.clone();
        
        for (key, value) in &self.replacements {
            prompt = prompt.replace(key, value);
        }
        
        prompt
    }
    
    /// Get default prompt template for converting code snippets to executable code
    pub fn default_code_extraction_template() -> String {
        r"
You are a skilled programmer tasked with converting research paper code snippets into fully executable code.

The following code was extracted from a research paper:

```{{LANGUAGE}}
{{CODE}}
```

Please convert this code snippet into fully executable {{LANGUAGE}} code. Follow these guidelines:

1. Fill in any missing functions, imports, or definitions
2. Add appropriate error handling
3. Make the code work as a standalone program
4. Add clear comments explaining how the code works
5. Preserve the original algorithm and approach
6. Add code documentation following best practices for {{LANGUAGE}}

Your output should be ONLY the complete, executable code. Do not include any explanations outside of code comments.
".to_string()
    }
    
    /// Template for code block detection and improvement
    pub fn code_detection_template() -> String {
        r"
You are a skilled programmer tasked with identifying and extracting code from research papers.

The following text was extracted from a research paper:

```
{{TEXT}}
```

Please identify any code blocks in this text. For each code block:
1. Extract the code, preserving its structure
2. Fix any obvious formatting issues, indentation, or syntax errors
3. Identify the programming language if possible
4. Present the code as a properly formatted code block

Your output should be ONLY the extracted, formatted code blocks. 
Return each block with the language identified (if possible) in markdown code block format.
".to_string()
    }
    
    /// Template for adding documentation to code
    pub fn documentation_template() -> String {
        r"
You are a skilled technical writer tasked with documenting code from a research paper.

The following code was extracted and converted from a research paper:

```{{LANGUAGE}}
{{CODE}}
```

Please add comprehensive documentation to this code. Follow these guidelines:

1. Add a detailed header comment explaining the purpose of the code
2. Document each function with clear descriptions of parameters and return values
3. Explain complex algorithms or data structures
4. Add inline comments for non-obvious code sections
5. Follow documentation best practices for {{LANGUAGE}}
6. Do not change the functionality of the code

Your output should be ONLY the documented code.
".to_string()
    }
    
    /// Template for bug fixing
    pub fn bug_fixing_template() -> String {
        r"
You are a skilled programmer tasked with fixing bugs in code extracted from a research paper.

The following code was extracted from a research paper:

```{{LANGUAGE}}
{{CODE}}
```

This code may contain bugs, errors, or inconsistencies. Please fix any issues with this code while preserving the original functionality and intent. Follow these guidelines:

1. Identify and fix syntax errors
2. Correct logical bugs
3. Handle edge cases appropriately
4. Ensure the code is robust and works as intended
5. Add appropriate error handling if missing
6. Add comments explaining your fixes

Your output should be ONLY the fixed, executable code.
".to_string()
    }
    
    /// Template for comparing and merging implementations
    pub fn merge_implementations_template() -> String {
        r"
You are a skilled programmer tasked with merging two implementations of the same algorithm from a research paper.

Original code from the paper:
```{{LANGUAGE}}
{{ORIGINAL}}
```

First implementation:
```{{LANGUAGE}}
{{IMPL1}}
```

Second implementation:
```{{LANGUAGE}}
{{IMPL2}}
```

Please merge these implementations into a single, optimized version. Follow these guidelines:

1. Keep the best aspects of each implementation
2. Ensure correctness and completeness
3. Optimize for performance and readability
4. Add comprehensive documentation
5. Ensure the code is fully executable
6. Follow best practices for {{LANGUAGE}}

Your output should be ONLY the merged, executable code.
".to_string()
    }
}