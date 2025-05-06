// src/text/code_detector.rs
use crate::error::AppError;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Represents a detected code block from a research paper
#[derive(Debug, Clone)]
pub struct CodeBlock {
    pub content: String,
    pub language: Option<String>,
    pub line_start: usize,
    pub line_end: usize,
    pub page_number: Option<u32>,
    pub confidence: f64,
    pub metadata: HashMap<String, String>,
}

impl CodeBlock {
    /// Creates a new code block with the specified content
    pub fn new(
        content: String, 
        language: Option<String>, 
        line_start: usize, 
        line_end: usize,
        page_number: Option<u32>,
    ) -> Self {
        Self {
            content,
            language,
            line_start,
            line_end,
            page_number,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }
    
    /// Adds metadata to the code block
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Sets the confidence score for this code block
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Responsible for detecting code blocks in extracted text
pub struct CodeDetector {
    patterns: Vec<Arc<Regex>>,
    language_patterns: HashMap<String, Arc<Regex>>,
    min_confidence: f64,
    min_lines: usize,
}

impl Default for CodeDetector {
    fn default() -> Self {
        let mut detector = Self {
            patterns: Vec::new(),
            language_patterns: HashMap::new(),
            min_confidence: 0.7,
            min_lines: 3,
        };
        
        // Initialize with default patterns
        detector.init_default_patterns();
        detector
    }
}

impl CodeDetector {
    pub fn new(min_confidence: f64, min_lines: usize) -> Self {
        let mut detector = Self {
            patterns: Vec::new(),
            language_patterns: HashMap::new(),
            min_confidence,
            min_lines,
        };
        
        detector.init_default_patterns();
        detector
    }
    
    /// Initialize default regex patterns for code detection
    fn init_default_patterns(&mut self) {
        // Markdown code blocks
        let _ = self.add_pattern(r"```(?P<lang>\w+)?\s*\n(?P<code>[\s\S]+?)\n```");
        
        // Indented code blocks
        let _ = self.add_pattern(r"(?m)^( {4}|\t)(.+)$");
        
        // Line numbered code
        let _ = self.add_pattern(r"(?m)^\s*(\d+)\s*[\:\|]\s*(.+)$");
        
        // Language-specific patterns
        let _ = self.add_language_pattern("python", r"(?m)(^|\n)(import\s+[\w\.]+|from\s+[\w\.]+\s+import\s+[\w\.\*]+|def\s+\w+\s*\(.*\)\s*\:|class\s+\w+(\s*\(.*\))?\s*\:)");
        let _ = self.add_language_pattern("java", r"(?m)(^|\n)(public\s+class|private\s+class|protected\s+class|class\s+\w+|public\s+static\s+void\s+main)");
        let _ = self.add_language_pattern("rust", r"(?m)(^|\n)(fn\s+\w+|struct\s+\w+|enum\s+\w+|impl\s+\w+|use\s+[\w\:\:]+;|pub\s+fn)");
        let _ = self.add_language_pattern("cpp", r#"(?m)(^|\n)(#include\s*[<"][a-zA-Z0-9\.]+[>"]\s*|namespace\s+[a-zA-Z0-9_]+|class\s+[a-zA-Z0-9_]+|void\s+[a-zA-Z0-9_]+\s*\()"#);
    }
    
    /// Add a new pattern for code block detection
    pub fn add_pattern(&mut self, pattern: &str) -> Result<(), AppError> {
        let regex = Regex::new(pattern)
            .map_err(|e| AppError::RegexError(e.to_string()))?;
        self.patterns.push(Arc::new(regex));
        Ok(())
    }
    
    /// Add a language-specific pattern
    pub fn add_language_pattern(&mut self, language: &str, pattern: &str) -> Result<(), AppError> {
        let regex = Regex::new(pattern)
            .map_err(|e| AppError::RegexError(e.to_string()))?;
        self.language_patterns.insert(language.to_string(), Arc::new(regex));
        Ok(())
    }
    
    /// Detect code blocks in a text chunk
    pub fn detect_code_blocks(&self, text: &str) -> Vec<CodeBlock> {
        let mut code_blocks = Vec::new();
        
        // Find code blocks using general patterns
        for pattern in &self.patterns {
            for capture in pattern.captures_iter(text) {
                if let Some(code_match) = capture.get(0) {
                    let content = code_match.as_str().to_string();
                    
                    // Extract language if available (from pattern like ```python)
                    let language = capture.name("lang").map(|m| m.as_str().to_string());
                    
                    // Count line numbers for the block
                    let text_before = &text[..code_match.start()];
                    let line_start = text_before.chars().filter(|&c| c == '\n').count() + 1;
                    let line_count = content.chars().filter(|&c| c == '\n').count();
                    let line_end = line_start + line_count;
                    
                    // Skip if block is too short
                    if line_count < self.min_lines {
                        continue;
                    }
                    
                    // Create the code block with initial confidence
                    let mut block = CodeBlock::new(
                        content,
                        language,
                        line_start,
                        line_end,
                        None,
                    ).with_confidence(0.8);
                    
                    // Try to detect language if not already known
                    if block.language.is_none() {
                        block.language = self.detect_language(&block.content);
                    }
                    
                    code_blocks.push(block);
                }
            }
        }
        
        // Filter out blocks with low confidence
        code_blocks.retain(|block| block.confidence >= self.min_confidence);
        
        code_blocks
    }
    
    /// Try to detect the programming language of a code block
    fn detect_language(&self, code: &str) -> Option<String> {
        let mut best_match = None;
        let mut highest_score = 0.0;
        
        for (language, pattern) in &self.language_patterns {
            if pattern.is_match(code) {
                // Count how many distinct pattern matches we get
                let match_count = pattern.find_iter(code).count();
                
                // Simple scoring: more matches = higher confidence
                let score = match_count as f64 * 0.1;
                
                if score > highest_score {
                    highest_score = score;
                    best_match = Some(language.clone());
                }
            }
        }
        
        best_match
    }
    
    /// Process text chunks in parallel to detect code blocks
    pub async fn detect_code_blocks_parallel(&self, chunks: &[String]) -> Vec<CodeBlock> {
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];
        
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk = chunk.clone();
            let results_clone = Arc::clone(&results);
            let detector = self.clone();
            
            let handle = tokio::spawn(async move {
                let blocks = detector.detect_code_blocks(&chunk);
                
                // Add chunk index metadata to help with merging
                let blocks: Vec<CodeBlock> = blocks.into_iter()
                    .map(|b| b.with_metadata("chunk_index", &chunk_idx.to_string()))
                    .collect();
                
                let mut results = results_clone.lock().await;
                results.extend(blocks);
            });
            
            handles.push(handle);
        }
        
        // Wait for all detection tasks to complete
        for handle in handles {
            let _ = handle.await;
        }
        
        let blocks = Arc::try_unwrap(results)
            .expect("Failed to unwrap Arc")
            .into_inner();
        
        blocks
    }
}

impl Clone for CodeDetector {
    fn clone(&self) -> Self {
        Self {
            patterns: self.patterns.clone(),
            language_patterns: self.language_patterns.clone(),
            min_confidence: self.min_confidence,
            min_lines: self.min_lines,
        }
    }
} 