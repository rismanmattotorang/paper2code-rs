// tests/text_tests.rs
use paper2code_rs::{
    text::{
        code_detector::CodeDetector,
        processor::TextProcessor,
    },
    AppResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_code_detection() {
        let detector = CodeDetector::new(0.7, 3);
        
        // Python code example
        let python_code = r#"
        def hello_world():
            print("Hello, world!")
            return 0
        
        if __name__ == "__main__":
            hello_world()
        "#;
        
        // Test language detection through the public detect_code_blocks method
        let blocks = detector.detect_code_blocks(python_code);
        assert!(!blocks.is_empty(), "Should detect at least one code block");
        
        if !blocks.is_empty() {
            let block = &blocks[0];
            assert!(block.confidence >= 0.7, "Python code confidence should be high: {}", block.confidence);
            assert_eq!(block.language.as_deref().unwrap_or_default(), "python", "Should detect Python language");
        }
        
        // Regular text
        let regular_text = r#"
        This is a paragraph about coding, but it's not actual code.
        It mentions variables and functions, but in prose format.
        "#;
        
        // Regular text should not produce code blocks above minimum confidence
        let blocks = detector.detect_code_blocks(regular_text);
        assert!(blocks.is_empty(), "Should not detect code blocks in regular text");
    }
    
    #[tokio::test]
    async fn test_text_processing() -> AppResult<()> {
        let detector = CodeDetector::new(0.7, 3);
        let processor = TextProcessor::new(detector, 1000);
        
        let chunks = vec![
            "This is regular text with no code.".to_string(),
            r#"
            Here's some Python code:
            
            def multiply(a, b):
                return a * b
                
            print(multiply(5, 3))
            "#.to_string(),
            "More regular text here.".to_string(),
        ];
        
        let blocks = processor.process_chunks(&chunks).await?;
        assert_eq!(blocks.len(), 1, "Should detect exactly one code block");
        
        let block = &blocks[0];
        assert_eq!(block.language.as_deref().unwrap_or_default(), "python", "Should be Python code");
        assert!(block.content.contains("def multiply"), "Should contain the function definition");
        
        Ok(())
    }
}
