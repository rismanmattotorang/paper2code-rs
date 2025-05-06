// src/text/mod.rs
pub mod processor;
pub mod code_detector;

pub use processor::TextProcessor;
pub use code_detector::{CodeBlock, CodeDetector};