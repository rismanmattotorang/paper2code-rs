// src/text/mod.rs
pub mod processor;
pub mod code_detector;
pub mod domain_detector;

pub use processor::TextProcessor;
pub use code_detector::{CodeBlock, CodeDetector};
pub use domain_detector::{ComputationalDomain, DomainDetector};