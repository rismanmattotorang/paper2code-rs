// src/code/mod.rs

// Public submodules
pub mod generator;
pub mod domain_aware_generator;
pub mod writer;

// Re-export commonly used items
pub use generator::CodeGenerator;
pub use domain_aware_generator::DomainAwareCodeGenerator;
pub use writer::CodeWriter;