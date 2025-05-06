// src/code/mod.rs

// Public submodules
pub mod generator;
pub mod writer;

// Re-export commonly used items
pub use generator::CodeGenerator;
pub use writer::CodeWriter;