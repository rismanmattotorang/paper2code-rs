// src/cli.rs
use clap::{Parser, Subcommand, Args};
use std::path::PathBuf;

/// CLI Application for Paper2Code
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    
    /// Path to configuration file
    #[arg(short, long, global = true, default_value = "config.toml")]
    pub config: PathBuf,
    
    /// Verbose output mode
    #[arg(short, long, global = true)]
    pub verbose: bool,
}

/// Available commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Extract code from PDF papers
    Extract(ExtractArgs),
    
    /// Generate configuration file
    Config(ConfigArgs),
    
    /// Test LLM connectivity
    Test(TestArgs),
}

/// Arguments for the extract command
#[derive(Args, Debug)]
pub struct ExtractArgs {
    /// Input PDF files to process
    #[arg(short, long, required = true)]
    pub input: Vec<PathBuf>,
    
    /// Output directory for generated code
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    
    /// Force overwrite of existing files
    #[arg(short, long)]
    pub force: bool,
    
    /// LLM strategy to use
    #[arg(long)]
    pub strategy: Option<String>,
    
    /// Programming language to target (auto-detect if not specified)
    #[arg(long)]
    pub language: Option<String>,
}

/// Arguments for the config command
#[derive(Args, Debug)]
pub struct ConfigArgs {
    /// Generate a new configuration file
    #[arg(short, long)]
    pub generate: bool,
    
    /// Path to save the configuration file
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    
    /// Force overwrite of existing configuration file
    #[arg(short, long)]
    pub force: bool,
}

/// Arguments for the test command
#[derive(Args, Debug)]
pub struct TestArgs {
    /// Test OpenAI API connectivity
    #[arg(long)]
    pub openai: bool,
    
    /// Test Claude API connectivity
    #[arg(long)]
    pub claude: bool,
    
    /// Test with a specific prompt
    #[arg(short, long)]
    pub prompt: Option<String>,
}