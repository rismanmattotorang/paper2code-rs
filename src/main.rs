// src/main.rs
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tokio::fs;
use tracing::{info, error, warn, Level};
use tracing_subscriber::FmtSubscriber;
use uuid::Uuid;

// Import from paper2code_rs crate
use paper2code_rs::{
    AppConfig,
    AppError,
    cli::Commands,
    CodeGenerator,
    CodeBlock,
    CodeDetector,
    PdfExtractor,
    TextProcessor,
    cli::{Cli, ExtractArgs, ConfigArgs, TestArgs},
    llm::{
        client::{LlmClient, ClaudeClient, OpenAiClient, MultiLlmClient},
        strategy::LlmStrategy,
        prompt::PromptBuilder,
    },
};

// Import from local modules

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let cli = Cli::parse();
    
    // Initialize logging based on verbosity
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
    
    // Process commands
    match &cli.command {
        Commands::Extract(args) => {
            process_extract_command(args, &cli.config).await?;
        },
        Commands::Config(args) => {
            process_config_command(args, &cli.config).await?;
        },
        Commands::Test(args) => {
            process_test_command(args, &cli.config).await?;
        },
    }
    
    Ok(())
}

/// Process the extract command
async fn process_extract_command(args: &ExtractArgs, config_path: &PathBuf) -> Result<()> {
    info!("Starting code extraction process");
    
    // Load configuration
    let mut config = match AppConfig::load(config_path) {
        Ok(config) => {
            info!("Loaded configuration from {:?}", config_path);
            config
        },
        Err(e) => {
            // If config failed to load but file exists, it's a real error
            if config_path.exists() {
                error!("Failed to load configuration: {}", e);
                return Err(e.into());
            }
            
            // Otherwise, use default config
            info!("No configuration found, using defaults");
            AppConfig::default()
        }
    };
    
    // Override output path if specified in CLI
    if let Some(output_path) = &args.output {
        config.output.path = output_path.to_string_lossy().to_string();
    }
    
    // Override strategy if specified
    let strategy = if let Some(strategy_name) = &args.strategy {
        match strategy_name.as_str() {
            "openai_only" => LlmStrategy::OpenAiOnly,
            "claude_only" => LlmStrategy::ClaudeOnly,
            "openai_first" => LlmStrategy::OpenAiFirstClaudeSecond,
            "claude_first" => LlmStrategy::ClaudeFirstOpenAiSecond,
            "compare_and_merge" => LlmStrategy::CompareAndMerge,
            "adaptive" => config.get_llm_strategy(),
            _ => {
                warn!("Unknown strategy '{}', using default", strategy_name);
                config.get_llm_strategy()
            }
        }
    } else {
        config.get_llm_strategy()
    };
    
    // Override force flag
    if args.force {
        config.output.overwrite = true;
    }
    
    // Ensure output directory exists
    fs::create_dir_all(&config.output.path).await?;
    
    // Initialize LLM clients
    let multi_llm_client = create_multi_llm_client(&config, strategy.clone()).await?;
    
    // Initialize components
    let pdf_extractor = PdfExtractor::new(config.pdf.chunk_size);
    
    let text_processor = TextProcessor::new(
        CodeDetector::new(
            config.code_detection.min_confidence,
            config.code_detection.min_lines,
        ),
        config.pdf.chunk_size,
    );
    
    let code_generator = CodeGenerator::new(
        Box::new(multi_llm_client),
        config.output.path.clone(),
        strategy,
    );
    
    // Process each input PDF
    let total_start_time = Instant::now();
    let mut successful = 0;
    let mut failed = 0;
    
    for input_path in &args.input {
        info!("Processing PDF: {:?}", input_path);
        
        match process_pdf(
            input_path,
            &pdf_extractor,
            &text_processor,
            &code_generator,
            &config,
            args.language.as_deref(),
        ).await {
            Ok(code_count) => {
                info!("Successfully processed {:?} - generated {} code files", input_path, code_count);
                successful += 1;
            },
            Err(e) => {
                error!("Failed to process {:?}: {}", input_path, e);
                eprintln!("Error processing {}: {}", input_path.display(), e);
                failed += 1;
            }
        }
    }
    
    let total_elapsed = total_start_time.elapsed();
    info!("Processing completed in {:.2?}", total_elapsed);
    info!("Summary: {} successful, {} failed", successful, failed);
    
    if successful > 0 {
        println!("Successfully processed {} PDF files. Generated code is in: {}", 
            successful, config.output.path);
    }
    
    Ok(())
}

/// Process the config command
async fn process_config_command(args: &ConfigArgs, config_path: &PathBuf) -> Result<()> {
    if args.generate {
        // Determine output path
        let output_path = args.output.as_ref().unwrap_or(config_path);
        
        // Check if file exists and we're not forcing overwrite
        if output_path.exists() && !args.force {
            error!("Configuration file already exists: {:?}", output_path);
            eprintln!("Configuration file already exists. Use --force to overwrite.");
            return Err(anyhow::anyhow!("Configuration file already exists"));
        }
        
        // Create default config
        let default_config = AppConfig::default();
        
        // Ensure parent directory exists
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await?;
            }
        }
        
        // Serialize and save config
        let config_str = toml::to_string_pretty(&default_config)
            .map_err(|e| AppError::ConfigError(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(output_path, config_str).await?;
        
        info!("Created configuration file: {:?}", output_path);
        println!("Created configuration file: {}", output_path.display());
    } else {
        // Show current config
        match AppConfig::load(config_path) {
            Ok(config) => {
                let config_str = toml::to_string_pretty(&config)
                    .map_err(|e| AppError::ConfigError(format!("Failed to serialize config: {}", e)))?;
                
                println!("Current configuration:\n\n{}", config_str);
            },
            Err(e) => {
                error!("Failed to load configuration: {}", e);
                eprintln!("Failed to load configuration: {}", e);
                return Err(e.into());
            }
        }
    }
    
    Ok(())
}

/// Process the test command
async fn process_test_command(args: &TestArgs, config_path: &PathBuf) -> Result<()> {
    // Load configuration
    let config = match AppConfig::load(config_path) {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            eprintln!("Failed to load configuration: {}", e);
            return Err(e.into());
        }
    };
    
    // Default test prompt
    let prompt_text = args.prompt.clone().unwrap_or_else(|| 
        "Write a simple 'Hello, World!' program.".to_string()
    );
    
    let prompt = PromptBuilder::new()
        .with_template("{{PROMPT}}")
        .with_replacement("{{PROMPT}}", &prompt_text)
        .with_language("python");  // Default language for testing
    
    // Test OpenAI if requested
    if args.openai || (!args.openai && !args.claude) {
        println!("Testing OpenAI API connection...");
        
        if let Some(openai_config) = &config.openai {
            match OpenAiClient::new(openai_config.clone()) {
                Ok(client) => {
                    match client.generate_code(&prompt).await {
                        Ok(response) => {
                            println!("OpenAI API test successful!");
                            println!("\nResponse:\n{}", response);
                        },
                        Err(e) => {
                            error!("OpenAI API test failed: {}", e);
                            eprintln!("OpenAI API test failed: {}", e);
                        }
                    }
                },
                Err(e) => {
                    error!("Failed to initialize OpenAI client: {}", e);
                    eprintln!("Failed to initialize OpenAI client: {}", e);
                }
            }
        } else {
            warn!("OpenAI API not configured");
            eprintln!("OpenAI API not configured in config file");
        }
    }
    
    // Test Claude if requested
    if args.claude || (!args.openai && !args.claude) {
        println!("\nTesting Claude API connection...");
        
        if let Some(claude_config) = &config.claude {
            match ClaudeClient::new(claude_config.clone()) {
                Ok(client) => {
                    match client.generate_code(&prompt).await {
                        Ok(response) => {
                            println!("Claude API test successful!");
                            println!("\nResponse:\n{}", response);
                        },
                        Err(e) => {
                            error!("Claude API test failed: {}", e);
                            eprintln!("Claude API test failed: {}", e);
                        }
                    }
                },
                Err(e) => {
                    error!("Failed to initialize Claude client: {}", e);
                    eprintln!("Failed to initialize Claude client: {}", e);
                }
            }
        } else {
            warn!("Claude API not configured");
            eprintln!("Claude API not configured in config file");
        }
    }
    
    Ok(())
}

/// Create a Multi-LLM client from config
async fn create_multi_llm_client(
    config: &AppConfig,
    strategy: LlmStrategy,
) -> Result<MultiLlmClient, AppError> {
    // Initialize OpenAI client if configured
    let openai_client = if let Some(openai_config) = &config.openai {
        match OpenAiClient::new(openai_config.clone()) {
            Ok(client) => {
                info!("Initialized OpenAI API client");
                Some(Box::new(client) as Box<dyn LlmClient>)
            },
            Err(e) => {
                warn!("Failed to initialize OpenAI API client: {}", e);
                None
            }
        }
    } else {
        info!("OpenAI API not configured");
        None
    };
    
    // Initialize Claude client if configured
    let claude_client = if let Some(claude_config) = &config.claude {
        match ClaudeClient::new(claude_config.clone()) {
            Ok(client) => {
                info!("Initialized Claude API client");
                Some(Box::new(client) as Box<dyn LlmClient>)
            },
            Err(e) => {
                warn!("Failed to initialize Claude client: {}", e);
                None
            }
        }
    } else {
        info!("Claude API not configured");
        None
    };
    
    // Ensure at least one client is configured
    if openai_client.is_none() && claude_client.is_none() {
        return Err(AppError::ConfigError(
            "No LLM API clients configured. Please configure at least one of OpenAI or Claude API.".to_string()
        ));
    }
    
    // Create multi-client
    let multi_client = MultiLlmClient::new(openai_client, claude_client, strategy);
    
    Ok(multi_client)
}

/// Process a single PDF file
async fn process_pdf(
    input_path: &PathBuf,
    pdf_extractor: &PdfExtractor,
    text_processor: &TextProcessor,
    code_generator: &CodeGenerator,
    config: &AppConfig,
    target_language: Option<&str>,
) -> Result<usize, AppError> {
    let start_time = Instant::now();
    
    // 1. Validate PDF file
    if !input_path.exists() {
        return Err(AppError::FileError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("File not found: {:?}", input_path)
        )));
    }
    
    // Create PDF-specific output directory
    let pdf_name = input_path.file_stem()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| format!("pdf_{}", Uuid::new_v4()));
    
    let output_dir = format!("{}/{}", config.output.path, pdf_name);
    fs::create_dir_all(&output_dir).await?;
    
    // 2. Extract text from PDF with error recovery
    info!("Extracting text from PDF");
    let extract_start = Instant::now();
    
    // Try parallel extraction first, fall back to regular if it fails
    let text_chunks = match pdf_extractor.extract_text_from_file(input_path).await {
        Ok(chunks) => {
            info!("Extraction successful - {} chunks", chunks.len());
            chunks
        },
        Err(e) => {
            warn!("Extraction failed: {}. Falling back to retry with recovery.", e);
            // Use the error recovery module to retry with backoff
            paper2code_rs::utils::error_recovery::pdf::extract_text_with_recovery(
                pdf_extractor, 
                input_path
            ).await?
        }
    };
    
    let extract_elapsed = extract_start.elapsed();
    info!("Text extraction completed in {:.2?} - {} chunks", extract_elapsed, text_chunks.len());
    
    if text_chunks.is_empty() {
        return Err(AppError::PdfExtractError(format!(
            "No text extracted from PDF: {:?}", input_path
        )));
    }
    
    // 3. Process text to find code blocks
    info!("Detecting code blocks");
    let detection_start = Instant::now();
    let mut code_blocks = text_processor.process_chunks(&text_chunks).await?;
    let detection_elapsed = detection_start.elapsed();
    
    info!("Code detection completed in {:.2?} - {} blocks found", 
        detection_elapsed, code_blocks.len());
    
    if code_blocks.is_empty() {
        warn!("No code blocks detected automatically, using LLM to search for code");
        
        // Use LLM to try to find code blocks
        let code_blocks_from_llm = detect_code_blocks_with_llm(
            &text_chunks, 
            code_generator.client(),
            &config.prompt.detection_template
        ).await?;
        
        if !code_blocks_from_llm.is_empty() {
            info!("LLM detected {} code blocks", code_blocks_from_llm.len());
            code_blocks = code_blocks_from_llm;
        } else {
            warn!("No code blocks found in PDF: {:?}", input_path);
            return Ok(0);
        }
    }
    
    // If target language is specified, filter code blocks
    if let Some(lang) = target_language {
        info!("Filtering code blocks for language: {}", lang);
        let lang = lang.to_lowercase();
        
        // Keep blocks with matching language or without detected language
        code_blocks.retain(|block| {
            if let Some(block_lang) = &block.language {
                block_lang.to_lowercase() == lang
            } else {
                true // Keep blocks without detected language
            }
        });
        
        info!("Filtered to {} code blocks", code_blocks.len());
        
        if code_blocks.is_empty() {
            warn!("No code blocks found for language {}: {:?}", lang, input_path);
            return Ok(0);
        }
    }
    
    // 4. Generate executable code
    info!("Generating executable code from {} blocks", code_blocks.len());
    let generation_start = Instant::now();
    
    // Create a customized prompt with PDF filename
    let mut prompt_template = config.prompt.extraction_template.clone();
    prompt_template = prompt_template.replace(
        "You are a skilled programmer", 
        &format!("You are a skilled programmer analyzing the paper '{}'", pdf_name)
    );
    
    // Set output directory to PDF-specific directory
    let modified_code_generator = CodeGenerator::new(
        code_generator.client().clone(),
        output_dir,
        code_generator.strategy().clone(),
    );
    
    let result_paths = modified_code_generator.generate_from_blocks(
        code_blocks.as_slice(), 
        &prompt_template,
        target_language,
    ).await?;
    
    let generation_elapsed = generation_start.elapsed();
    info!("Code generation completed in {:.2?} - {} files generated", 
        generation_elapsed, result_paths.len());
    
    // 5. Report completion
    let total_elapsed = start_time.elapsed();
    info!("PDF processing completed in {:.2?}", total_elapsed);
    
    Ok(result_paths.len())
}

/// Use LLM to detect code blocks in text
async fn detect_code_blocks_with_llm(
    text_chunks: &[String],
    llm_client: &Box<dyn LlmClient>,
    template: &str,
) -> Result<Vec<CodeBlock>, AppError> {
    let mut code_blocks = Vec::new();
    
    // Process each chunk with the LLM
    for (i, chunk) in text_chunks.iter().enumerate() {
        // Skip if chunk is too small
        if chunk.len() < 100 {
            continue;
        }
        
        // Create prompt for code detection
        let prompt = PromptBuilder::new()
            .with_template(template)
            .with_replacement("{{TEXT}}", chunk);
        
        // Send to LLM
        match llm_client.generate_code(&prompt).await {
            Ok(response) => {
                // Parse the response to find code blocks
                let blocks = parse_code_blocks_from_llm_response(&response, i);
                code_blocks.extend(blocks);
            },
            Err(e) => {
                warn!("Failed to detect code in chunk {}: {}", i, e);
                // Continue with other chunks
            }
        }
    }
    
    Ok(code_blocks)
}

/// Parse code blocks from LLM response
fn parse_code_blocks_from_llm_response(
    response: &str,
    chunk_index: usize,
) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    
    // Find code blocks in markdown format: ```language
    let mut line_num = 1;
    let mut in_code_block = false;
    let mut current_block = String::new();
    let mut current_language = None;
    let mut start_line = 0;
    
    for line in response.lines() {
        if line.starts_with("```") && !in_code_block {
            // Start of code block
            in_code_block = true;
            start_line = line_num;
            
            // Try to detect language
            let lang = line.trim_start_matches('`').trim();
            if !lang.is_empty() {
                current_language = Some(lang.to_string());
            }
        } else if line.starts_with("```") && in_code_block {
            // End of code block
            in_code_block = false;
            
            // Add block if not empty
            if !current_block.is_empty() {
                let block = CodeBlock::new(
                    current_block.clone(),
                    current_language.clone(),
                    start_line,
                    line_num,
                    None,
                )
                .with_metadata("chunk_index", &chunk_index.to_string())
                .with_confidence(0.9);
                
                blocks.push(block);
                
                // Reset for next block
                current_block.clear();
                current_language = None;
            }
        } else if in_code_block {
            // Content of code block
            current_block.push_str(line);
            current_block.push('\n');
        }
        
        line_num += 1;
    }
    
    // Handle unclosed code block
    if in_code_block && !current_block.is_empty() {
        let block = CodeBlock::new(
            current_block,
            current_language,
            start_line,
            line_num,
            None,
        )
        .with_metadata("chunk_index", &chunk_index.to_string())
        .with_confidence(0.8);  // Lower confidence for unclosed block
        
        blocks.push(block);
    }
    
    blocks
}