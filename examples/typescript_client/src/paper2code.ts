/**
 * TypeScript client for the paper2code-rs library
 * 
 * This client provides a TypeScript/JavaScript interface to the paper2code-rs library.
 * It uses child_process to call the paper2code-rs CLI and processes the output to provide
 * a consistent API regardless of CLI version or output format.
 */

// Node.js modules with proper TypeScript imports
import * as childProcess from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

/**
 * Custom error class for Paper2Code client errors
 */
export class Paper2CodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'Paper2CodeError';
    // Restore prototype chain
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Error thrown when the paper2code-rs binary is not found
 */
export class BinaryNotFoundError extends Paper2CodeError {
  constructor(message: string) {
    super(message);
    this.name = 'BinaryNotFoundError';
  }
}

/**
 * Error thrown when a command fails to execute
 */
export class CommandExecutionError extends Paper2CodeError {
  constructor(message: string) {
    super(message);
    this.name = 'CommandExecutionError';
  }
}

/**
 * Error thrown when invalid input is provided
 */
export class InvalidInputError extends Paper2CodeError {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidInputError';
  }
}

/**
 * Code block extracted from a document
 */
export interface CodeBlock {
  content: string;
  language?: string;
  line_start: number;
  line_end: number;
  page_number?: number;
  confidence: number;
  metadata: Record<string, string>;
}

/**
 * Static method to extract a code block from text using regex
 */
export function extractCodeBlockFromText(text: string): CodeBlock {
  // Try to determine language from markdown code block format
  const langMatch = text.match(/```(\w+)/);
  const language = langMatch ? langMatch[1] : undefined;
  
  // Extract content between code markers
  const contentMatch = text.match(/```(?:\w+)?\n([\s\S]+?)\n```/s);
  const content = contentMatch ? contentMatch[1] : text;
  
  return {
    content,
    language,
    line_start: 0,
    line_end: content.split('\n').length,
    confidence: 1.0,
    metadata: {}
  };
}

/**
 * Result from code extraction
 */
export interface ExtractionResult {
  code_blocks: CodeBlock[];
  source_file?: string;
  raw_output?: string;
  success: boolean;
}

/**
 * Configuration for OpenAI
 */
export interface OpenAIConfig {
  api_key: string;
  model?: string;
  timeout_seconds?: number;
}

/**
 * Configuration for Claude
 */
export interface ClaudeConfig {
  api_key: string;
  model?: string;
  timeout_seconds?: number;
}

/**
 * Application configuration
 */
export interface AppConfig {
  openai?: OpenAIConfig;
  claude?: ClaudeConfig;
  output_dir?: string;
  raw_config?: string;
  success?: boolean;
}

/**
 * Options for extracting code from PDF
 */
export interface ExtractOptions {
  outputDir?: string;
  language?: string;
  strategy?: string;
  force?: boolean;
}

/**
 * Options for generating a configuration file
 */
export interface ConfigOptions {
  generate?: boolean;
  outputPath?: string;
  force?: boolean;
}

/**
 * Options for testing LLM connectivity
 */
export interface TestOptions {
  openai?: boolean;
  claude?: boolean;
  prompt?: string;
}

/**
 * Client for interacting with the paper2code-rs library
 */
export class Paper2CodeClient {
  private binaryPath: string;
  private configPath: string;
  private verbose: boolean;
  
  /**
   * Create a new Paper2CodeClient
   * 
   * @param binaryPath Path to the paper2code-rs binary
   * @param configPath Path to the configuration file
   * @param verbose Whether to enable verbose output
   */
  constructor(binaryPath: string = 'paper2code-rs', configPath: string = 'config.toml', verbose: boolean = false) {
    this.binaryPath = binaryPath;
    this.configPath = configPath;
    this.verbose = verbose;
  }

  /**
   * Check if the paper2code-rs binary exists and is executable
   */
  public async checkBinary(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      childProcess.exec(`${this.binaryPath} --version`, (error: Error | null, stdout: string, stderr: string) => {
        if (error) {
          reject(new BinaryNotFoundError(`Failed to run paper2code-rs: ${error.message}. Make sure it's installed and on your PATH.`));
        } else {
          console.log(`Using paper2code-rs version: ${stdout.trim()}`);
          resolve();
        }
      });
    });
  }

  /**
   * Run a paper2code-rs command and return the processed output
   * 
   * @param args Command arguments
   * @param stdinData Optional data to pass to stdin
   * @returns Processed command output
   */
  private async runCommand<T>(args: string[], stdinData?: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      // Prepare command with global options
      let allArgs: string[] = [];
      
      // Add config path if not default
      if (this.configPath !== 'config.toml') {
        allArgs.push('--config', this.configPath);
      }
      
      // Add verbose flag if enabled
      if (this.verbose) {
        allArgs.push('--verbose');
      }
      
      // Add command-specific arguments
      allArgs = [...allArgs, ...args];
      
      console.log(`Running command: ${this.binaryPath} ${allArgs.join(' ')}`);
      
      const cmd = childProcess.spawn(this.binaryPath, allArgs);
      
      let stdout = '';
      let stderr = '';
      
      cmd.stdout.on('data', (data: Buffer) => {
        stdout += data.toString();
      });
      
      cmd.stderr.on('data', (data: Buffer) => {
        stderr += data.toString();
      });
      
      if (stdinData) {
        cmd.stdin.write(stdinData);
        cmd.stdin.end();
      }
      
      cmd.on('close', (code: number) => {
        if (code !== 0) {
          reject(new CommandExecutionError(`Command failed with error: ${stderr.trim()}`));
          return;
        }
        
        // Process output based on command
        try {
          // Try to parse as JSON first (not likely with current CLI)
          const result = JSON.parse(stdout.trim());
          resolve(result as T);
        } catch (e) {
          // If this was an extract command, parse using regex for code blocks
          if (args[0] === 'extract') {
            resolve(this.parseExtractOutput(stdout) as unknown as T);
          } else {
            // For other commands, return standardized dictionary
            resolve({
              output: stdout.trim(),
              success: true,
              command: args[0] || ''
            } as unknown as T);
          }
        }
      });
      
      cmd.on('error', (err: Error) => {
        reject(new CommandExecutionError(`Failed to execute command: ${err.message}`));
      });
    });
  }
  
  /**
   * Parse output from extract command to find code blocks
   * 
   * @param output Raw output from extract command
   * @returns Standardized extraction result
   */
  private parseExtractOutput(output: string): ExtractionResult {
    // Initialize result
    const result: ExtractionResult = {
      code_blocks: [],
      raw_output: output,
      success: true
    };
    
    // Extract code blocks using regex pattern for markdown code blocks
    const codeBlockPattern = /```(\w*)\n([\s\S]+?)\n```/g;
    let match;
    
    while ((match = codeBlockPattern.exec(output)) !== null) {
      const language = match[1] || 'unknown';
      const content = match[2];
      
      // Create CodeBlock object
      const block: CodeBlock = {
        content,
        language,
        line_start: 0,  // We don't have line information
        line_end: content.split('\n').length,
        confidence: 1.0,  // We don't have confidence information
        metadata: {}
      };
      
      // Add to result
      result.code_blocks.push(block);
    }
    
    return result;
  }

  /**
   * Extract code from a PDF file and return structured results
   * 
   * @param pdfPath Path to the PDF file
   * @param options Optional parameters for the extraction
   * @returns Extraction results with code blocks and metadata
   */
  public async extractCodeFromPdf(
    pdfPath: string, 
    options?: ExtractOptions
  ): Promise<ExtractionResult> {
    try {
      // Validate input
      if (!pdfPath || !pdfPath.trim()) {
        throw new InvalidInputError('PDF path cannot be empty');
      }
      
      if (!fs.existsSync(pdfPath)) {
        throw new InvalidInputError(`PDF file does not exist: ${pdfPath}`);
      }
      
      await this.checkBinary();
      
      const args = ['extract', '--input', pdfPath];
      
      if (options?.outputDir) {
        args.push('--output', options.outputDir);
      }
      
      if (options?.language) {
        args.push('--language', options.language);
      }
      
      if (options?.strategy) {
        args.push('--strategy', options.strategy);
      }
      
      if (options?.force) {
        args.push('--force');
      }
      
      return await this.runCommand<ExtractionResult>(args);
    } catch (error) {
      if (error instanceof Paper2CodeError) {
        throw error;
      } else {
        throw new CommandExecutionError(`Failed to extract code from PDF: ${(error as Error).message}`);
      }
    }
  }

  /**
   * Extract code from text input
   * 
   * @param text Text containing code to extract
   * @param options Optional parameters for the extraction
   * @returns Extraction result
   */
  public async extractCodeFromText(
    text: string, 
    options?: ExtractOptions
  ): Promise<ExtractionResult> {
    try {
      // Validate input
      if (!text || !text.trim()) {
        throw new InvalidInputError('Text input cannot be empty');
      }
      
      await this.checkBinary();
      
      // Create a temporary file for the text
      const tmpPath = path.join(os.tmpdir(), `paper2code-${Date.now()}.txt`);
      fs.writeFileSync(tmpPath, text);
      
      try {
        const args = ['extract', '--input', tmpPath];
        
        if (options?.outputDir) {
          args.push('--output', options.outputDir);
        }
        
        if (options?.language) {
          args.push('--language', options.language);
        }
        
        if (options?.strategy) {
          args.push('--strategy', options.strategy);
        }
        
        if (options?.force) {
          args.push('--force');
        }
        
        return await this.runCommand<ExtractionResult>(args);
      } finally {
        // Clean up the temporary file
        try {
          fs.unlinkSync(tmpPath);
        } catch (e) {
          console.warn(`Failed to delete temporary file ${tmpPath}:`, e);
        }
      }
    } catch (error) {
      if (error instanceof Paper2CodeError) {
        throw error;
      } else {
        throw new CommandExecutionError(`Failed to extract code from text: ${(error as Error).message}`);
      }
    }
  }

  /**
   * Test connection to LLM provider
   * 
   * @param options Options for testing LLM connectivity
   * @returns Test result
   */
  public async testLlmConnection(options?: TestOptions): Promise<any> {
    try {
      await this.checkBinary();
      
      const args = ['test'];
      
      if (options?.openai) {
        args.push('--openai');
      } else if (options?.claude) {
        args.push('--claude');
      }
      
      if (options?.prompt) {
        args.push('--prompt', options.prompt);
      }
      
      return await this.runCommand<any>(args);
    } catch (error) {
      if (error instanceof Paper2CodeError) {
        throw error;
      } else {
        throw new CommandExecutionError(`Failed to test LLM connection: ${(error as Error).message}`);
      }
    }
  }

  /**
   * Generate executable code from code snippets
   * 
   * @param codeSnippets Array of code snippets to convert
   * @param options Optional parameters for code generation
   * @returns Generation result
   */
  public async generateCode(
    codeSnippets: string[],
    options?: ExtractOptions
  ): Promise<any> {
    try {
      // Validate input
      if (!codeSnippets || codeSnippets.length === 0 || codeSnippets.every(s => !s.trim())) {
        throw new InvalidInputError('Code snippets cannot be empty');
      }
      
      await this.checkBinary();
      
      // Create a temporary file for the code snippets
      const tmpPath = path.join(os.tmpdir(), `paper2code-${Date.now()}.txt`);
      
      // Write code snippets to the file
      const content = codeSnippets.join('\n\n---\n\n');
      fs.writeFileSync(tmpPath, content);
      
      try {
        // Similar to the Python client, use extract instead of generate
        const args = ['extract', '--input', tmpPath];
        
        if (options?.outputDir) {
          args.push('--output', options.outputDir);
        }
        
        if (options?.language) {
          args.push('--language', options.language);
        }
        
        if (options?.strategy) {
          args.push('--strategy', options.strategy);
        }
        
        if (options?.force) {
          args.push('--force');
        }
        
        return await this.runCommand<any>(args);
      } finally {
        // Clean up the temporary file
        try {
          fs.unlinkSync(tmpPath);
        } catch (e) {
          console.warn(`Failed to delete temporary file ${tmpPath}:`, e);
        }
      }
    } catch (error) {
      if (error instanceof Paper2CodeError) {
        throw error;
      } else {
        throw new CommandExecutionError(`Failed to generate code: ${(error as Error).message}`);
      }
    }
  }

  /**
   * Get the current configuration
   * 
   * @returns Current configuration
   */
  public async getConfig(): Promise<AppConfig> {
    try {
      await this.checkBinary();
      
      // Try to read the config file directly
      if (fs.existsSync(this.configPath)) {
        try {
          const configContent = fs.readFileSync(this.configPath, 'utf-8');
          try {
            // Try to parse as JSON
            return JSON.parse(configContent) as AppConfig;
          } catch (e) {
            // Return raw content if parsing fails
            return {
              raw_config: configContent,
              success: true
            };
          }
        } catch (e) {
          // If reading fails, try to generate a new config
          return await this.generateConfig();
        }
      } else {
        // Config doesn't exist, generate a new one
        return await this.generateConfig();
      }
    } catch (error) {
      if (error instanceof Paper2CodeError) {
        throw error;
      } else {
        throw new CommandExecutionError(`Failed to get configuration: ${(error as Error).message}`);
      }
    }
  }
  
  /**
   * Generate a new configuration file
   * 
   * @param options Options for generating the configuration file
   * @returns Generated configuration
   */
  public async generateConfig(options?: ConfigOptions): Promise<AppConfig> {
    try {
      await this.checkBinary();
      
      const args = ['config', '--generate'];
      
      if (options?.outputPath) {
        args.push('--output', options.outputPath);
      }
      
      if (options?.force) {
        args.push('--force');
      }
      
      return await this.runCommand<AppConfig>(args);
    } catch (error) {
      if (error instanceof Paper2CodeError) {
        throw error;
      } else {
        throw new CommandExecutionError(`Failed to generate configuration: ${(error as Error).message}`);
      }
    }
  }

  /**
   * Set API key for an LLM provider
   * 
   * @param provider LLM provider (claude, openai)
   * @param apiKey API key
   * @returns Updated configuration
   */
  public async setApiKey(provider: string, apiKey: string): Promise<AppConfig> {
    try {
      // Validate input
      if (provider !== 'claude' && provider !== 'openai') {
        throw new InvalidInputError(`Invalid provider: ${provider}. Must be 'claude' or 'openai'.`);
      }
      
      if (!apiKey || !apiKey.trim()) {
        throw new InvalidInputError('API key cannot be empty');
      }
      
      await this.checkBinary();
      
      // The CLI doesn't have a 'set' subcommand for config, so we'll need to modify the config file directly
      const config = await this.getConfig();
      
      // Add provider section if it doesn't exist
      if (!config[provider as keyof AppConfig]) {
        (config as any)[provider] = {};
      }
      
      // Set API key
      (config as any)[provider]['api_key'] = apiKey;
      
      // Write updated config back to file
      fs.writeFileSync(this.configPath, JSON.stringify(config, null, 2));
      
      return config;
    } catch (error) {
      if (error instanceof Paper2CodeError) {
        throw error;
      } else {
        throw new CommandExecutionError(`Failed to set API key: ${(error as Error).message}`);
      }
    }
  }
}

// Export default client
export default Paper2CodeClient;
