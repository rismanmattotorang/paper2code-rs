/**
 * TypeScript client for the paper2code-rs library
 * 
 * This client provides a TypeScript/JavaScript interface to the paper2code-rs library.
 * It uses child_process to call the paper2code-rs CLI and parses the output.
 */

// Import directly from JavaScript without relying on type definitions
// This approach is simpler but doesn't provide type checking for node modules
// eslint-disable-next-line @typescript-eslint/no-var-requires
const childProcess = require('child_process');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const fs = require('fs');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const path = require('path');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const os = require('os');

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
 * Result from code extraction
 */
export interface ExtractionResult {
  code_blocks: CodeBlock[];
  source_file?: string;
  total_pages?: number;
  processing_time_ms: number;
}

/**
 * Configuration for OpenAI
 */
export interface OpenAIConfig {
  api_key: string;
  model: string;
  max_tokens: number;
  temperature: number;
  timeout_seconds: number;
  max_concurrent_requests: number;
}

/**
 * Configuration for Claude
 */
export interface ClaudeConfig {
  api_key: string;
  model: string;
  max_tokens: number;
  temperature: number;
  timeout_seconds: number;
  max_concurrent_requests: number;
}

/**
 * Application configuration
 */
export interface AppConfig {
  openai?: OpenAIConfig;
  claude?: ClaudeConfig;
  default_provider: string;
  output_dir: string;
  parallel_requests: number;
  chunk_size: number;
}

/**
 * Options for extracting code from PDF
 */
export interface PdfExtractionOptions {
  outputDir?: string;
  pageRange?: string;
}

/**
 * Options for generating code
 */
export interface CodeGenerationOptions {
  language?: string;
  outputDir?: string;
}

/**
 * Client for interacting with the paper2code-rs library
 */
export class Paper2CodeClient {
  private binaryPath: string;

  /**
   * Create a new Paper2CodeClient
   * 
   * @param binaryPath Path to the paper2code-rs binary
   */
  constructor(binaryPath: string = 'paper2code-rs') {
    this.binaryPath = binaryPath;
  }

  /**
   * Check if the paper2code-rs binary exists and is executable
   */
  public async checkBinary(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      childProcess.exec(`${this.binaryPath} --version`, (error: Error | null) => {
        if (error) {
          reject(new Error(`Failed to run paper2code-rs: ${error.message}. Make sure it's installed and on your PATH.`));
        } else {
          resolve();
        }
      });
    });
  }

  /**
   * Run a paper2code-rs command and return the JSON output
   * 
   * @param args Command arguments
   * @param stdinData Optional data to pass to stdin
   * @returns Parsed JSON output
   */
  private async runCommand<T>(args: string[], stdinData?: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      // Add JSON format option
      const allArgs = [...args, '--format', 'json'];
      
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
          reject(new Error(`Command failed with error: ${stderr.trim()}`));
          return;
        }
        
        try {
          const result = JSON.parse(stdout.trim());
          resolve(result);
        } catch (e) {
          // Return raw output if not JSON
          resolve({ output: stdout.trim() } as unknown as T);
        }
      });
    });
  }

  /**
   * Extract code from a PDF file
   * 
   * @param pdfPath Path to the PDF file
   * @param options Extraction options
   * @returns Extraction result
   */
  public async extractCodeFromPdf(
    pdfPath: string, 
    options?: PdfExtractionOptions
  ): Promise<ExtractionResult> {
    const args = ['extract', pdfPath];
    
    if (options?.outputDir) {
      args.push('--output', options.outputDir);
    }
    
    if (options?.pageRange) {
      args.push('--pages', options.pageRange);
    }
    
    return this.runCommand<ExtractionResult>(args);
  }

  /**
   * Extract code from text
   * 
   * @param text Text containing code snippets
   * @returns Extraction result
   */
  public async extractCodeFromText(text: string): Promise<ExtractionResult> {
    const args = ['extract', '--from-text'];
    return this.runCommand<ExtractionResult>(args, text);
  }

  /**
   * Test connection to LLM provider
   * 
   * @param provider LLM provider (auto, claude, openai)
   * @returns Test result
   */
  public async testLlmConnection(provider: string = 'auto'): Promise<any> {
    const args = ['test', 'llm', '--provider', provider];
    return this.runCommand<any>(args);
  }

  /**
   * Generate executable code from code snippets
   * 
   * @param codeSnippets List of code snippets
   * @param options Generation options
   * @returns Generation result
   */
  public async generateCode(
    codeSnippets: string[], 
    options?: CodeGenerationOptions
  ): Promise<any> {
    let tmpFile = '';
    
    try {
      // Create a temporary file with the code snippets
      const tmpDir = os.tmpdir();
      tmpFile = path.join(tmpDir, `paper2code-${Date.now()}.txt`);
      
      // Write code snippets to the temporary file
      fs.writeFileSync(
        tmpFile, 
        codeSnippets.join('\n\n---\n\n')
      );
      
      const args = ['generate', '--input', tmpFile];
      
      if (options?.language) {
        args.push('--language', options.language);
      }
      
      if (options?.outputDir) {
        args.push('--output', options.outputDir);
      }
      
      return this.runCommand<any>(args);
    } finally {
      // Clean up the temporary file if it exists
      if (tmpFile) {
        try {
          fs.unlinkSync(tmpFile);
        } catch (err) {
          // Ignore errors in cleanup
        }
      }
    }
  }

  /**
   * Get the current configuration
   * 
   * @returns Current configuration
   */
  public async getConfig(): Promise<AppConfig> {
    const args = ['config', 'show'];
    return this.runCommand<AppConfig>(args);
  }

  /**
   * Set API key for an LLM provider
   * 
   * @param provider LLM provider (claude, openai)
   * @param apiKey API key
   * @returns Updated configuration
   */
  public async setApiKey(provider: string, apiKey: string): Promise<AppConfig> {
    const args = ['config', 'set', `${provider}.api_key`, apiKey];
    return this.runCommand<AppConfig>(args);
  }
}

// Export default client
export default Paper2CodeClient; 