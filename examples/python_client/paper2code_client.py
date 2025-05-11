#!/usr/bin/env python3
"""
Python client for paper2code-rs

This client provides a Python interface to the paper2code-rs library.
It uses subprocess to call the paper2code-rs CLI and processes the output.
"""

import subprocess
import json
import os
import tempfile
import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger("paper2code_client")


class Paper2CodeError(Exception):
    """Base exception for Paper2Code client errors."""
    pass


class BinaryNotFoundError(Paper2CodeError):
    """Raised when the paper2code-rs binary cannot be found."""
    pass


class CommandExecutionError(Paper2CodeError):
    """Raised when a command fails to execute."""
    pass


class InvalidInputError(Paper2CodeError):
    """Raised when invalid input is provided to a method."""
    pass


class CodeBlock:
    """Represents a code block extracted from a document."""
    
    def __init__(self, content: str, language: Optional[str] = None,
                 line_start: int = 0, line_end: int = 0,
                 page_number: Optional[int] = None, confidence: float = 1.0):
        self.content = content
        self.language = language
        self.line_start = line_start
        self.line_end = line_end
        self.page_number = page_number
        self.confidence = confidence
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert CodeBlock to dictionary."""
        return {
            "content": self.content,
            "language": self.language,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "page_number": self.page_number,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_text(cls, text: str) -> 'CodeBlock':
        """Extract code block from text using regex."""
        # Try to determine language from markdown code block format
        lang_match = re.search(r'```(\w+)', text)
        language = lang_match.group(1) if lang_match else None
        
        # Extract content between code markers
        content_match = re.search(r'```(?:\w+)?\n(.+?)\n```', text, re.DOTALL)
        content = content_match.group(1) if content_match else text
        
        return cls(content=content, language=language)


class Paper2CodeClient:
    """Client for interacting with the paper2code-rs library."""

    def __init__(self, binary_path: str = "paper2code-rs", config_path: str = "config.toml", verbose: bool = False):
        """
        Initialize the client.

        Args:
            binary_path: Path to the paper2code-rs binary. Defaults to "paper2code-rs".
            config_path: Path to the configuration file. Defaults to "config.toml".
            verbose: Whether to enable verbose output. Defaults to False.
        """
        self.binary_path = binary_path
        self.config_path = config_path
        self.verbose = verbose
        self._check_binary()

    def _check_binary(self) -> None:
        """Check if the paper2code-rs binary exists and is executable."""
        try:
            result = subprocess.run(
                [self.binary_path, "--version"],
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode != 0:
                raise BinaryNotFoundError(f"Failed to run paper2code-rs: {result.stderr}")
            
            # Log version information
            version = result.stdout.strip()
            logger.info(f"Using paper2code-rs version: {version}")
            
        except FileNotFoundError:
            raise BinaryNotFoundError(
                f"Could not find paper2code-rs binary at {self.binary_path}. "
                "Make sure it's installed and on your PATH."
            )

    def _run_command(self, args: List[str], stdin_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a paper2code-rs command and return the output.

        Args:
            args: List of arguments to pass to paper2code-rs
            stdin_data: Optional data to pass to stdin

        Returns:
            Dictionary containing the parsed output
        """
        # Add global options
        cmd = [self.binary_path]
        
        # Add config path if not default
        if self.config_path != "config.toml":
            cmd.extend(["--config", self.config_path])
            
        # Add verbose flag if enabled
        if self.verbose:
            cmd.append("--verbose")
            
        # Add command-specific arguments
        cmd.extend(args)
        
        # Log the command
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        stdin = subprocess.PIPE if stdin_data else None
        
        try:
            result = subprocess.run(
                cmd,
                input=stdin_data.encode() if stdin_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=stdin,
                check=False
            )
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            raise CommandExecutionError(f"Failed to execute command: {e}") from e
        
        if result.returncode != 0:
            error_msg = result.stderr.decode().strip()
            logger.error(f"Command failed with error: {error_msg}")
            raise CommandExecutionError(f"Command failed with error: {error_msg}")
        
        output = result.stdout.decode().strip()
        
        # Process based on command type to create a standardized output format
        try:
            # Try to parse as JSON first (not likely with current CLI)
            return json.loads(output)
        except json.JSONDecodeError:
            # Extract code blocks using pattern matching if this was an extract command
            if "extract" in args:
                return self._parse_extract_output(output)
            
            # For other commands, return a standardized dictionary
            return {
                "output": output,
                "success": True,
                "command": args[0] if args else ""
            }
        
    def _parse_extract_output(self, output: str) -> Dict[str, Any]:
        """
        Parse output from the extract command to find code blocks.
        
        Args:
            output: Raw output from the extract command
            
        Returns:
            Dictionary containing parsed code blocks and metadata
        """
        # Initialize result
        result = {
            "code_blocks": [],
            "processing_time_ms": 0,
            "raw_output": output
        }
        
        # Extract code blocks using regex pattern for markdown code blocks
        code_block_pattern = r'```(\w*)\n(.+?)\n```'
        matches = re.finditer(code_block_pattern, output, re.DOTALL)
        
        for i, match in enumerate(matches):
            language = match.group(1) or "unknown"
            content = match.group(2)
            
            # Create CodeBlock object
            block = CodeBlock(
                content=content,
                language=language,
                line_start=0,  # We don't have line information
                line_end=len(content.split('\n')),
                confidence=1.0  # We don't have confidence information
            )
            
            # Add to result
            result["code_blocks"].append(block.to_dict())
        
        return result

    def extract_code_from_pdf(self, pdf_path: Union[str, Path], 
                              output_dir: Optional[Union[str, Path]] = None,
                              language: Optional[str] = None,
                              strategy: Optional[str] = None,
                              force: bool = False) -> Dict[str, Any]:
        """
        Extract code from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional directory to save the extracted code
            language: Optional programming language to target (auto-detected if not specified)
            strategy: Optional LLM strategy to use
            force: Whether to force overwrite of existing files

        Returns:
            Dictionary containing the extracted code blocks
        """
        # Validate inputs
        if not pdf_path or not os.path.exists(str(pdf_path)):
            raise InvalidInputError(f"PDF file not found: {pdf_path}")
        
        # Prepare command
        args = ["extract", "--input", str(pdf_path)]
        
        # Add optional arguments
        if output_dir:
            args.extend(["--output", str(output_dir)])
        
        if language:
            args.extend(["--language", language])
            
        if strategy:
            args.extend(["--strategy", strategy])
            
        if force:
            args.append("--force")
        
        # Execute command
        try:
            return self._run_command(args)
        except CommandExecutionError as e:
            logger.error(f"Failed to extract code from PDF: {e}")
            # Re-raise with more specific message
            raise CommandExecutionError(f"Failed to extract code from PDF {pdf_path}: {e}") from e

    def extract_code_from_text(self, text: str, 
                           output_dir: Optional[Union[str, Path]] = None,
                           language: Optional[str] = None,
                           strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract code from text.

        Args:
            text: Text containing code snippets
            output_dir: Optional directory to save the extracted code
            language: Optional programming language to target
            strategy: Optional LLM strategy to use

        Returns:
            Dictionary containing the extracted code blocks
        """
        if not text or not text.strip():
            raise InvalidInputError("Text content cannot be empty")
            
        # Create a temporary file with the text
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
            tmp.write(text)
            tmp_path = tmp.name
        
        try:
            # Prepare command
            args = ["extract", "--input", tmp_path]
            
            # Add optional arguments
            if output_dir:
                args.extend(["--output", str(output_dir)])
            
            if language:
                args.extend(["--language", language])
                
            if strategy:
                args.extend(["--strategy", strategy])
                
            # Execute command
            result = self._run_command(args)
            
            # If no code blocks were extracted via regex but text contains markdown code blocks,
            # manually extract them
            if not result.get("code_blocks") and "```" in text:
                # Process the raw text ourselves to extract code blocks
                code_blocks = []
                code_block = CodeBlock.from_text(text)
                if code_block.content:
                    code_blocks.append(code_block.to_dict())
                    
                # Update result if we found code blocks manually
                if code_blocks:
                    result["code_blocks"] = code_blocks
                    
            return result
        except CommandExecutionError as e:
            logger.error(f"Failed to extract code from text: {e}")
            raise CommandExecutionError(f"Failed to extract code from text: {e}") from e
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")

    def test_llm_connection(self, provider: Optional[str] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Test connection to LLM provider.

        Args:
            provider: LLM provider to test (openai, claude)
            prompt: Test with a specific prompt

        Returns:
            Dictionary containing the test results
        """
        args = ["test"]
        
        # Add provider-specific flags
        if provider == "openai":
            args.append("--openai")
        elif provider == "claude":
            args.append("--claude")
            
        # Add prompt if provided
        if prompt:
            args.extend(["--prompt", prompt])
            
        try:
            return self._run_command(args)
        except CommandExecutionError as e:
            logger.error(f"Failed to test LLM connection: {e}")
            raise CommandExecutionError(f"Failed to test LLM connection: {e}") from e

    def generate_code(self, code_snippets: List[str], 
                     language: Optional[str] = None,
                     output_dir: Optional[Union[str, Path]] = None,
                     strategy: Optional[str] = None,
                     force: bool = False) -> Dict[str, Any]:
        """
        Generate executable code from code snippets.

        Args:
            code_snippets: List of code snippets to convert
            language: Optional programming language
            output_dir: Optional directory to save the generated code
            strategy: Optional LLM strategy to use
            force: Whether to force overwrite existing files

        Returns:
            Dictionary containing the generated code
        """
        if not code_snippets or all(not snippet.strip() for snippet in code_snippets):
            raise InvalidInputError("Code snippets cannot be empty")
        
        # Create a temporary file with the code snippets
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
            for snippet in code_snippets:
                tmp.write(snippet + "\n\n---\n\n")
            tmp_path = tmp.name
        
        try:
            # Prepare command - note that 'generate' command isn't shown in help
            # but we'll use 'extract' which seems to be the closest match based on our testing
            args = ["extract", "--input", tmp_path]
            
            if language:
                args.extend(["--language", language])
            
            if output_dir:
                args.extend(["--output", str(output_dir)])
                
            if strategy:
                args.extend(["--strategy", strategy])
                
            if force:
                args.append("--force")
            
            try:
                return self._run_command(args)
            except CommandExecutionError as e:
                logger.error(f"Failed to generate code: {e}")
                raise CommandExecutionError(f"Failed to generate code: {e}") from e
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Dictionary containing the current configuration
        """
        # The CLI doesn't have a 'show' subcommand, so we'll need to read the config file directly
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    content = f.read()
                
                # Try to parse as TOML or JSON
                try:
                    import toml
                    return toml.loads(content)
                except (ImportError, ValueError):
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return {"raw_config": content}
            else:
                # Config file doesn't exist, so we'll invoke the config command to generate one
                return self.generate_config()
        except Exception as e:
            logger.error(f"Failed to read config file: {e}")
            raise CommandExecutionError(f"Failed to read config: {e}") from e
    
    def generate_config(self, output_path: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """
        Generate a new configuration file.
        
        Args:
            output_path: Path to save the configuration file. Defaults to the path specified during initialization.
            force: Whether to force overwrite of existing configuration file. Defaults to False.
            
        Returns:
            Dictionary containing the generated configuration
        """
        args = ["config", "--generate"]
        
        if output_path:
            args.extend(["--output", output_path])
        
        if force:
            args.append("--force")
            
        try:
            result = self._run_command(args)
            # After generating, read the config file
            if os.path.exists(output_path or self.config_path):
                return self.get_config()
            return result
        except CommandExecutionError as e:
            logger.error(f"Failed to generate config: {e}")
            raise CommandExecutionError(f"Failed to generate config: {e}") from e
            
    def set_api_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        """
        Set API key for an LLM provider.

        Args:
            provider: LLM provider (claude, openai)
            api_key: API key

        Returns:
            Dictionary containing the updated configuration
        """
        if provider not in ["claude", "openai"]:
            raise InvalidInputError(f"Invalid provider: {provider}. Must be 'claude' or 'openai'.")
            
        if not api_key.strip():
            raise InvalidInputError("API key cannot be empty")
            
        # The CLI doesn't have a 'set' subcommand for config, so we'll need to modify the config file directly
        try:
            config = self.get_config()
            
            # Add provider section if it doesn't exist
            if provider not in config:
                config[provider] = {}
                
            # Set API key
            config[provider]["api_key"] = api_key
            
            # Write updated config back to file
            try:
                import toml
                with open(self.config_path, 'w') as f:
                    toml.dump(config, f)
            except ImportError:
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    
            return config
        except Exception as e:
            logger.error(f"Failed to set API key: {e}")
            raise CommandExecutionError(f"Failed to set API key: {e}") from e


# Example usage
if __name__ == "__main__":
    client = Paper2CodeClient()
    
    # Test the client with a simple example
    try:
        config = client.get_config()
        print(f"Current configuration: {json.dumps(config, indent=2)}")
        
        # Extract code from a sample PDF
        # result = client.extract_code_from_pdf("sample.pdf")
        # print(f"Extracted code: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error: {e}") 