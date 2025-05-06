#!/usr/bin/env python3
"""
Python client for paper2code-rs

This client provides a Python interface to the paper2code-rs library.
It uses subprocess to call the paper2code-rs CLI and parses the output.
"""

import subprocess
import json
import os
import tempfile
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class Paper2CodeClient:
    """Client for interacting with the paper2code-rs library."""

    def __init__(self, binary_path: str = "paper2code-rs"):
        """
        Initialize the client.

        Args:
            binary_path: Path to the paper2code-rs binary. Defaults to "paper2code-rs".
        """
        self.binary_path = binary_path
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
                raise RuntimeError(f"Failed to run paper2code-rs: {result.stderr}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find paper2code-rs binary at {self.binary_path}. "
                "Make sure it's installed and on your PATH."
            )

    def _run_command(self, args: List[str], stdin_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a paper2code-rs command and return the JSON output.

        Args:
            args: List of arguments to pass to paper2code-rs
            stdin_data: Optional data to pass to stdin

        Returns:
            Dictionary containing the parsed JSON output
        """
        cmd = [self.binary_path] + args + ["--format", "json"]
        
        stdin = subprocess.PIPE if stdin_data else None
        
        result = subprocess.run(
            cmd,
            input=stdin_data.encode() if stdin_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=stdin,
            check=False
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode().strip()
            raise RuntimeError(f"Command failed with error: {error_msg}")
        
        output = result.stdout.decode().strip()
        
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # Return raw output if not JSON
            return {"output": output}

    def extract_code_from_pdf(self, pdf_path: Union[str, Path], 
                              output_dir: Optional[Union[str, Path]] = None,
                              page_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract code from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional directory to save the extracted code
            page_range: Optional page range to extract from (e.g. "1-5")

        Returns:
            Dictionary containing the extracted code blocks
        """
        args = ["extract", str(pdf_path)]
        
        if output_dir:
            args.extend(["--output", str(output_dir)])
        
        if page_range:
            args.extend(["--pages", page_range])
        
        return self._run_command(args)

    def extract_code_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract code from text.

        Args:
            text: Text containing code snippets

        Returns:
            Dictionary containing the extracted code blocks
        """
        args = ["extract", "--from-text"]
        return self._run_command(args, stdin_data=text)

    def test_llm_connection(self, provider: str = "auto") -> Dict[str, Any]:
        """
        Test connection to LLM provider.

        Args:
            provider: LLM provider to test (auto, claude, openai)

        Returns:
            Dictionary containing the test results
        """
        args = ["test", "llm", "--provider", provider]
        return self._run_command(args)

    def generate_code(self, code_snippets: List[str], 
                     language: Optional[str] = None,
                     output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Generate executable code from code snippets.

        Args:
            code_snippets: List of code snippets to convert
            language: Optional programming language
            output_dir: Optional directory to save the generated code

        Returns:
            Dictionary containing the generated code
        """
        # Create a temporary file with the code snippets
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
            for snippet in code_snippets:
                tmp.write(snippet + "\n\n---\n\n")
            tmp_path = tmp.name
        
        try:
            args = ["generate", "--input", tmp_path]
            
            if language:
                args.extend(["--language", language])
            
            if output_dir:
                args.extend(["--output", str(output_dir)])
            
            return self._run_command(args)
        finally:
            os.unlink(tmp_path)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Dictionary containing the current configuration
        """
        args = ["config", "show"]
        return self._run_command(args)

    def set_api_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        """
        Set API key for an LLM provider.

        Args:
            provider: LLM provider (claude, openai)
            api_key: API key

        Returns:
            Dictionary containing the updated configuration
        """
        args = ["config", "set", f"{provider}.api_key", api_key]
        return self._run_command(args)


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