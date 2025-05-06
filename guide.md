# Paper2Code CLI Tool - Setup and Usage Guide

Paper2Code is a command-line tool that extracts code snippets from research papers and converts them into fully executable code using advanced LLMs (Large Language Models).

## Features

- Extract code from PDF research papers
- Use multiple LLMs (OpenAI GPT and Claude) for higher quality code extraction
- Convert code snippets to executable programs
- Intelligent language detection
- Customizable prompt templates
- Configure different strategies for each extraction phase

## Setup Instructions

### Prerequisites

- Rust toolchain (1.70.0 or newer)
- API keys for at least one of:
  - OpenAI API
  - Anthropic Claude API

### Installation

#### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paper2code-rs.git
   cd paper2code-rs
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Install the binary:
   ```bash
   cargo install --path .
   ```

#### Using Cargo

```bash
cargo install paper2code
```

### Configuration

Generate a default configuration file:

```bash
paper2code config --generate
```

This will create a `config.toml` file in the current directory. Open this file and add your API keys and other preferences.

#### API Keys

1. For OpenAI:
   - Go to https://platform.openai.com/api-keys
   - Create a new API key
   - Add it to the `config.toml` file in the `openai.api_key` field

2. For Claude:
   - Go to https://console.anthropic.com/
   - Generate an API key
   - Add it to the `config.toml` file in the `claude.api_key` field

#### Testing Your Setup

Verify your API connections:

```bash
# Test both APIs
paper2code test

# Test only OpenAI
paper2code test --openai

# Test only Claude
paper2code test --claude

# Test with a custom prompt
paper2code test --prompt "Write a bubble sort algorithm in Python"
```

## Usage

### Basic Usage

Extract code from a PDF:

```bash
paper2code extract --input paper.pdf
```

Multiple PDFs can be processed in one command:

```bash
paper2code extract --input paper1.pdf paper2.pdf paper3.pdf
```

### Advanced Options

Specify output directory:

```bash
paper2code extract --input paper.pdf --output ./extracted-code
```

Filter for a specific programming language:

```bash
paper2code extract --input paper.pdf --language python
```

Change the LLM strategy:

```bash
paper2code extract --input paper.pdf --strategy openai_only
```

Available strategies:
- `adaptive` - Use the best LLM for each task (default)
- `openai_only` - Use only OpenAI GPT
- `claude_only` - Use only Claude
- `openai_first` - Generate with OpenAI, refine with Claude
- `claude_first` - Generate with Claude, refine with OpenAI
- `compare_and_merge` - Use both and merge results

### Verbose Output

For detailed logs:

```bash
paper2code extract --input paper.pdf --verbose
```

## Environment Variables

Instead of using the config file, you can set environment variables:

```bash
export OPENAI_API_KEY=your_key_here
export CLAUDE_API_KEY=your_key_here
paper2code extract --input paper.pdf
```

## Building for Different Platforms

### macOS

Building for macOS is straightforward:

```bash
cargo build --release
```

### Linux

For most Linux distributions:

```bash
cargo build --release
```

For static linking:

```bash
cargo build --release --target x86_64-unknown-linux-musl
```

### Windows

For Windows:

```bash
cargo build --release
```

Cross-compile from Linux or macOS:

```bash
cargo build --release --target x86_64-pc-windows-gnu
```

## Troubleshooting

### Common Issues

1. **API Authentication Errors**
   - Verify your API keys are correct
   - Check network connectivity
   - Make sure you have sufficient credits/quota

2. **PDF Extraction Issues**
   - Some PDFs may have security features that prevent extraction
   - Try flattening or OCR-processing problematic PDFs

3. **No Code Blocks Detected**
   - Try using the `--verbose` flag to see detailed logs
   - Some papers format code in ways that are hard to detect automatically
   - Consider manually specifying sections with the `--language` flag

4. **Memory Usage Issues**
   - For large PDFs, ensure sufficient memory is available
   - Process one PDF at a time for very large documents

### Getting Help

If you encounter issues not covered here, please file an issue on the GitHub repository with:
- The exact command you ran
- Your configuration (with API keys removed)
- Error messages
- Sample PDF if possible (or description if it contains sensitive information)

## LLM Strategy Recommendations

For optimal results when extracting code from papers, we recommend these strategies:

1. **Machine Learning/AI Papers**: Use `--strategy claude_first` as Claude often better understands mathematical notation and algorithms

2. **Systems/Low-level Papers**: Use `--strategy openai_first` as OpenAI GPT often performs better with systems programming concepts

3. **Mixed Content Papers**: Use the default `--strategy adaptive` which selects the best LLM for each step

4. **Complex Algorithms**: Use `--strategy compare_and_merge` to get the best implementation by combining insights from both models

## Contributing

Contributions are welcome! Please see the CONTRIBUTING.md file in the repository for guidelines.


# Multi-LLM Strategy Guide for Paper2Code

This guide explains how Paper2Code uses multiple LLMs (OpenAI GPT and Claude) to achieve optimal code extraction and generation from research papers.

## LLM Strengths and Weaknesses

Each LLM has different strengths and weaknesses, which we leverage for specific tasks:

### OpenAI GPT

**Strengths:**
- Superior code structure and organization
- More up-to-date syntax and API usage
- Better at identifying and fixing bugs
- Stronger with systems programming and low-level concepts
- More consistent with performance optimizations

**Weaknesses:**
- Sometimes misses nuanced algorithmic details from papers
- May overly simplify complex mathematical concepts
- Can be more verbose in comments and documentation

### Claude

**Strengths:**
- Better at understanding academic and research context
- Superior handling of mathematical notation and algorithms
- More accurate preservation of the paper's original intent
- More comprehensive documentation
- Better at identifying code blocks in dense text

**Weaknesses:**
- Sometimes uses deprecated APIs or syntax
- May over-complicate simple implementations
- Can be inconsistent with error handling

## Tasks in the Extraction Pipeline

Paper2Code divides the extraction and generation process into five main tasks:

1. **Code Detection**: Identifying code blocks within the extracted text
2. **Code Improvement**: Fixing formatting, indentation, and basic errors
3. **Code Generation**: Converting snippets to executable programs
4. **Documentation**: Adding comprehensive comments and documentation
5. **Bug Fixing**: Identifying and fixing logical errors or edge cases

## Default Strategy

By default, Paper2Code uses an "adaptive" strategy that assigns different LLMs to different tasks:

| Task | Default LLM | Reasoning |
|------|-------------|-----------|
| Code Detection | Claude | Better at identifying code in research text and preserving mathematical notation |
| Code Improvement | OpenAI | Superior code structuring and modern syntax |
| Code Generation | OpenAI | Better at implementing complete, working code |
| Documentation | Claude | More comprehensive explanations with research context |
| Bug Fixing | OpenAI | Better at identifying and fixing logical errors |

## Available Strategies

Paper2Code supports several strategies for using multiple LLMs:

### 1. Adaptive (Default)

Uses the best LLM for each task as described above. This is the recommended approach for most papers.

### 2. OpenAI Only

Uses only OpenAI for all tasks. Best for:
- Systems programming papers
- Implementation-focused papers with minimal mathematics
- When Claude API is unavailable

### 3. Claude Only

Uses only Claude for all tasks. Best for:
- Highly mathematical papers
- Algorithm-focused research
- When OpenAI API is unavailable

### 4. OpenAI First, Claude Second

Generates initial code with OpenAI, then refines it with Claude. Best for:
- Papers requiring solid implementation but with mathematical nuances
- When you want readable code with comprehensive documentation

### 5. Claude First, OpenAI Second

Generates initial code with Claude, then refines it with OpenAI. Best for:
- Algorithm-heavy papers that need implementation polish
- Preserving mathematical concepts while ensuring modern code practices

### 6. Compare and Merge

Generates code with both LLMs and merges the results. Best for:
- Complex algorithms where different approaches might be valuable
- Critical implementations requiring the highest quality
- When computational budget allows for maximum quality

## Customizing the Strategy

You can customize the strategy in two ways:

### 1. Command-Line Option

```bash
paper2code extract --input paper.pdf --strategy openai_first
```

### 2. Configuration File

In your `config.toml`:

```toml
[llm_strategy]
strategy_type = "adaptive"
code_detection_preference = "prefer_claude"
code_improvement_preference = "prefer_openai"
code_generation_preference = "prefer_openai"
documentation_preference = "prefer_claude"
bug_fixing_preference = "prefer_openai"
```

## Strategy Selection Guidelines

Here are some guidelines for choosing the right strategy based on paper type:

### Machine Learning Papers

ML papers often contain complex mathematics and algorithms.

**Recommended strategy:** `claude_first` or `adaptive`

These papers benefit from Claude's strong mathematical understanding followed by OpenAI's implementation skills.

### Systems/Low-Level Programming Papers

Papers on operating systems, networking, or low-level programming.

**Recommended strategy:** `openai_first` or `openai_only`

These papers benefit from OpenAI's strong systems programming knowledge.

### Algorithm Papers

Papers focused on novel algorithms or data structures.

**Recommended strategy:** `compare_and_merge`

These papers benefit from seeing both models' interpretations of the algorithm.

### Applied Research Papers

Papers that apply existing methods to new problems.

**Recommended strategy:** `adaptive`

These papers benefit from the balanced approach of using each LLM for its strengths.

## Performance vs. Quality Tradeoffs

- **Fastest:** `openai_only` or `claude_only` (using whichever API responds faster in your region)
- **Balanced:** `adaptive` (uses each LLM efficiently)
- **Highest Quality:** `compare_and_merge` (uses both LLMs for each task)

Choose based on your priorities for speed vs. quality.

## Debugging LLM Issues

If you encounter problems with the generated code:

1. Try switching strategies (e.g., from `openai_only` to `claude_only`)
2. For specific issues:
   - **Mathematical errors:** Try `claude_first`
   - **Implementation bugs:** Try `openai_first`
   - **Incomplete code:** Try `openai_only`
   - **Missing context:** Try `claude_only`

## Future Improvements

The Paper2Code multi-LLM strategy system is designed to be extensible. In the future, we plan to add:

- Support for additional LLMs
- Fine-grained language-specific preferences
- Learning from feedback to improve LLM selection
- Interactive mode to choose strategy during extraction

# Cross-Platform Build Guide for Paper2Code

This guide provides detailed instructions for building the Paper2Code CLI application on macOS, Windows, and Linux, as well as creating cross-platform releases.

## Prerequisites

- Rust toolchain (1.70.0 or newer)
- Cargo (comes with Rust)
- Git

## Installing Rust

If you haven't installed Rust yet, use rustup:

```bash
# macOS/Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Windows
# Download and run rustup-init.exe from https://rustup.rs/
```

## Building for macOS

### Development Build

```bash
# Clone the repository
git clone https://github.com/yourusername/paper2code-rs.git
cd paper2code-rs

# Build in debug mode
cargo build

# Run the application
./target/debug/paper2code --help
```

### Release Build

```bash
# Build optimized release
cargo build --release

# The binary will be at ./target/release/paper2code
```

### Universal Binary (Intel and Apple Silicon)

To create a universal binary that works on both Intel Macs and Apple Silicon:

```bash
# Install required targets
rustup target add x86_64-apple-darwin aarch64-apple-darwin

# Build for both architectures
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin

# Combine into universal binary (requires lipo)
lipo -create -output paper2code-universal \
  ./target/x86_64-apple-darwin/release/paper2code \
  ./target/aarch64-apple-darwin/release/paper2code
```

### Creating a macOS Package

Create a simple package with [Homebrew](https://brew.sh/):

1. Create a homebrew formula in a new repository:

```ruby
class Paper2code < Formula
  desc "Extract and convert code from research papers using multiple LLMs"
  homepage "https://github.com/yourusername/paper2code-rs"
  url "https://github.com/yourusername/paper2code-rs/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "your_package_sha256_here"
  license "MIT"

  depends_on "rust" => :build

  def install
    system "cargo", "install", "--locked", "--root", prefix, "--path", "."
  end

  test do
    system "#{bin}/paper2code", "--version"
  end
end
```

2. Users can then install with:

```bash
brew tap yourusername/paper2code
brew install paper2code
```

## Building for Linux

### Development Build

```bash
# Clone the repository
git clone https://github.com/yourusername/paper2code-rs.git
cd paper2code-rs

# Build in debug mode
cargo build

# Run the application
./target/debug/paper2code --help
```

### Release Build

```bash
# Build optimized release
cargo build --release

# The binary will be at ./target/release/paper2code
```

### Static Linking for Maximum Compatibility

To create a portable Linux binary that works across different distributions:

```bash
# Install musl target
rustup target add x86_64-unknown-linux-musl

# Build with musl for static linking
cargo build --release --target x86_64-unknown-linux-musl

# The binary will be at ./target/x86_64-unknown-linux-musl/release/paper2code
```

### Debian/Ubuntu Package

Create a Debian package using cargo-deb:

```bash
# Install cargo-deb
cargo install cargo-deb

# Generate Debian package
cargo deb

# The .deb file will be in ./target/debian/
```

### RPM Package (Fedora/CentOS/RHEL)

Create an RPM package using cargo-rpm:

```bash
# Install cargo-rpm
cargo install cargo-rpm

# Initialize RPM configuration (first time only)
cargo rpm init

# Build the RPM package
cargo rpm build

# The .rpm file will be in ./target/release/rpmbuild/RPMS/x86_64/
```

## Building for Windows

### Development Build

```powershell
# Clone the repository
git clone https://github.com/yourusername/paper2code-rs.git
cd paper2code-rs

# Build in debug mode
cargo build

# Run the application
.\target\debug\paper2code.exe --help
```

### Release Build

```powershell
# Build optimized release
cargo build --release

# The binary will be at .\target\release\paper2code.exe
```

### Installer with WiX Toolset

Create a Windows installer using cargo-wix:

```powershell
# Install cargo-wix
cargo install cargo-wix

# Create the Windows installer (.msi)
cargo wix

# The .msi file will be in .\target\wix\
```

### Portable Executable

For a portable Windows .exe that doesn't require installation:

```powershell
# Build release
cargo build --release

# Copy the executable
copy .\target\release\paper2code.exe .\paper2code.exe

# Create a simple batch file wrapper (optional)
echo @echo off > paper2code.bat
echo .\paper2code.exe %* >> paper2code.bat
```

## Cross-Compilation

### From Linux to Windows

```bash
# Install Windows target
rustup target add x86_64-pc-windows-gnu

# Install MinGW cross-compiler
sudo apt-get install mingw-w64

# Build for Windows
cargo build --release --target x86_64-pc-windows-gnu

# The binary will be at ./target/x86_64-pc-windows-gnu/release/paper2code.exe
```

### From Linux to macOS

Cross-compiling for macOS from Linux is complex due to Apple-specific requirements. The recommended approach is to build on a macOS machine or use a CI service with macOS runners.

## GitHub Actions for Continuous Integration

Here's a GitHub Actions workflow to build releases for all platforms:

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: x86_64-unknown-linux-musl
          override: true
      - uses: actions-rs/cargo@v1
        with:
          command: build
          args: --release --target x86_64-unknown-linux-musl
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: paper2code-linux
          path: target/x86_64-unknown-linux-musl/release/paper2code

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      - uses: actions-rs/cargo@v1
        with:
          command: build
          args: --release
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: paper2code-macos
          path: target/release/paper2code

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      - uses: actions-rs/cargo@v1
        with:
          command: build
          args: --release
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: paper2code-windows
          path: target/release/paper2code.exe

  create-release:
    needs: [build-linux, build-macos, build-windows]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v3
      - name: Prepare release assets
        run: |
          mkdir release
          cp paper2code-linux/paper2code release/paper2code-linux
          cp paper2code-macos/paper2code release/paper2code-macos
          cp paper2code-windows/paper2code.exe release/paper2code-windows.exe
          cd release
          chmod +x paper2code-linux paper2code-macos
          tar czf paper2code-linux.tar.gz paper2code-linux
          tar czf paper2code-macos.tar.gz paper2code-macos
          zip paper2code-windows.zip paper2code-windows.exe
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            release/paper2code-linux.tar.gz
            release/paper2code-macos.tar.gz
            release/paper2code-windows.zip
```

## Installation from Release Binaries

### macOS

```bash
# Download the macOS release
curl -LO https://github.com/yourusername/paper2code-rs/releases/download/v0.1.0/paper2code-macos.tar.gz

# Extract
tar xzf paper2code-macos.tar.gz

# Make executable
chmod +x paper2code-macos

# Move to a directory in your PATH
sudo mv paper2code-macos /usr/local/bin/paper2code
```

### Linux

```bash
# Download the Linux release
curl -LO https://github.com/yourusername/paper2code-rs/releases/download/v0.1.0/paper2code-linux.tar.gz

# Extract
tar xzf paper2code-linux.tar.gz

# Make executable
chmod +x paper2code-linux

# Move to a directory in your PATH
sudo mv paper2code-linux /usr/local/bin/paper2code
```

### Windows

1. Download the Windows release from GitHub
2. Extract the ZIP file
3. Move the .exe to your desired location
4. (Optional) Add the location to your PATH environment variable

## Docker Container

Create a Dockerfile:

```Dockerfile
FROM rust:1.70-slim as builder
WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
COPY --from=builder /usr/src/app/target/release/paper2code /usr/local/bin/
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["paper2code"]
```

Build and use the Docker image:

```bash
# Build the image
docker build -t paper2code .

# Run the container
docker run -v $(pwd):/data paper2code extract --input /data/paper.pdf --output /data/output
```

## Troubleshooting Common Build Issues

### Missing Dependencies on Linux

If you encounter errors about missing libraries:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# Fedora/RHEL/CentOS
sudo dnf install openssl-devel gcc make
```

### PDF Library Issues

For PDF processing library issues:

```bash
# Ubuntu/Debian
sudo apt-get install libpoppler-dev

# macOS
brew install poppler

# Windows
# Download and install poppler from http://blog.alivate.com.au/poppler-windows/
```

### Cross-Compilation Issues

For cross-compilation problems, ensure you have the right toolchains:

```bash
# For Windows targets from Linux
sudo apt-get install mingw-w64

# For ARM targets
sudo apt-get install gcc-arm-linux-gnueabihf libc6-dev-armhf-cross
```

## Optimizing the Release Binary

For smaller, faster binaries:

```bash
# Add to Cargo.toml
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
strip = true
```

# Cross-Platform Build Guide for Paper2Code

This guide provides detailed instructions for building the Paper2Code CLI application on macOS, Windows, and Linux, as well as creating cross-platform releases.

## Prerequisites

- Rust toolchain (1.70.0 or newer)
- Cargo (comes with Rust)
- Git

## Installing Rust

If you haven't installed Rust yet, use rustup:

```bash
# macOS/Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Windows
# Download and run rustup-init.exe from https://rustup.rs/
```

## Building for macOS

### Development Build

```bash
# Clone the repository
git clone https://github.com/yourusername/paper2code-rs.git
cd paper2code-rs

# Build in debug mode
cargo build

# Run the application
./target/debug/paper2code --help
```

### Release Build

```bash
# Build optimized release
cargo build --release

# The binary will be at ./target/release/paper2code
```

### Universal Binary (Intel and Apple Silicon)

To create a universal binary that works on both Intel Macs and Apple Silicon:

```bash
# Install required targets
rustup target add x86_64-apple-darwin aarch64-apple-darwin

# Build for both architectures
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin

# Combine into universal binary (requires lipo)
lipo -create -output paper2code-universal \
  ./target/x86_64-apple-darwin/release/paper2code \
  ./target/aarch64-apple-darwin/release/paper2code
```

### Creating a macOS Package

Create a simple package with [Homebrew](https://brew.sh/):

1. Create a homebrew formula in a new repository:

```ruby
class Paper2code < Formula
  desc "Extract and convert code from research papers using multiple LLMs"
  homepage "https://github.com/yourusername/paper2code-rs"
  url "https://github.com/yourusername/paper2code-rs/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "your_package_sha256_here"
  license "MIT"

  depends_on "rust" => :build

  def install
    system "cargo", "install", "--locked", "--root", prefix, "--path", "."
  end

  test do
    system "#{bin}/paper2code", "--version"
  end
end
```

2. Users can then install with:

```bash
brew tap yourusername/paper2code
brew install paper2code
```

## Building for Linux

### Development Build

```bash
# Clone the repository
git clone https://github.com/yourusername/paper2code-rs.git
cd paper2code-rs

# Build in debug mode
cargo build

# Run the application
./target/debug/paper2code --help
```

### Release Build

```bash
# Build optimized release
cargo build --release

# The binary will be at ./target/release/paper2code
```

### Static Linking for Maximum Compatibility

To create a portable Linux binary that works across different distributions:

```bash
# Install musl target
rustup target add x86_64-unknown-linux-musl

# Build with musl for static linking
cargo build --release --target x86_64-unknown-linux-musl

# The binary will be at ./target/x86_64-unknown-linux-musl/release/paper2code
```

### Debian/Ubuntu Package

Create a Debian package using cargo-deb:

```bash
# Install cargo-deb
cargo install cargo-deb

# Generate Debian package
cargo deb

# The .deb file will be in ./target/debian/
```

### RPM Package (Fedora/CentOS/RHEL)

Create an RPM package using cargo-rpm:

```bash
# Install cargo-rpm
cargo install cargo-rpm

# Initialize RPM configuration (first time only)
cargo rpm init

# Build the RPM package
cargo rpm build

# The .rpm file will be in ./target/release/rpmbuild/RPMS/x86_64/
```

## Building for Windows

### Development Build

```powershell
# Clone the repository
git clone https://github.com/yourusername/paper2code-rs.git
cd paper2code-rs

# Build in debug mode
cargo build

# Run the application
.\target\debug\paper2code.exe --help
```

### Release Build

```powershell
# Build optimized release
cargo build --release

# The binary will be at .\target\release\paper2code.exe
```

### Installer with WiX Toolset

Create a Windows installer using cargo-wix:

```powershell
# Install cargo-wix
cargo install cargo-wix

# Create the Windows installer (.msi)
cargo wix

# The .msi file will be in .\target\wix\
```

### Portable Executable

For a portable Windows .exe that doesn't require installation:

```powershell
# Build release
cargo build --release

# Copy the executable
copy .\target\release\paper2code.exe .\paper2code.exe

# Create a simple batch file wrapper (optional)
echo @echo off > paper2code.bat
echo .\paper2code.exe %* >> paper2code.bat
```

## Cross-Compilation

### From Linux to Windows

```bash
# Install Windows target
rustup target add x86_64-pc-windows-gnu

# Install MinGW cross-compiler
sudo apt-get install mingw-w64

# Build for Windows
cargo build --release --target x86_64-pc-windows-gnu

# The binary will be at ./target/x86_64-pc-windows-gnu/release/paper2code.exe
```

### From Linux to macOS

Cross-compiling for macOS from Linux is complex due to Apple-specific requirements. The recommended approach is to build on a macOS machine or use a CI service with macOS runners.

## GitHub Actions for Continuous Integration

Here's a GitHub Actions workflow to build releases for all platforms:

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: x86_64-unknown-linux-musl
          override: true
      - uses: actions-rs/cargo@v1
        with:
          command: build
          args: --release --target x86_64-unknown-linux-musl
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: paper2code-linux
          path: target/x86_64-unknown-linux-musl/release/paper2code

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      - uses: actions-rs/cargo@v1
        with:
          command: build
          args: --release
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: paper2code-macos
          path: target/release/paper2code

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      - uses: actions-rs/cargo@v1
        with:
          command: build
          args: --release
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: paper2code-windows
          path: target/release/paper2code.exe

  create-release:
    needs: [build-linux, build-macos, build-windows]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v3
      - name: Prepare release assets
        run: |
          mkdir release
          cp paper2code-linux/paper2code release/paper2code-linux
          cp paper2code-macos/paper2code release/paper2code-macos
          cp paper2code-windows/paper2code.exe release/paper2code-windows.exe
          cd release
          chmod +x paper2code-linux paper2code-macos
          tar czf paper2code-linux.tar.gz paper2code-linux
          tar czf paper2code-macos.tar.gz paper2code-macos
          zip paper2code-windows.zip paper2code-windows.exe
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            release/paper2code-linux.tar.gz
            release/paper2code-macos.tar.gz
            release/paper2code-windows.zip
```

## Installation from Release Binaries

### macOS

```bash
# Download the macOS release
curl -LO https://github.com/yourusername/paper2code-rs/releases/download/v0.1.0/paper2code-macos.tar.gz

# Extract
tar xzf paper2code-macos.tar.gz

# Make executable
chmod +x paper2code-macos

# Move to a directory in your PATH
sudo mv paper2code-macos /usr/local/bin/paper2code
```

### Linux

```bash
# Download the Linux release
curl -LO https://github.com/yourusername/paper2code-rs/releases/download/v0.1.0/paper2code-linux.tar.gz

# Extract
tar xzf paper2code-linux.tar.gz

# Make executable
chmod +x paper2code-linux

# Move to a directory in your PATH
sudo mv paper2code-linux /usr/local/bin/paper2code
```

### Windows

1. Download the Windows release from GitHub
2. Extract the ZIP file
3. Move the .exe to your desired location
4. (Optional) Add the location to your PATH environment variable

## Docker Container

Create a Dockerfile:

```Dockerfile
FROM rust:1.70-slim as builder
WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
COPY --from=builder /usr/src/app/target/release/paper2code /usr/local/bin/
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["paper2code"]
```

Build and use the Docker image:

```bash
# Build the image
docker build -t paper2code .

# Run the container
docker run -v $(pwd):/data paper2code extract --input /data/paper.pdf --output /data/output
```

## Troubleshooting Common Build Issues

### Missing Dependencies on Linux

If you encounter errors about missing libraries:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# Fedora/RHEL/CentOS
sudo dnf install openssl-devel gcc make
```

### PDF Library Issues

For PDF processing library issues:

```bash
# Ubuntu/Debian
sudo apt-get install libpoppler-dev

# macOS
brew install poppler

# Windows
# Download and install poppler from http://blog.alivate.com.au/poppler-windows/
```

### Cross-Compilation Issues

For cross-compilation problems, ensure you have the right toolchains:

```bash
# For Windows targets from Linux
sudo apt-get install mingw-w64

# For ARM targets
sudo apt-get install gcc-arm-linux-gnueabihf libc6-dev-armhf-cross
```

## Optimizing the Release Binary

For smaller, faster binaries:

```bash
# Add to Cargo.toml
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
strip = true
```