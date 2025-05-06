// src/code/writer.rs
use crate::error::AppError;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::info;

/// Handles writing generated code to files
pub struct CodeWriter {
    output_dir: PathBuf,
    overwrite: bool,
}

impl CodeWriter {
    /// Create a new CodeWriter
    pub fn new<P: AsRef<Path>>(output_dir: P, overwrite: bool) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            overwrite,
        }
    }
    
    /// Get the output directory
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
    
    /// Check if overwriting is enabled
    pub fn overwrite(&self) -> bool {
        self.overwrite
    }
    
    /// Write code to a file with the given filename
    pub async fn write_file<P: AsRef<Path>>(&self, filename: P, content: &str) -> Result<PathBuf, AppError> {
        // Create full path
        let path = self.output_dir.join(filename);
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await
                .map_err(AppError::from_io_error)?;
        }
        
        // Check if file exists and we're not overwriting
        if path.exists() && !self.overwrite {
            return Err(AppError::FileError(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("File {} already exists and overwrite is disabled", path.display())
            )));
        }
        
        // Write content to file
        fs::write(&path, content).await
            .map_err(AppError::from_io_error)?;
        
        info!("Wrote code to {}", path.display());
        
        Ok(path)
    }
    
    /// Generate a unique filename
    fn _generate_unique_filename(&self, prefix: &str, extension: &str) -> String {
        // Generate a UUID to ensure uniqueness
        let uuid = uuid::Uuid::new_v4();
        // Extract just the first segment for a shorter filename
        let uuid_str = uuid.to_string();
        let short_uuid = uuid_str.split('-').next().unwrap();
        
        format!("{}_{}.{}", prefix, short_uuid, extension)
    }
} 