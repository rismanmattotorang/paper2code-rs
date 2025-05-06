// src/pdf/parser.rs
use crate::error::AppError;
use lopdf::{Document, Object};
use std::path::Path;

/// Handles parsing of PDF document structure
pub struct PdfParser;

impl PdfParser {
    pub fn new() -> Self {
        Self
    }
    
    /// Parse PDF metadata
    pub async fn extract_metadata<P: AsRef<Path>>(
        &self, 
        path: P
    ) -> Result<PdfMetadata, AppError> {
        let buffer = tokio::fs::read(path).await.map_err(AppError::from_io_error)?;
        
        let metadata = tokio::task::spawn_blocking(move || -> Result<PdfMetadata, AppError> {
            let doc = Document::load_mem(&buffer)
                .map_err(|e| AppError::PdfParseError(e.to_string()))?;
                
            // Try to get the Info dictionary
            let info_dict = match doc.trailer.get(b"Info") {
                Ok(info_ref) => {
                    match info_ref {
                        Object::Reference(id) => {
                            match doc.get_object(*id) {
                                Ok(obj) => obj.as_dict().ok(),
                                Err(_) => None,
                            }
                        },
                        _ => None,
                    }
                },
                Err(_) => None,
            };
                
            let metadata = PdfMetadata {
                page_count: doc.get_pages().len() as u32,
                title: extract_text_from_dict(info_dict, "Title"),
                author: extract_text_from_dict(info_dict, "Author"),
                subject: extract_text_from_dict(info_dict, "Subject"),
                keywords: extract_text_from_dict(info_dict, "Keywords"),
                creator: extract_text_from_dict(info_dict, "Creator"),
                producer: extract_text_from_dict(info_dict, "Producer"),
            };
            
            Ok(metadata)
        })
        .await
        .map_err(|e| AppError::TaskJoinError(e.to_string()))??;
        
        Ok(metadata)
    }
}

/// Represents metadata extracted from a PDF document
#[derive(Debug, Clone)]
pub struct PdfMetadata {
    pub page_count: u32,
    pub title: Option<String>,
    pub author: Option<String>,
    pub subject: Option<String>,
    pub keywords: Option<String>,
    pub creator: Option<String>,
    pub producer: Option<String>,
}

impl PdfMetadata {
    fn _default(page_count: u32) -> Self {
        Self {
            page_count,
            title: None,
            author: None,
            subject: None,
            keywords: None,
            creator: None,
            producer: None,
        }
    }
}

/// Helper function to extract text from a PDF dictionary
fn extract_text_from_dict(dict: Option<&lopdf::Dictionary>, key: &str) -> Option<String> {
    match dict {
        Some(d) => {
            match d.get(key.as_bytes()) {
                Ok(obj) => {
                    match obj {
                        Object::String(s, _) => String::from_utf8(s.clone()).ok(),
                        _ => None,
                    }
                },
                Err(_) => None,
            }
        },
        None => None,
    }
} 