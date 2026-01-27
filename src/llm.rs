//! LLM
//!
//! This module provides access to load and run inference on most common models and
//! their available formats. For now the LLM loader supports `*.safetensors`  files
//! and text generation models.

use crate::{runtime::LLMRuntime, Error, LLMRuntimeConfig};
use std::path::Path;

pub mod loaders;
pub mod runtime;

/// LLMServices manages runtime instances
#[allow(dead_code)]
pub struct LLMService {
    configs: Option<Vec<LLMRuntimeConfig>>,
    active: Option<LLMRuntime>,
}

#[allow(dead_code)]
impl LLMService {
    /// Creates a new [`LLMService`] by providing a path to the directory containing individual
    /// configurations for [`LLMRuntimes`]
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    /// let service = LLMService::from_path("path/to/configs").expect("Could not load LLMService");
    ///
    /// ```
    ///
    /// # Errors
    ///
    /// Following errors can occur:
    /// - configuration file cannot be found
    /// - configuration is malformed / contains invalid values
    pub fn from_path<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let filepaths = std::fs::read_dir(path)?;
        let mut configs = vec![];

        for entry in filepaths {
            if let Err(error) = entry {
                tracing::error!("reading file path entry returned an error: {error}");
                continue;
            }

            let entry = entry.unwrap();

            let filetype = match entry.file_type() {
                Ok(f) => f,
                Err(error) => {
                    tracing::error!("Reading file type of entry returned error: {error}");
                    continue;
                }
            };

            if !filetype.is_file() {
                continue;
            }

            let filename = match entry.file_name().into_string() {
                Ok(filename) => filename,
                Err(os_filename) => {
                    tracing::warn!(
                        "File name cannot be converted into Rust UTF-8 String. ({os_filename:?})"
                    );
                    continue;
                }
            };

            if filename.ends_with(".json") {
                match LLMRuntimeConfig::from_path(entry.path()) {
                    Ok(llm_config) => {
                        configs.push(llm_config);
                    }
                    Err(error) => {
                        tracing::error!(
                            "Reading LLMRuntimeConfig from file returned an error: {error}"
                        );
                        continue;
                    }
                }
            }
        }

        if configs.is_empty() {
            return Err(Error::MissingConfigLLM(
                "No valid configuration files found".to_string(),
            ));
        }

        Ok(Self {
            configs: Some(configs),
            active: None,
        })
    }
}
