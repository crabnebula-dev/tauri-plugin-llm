//! LLM
//!
//! This module provides access to load and run inference on most common models and
//! their available formats. For now the LLM loader supports `*.safetensors`  files
//! and text generation models.

use crate::{runtime::LLMRuntime, Error, LLMRuntimeConfig};
use std::{collections::HashMap, path::Path};

pub mod backend;
pub mod loaders;
pub mod runtime;
pub mod tool_call;

/// LLMServices manages runtime instances
pub struct LLMService {
    configs: Option<HashMap<String, LLMRuntimeConfig>>,
    active: Option<LLMRuntime>,
}

impl LLMService {
    /// Creates a new [`LLMService`] by providing a path to the directory containing individual
    /// configurations for [`LLMRuntimes`]
    ///
    /// # Example
    ///
    /// ```
    /// use tauri_plugin_llm::LLMService;
    /// let service = LLMService::from_dir("tests/fixtures/").expect("Could not load LLMService");
    ///
    /// ```
    ///
    /// # Errors
    ///
    /// Following errors can occur:
    /// - configuration file cannot be found
    /// - configuration is malformed / contains invalid values
    pub fn from_dir<P>(dir: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let filepaths = std::fs::read_dir(dir)?;
        let mut configs = HashMap::new();

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
                        configs.insert(llm_config.name.clone(), llm_config);
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

    /// Loads a single [`LLMRuntimeConfig`] from path
    pub fn from_path<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let config = LLMRuntimeConfig::from_path(path)?;

        Ok(Self {
            configs: Some([config].into_iter().map(|c| (c.name.clone(), c)).collect()),
            active: None,
        })
    }

    /// Loads multiple [`LLMRuntimeConfig`] from paths.
    pub fn from_path_multiple<P>(paths: &[P]) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut configs = HashMap::default();

        for p in paths {
            let config = LLMRuntimeConfig::from_path(p)?;
            let name = config.name.clone();

            configs.insert(name, config);
        }

        Ok(Self {
            configs: Some(configs),
            active: None,
        })
    }

    /// Initializes [`LLMService`] with already preloaded [`LLMRuntimeConfig`]s.
    ///
    /// This initializer function takes a [`Vec`] of [`LLMRuntimeConfig`] and maps the model name
    /// to
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    ///
    /// ```
    pub fn from_runtime_configs(configs: &[LLMRuntimeConfig]) -> Self {
        let mappings = configs
            .iter()
            .map(|c| (c.name.clone(), c.clone()))
            .collect();

        Self {
            configs: Some(mappings),
            active: None,
        }
    }
}

impl LLMService {
    /// Returns the currently active [`LLMRuntime`], or `None`
    pub fn runtime(&mut self) -> Option<&mut LLMRuntime> {
        self.active.as_mut()
    }

    /// Returns a list of available model names
    pub fn list_models(&self) -> Vec<String> {
        self.configs
            .as_ref()
            .map(|configs| configs.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Shuts down the currently active runtime if one exists
    fn shutdown_active(&mut self) {
        if let Some(runtime) = self.active.take() {
            tracing::debug!("Shutting down active runtime");
            runtime.shutdown();
        }
    }

    /// Add a [`LLMRuntimeConfig`] to the service at runtime.
    pub fn add_config(&mut self, config: String) -> Result<(), Error> {
        let c = LLMRuntimeConfig::from_raw(config)?;

        match self.configs {
            Some(ref mut inner) => {
                inner.insert(c.name.clone(), c);
            }
            None => {
                let mut configs = HashMap::new();
                configs.insert(c.name.clone(), c);

                self.configs = Some(configs)
            }
        }

        Ok(())
    }

    /// Activates the target [`LLMRuntime`]
    ///
    /// Calling this function does a few things interally:
    ///
    /// - Checking, if there is an active [`LLMRuntime`] and shutting it down
    /// - Loading the desired [`LLMRuntimeConfig`] by `id`
    /// - Initialize the [`LLMRuntime`]
    /// - Return a mutable reference to the [`LLMRuntime`]
    /// - Set the new [`Runtime`] active
    pub fn activate(&mut self, id: String) -> Result<&mut LLMRuntime, Error> {
        // Shutdown active runtime if exists
        self.shutdown_active();

        // Look up config by id
        let config = self
            .configs
            .as_ref()
            .and_then(|configs| configs.get(&id))
            .ok_or_else(|| {
                Error::MissingConfigLLM(format!("No configuration found for model: {}", id))
            })?
            .clone();

        tracing::debug!("Activating runtime for model: {}", id);

        // Create new runtime from config
        let mut runtime = LLMRuntime::from_config(config)?;

        // Start the worker thread and load model weights
        runtime.run_stream()?;

        // Store in active field
        self.active = Some(runtime);

        // Return mutable reference to active runtime
        self.active.as_mut().ok_or(Error::MissingActiveRuntime)
    }
}
