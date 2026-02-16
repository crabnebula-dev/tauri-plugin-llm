use crate::{error::Error, TemplateProcessor};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};
use tokenizers::AddedToken;

/// A query type.
#[derive(Serialize, Deserialize, Debug, Clone)]
// #[serde(tag = "type")] // we changed this to only see the values
#[serde(untagged)]
pub enum Query {
    Prompt {
        messages: Vec<QueryMessage>,

        /// We keep the tools info as generic as possible.
        /// This may change in the future. For now a model can be
        /// informed about available tools by a json encoded message
        /// as defined by the MCP standard
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tools: Vec<String>,

        chunk_size: Option<usize>,

        timestamp: Option<u64>,

        max_tokens: Option<usize>,

        temperature: Option<f32>,

        top_k: Option<f32>,

        top_p: Option<f32>,

        #[serde(default)]
        think: bool,

        #[serde(default)]
        stream: bool,

        model: Option<String>,

        /// Repetition penalty. Defaults to 1.1 if not provided.
        penalty: Option<f32>,

        /// Generation seed. Defaults to Random if not provided.
        seed: Option<GenerationSeed>,

        /// Sampling configuration. Defaults to All if not provided.
        sampling_config: Option<SamplingConfig>,
    },

    Response {
        error: Option<String>,
        messages: Vec<QueryMessage>,
        tools: Vec<String>,
    },

    Chunk {
        id: usize,
        data: Vec<u8>,
        timestamp: Option<u64>,
        kind: QueryChunkType,
    },

    End {
        usage: Option<TokenUsage>,
    },
    Exit,
    Status {
        msg: String,
    },
}

/// Represents a parsed tool call extracted from model output.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(id: String, name: String, arguments: serde_json::Value) -> Self {
        Self {
            id,
            name,
            arguments,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[serde(untagged)]
pub enum QueryChunkType {
    String,
    Bytes,
    ToolCall,
}

/// Metrics on actual token usage
#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct TokenUsage {
    /// The number of input tokens
    pub prompt_tokens: usize,

    /// The number of tokens generated
    pub completion_tokens: usize,

    /// the total number of tokens used (prompt + completion)
    pub total_tokens: usize,
}

impl Query {
    /// Applies [`Self`] with the given template and returns the rendered version as String
    pub fn apply_template(&self, template: &str, tp: &TemplateProcessor) -> Result<String, Error> {
        let json_context = serde_json::to_string(self)?;
        tp.render(template, &json_context)
    }

    pub fn try_render_as_event_name(&self) -> Result<String, Error> {
        match self {
            Query::Chunk { .. } => Ok("query-stream-chunk".to_string()),
            Query::End { .. } => Ok("query-stream-end".to_string()),
            Query::Status { .. } => Ok("query-stream-error".to_string()),

            Query::Prompt { .. } | Query::Response { .. } | Query::Exit => {
                Err(Error::UndefinedClientEvent(format!("{self:?}")))
            }
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct LLMRuntimeConfig {
    /// Name of the Model
    ///
    /// This setting is being used to detect which model loader to use.
    pub name: String,

    /// Path to `tokenizer.json`
    pub tokenizer_file: Option<PathBuf>,

    /// Path to `tokenizer_config.json`
    pub tokenizer_config_file: Option<PathBuf>,

    /// path to `config.json`
    pub model_config_file: Option<PathBuf>,

    /// Path to `model.safetensors.index.json`.
    /// If present, the model format is inferred as Safetensors.
    pub model_index_file: Option<PathBuf>,

    /// Path to `model.EXTENSION` (e.g. `.gguf`).
    /// If present (and no `model_index_file`), the model format is inferred as Safetensors.
    pub model_file: Option<PathBuf>,

    /// Path to model directory
    ///
    /// Use this setting if the model files are distributed with sharded files eg. `*.safetensors`
    pub model_dir: Option<PathBuf>,

    /// If the models ships with a separate template file, this can be configured here.
    /// Given a `tokenizer_config_file`, the template file setting will be ignored.
    pub template_file: Option<PathBuf>,
}

#[derive(Deserialize, Serialize, Debug, Default, Clone)]
pub enum GenerationSeed {
    Fixed(usize),

    #[default]
    Random,
}

/// SampleConfig is copied 1:1 from
/// candle_transformers::generation::Sampling
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub enum SamplingConfig {
    ArgMax,

    #[default]
    All, // { temperature: f64 },
    TopK,          // { k: usize, temperature: f64 },
    TopP,          // { p: f64, temperature: f64 },
    TopKThenTopP,  // { k: usize, p: f64, temperature: f64 },
    GumbelSoftmax, // { temperature: f64 },
}

/// Use this to deserialize the `tokenizer_config.json`
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct TokenizerConfig {
    pub bos_token: Option<String>,
    pub chat_template: Option<String>,
    pub clean_up_tokenization_spaces: bool,
    pub eos_token: Option<String>,

    // TODO: this field is not being used, maybe we just skip the
    // deserialization?
    pub model_max_length: Option<u128>,
    pub tokenizer_class: Option<String>,

    pub added_tokens_decoder: Option<HashMap<String, AddedToken>>,
}

impl LLMRuntimeConfig {
    /// Returns true if a `model_index_file` is present, indicating Safetensors format
    pub fn is_safetensors_with_index_file(&self) -> bool {
        self.model_index_file
            .as_ref()
            .map(|p| !p.as_os_str().is_empty())
            .unwrap_or(false)
    }

    pub fn is_safetensors_inidividual_file(&self) -> bool {
        self.model_file
            .as_ref()
            .filter(|inner| {
                inner
                    .file_name()
                    .eq(&Some(std::ffi::OsStr::new("model.safetensors")))
            })
            .is_some()
    }

    /// Returns true if a `model_file` is present (and no index file), indicating GGUF format
    pub fn is_gguf(&self) -> bool {
        self.model_file
            .as_ref()
            .map(|p| !p.as_os_str().is_empty())
            .unwrap_or(false)
    }

    /// Loads a config from path
    pub fn from_path<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut file =
            File::open(path.as_ref()).map_err(|e| Error::ExecutionError(e.to_string()))?;
        serde_json::from_reader(&mut file).map_err(|e| Error::ExecutionError(e.to_string()))
    }

    /// Loads [`Self`] from a raw String.
    pub fn from_raw<S>(content: S) -> Result<Self, Error>
    where
        S: AsRef<str>,
    {
        Ok(serde_json::from_str(content.as_ref())?)
    }

    /// Tries to derive a [`LLMRuntomeConfig`] by a package location
    /// provided by hf_hub
    pub fn from_hf_local_cache<S, P>(model: S, cache_dir: Option<P>) -> Result<Self, Error>
    where
        S: AsRef<str>,
        P: AsRef<Path>,
    {
        let model_id = model.as_ref().to_owned();

        // enable this feature flag to disable inclusion in production build
        // to bypass model validation check. By passing Mock, no actual model
        // will be loaded, but a mock runtime will be returned.
        #[cfg(test)]
        if model.as_ref().eq("Mock") {
            return Ok(LLMRuntimeConfig {
                name: model_id,
                ..Default::default()
            });
        }

        Self::validate_model_name(&model_id)?;

        let cache = cache_dir.map_or_else(hf_hub::Cache::default, |path| {
            hf_hub::Cache::new(path.as_ref().into())
        });
        let cache_dir = cache.path().clone();
        let cache_repo = cache.model(model_id.clone());
        let tokenizer_file = cache_repo.get("tokenizer.json");
        let tokenizer_config_file = cache_repo.get("tokenizer_config.json");
        let model_config_file = cache_repo.get("config.json");
        let model_index_file = cache_repo.get("model.safetensors.index.json");
        let model_file = cache_repo.get("model.safetensors");
        // let template_file = cache_repo.get("chat_template.jinja");

        // Defense in depth: verify all resolved paths are within the cache directory.
        // This guards against symlink attacks or future changes in hf_hub's path logic.
        let resolved_files: Vec<&Option<std::path::PathBuf>> = vec![
            &tokenizer_file,
            &tokenizer_config_file,
            &model_config_file,
            &model_index_file,
            &model_file,
        ];
        for path in resolved_files.into_iter().flatten() {
            Self::ensure_within_cache(path, &cache_dir)?;
        }

        // For sharded models: if the index file exists but a single model file doesn't,
        // use the parent directory of the index file as model_dir
        let model_dir = if model_file.is_none() {
            model_index_file
                .as_ref()
                .and_then(|index_path| index_path.parent().map(|p| p.to_path_buf()))
        } else {
            None
        };

        if let Some(ref dir) = model_dir {
            Self::ensure_within_cache(dir, &cache_dir)?;
        }

        // Require at minimum a tokenizer and model weights (either single file or sharded dir)
        if tokenizer_file.is_none() || (model_file.is_none() && model_dir.is_none()) {
            return Err(Error::MissingConfigLLM(
               format!("Required model files not found in local cache for '{model_id}': tokenizer {tokenizer_file:?}, model_file {model_file:?}, model_dir {model_dir:?} \
               Ensure the model has been downloaded first.")
            ));
        }

        if model_file.is_some() {
            tracing::debug!("Model File for {} found {model_file:?}", model.as_ref())
        }

        Ok(LLMRuntimeConfig {
            name: model_id,
            tokenizer_file,
            tokenizer_config_file,
            model_config_file,
            model_index_file,
            model_file,
            model_dir,
            template_file: None,
        })
    }

    /// Verifies that a resolved path is within the expected cache directory.
    ///
    /// Uses path canonicalization to resolve symlinks and `..` segments, then
    /// checks that the result is still a descendant of `cache_dir`. This is a
    /// defense-in-depth measure — even if `validate_model_name` is bypassed or
    /// `hf_hub` changes its internal path construction, files outside the cache
    /// cannot be reached.
    fn ensure_within_cache(path: &Path, cache_dir: &Path) -> Result<(), Error> {
        let canonical = path.canonicalize()?;
        let cache_canonical = cache_dir.canonicalize()?;
        if !canonical.starts_with(&cache_canonical) {
            return Err(Error::MissingConfigLLM(format!(
                "Resolved path '{}' escapes the cache directory '{}'",
                canonical.display(),
                cache_canonical.display()
            )));
        }
        Ok(())
    }

    pub fn validate_model_name(name: &str) -> Result<(), Error> {
        let parts: Vec<&str> = name.split('/').collect();
        if parts.len() != 2 {
            return Err(Error::MissingConfigLLM(format!(
                "Model name must be in 'org/model' format, got: '{name}'"
            )));
        }

        let is_valid_segment = |s: &str| -> bool {
            !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
                && !s.contains("..")
        };

        if !is_valid_segment(parts[0]) || !is_valid_segment(parts[1]) {
            return Err(Error::MissingConfigLLM(
                "Model name contains invalid characters. Only alphanumeric, hyphens, underscores, and dots are allowed."
                    .to_string(),
            ));
        }

        let Some((segment1, segment2)) = name.split_once('/') else {
            return Err(Error::MissingConfigLLM(format!(
                "Model name must be in 'org/model' format, got: '{name}'"
            )));
        };

        for s in [segment1, segment2] {
            if s.is_empty()
                || !s
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
                || s.contains("..")
            {
                return Err(Error::MissingConfigLLM(
                    "Model name contains invalid characters. Only alphanumeric, hyphens, underscores, and dots are allowed."
                        .to_string(),
                ));
            }
        }

        Ok(())
    }
}
