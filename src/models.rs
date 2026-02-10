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

        tracing::debug!("Query as JSON: {}", json_context);
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
    /// If present (and no `model_index_file`), the model format is inferred as GGUF.
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
    pub model_max_length: Option<usize>,
    pub tokenizer_class: Option<String>,

    pub added_tokens_decoder: Option<HashMap<String, AddedToken>>,
}

impl LLMRuntimeConfig {
    /// Returns true if a `model_index_file` is present, indicating Safetensors format
    pub fn is_safetensors(&self) -> bool {
        self.model_index_file
            .as_ref()
            .map(|p| !p.as_os_str().is_empty())
            .unwrap_or(false)
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
}
