use crate::{error::Error, TemplateProcessor};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum Query {
    Prompt {
        messages: Vec<QueryMessage>,

        /// We keep the tools info as generic as possible.
        /// This may change in the future. For now a model can be
        /// informed about available tools by a json encoded message
        /// as defined by the MCP standard
        tools: Vec<String>,

        /// Optional config for the query.
        /// If no value has been set, the default is assumed
        #[serde(default, deserialize_with = "null_to_default")]
        config: Option<QueryConfig>,
    },
    Binary {},
    Response {
        error: Option<String>,
        messages: Vec<QueryMessage>,
        tools: Vec<String>,
    },
    Exit,
    Status,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct QueryConfig {
    pub generate_num_samples: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryMessage {
    pub role: String,
    pub content: String,
}

// #[derive(Serialize, Deserialize, Debug, Clone)]
// pub struct QueryMessageContentType {
//     pub r#type: String,
//     pub content: String,
// }

impl Query {
    /// Renders [`Self`] with the given template and returns the rendered version as String
    pub fn render(&self, template: &String, tp: &TemplateProcessor) -> Result<String, Error> {
        let json_context = serde_json::to_string(self)?;

        tracing::debug!("Query as JSON: {}", json_context);
        tp.render(&template, &json_context)
    }
}

impl Default for QueryConfig {
    fn default() -> Self {
        QueryConfig {
            generate_num_samples: 100,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct LLMRuntimeConfig {
    /// Path to `tokenizer.json`
    pub tokenizer_file: Option<PathBuf>,

    /// Path to `tokenizer_config.json`
    pub tokenizer_config_file: Option<PathBuf>,

    /// path to `config.json`
    pub model_config_file: Option<PathBuf>,

    /// Path to `model.safetensors.index.json`
    pub model_index_file: Option<PathBuf>,

    /// Path to `model.EXTENSION`
    ///
    /// The path to the model file depends on the model type. Some models use sharded
    /// model files eg. `*.safetensors`. For split files use `model_dir`
    pub model_file: Option<PathBuf>,

    /// Path to model directory
    ///
    /// Use this setting, if the model files are distributed with sharded files eg. `*.safetensors`
    pub model_dir: Option<PathBuf>,

    /// The Modelconfiguration
    pub model_config: ModelConfig,

    /// Enables logging
    pub verbose: bool,

    /// If the models ships with a separate template file, this can be configure here
    /// Given a `tokenizer_config_file`, the template file setting will be ignored
    pub template_file: Option<PathBuf>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct ModelConfig {
    /// Limits sampling to the K most likely next tokens.
    pub top_k: usize,

    /// Nucleus sampling. Dynamically selects the smallest
    /// set of tokens whose cumulative probability exceeds P
    pub top_p: f64,

    /// Lower Temperatures 0.1 - 0.5 select tokens with high confidence
    /// Higher Temperatures 0.5 - 1.0 consider more possibilities for the next token
    pub temperature: f64,

    /// Name of the Model
    ///
    /// This setting is being used to detect which model loader to use.
    pub name: String,

    /// Depending on the mode file type, a different loader will be selected internally
    pub file_type: ModelFileType,

    /// Repetition penalty
    pub penalty: f32,

    /// Some models expect a generation seed.
    ///
    /// This can either be a fixed value or random where random is the default, if no explicit
    /// value has been set.
    pub seed: GenerationSeed,

    /// Enabl thinking mode, if model supports it
    pub thinking: bool,

    /// Enable streaming responses
    pub streaming: bool,

    /// Sampling configuration
    pub sampling_config: SamplingConfig,
}

#[derive(Deserialize, Serialize, Debug, Default, Clone)]
pub enum GenerationSeed {
    Fixed(usize),

    #[default]
    Random,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub enum ModelFileType {
    // *.gguf
    GGUF,

    // *.safetensors
    #[default]
    Safetensors,

    // *.pth
    Pickle,
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
}

impl LLMRuntimeConfig {
    ///Loads a config from path
    pub fn from_path<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut file =
            File::open(path.as_ref()).map_err(|e| Error::ExecutionError(e.to_string()))?;
        serde_json::from_reader(&mut file).map_err(|e| Error::ExecutionError(e.to_string()))
    }
}

fn null_to_default<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Ok(Some(Option::deserialize(deserializer)?.unwrap_or_default()))
}
