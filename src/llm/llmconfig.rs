use crate::error::Error;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct LLMRuntimeConfig {
    /// Path to `tokenizer.json`
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
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ModelConfig {
    /// Limits sampling to the K most likely next tokens.
    pub top_k: usize,

    /// Nucleus sampling. Dynamically selects the smallest
    /// set of tokens whose cumulative probability exceeds P
    pub top_p: f32,

    /// Lower Temperatures 0.1 - 0.5 select tokens with high confidence
    /// Higher Temperatures 0.5 - 1.0 consider more possibilities for the next token
    pub temperature: f32,

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
}

#[derive(Deserialize, Serialize, Debug, Default, Clone)]
pub enum GenerationSeed {
    Fixed(usize),

    #[default]
    Random,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum ModelFileType {
    // *.gguf
    GGUF,

    // *.safetensors
    Safetensors,

    // *.pth
    Pickle,
}

impl LLMRuntimeConfig {
    ///Loads a config from path
    pub fn from_path<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut file = File::open(path.as_ref()).map_err(|_| Error::ExecutionError)?;
        serde_json::from_reader(&mut file).map_err(|_| Error::ExecutionError)
    }
}
