use std::fs::File;

use crate::error::Error;
use crate::llmconfig::LLMRuntimeConfig;
use crate::runtime::{LLMRuntimeModel, LlmMessage};
use candle_core::Device;
use candle_core::{quantized::gguf_file, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use tokenizers::Tokenizer;

pub struct Qwen3Model {
    pub(crate) streaming: bool,
    pub(crate) device: Option<Device>,
    pub(crate) tokenizer: Option<Tokenizer>,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) temperature: f32,
    pub(crate) thinking: bool,
    pub(crate) weights: Option<Qwen3>,
}

impl LLMRuntimeModel for Qwen3Model {
    fn execute(&self, message: LlmMessage) -> Result<LlmMessage, Error> {
        if let LlmMessage::Prompt { system: _, message } = message {
            tracing::debug!("Processing Message: {:?}", message);

            // do inference here
            return Ok(LlmMessage::Response {
                error: String::new(),
                message,
            });
        }

        Err(Error::ExecutionError)
    }

    fn init(&mut self, config: &LLMRuntimeConfig) -> Result<(), crate::Error> {
        self.tokenizer = Some(
            Tokenizer::from_file(&config.tokenizer_config_file.as_ref().ok_or(
                Error::MissingConfig("Tokenizer config is missing".to_owned()),
            )?)
            .map_err(|e| Error::LoadingFile(e.to_string()))?,
        );

        self.weights = {
            let mut model_file = File::open(config.model_file.as_ref().ok_or(
                Error::MissingConfig("Model config file is missing".to_owned()),
            )?)?;
            let model = gguf_file::Content::read(&mut model_file)
                .map_err(|e| Error::LoadingFile(e.to_string()))?;

            Some(
                Qwen3::from_gguf(
                    model,
                    &mut model_file,
                    self.device.as_ref().ok_or(Error::MissingDevice)?,
                )
                .map_err(|e| Error::LoadingFile(e.to_string()))?,
            )
        };
        Ok(())
    }
}
