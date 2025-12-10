use std::fs::File;

use crate::error::Error;
use crate::llmconfig::{LLMRuntimeConfig, ModelConfig};
use crate::runtime::{LLMRuntimeModel, LlmMessage};
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_core::{safetensors, Tensor};
use candle_transformers::quantized_var_builder::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use rand::Rng;
use tokenizers::Tokenizer;

pub struct LLama3Model {
    pub(crate) streaming: bool,
    pub(crate) device: Option<Device>,
    pub(crate) tokenizer: Option<Tokenizer>,
    pub(crate) top_k: usize,
    pub(crate) top_p: f64,
    pub(crate) temperature: f64,
    pub(crate) thinking: bool,
    pub(crate) weights: Option<Qwen3>,
    pub(crate) logits_processor: Option<LogitsProcessor>,
}

impl LLMRuntimeModel for LLama3Model {
    fn execute(&mut self, message: LlmMessage) -> anyhow::Result<LlmMessage, Error> {
        todo!()
    }

    fn init(&mut self, config: &LLMRuntimeConfig) -> anyhow::Result<(), Error> {
        let ModelConfig {
            seed,
            sampling_config,
            ..
        } = config.model_config.clone();

        // Initialize the tokenizer
        self.tokenizer = Some(
            Tokenizer::from_file(&config.tokenizer_config_file.as_ref().ok_or(
                Error::MissingConfigLLM("Tokenizer config is missing".to_owned()),
            )?)
            .map_err(|e| {
                Error::LoadingFile(format!("{:?}", config.tokenizer_config_file), e.to_string())
            })?,
        );

        // Load weights
        self.weights = {
            let mut model_file = File::open(config.model_file.as_ref().ok_or(
                Error::MissingConfigLLM("Model config file is missing".to_owned()),
            )?)?;

            

            let model = gguf_file::Content::read(&mut model_file).map_err(|e| {
                Error::LoadingFile(format!("{:?}", config.model_file), e.to_string())
            })?;

            Some(
                Qwen3::from_gguf(
                    model,
                    &mut model_file,
                    self.device.as_ref().ok_or(Error::MissingDevice)?,
                )
                .map_err(|e| Error::LoadingFile(format!("{:?}", model_file), e.to_string()))?,
            )
        };

        // Initialize Logits Processor
        self.logits_processor = {
            let sampling = match sampling_config {
                crate::llmconfig::SamplingConfig::ArgMax => Sampling::ArgMax,
                crate::llmconfig::SamplingConfig::All => Sampling::All {
                    temperature: self.temperature,
                },
                crate::llmconfig::SamplingConfig::TopK => Sampling::TopK {
                    k: self.top_k,
                    temperature: self.temperature,
                },
                crate::llmconfig::SamplingConfig::TopP => Sampling::TopP {
                    p: self.top_p,
                    temperature: self.temperature,
                },
                crate::llmconfig::SamplingConfig::TopKThenTopP => Sampling::TopKThenTopP {
                    k: self.top_k,
                    p: self.top_p,
                    temperature: self.temperature,
                },
                crate::llmconfig::SamplingConfig::GumbelSoftmax => Sampling::GumbelSoftmax {
                    temperature: self.temperature,
                },
            };

            let seed = match seed {
                crate::llmconfig::GenerationSeed::Fixed(inner) => inner as u64,
                crate::llmconfig::GenerationSeed::Random => {
                    let mut rng = rand::rng();
                    let seed = rng.random_range(1..1e10 as u64);
                    tracing::debug!("Using seed for Logits Processor: {seed}");

                    seed
                }
            };

            Some(LogitsProcessor::from_sampling(seed, sampling))
        };

        Ok(())
    }

    fn apply_chat_template(&mut self, template: String) {
        todo!()
    }
}
