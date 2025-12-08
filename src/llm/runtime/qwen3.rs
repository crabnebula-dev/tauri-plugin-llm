use std::fs::File;

use crate::error::Error;
use crate::llmconfig::{LLMRuntimeConfig, ModelConfig};
use crate::runtime::{LLMRuntimeModel, LlmMessage};
use candle_core::Device;
use candle_core::{quantized::gguf_file, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use rand::Rng;
use tokenizers::Tokenizer;

pub struct Qwen3Model {
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

impl LLMRuntimeModel for Qwen3Model {
    /// TODO:
    /// - apply penalty for repetitions
    /// - enable thinking mode
    /// - enable setting a system message
    fn execute(&mut self, message: LlmMessage) -> Result<LlmMessage, Error> {
        if let LlmMessage::Prompt {
            system: _,
            message,
            num_samples,
        } = message
        {
            tracing::debug!("Processing Message: {:?}", message);

            // get defaults
            let tokenizer = self.tokenizer.as_ref().unwrap();
            let model = self.weights.as_mut().unwrap();
            let logits_processor = self.logits_processor.as_mut().unwrap();
            let device = self.device.as_ref().unwrap();

            // encode message
            let tokens = tokenizer
                .encode(message, true)
                .map_err(|e| Error::MessageEncodingError(e.to_string()))?;

            let tokens = tokens.get_ids();

            // set next token
            let mut next_token = {
                let input = Tensor::new(tokens, &device)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                let logits = model
                    .forward(&input, 0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                logits_processor
                    .sample(&logits)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
            };

            let mut all_tokens = vec![];
            all_tokens.push(next_token);

            // set end of stream token
            let eos_token = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();

            // Start sampling
            for index in 0..num_samples {
                let input = Tensor::new(&[next_token], &device)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                let logits = model
                    .forward(&input, tokens.len() + index)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                next_token = logits_processor
                    .sample(&logits)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                all_tokens.push(next_token);

                if next_token == eos_token {
                    break;
                }
            }

            let message = match tokenizer.decode(&all_tokens, true) {
                Ok(str) => str,
                Err(e) => return Err(Error::ExecutionError(e.to_string())),
            };

            return Ok(LlmMessage::Response {
                error: None,
                message,
            });
        }

        Err(Error::ExecutionError("".to_string()))
    }

    fn init(&mut self, config: &LLMRuntimeConfig) -> Result<(), crate::Error> {
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
            .map_err(|e| Error::LoadingFile(e.to_string()))?,
        );

        // Load weights
        self.weights = {
            let mut model_file = File::open(config.model_file.as_ref().ok_or(
                Error::MissingConfigLLM("Model config file is missing".to_owned()),
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
}
