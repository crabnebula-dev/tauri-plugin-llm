use std::fs::File;

use crate::error::Error;
use crate::llmconfig::{LLMRuntimeConfig, ModelConfig};
use crate::loaders::safetensor::IndexFile;
use crate::runtime::{LLMRuntimeModel, LlmMessage};
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarBuilder;
// use candle_transformers::quantized_var_builder::VarBuilder;

use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};

use candle_transformers::generation::{LogitsProcessor, Sampling};
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
    pub(crate) weights: Option<Llama>,
    pub(crate) logits_processor: Option<LogitsProcessor>,
    pub(crate) cache: Option<model::Cache>,
    pub(crate) penalty: f32,
}

impl LLMRuntimeModel for LLama3Model {
    fn execute(&mut self, message: LlmMessage) -> anyhow::Result<LlmMessage, Error> {
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
                    .forward(&input, 0, self.cache.as_mut().unwrap())
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

            // TODO: set end of stream token
            let eos_token = *tokenizer.get_vocab(true).get("<|end_of_text|>").unwrap();

            // Start sampling
            for index in 0..num_samples {
                let input = Tensor::new(&[next_token], &device)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                let logits = model
                    .forward(&input, tokens.len() + index, self.cache.as_mut().unwrap())
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                let logits = candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.penalty,
                    &all_tokens[0..],
                )
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

    fn init(&mut self, config: &LLMRuntimeConfig) -> anyhow::Result<(), Error> {
        let LLMRuntimeConfig {
            tokenizer_config_file: _,
            model_config_file,
            model_index_file,
            model_file: _,
            model_dir,
            model_config: _,
            verbose: _,
        } = config;

        let ModelConfig {
            seed,
            sampling_config,
            penalty,
            ..
        } = config.model_config.clone();

        self.penalty = penalty;

        // Initialize the tokenizer
        self.tokenizer = Some(
            Tokenizer::from_file(&config.tokenizer_config_file.as_ref().ok_or(
                Error::MissingConfigLLM("Tokenizer config is missing".to_owned()),
            )?)
            .map_err(|e| {
                Error::LoadingFile(format!("{:?}", config.tokenizer_config_file), e.to_string())
            })?,
        );

        let mut llama_config_file = File::open(model_config_file.as_ref().unwrap())?;
        let cfg: LlamaConfig = serde_json::from_reader(&mut llama_config_file)?;
        let cfg = cfg.into_config(false);

        // Load weights
        self.weights = {
            let mut index_file = IndexFile::from_path(model_index_file.as_ref().unwrap())?;
            let paths = index_file.files(model_dir.as_ref().unwrap());

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &paths,
                    candle_core::DType::BF16,
                    self.device.as_ref().unwrap(),
                )
                .map_err(|e| Error::ExecutionError(e.to_string()))?
            };

            Llama::load(vb, &cfg)
                .map_err(|e| Error::ExecutionError(e.to_string()))
                .ok()
        };

        self.cache = {
            Some(
                model::Cache::new(
                    false,
                    candle_core::DType::BF16,
                    &cfg,
                    self.device.as_ref().unwrap(),
                )
                .map_err(|e| Error::ExecutionError(e.to_string()))?,
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

    fn apply_chat_template(&mut self, template: String) {}
}
