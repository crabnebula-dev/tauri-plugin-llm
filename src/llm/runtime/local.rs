//! Generic LocalRuntime that can load models in different weight formats (GGUF, Safetensors, Pickle)

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::Error;
use crate::loaders::IndexFile;
use crate::runtime::{LLMRuntimeModel, Query};
use crate::{
    GenerationSeed, LLMRuntimeConfig, SamplingConfig, TemplateProcessor, TokenUsage,
    TokenizerConfig,
};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use rand::Rng;
use tokenizers::Tokenizer;

// Model weight types supported by candle
use candle_transformers::models::llama::{self as llama_model, Llama, LlamaConfig};
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlamaWeights;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3Weights;
use candle_transformers::models::qwen3::{self as qwen3_model, Config as Qwen3Config};

/// Abstraction over different model weight types
pub enum ModelWeights {
    /// Quantized Llama weights (GGUF format)
    QuantizedLlama(QuantizedLlamaWeights),
    /// Quantized Qwen3 weights (GGUF format)
    QuantizedQwen3(QuantizedQwen3Weights),
    /// Full precision Llama weights (Safetensors format)
    Llama {
        model: Llama,
        cache: llama_model::Cache,
    },
    /// Full precision Qwen3 weights (Safetensors format)
    Qwen3(qwen3_model::Model),
}

impl ModelWeights {
    /// Forward pass through the model
    pub fn forward(&mut self, input: &Tensor, index: usize) -> Result<Tensor, Error> {
        match self {
            ModelWeights::QuantizedLlama(model) => model
                .forward(input, index)
                .map_err(|e| Error::ExecutionError(e.to_string())),
            ModelWeights::QuantizedQwen3(model) => model
                .forward(input, index)
                .map_err(|e| Error::ExecutionError(e.to_string())),
            ModelWeights::Llama { model, cache } => {
                let logits = model
                    .forward(input, index, cache)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                Self::extract_last_token_logits(logits)
            }
            ModelWeights::Qwen3(model) => {
                let logits = model
                    .forward(input, index)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                Self::extract_last_token_logits(logits)
            }
        }
    }

    /// Extracts the last token's logits from the model output.
    /// Handles both 3D [batch, seq_len, vocab] and 2D [batch, vocab] outputs.
    fn extract_last_token_logits(logits: Tensor) -> Result<Tensor, Error> {
        if logits.dims().len() == 3 {
            let seq_len = logits
                .dim(1)
                .map_err(|e| Error::ExecutionError(e.to_string()))?;
            logits
                .narrow(1, seq_len - 1, 1)
                .and_then(|t| t.squeeze(1))
                .map_err(|e| Error::ExecutionError(e.to_string()))
        } else {
            Ok(logits)
        }
    }
}

impl ModelWeights {
    /// Clear the KV cache if the model supports it
    pub fn clear_kv_cache(&mut self) {
        match self {
            ModelWeights::QuantizedLlama(_) => {}
            ModelWeights::QuantizedQwen3(model) => model.clear_kv_cache(),
            ModelWeights::Llama { .. } => {}
            ModelWeights::Qwen3(_) => {
                // qwen3::Model::clear_kv_cache is private in candle_transformers
            }
        }
    }
}

/// A generic local runtime that can load models in different formats
pub struct LocalRuntime {
    pub(crate) device: Option<Device>,
    pub(crate) tokenizer: Option<Tokenizer>,
    pub(crate) weights: Option<ModelWeights>,
    pub(crate) template: Option<String>,
    pub(crate) template_proc: Option<TemplateProcessor>,
    pub(crate) seed: GenerationSeed,
    pub(crate) sampling_config: SamplingConfig,
    pub(crate) penalty: f32,
    pub(crate) eos_token: Option<String>,
}

impl Default for LocalRuntime {
    fn default() -> Self {
        Self {
            device: None,
            tokenizer: None,
            weights: None,
            template: None,
            template_proc: None,
            seed: GenerationSeed::Random,
            sampling_config: SamplingConfig::default(),
            penalty: 1.1,
            eos_token: None,
        }
    }
}

impl LocalRuntime {
    pub fn new(device: Device) -> Self {
        Self {
            device: Some(device),
            ..Default::default()
        }
    }

    /// Creates a LogitsProcessor based on runtime parameters from Query::Prompt
    fn create_logits_processor(
        &self,
        temperature: Option<f32>,
        top_k: Option<f32>,
        top_p: Option<f32>,
    ) -> LogitsProcessor {
        let temperature = temperature.map(|t| t as f64).unwrap_or(0.7);
        let top_k = top_k.map(|k| k as usize).unwrap_or(40);
        let top_p = top_p.map(|p| p as f64).unwrap_or(0.9);

        let sampling = match &self.sampling_config {
            SamplingConfig::ArgMax => Sampling::ArgMax,
            SamplingConfig::All => Sampling::All { temperature },
            SamplingConfig::TopK => Sampling::TopK {
                k: top_k,
                temperature,
            },
            SamplingConfig::TopP => Sampling::TopP {
                p: top_p,
                temperature,
            },
            SamplingConfig::TopKThenTopP => Sampling::TopKThenTopP {
                k: top_k,
                p: top_p,
                temperature,
            },
            SamplingConfig::GumbelSoftmax => Sampling::GumbelSoftmax { temperature },
        };

        let seed = match &self.seed {
            GenerationSeed::Fixed(inner) => *inner as u64,
            GenerationSeed::Random => {
                let mut rng = rand::rng();
                let seed = rng.random_range(1..1e10 as u64);
                tracing::debug!("Using seed for Logits Processor: {seed}");
                seed
            }
        };

        LogitsProcessor::from_sampling(seed, sampling)
    }

    /// Load weights from a GGUF file
    fn load_gguf_weights(
        &self,
        model_file: &PathBuf,
        model_name: &str,
    ) -> Result<ModelWeights, Error> {
        let mut file = File::open(model_file)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| Error::LoadingFile(format!("{:?}", model_file), e.to_string()))?;

        let device = self.device.as_ref().ok_or(Error::MissingDevice)?;

        // Determine model type from name
        if model_name.contains("Qwen") {
            let weights = QuantizedQwen3Weights::from_gguf(content, &mut file, device)
                .map_err(|e| Error::LoadingFile(format!("{:?}", model_file), e.to_string()))?;
            Ok(ModelWeights::QuantizedQwen3(weights))
        } else {
            // Default to Llama for GGUF
            let weights = QuantizedLlamaWeights::from_gguf(content, &mut file, device)
                .map_err(|e| Error::LoadingFile(format!("{:?}", model_file), e.to_string()))?;
            Ok(ModelWeights::QuantizedLlama(weights))
        }
    }

    /// Load weights from sharded safetensors files.
    /// Dispatches to the correct model loader based on model name.
    fn load_safetensor_weights(
        &self,
        model_index_file: &PathBuf,
        model_dir: &PathBuf,
        model_config_file: &PathBuf,
        model_name: &str,
    ) -> Result<ModelWeights, Error> {
        let device = self.device.as_ref().ok_or(Error::MissingDevice)?;

        // Load sharded weights via index file
        let mut index_file = IndexFile::from_path(model_index_file)?;
        let paths = index_file.files(model_dir);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::BF16, device)
                .map_err(|e| Error::ExecutionError(e.to_string()))?
        };

        if model_name.contains("Qwen") {
            tracing::info!("Loading Qwen3 safetensors model");
            let mut config_file = File::open(model_config_file)?;
            let qwen3_config: Qwen3Config = serde_json::from_reader(&mut config_file)?;

            let model = qwen3_model::Model::new(&qwen3_config, vb)
                .map_err(|e| Error::ExecutionError(e.to_string()))?;

            Ok(ModelWeights::Qwen3(model))
        } else {
            tracing::info!("Loading Llama safetensors model");
            let mut config_file = File::open(model_config_file)?;
            let llama_config: LlamaConfig = serde_json::from_reader(&mut config_file)?;
            let config = llama_config.into_config(false);

            let model =
                Llama::load(vb, &config).map_err(|e| Error::ExecutionError(e.to_string()))?;
            let cache = llama_model::Cache::new(false, candle_core::DType::BF16, &config, device)
                .map_err(|e| Error::ExecutionError(e.to_string()))?;

            Ok(ModelWeights::Llama { model, cache })
        }
    }
}

impl LLMRuntimeModel for LocalRuntime {
    fn init(&mut self, config: &LLMRuntimeConfig) -> Result<(), Error> {
        let name = &config.name;

        self.penalty = config.penalty.max(0.1);
        self.seed = config.seed.clone();
        self.sampling_config = config.sampling_config.clone();

        // Set EOS token based on model name
        self.eos_token = if name.contains("Llama") {
            Some("<|eot_id|>".to_string())
        } else if name.contains("Qwen") {
            Some("<|im_end|>".to_string())
        } else {
            Some("</s>".to_string())
        };

        // Load template
        self.template = {
            if let Some(t) = &config.tokenizer_config_file {
                let mut file = File::open(t)?;
                let tokenizer_config_json: TokenizerConfig = serde_json::from_reader(&mut file)?;

                if let Some(template) = tokenizer_config_json.chat_template {
                    tracing::info!("Loaded Template from tokenizer_config file");
                    Some(template)
                } else {
                    tracing::info!("The tokenizer_config file does not provide a chat template");
                    None
                }
            } else if let Some(t) = &config.template_file {
                tracing::info!("Loading extra provided template file");
                Some(std::fs::read_to_string(t)?)
            } else {
                tracing::info!("No extra template file has been provided");
                None
            }
        };
        tracing::info!("Loading template processor");
        self.template_proc = if config.tokenizer_config_file.is_some() && self.template.is_some() {
            Some(TemplateProcessor::with_jinja_template())
        } else {
            None
        };

        // Initialize tokenizer
        tracing::info!("Loading Tokenizer");
        self.tokenizer =
            Some(
                Tokenizer::from_file(config.tokenizer_file.as_ref().ok_or(
                    Error::MissingConfigLLM("Tokenizer file is missing".to_owned()),
                )?)
                .map_err(|e| {
                    Error::LoadingFile(format!("{:?}", config.tokenizer_file), e.to_string())
                })?,
            );

        // Load weights — infer format from config fields
        tracing::info!("Loading Model File");
        self.weights = Some(if config.is_safetensors() {
            let model_index = config
                .model_index_file
                .as_ref()
                .ok_or(Error::MissingConfigLLM(
                    "Model index file is missing for Safetensors".to_owned(),
                ))?;
            let model_dir = config.model_dir.as_ref().ok_or(Error::MissingConfigLLM(
                "Model directory is missing for Safetensors".to_owned(),
            ))?;
            let model_config_file =
                config
                    .model_config_file
                    .as_ref()
                    .ok_or(Error::MissingConfigLLM(
                        "Model config file is missing for Safetensors".to_owned(),
                    ))?;

            tracing::info!("Loading Safetensor Weights");
            self.load_safetensor_weights(model_index, model_dir, model_config_file, name)?
        } else if config.is_gguf() {
            let model_file = config.model_file.as_ref().ok_or(Error::MissingConfigLLM(
                "Model file is missing for GGUF".to_owned(),
            ))?;
            self.load_gguf_weights(model_file, name)?
        } else {
            return Err(Error::ExecutionError(
                "Cannot infer model format: neither model_index_file nor model_file is set"
                    .to_owned(),
            ));
        });

        tracing::info!("Runtime has been initialized");

        Ok(())
    }

    fn execute(
        &mut self,
        q: Query,
        response_tx: Arc<std::sync::mpsc::Sender<Query>>,
    ) -> Result<(), Error> {
        let usage = self.inference(q, response_tx.clone())?;

        tracing::debug!("LocalRuntime inference ended. Sending end termination");
        response_tx
            .send(Query::End { usage })
            .map_err(|e| Error::StreamError(e.to_string()))?;

        Ok(())
    }

    fn inference(
        &mut self,
        message: Query,
        response_tx: Arc<std::sync::mpsc::Sender<Query>>,
    ) -> Result<Option<TokenUsage>, Error> {
        tracing::debug!("LocalRuntime got message: {:?}", message);

        if let Query::Prompt {
            messages,
            tools: _,
            chunk_size,
            timestamp,
            max_tokens,
            temperature,
            top_k,
            top_p,
            think: _,
            stream: _,
            model: _,
        } = message.clone()
        {
            let chunk_size = chunk_size.unwrap_or(self.default_chunksize());

            // Create logits processor with runtime parameters
            let mut logits_processor = self.create_logits_processor(temperature, top_k, top_p);

            // Preprocess message by applying chat template
            let processed_message = {
                match self.template.as_ref() {
                    Some(template) => {
                        let proc = self.template_proc.as_ref().ok_or(Error::ExecutionError(
                            "Template processor is not initialized".to_string(),
                        ))?;
                        message.apply_template(template, proc)?
                    }
                    None => {
                        tracing::warn!("No template found. Using plain message content");
                        messages
                            .iter()
                            .map(|m| format!("{}: {}", m.role, m.content))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
            };

            let generate_num_samples = max_tokens.unwrap_or(500);

            tracing::debug!("Processing Message: {:?}", processed_message);

            let tokenizer = self.tokenizer.as_ref().unwrap();
            let weights = self.weights.as_mut().unwrap();
            let device = self.device.as_ref().unwrap();

            // Encode message
            let tokens = tokenizer
                .encode(processed_message, true)
                .map_err(|e| Error::MessageEncodingError(e.to_string()))?;

            let tokens = tokens.get_ids();

            // Clear cache for fresh generation
            weights.clear_kv_cache();

            // Get first token
            let mut next_token = {
                let input = Tensor::new(tokens, device)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                let logits = weights.forward(&input, 0)?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                logits_processor
                    .sample(&logits)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
            };

            let mut all_tokens = vec![];
            all_tokens.push(next_token);

            let mut chunks = Vec::with_capacity(chunk_size);
            chunks.push(next_token);

            // Get EOS token
            let eos_token = self
                .eos_token
                .as_ref()
                .and_then(|eos| tokenizer.get_vocab(true).get(eos).copied())
                .unwrap_or(u32::MAX);

            let mut id = 0;

            // Start sampling
            for index in 0..generate_num_samples {
                let input = Tensor::new(&[next_token], device)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                let logits = weights.forward(&input, tokens.len() + index)?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                // Apply repetition penalty
                let start_at = all_tokens.len().saturating_sub(128);
                let logits = candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.penalty,
                    &all_tokens[start_at..],
                )
                .map_err(|e| Error::ExecutionError(e.to_string()))?;

                next_token = logits_processor
                    .sample(&logits)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;
                all_tokens.push(next_token);
                chunks.push(next_token);

                if chunks.len().eq(&chunk_size) {
                    let data = match tokenizer.decode(&chunks, true) {
                        Ok(str) => str.as_bytes().to_vec(),
                        Err(e) => return Err(Error::ExecutionError(e.to_string())),
                    };

                    id += 1;

                    tracing::debug!("Sending Chunk");

                    if let Err(e) = response_tx.send(Query::Chunk {
                        id,
                        kind: crate::QueryChunkType::String,
                        data,
                        timestamp,
                    }) {
                        tracing::error!("Error sending chunk: {e}");
                        return Err(Error::StreamError(e.to_string()));
                    }

                    chunks.truncate(0);
                }

                if next_token == eos_token {
                    break;
                }
            }

            // Send remaining chunks
            if !chunks.is_empty() {
                let data = match tokenizer.decode(&chunks, true) {
                    Ok(str) => str.as_bytes().to_vec(),
                    Err(e) => return Err(Error::ExecutionError(e.to_string())),
                };

                if let Err(e) = response_tx.send(Query::Chunk {
                    id: id + 1,
                    kind: crate::QueryChunkType::String,
                    data,
                    timestamp,
                }) {
                    return Err(Error::StreamError(e.to_string()));
                }
            }

            let prompt_tokens = tokens.len();
            let completion_tokens = all_tokens.len();

            tracing::debug!(
                "Finished inference. Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}"
            );

            return Ok(Some(TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }));
        }

        tracing::warn!(
            "Got unhandled `Query` type: discriminant({:?}), actual type: ({:?})",
            std::mem::discriminant(&message),
            message
        );

        Ok(None)
    }
}
