//! Generic LocalRuntime that can load models in different weight formats.

use std::fs::File;
use std::sync::Arc;

use crate::error::Error;
use crate::iter::IntoIterChunks;
use crate::runtime::{LLMRuntimeModel, Query};
use crate::{
    GenerationSeed, LLMRuntimeConfig, SamplingConfig, TemplateProcessor, TokenUsage,
    TokenizerConfig,
};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use rand::Rng;
use tokenizers::Tokenizer;

use super::backend::{self, ModelBackend};

/// A generic local runtime that can load models in different formats.
///
/// Model-specific logic (forward pass, KV cache, tool call parsing) is
/// delegated to the [`ModelBackend`] trait. This struct handles the
/// model-agnostic parts: tokenization, template rendering, sampling,
/// chunk streaming, and tool call post-processing.
pub struct LocalRuntime {
    pub(crate) device: Option<Device>,
    pub(crate) tokenizer: Option<Tokenizer>,
    pub(crate) backend: Option<Box<dyn ModelBackend>>,
    pub(crate) template: Option<String>,
    pub(crate) template_proc: Option<TemplateProcessor>,
    pub(crate) eos_tokens: Vec<String>,
}

impl Default for LocalRuntime {
    fn default() -> Self {
        Self {
            device: None,
            tokenizer: None,
            backend: None,
            template: None,
            template_proc: None,
            eos_tokens: Vec::new(),
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
        temperature: Option<f32>,
        top_k: Option<f32>,
        top_p: Option<f32>,
        seed: Option<GenerationSeed>,
        sampling_config: Option<SamplingConfig>,
    ) -> LogitsProcessor {
        // Use Candle reference defaults: temp=0.8, top_k=40, top_p=0.9
        let temperature = temperature.map(|t| t as f64).unwrap_or(0.8);
        let top_k = top_k.map(|k| k as usize).unwrap_or(40);
        let top_p = top_p.map(|p| p as f64).unwrap_or(0.9);

        // If no sampling config specified, default to TopKThenTopP for better quality
        let sampling_config = sampling_config.unwrap_or(SamplingConfig::TopKThenTopP);
        let sampling = match &sampling_config {
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

        let seed = match seed.unwrap_or_default() {
            GenerationSeed::Fixed(inner) => inner as u64,
            GenerationSeed::Random => {
                let mut rng = rand::rng();
                let seed = rng.random_range(1..1e10 as u64);
                tracing::debug!("Using seed for Logits Processor: {seed}");
                seed
            }
        };

        LogitsProcessor::from_sampling(seed, sampling)
    }
}

impl LLMRuntimeModel for LocalRuntime {
    fn init(&mut self, config: &LLMRuntimeConfig) -> Result<(), Error> {
        let name = &config.name;

        // Load tokenizer config if available
        let tokenizer_config_json: Option<TokenizerConfig> =
            if let Some(t) = &config.tokenizer_config_file {
                let mut file = File::open(t)?;
                Some(serde_json::from_reader(&mut file)?)
            } else {
                None
            };

        // Set EOS tokens from tokenizer_config.json, falling back to defaults
        self.eos_tokens = if let Some(eos) = tokenizer_config_json
            .as_ref()
            .and_then(|tc| tc.eos_token.as_ref())
        {
            tracing::info!("Using EOS token from tokenizer_config: {eos}");
            vec![eos.clone()]
        } else {
            tracing::warn!("No EOS token in tokenizer_config, falling back to default");
            vec!["</s>".to_string()]
        };

        // Load template
        self.template = {
            if let Some(tc) = &tokenizer_config_json {
                if let Some(template) = &tc.chat_template {
                    tracing::info!("Loaded Template from tokenizer_config file");
                    Some(template.clone())
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

        // Load backend — infer format from config fields
        tracing::info!("Loading Model Backend");
        let device = self.device.as_ref().ok_or(Error::MissingDevice)?;

        self.backend = Some(if config.is_safetensors() {
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
            backend::create_backend(name, device, model_index, model_dir, model_config_file)?
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
            penalty,
            seed,
            sampling_config,
        } = message.clone()
        {
            let chunk_size = chunk_size.unwrap_or(self.default_chunksize());

            // Create logits processor with runtime parameters
            let mut logits_processor = Self::create_logits_processor(
                temperature,
                top_k,
                top_p,
                seed,
                sampling_config.clone(),
            );

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

            let tokenizer = self.tokenizer.as_ref().unwrap();
            let backend = self.backend.as_mut().unwrap();
            let device = self.device.as_ref().unwrap();

            // Encode message
            let tokens = tokenizer
                .encode(processed_message, true)
                .map_err(|e| Error::MessageEncodingError(e.to_string()))?;

            let tokens = tokens.get_ids();

            // Clear cache for fresh generation
            backend.clear_kv_cache();

            // Get first token
            let mut next_token = {
                let input = Tensor::new(tokens, device)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                let logits = backend.forward(&input, 0)?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                logits_processor
                    .sample(&logits)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
            };

            let mut all_tokens = vec![next_token];

            // Get EOS token IDs
            let eos_token_ids: Vec<u32> = self
                .eos_tokens
                .iter()
                .filter_map(|eos| tokenizer.get_vocab(true).get(eos).copied())
                .collect();

            if eos_token_ids.is_empty() {
                tracing::warn!("No EOS token IDs found in tokenizer vocabulary!");
            }

            let penalty = penalty.unwrap_or(1.1).max(0.1);
            let mut index = 0usize;
            let mut done = false;
            let mut sample_error: Option<Error> = None;

            // Token iterator: yields tokens until EOS, max_tokens, or error
            let token_iter = std::iter::once(next_token).chain(std::iter::from_fn(|| {
                if done || index >= generate_num_samples {
                    return None;
                }

                let current_index = index;
                index += 1;

                let result = (|| -> Result<u32, Error> {
                    let input = Tensor::new(&[next_token], device)
                        .map_err(|e| Error::ExecutionError(e.to_string()))?
                        .unsqueeze(0)
                        .map_err(|e| Error::ExecutionError(e.to_string()))?;

                    let logits = backend.forward(&input, tokens.len() + current_index)?;
                    let logits = logits
                        .squeeze(0)
                        .map_err(|e| Error::ExecutionError(e.to_string()))?;

                    let start_at = all_tokens.len().saturating_sub(128);
                    let logits = candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        penalty,
                        &all_tokens[start_at..],
                    )
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                    logits_processor
                        .sample(&logits)
                        .map_err(|e| Error::ExecutionError(e.to_string()))
                })();

                match result {
                    Ok(token) => {
                        next_token = token;
                        all_tokens.push(token);
                        if eos_token_ids.contains(&token) {
                            tracing::debug!("FOUND EOS TOKEN");
                            done = true;
                        }
                        Some(token)
                    }
                    Err(e) => {
                        sample_error = Some(e);
                        None
                    }
                }
            }));

            let mut last_chunk_id = 0usize;

            for (id, chunk) in token_iter.chunks(chunk_size).enumerate() {
                let chunk_tokens: Vec<u32> = chunk.into_iter().collect();
                let data = tokenizer
                    .decode(&chunk_tokens, true)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .as_bytes()
                    .to_vec();

                tracing::debug!("Sending Chunk {id}");
                last_chunk_id = id;

                if let Err(e) = response_tx.send(Query::Chunk {
                    id,
                    kind: crate::QueryChunkType::String,
                    data,
                    timestamp,
                }) {
                    tracing::error!("Error sending chunk: {e}");
                    return Err(Error::StreamError(e.to_string()));
                }
            }

            if let Some(e) = sample_error {
                return Err(e);
            }

            // Tool call post-processing: parse the full output for tool calls
            if let Some(parser) = backend.tool_call_parser() {
                let full_text = tokenizer
                    .decode(&all_tokens, true)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                if let Some(tool_calls) = parser.parse(&full_text) {
                    tracing::debug!("Detected {} tool call(s) in model output", tool_calls.len());

                    let data = serde_json::to_vec(&tool_calls)
                        .map_err(|e| Error::ExecutionError(e.to_string()))?;

                    if let Err(e) = response_tx.send(Query::Chunk {
                        id: last_chunk_id + 1,
                        kind: crate::QueryChunkType::ToolCall,
                        data,
                        timestamp,
                    }) {
                        tracing::error!("Error sending tool call chunk: {e}");
                        return Err(Error::StreamError(e.to_string()));
                    }
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
