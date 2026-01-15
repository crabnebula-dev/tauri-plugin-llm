use std::fs::File;

use crate::error::Error;
use crate::loaders::IndexFile;
use crate::runtime::{LLMRuntimeModel, Query};
use crate::{
    LLMRuntimeConfig, ModelConfig, QueryConfig, QueryMessage, TemplateProcessor, TokenizerConfig,
};
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};

use candle_transformers::generation::{LogitsProcessor, Sampling};
use rand::Rng;
use tokenizers::{AddedToken, Tokenizer};

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

    pub(crate) template: Option<String>,
    pub(crate) template_proc: Option<TemplateProcessor>,
}

impl LLMRuntimeModel for LLama3Model {
    fn execute(&mut self, message: Query) -> anyhow::Result<Query, Error> {
        if let Query::Prompt {
            messages: _,
            tools,
            config,
            chunk_size: _,
            timestamp: _,
        } = message.clone()
        {
            // preprocess message by applying chat template
            let message = {
                let template = self.template.as_ref().ok_or(Error::ExecutionError(format!(
                    "Template is missing in config!"
                )))?;
                let proc = self
                    .template_proc
                    .as_ref()
                    .ok_or(Error::ExecutionError(format!(
                        "Template processor is not intialized"
                    )))?;
                message.apply_template(&template, proc)?
            };

            let QueryConfig {
                generate_num_samples,
            } = config.unwrap();

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
            let eos_token = *tokenizer.get_vocab(true).get("<|eot_id|>").unwrap();

            tracing::info!("Encoded eos token: {eos_token}");

            // Start sampling
            for index in 0..generate_num_samples {
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

                // 128 = last n repeated tokens (this is just configuration)
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

                if next_token == eos_token {
                    break;
                }
            }

            let message = match tokenizer.decode(&all_tokens, true) {
                Ok(str) => str,
                Err(e) => return Err(Error::ExecutionError(e.to_string())),
            };

            return Ok(Query::Response {
                error: None,
                messages: vec![QueryMessage {
                    role: "assistant".to_owned(),
                    content: message,
                }],
                tools,
            });
        }

        Err(Error::ExecutionError(
            "Cannot handle Query type".to_string(),
        ))
    }

    fn init(&mut self, config: &LLMRuntimeConfig) -> anyhow::Result<(), Error> {
        let LLMRuntimeConfig {
            tokenizer_file: _,
            tokenizer_config_file,
            model_config_file,
            model_index_file,
            model_file: _,
            model_dir,
            model_config: _,
            verbose: _,
            template_file,
        } = config;

        let ModelConfig {
            seed,
            sampling_config,
            penalty,
            ..
        } = config.model_config.clone();

        self.penalty = penalty.max(0.1);

        // set template
        self.template = {
            if let Some(t) = tokenizer_config_file {
                let mut file = File::open(t)?;
                let tokenizer_config_json: TokenizerConfig = serde_json::from_reader(&mut file)?;

                if let Some(template) = tokenizer_config_json.chat_template {
                    tracing::info!("Loaded Template from tokenizer_config file");
                    Some(template)
                } else {
                    tracing::info!("The tokenizer_config file does not provide a chat template");
                    None
                }
            } else if let Some(t) = template_file {
                tracing::info!("Loading extra provided template file");
                Some(std::fs::read_to_string(t)?)
            } else {
                tracing::info!("No extra template file has been provided");
                None
            }
        };

        self.template_proc = if tokenizer_config_file.is_some() && self.template.is_some() {
            // the extra check for the `tokenizer_config_file` is to indicate the presence of a jinja template
            Some(TemplateProcessor::with_jinja_template())
        } else if template_file.is_some() {
            Some(TemplateProcessor::with_go_template())
        } else {
            None
        };

        // Initialize the tokenizer
        self.tokenizer = Some({
            let mut tokenizer = Tokenizer::from_file(&config.tokenizer_file.as_ref().ok_or(
                Error::MissingConfigLLM("Tokenizer config is missing".to_owned()),
            )?)
            .map_err(|e| {
                Error::LoadingFile(format!("{:?}", config.tokenizer_file), e.to_string())
            })?;

            if let Some(t) = tokenizer_config_file {
                let mut file = File::open(t)?;
                let tokenizer_config_json: TokenizerConfig = serde_json::from_reader(&mut file)?;

                if let Some(added_tokens) = tokenizer_config_json.added_tokens_decoder {
                    let added_tokens: Vec<AddedToken> = added_tokens.into_values().collect();

                    tokenizer.add_special_tokens(&added_tokens);
                }
            }

            tokenizer
        });

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
                    // TODO: this can also be set via config
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
                crate::SamplingConfig::ArgMax => Sampling::ArgMax,
                crate::SamplingConfig::All => Sampling::All {
                    temperature: self.temperature,
                },
                crate::SamplingConfig::TopK => Sampling::TopK {
                    k: self.top_k,
                    temperature: self.temperature,
                },
                crate::SamplingConfig::TopP => Sampling::TopP {
                    p: self.top_p,
                    temperature: self.temperature,
                },
                crate::SamplingConfig::TopKThenTopP => Sampling::TopKThenTopP {
                    k: self.top_k,
                    p: self.top_p,
                    temperature: self.temperature,
                },
                crate::SamplingConfig::GumbelSoftmax => Sampling::GumbelSoftmax {
                    temperature: self.temperature,
                },
            };

            let seed = match seed {
                crate::GenerationSeed::Fixed(inner) => inner as u64,
                crate::GenerationSeed::Random => {
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

    fn execute_streaming(
        &mut self,
        q: crate::Query,
        response_tx: std::sync::Arc<std::sync::mpsc::Sender<crate::Query>>,
    ) -> anyhow::Result<(), crate::Error> {
        self.inference(q, response_tx.clone())?;

        // send termination
        response_tx
            .send(crate::Query::End)
            .map_err(|e| crate::Error::StreamError(e.to_string()))?;

        Ok(())
    }

    fn inference(
        &mut self,
        message: Query,
        response_tx: std::sync::Arc<std::sync::mpsc::Sender<Query>>,
    ) -> anyhow::Result<(), Error> {
        if let Query::Prompt {
            messages: _,
            tools,
            config,
            chunk_size,
            timestamp: _,
        } = message.clone()
        {
            let chunk_size = chunk_size.unwrap_or(self.default_chunksize());

            let is_chunk_available =
                |num_tokens: usize, chunk_size: usize| -> bool { (num_tokens % chunk_size) == 0 };

            // preprocess message by applying chat template
            let message = {
                let template = self.template.as_ref().ok_or(Error::ExecutionError(format!(
                    "Template is missing in config!"
                )))?;
                let proc = self
                    .template_proc
                    .as_ref()
                    .ok_or(Error::ExecutionError(format!(
                        "Template processor is not intialized"
                    )))?;
                message.apply_template(&template, proc)?
            };

            let QueryConfig {
                generate_num_samples,
            } = config.unwrap();

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
            let eos_token = *tokenizer.get_vocab(true).get("<|eot_id|>").unwrap();

            tracing::info!("Encoded eos token: {eos_token}");

            let mut message_id = 0;

            // Start sampling
            for index in 0..generate_num_samples {
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

                // 128 = last n repeated tokens (this is just configuration)
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

                if is_chunk_available(all_tokens.len(), chunk_size) {
                    let data = match tokenizer.decode(&all_tokens, true) {
                        Ok(str) => str.as_bytes().to_vec(),
                        Err(e) => return Err(Error::ExecutionError(e.to_string())),
                    };
                    message_id += 1;
                    let id = message_id;

                    if let Err(e) = response_tx.send(crate::Query::Chunk {
                        id,
                        kind: crate::QueryChunkType::String,
                        data,
                    }) {
                        return Err(Error::StreamError(e.to_string()));
                    }
                }

                if next_token == eos_token {
                    break;
                }
            }

            // send last message.
            {
                let data = match tokenizer.decode(&all_tokens, true) {
                    Ok(str) => str.as_bytes().to_vec(),
                    Err(e) => return Err(Error::ExecutionError(e.to_string())),
                };
                message_id += 1;
                let id = message_id;

                if let Err(e) = response_tx.send(crate::Query::Chunk {
                    id,
                    kind: crate::QueryChunkType::String,
                    data,
                }) {
                    return Err(Error::StreamError(e.to_string()));
                }
            }
        }

        Err(Error::ExecutionError(
            "Cannot handle Query type".to_string(),
        ))
    }
}
