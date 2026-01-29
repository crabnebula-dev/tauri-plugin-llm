use std::fs::File;
use std::sync::Arc;

use crate::error::Error;
use crate::runtime::{LLMRuntimeModel, Query};
use crate::{LLMRuntimeConfig, ModelConfig, QueryConfig, TemplateProcessor, TokenizerConfig};
use candle_core::Device;
use candle_core::{quantized::gguf_file, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use rand::Rng;
use tokenizers::Tokenizer;

pub struct Qwen3Model {
    pub(crate) _streaming: bool,
    pub(crate) device: Option<Device>,
    pub(crate) tokenizer: Option<Tokenizer>,
    pub(crate) top_k: usize,
    pub(crate) top_p: f64,
    pub(crate) temperature: f64,
    pub(crate) _thinking: bool,
    pub(crate) weights: Option<Qwen3>,
    pub(crate) logits_processor: Option<LogitsProcessor>,

    pub(crate) template: Option<String>,
    pub(crate) template_proc: Option<TemplateProcessor>,
}

impl LLMRuntimeModel for Qwen3Model {
    fn init(&mut self, config: &LLMRuntimeConfig) -> Result<(), crate::Error> {
        let LLMRuntimeConfig {
            tokenizer_file: _,
            tokenizer_config_file,
            model_config_file: _,
            model_index_file: _,
            model_file: _,
            model_dir: _,
            model_config,
            verbose: _,
            template_file,
        } = config;

        let ModelConfig {
            seed,
            sampling_config,
            ..
        } = model_config.clone();

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
            Some(TemplateProcessor::with_jinja_template())
        } else if template_file.is_some() {
            Some(TemplateProcessor::with_go_template())
        } else {
            None
        };

        // Initialize the tokenizer
        self.tokenizer =
            Some(
                Tokenizer::from_file(config.tokenizer_file.as_ref().ok_or(
                    Error::MissingConfigLLM("Tokenizer config is missing".to_owned()),
                )?)
                .map_err(|e| {
                    Error::LoadingFile(format!("{:?}", config.tokenizer_file), e.to_string())
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

    fn execute(
        &mut self,
        q: Query,
        response_tx: Arc<std::sync::mpsc::Sender<Query>>,
    ) -> anyhow::Result<(), Error> {
        self.inference(q, response_tx.clone())?;

        tracing::debug!("Qwen3 inference ended. Sending end termination");
        response_tx
            .send(crate::Query::End)
            .map_err(|e| crate::Error::StreamError(e.to_string()))?;

        Ok(())
    }

    /// This executes inference in a streaming fashion.
    ///
    /// A [`Sender`] is provided to transmit [`StreamQuery::Chunk`]s over the channel.
    fn inference(
        &mut self,
        message: Query,
        response_tx: Arc<std::sync::mpsc::Sender<crate::Query>>,
    ) -> anyhow::Result<(), Error> {
        tracing::debug!("Qwen3 got message: {:?}", message);

        if let Query::Prompt {
            messages,
            tools: _,
            config,
            chunk_size,
            timestamp,
        } = message.clone()
        {
            let chunk_size = chunk_size.unwrap_or(self.default_chunksize());

            let is_chunk_available =
                |num_tokens: usize, chunk_size: usize| -> bool { (num_tokens % chunk_size) == 0 };

            // preprocess message by applying chat template
            let message = {
                match self.template.as_ref().ok_or(Error::ExecutionError(
                    "Template is missing in config!".to_string(),
                )) {
                    Ok(template) => {
                        let proc = self.template_proc.as_ref().ok_or(Error::ExecutionError(
                            "Template processor is not intialized".to_string(),
                        ))?;
                        message.apply_template(template, proc)?
                    }
                    Err(_) => {
                        tracing::warn!("No templates have been found. Sending plain query");

                        // FIXME: we don't want the plain prompt to be send to the model. Not loading the
                        // template from the plugin config indicates a deeper problem.
                        messages[0].content.clone()
                    }
                }
            };

            let QueryConfig {
                generate_num_samples,
            } = config.unwrap_or(QueryConfig {
                generate_num_samples: 5000,
            });

            tracing::debug!("Processing Message: {:?}", message);

            // get defaults
            let tokenizer = self.tokenizer.as_ref().unwrap();
            let model = self.weights.as_mut().unwrap();
            let logits_processor = self.logits_processor.as_mut().unwrap();
            let device = self.device.as_ref().unwrap();

            tracing::debug!("Encoding Message: {:?}", message);
            // encode message
            let tokens = tokenizer
                .encode(message, true)
                .map_err(|e| Error::MessageEncodingError(e.to_string()))?;

            let tokens = tokens.get_ids();

            // clear cache to have a reset?
            model.clear_kv_cache();

            // set next token
            let mut next_token = {
                let input = Tensor::new(tokens, device)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                // inference is here (successive calls fail)
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

            let mut message_id = 0;

            let mut window_index = 0;

            // Start sampling
            for index in 0..generate_num_samples {
                let input = Tensor::new(&[next_token], device)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| Error::ExecutionError(e.to_string()))?;

                // inference is here
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

                
                if is_chunk_available(all_tokens[window_index..].len(), chunk_size) {
                    let data = match tokenizer.decode(&all_tokens[window_index..], true) {
                        Ok(str) => str.as_bytes().to_vec(),
                        Err(e) => return Err(Error::ExecutionError(e.to_string())),
                    };

                    window_index += chunk_size;
                    message_id += 1;
                    let id = message_id;

                    if let Err(e) = response_tx.send(crate::Query::Chunk {
                        id,
                        kind: crate::QueryChunkType::String,
                        data,
                        timestamp,
                    }) {
                        tracing::error!("Error sending chunk: {e}");
                        return Err(Error::StreamError(e.to_string()));
                    }
                }

                if next_token == eos_token {
                    break;
                }
            }

            {
                let data = match tokenizer.decode(&all_tokens[window_index..], true) {
                    Ok(str) => str.as_bytes().to_vec(),
                    Err(e) => return Err(Error::ExecutionError(e.to_string())),
                };
                message_id += 1;
                let id = message_id;

                if let Err(e) = response_tx.send(crate::Query::Chunk {
                    id,
                    kind: crate::QueryChunkType::String,
                    data,
                    timestamp,
                }) {
                    return Err(Error::StreamError(e.to_string()));
                }
            }

            tracing::debug!("Finished inference");

            return Ok(());
        }

        tracing::warn!(
            "Got unhandled `Query` type: discriminant({:?}), actual type: ({:?})",
            std::mem::discriminant(&message),
            message
        );

        // Err(Error::ExecutionError(
        //     "Cannot handle Query type".to_string(),
        // ))

        Ok(())
    }
}
