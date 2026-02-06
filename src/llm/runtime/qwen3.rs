use std::fs::File;
use std::sync::Arc;

use crate::error::Error;
use crate::runtime::{LLMRuntimeModel, Query};
use crate::{
    LLMRuntimeConfig, ModelConfig, TemplateProcessor, TokenUsage, TokenizerConfig,
};
use candle_core::Device;
use candle_core::{quantized::gguf_file, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use rand::Rng;
use tokenizers::Tokenizer;

use crate::{GenerationSeed, SamplingConfig};

pub struct Qwen3Model {
    pub(crate) device: Option<Device>,
    pub(crate) tokenizer: Option<Tokenizer>,
    pub(crate) weights: Option<Qwen3>,
    pub(crate) template: Option<String>,
    pub(crate) template_proc: Option<TemplateProcessor>,
    pub(crate) seed: GenerationSeed,
    pub(crate) sampling_config: SamplingConfig,
}

impl Qwen3Model {
    /// Creates a LogitsProcessor based on runtime parameters from Query::Prompt
    fn create_logits_processor(
        &self,
        temperature: Option<f32>,
        top_k: Option<f32>,
        top_p: Option<f32>,
    ) -> LogitsProcessor {
        // Use provided values or defaults
        let temperature = temperature.map(|t| t as f64).unwrap_or(0.7);
        let top_k = top_k.map(|k| k as usize).unwrap_or(40);
        let top_p = top_p.map(|p| p as f64).unwrap_or(0.9);

        let sampling = match &self.sampling_config {
            SamplingConfig::ArgMax => Sampling::ArgMax,
            SamplingConfig::All => Sampling::All { temperature },
            SamplingConfig::TopK => Sampling::TopK { k: top_k, temperature },
            SamplingConfig::TopP => Sampling::TopP { p: top_p, temperature },
            SamplingConfig::TopKThenTopP => Sampling::TopKThenTopP { k: top_k, p: top_p, temperature },
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

        self.seed = seed;
        self.sampling_config = sampling_config;

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

        Ok(())
    }

    fn execute(
        &mut self,
        q: Query,
        response_tx: Arc<std::sync::mpsc::Sender<Query>>,
    ) -> anyhow::Result<(), Error> {
        let usage = self.inference(q, response_tx.clone())?;

        tracing::debug!("Qwen3 inference ended. Sending end termination");
        response_tx
            .send(crate::Query::End { usage })
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
    ) -> anyhow::Result<Option<TokenUsage>, Error> {
        tracing::debug!("Qwen3 got message: {:?}", message);

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

            let generate_num_samples = max_tokens.unwrap_or(5000);

            tracing::debug!("Processing Message: {:?}", message);

            // get defaults
            let tokenizer = self.tokenizer.as_ref().unwrap();
            let model = self.weights.as_mut().unwrap();
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

            let mut chunks = Vec::with_capacity(chunk_size);
            chunks.push(next_token);

            // set end of stream token
            let eos_token = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();
            let mut id = 0;

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
                chunks.push(next_token);

                if chunks.len().eq(&chunk_size) {
                    let data = match tokenizer.decode(&chunks, true) {
                        Ok(str) => str.as_bytes().to_vec(),
                        Err(e) => return Err(Error::ExecutionError(e.to_string())),
                    };

                    id += 1;

                    tracing::debug!("Sending Chunk");

                    if let Err(e) = response_tx.send(crate::Query::Chunk {
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

            if !chunks.is_empty() {
                let data = match tokenizer.decode(&chunks, true) {
                    Ok(str) => str.as_bytes().to_vec(),
                    Err(e) => return Err(Error::ExecutionError(e.to_string())),
                };

                if let Err(e) = response_tx.send(crate::Query::Chunk {
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
