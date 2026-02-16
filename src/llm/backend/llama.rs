use std::fs::File;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{self as llama_model, Llama, LlamaConfig};

use crate::error::Error;
use crate::llm::tool_call::{LlamaToolCallParser, ToolCallParser};

use super::{extract_last_token_logits, ModelBackend};

pub struct LlamaBackend {
    model: Llama,
    cache: llama_model::Cache,
    config: llama_model::Config,
    device: Device,
    tool_call_parser: LlamaToolCallParser,
}

impl LlamaBackend {
    /// Load Llama weights from sharded safetensors files.
    pub fn from_safetensors(
        vb: VarBuilder,
        model_config_file: &PathBuf,
        device: &Device,
    ) -> Result<Self, Error> {
        let mut config_file = File::open(model_config_file)?;
        let llama_config: LlamaConfig = serde_json::from_reader(&mut config_file)?;
        let config = llama_config.into_config(false);

        let model = Llama::load(vb, &config).map_err(|e| Error::ExecutionError(e.to_string()))?;
        let cache = llama_model::Cache::new(true, candle_core::DType::BF16, &config, device)
            .map_err(|e| Error::ExecutionError(e.to_string()))?;

        Ok(Self {
            model,
            cache,
            config,
            device: device.clone(),
            tool_call_parser: LlamaToolCallParser,
        })
    }
}

impl ModelBackend for LlamaBackend {
    fn forward(&mut self, input: &Tensor, index: usize) -> Result<Tensor, Error> {
        let logits = self
            .model
            .forward(input, index, &mut self.cache)
            .map_err(|e| Error::ExecutionError(e.to_string()))?;
        extract_last_token_logits(logits)
    }

    fn clear_kv_cache(&mut self) {
        self.cache =
            llama_model::Cache::new(true, candle_core::DType::BF16, &self.config, &self.device)
                .expect("Failed to recreate cache");
    }

    fn tool_call_parser(&self) -> Option<&dyn ToolCallParser> {
        Some(&self.tool_call_parser)
    }
}
