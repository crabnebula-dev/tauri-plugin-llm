use std::fs::File;
use std::path::PathBuf;

use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{self as qwen3_model, Config as Qwen3Config};

use crate::error::Error;
use crate::llm::tool_call::{Qwen3ToolCallParser, ToolCallParser};

use super::{extract_last_token_logits, ModelBackend};

pub struct Qwen3Backend {
    model: qwen3_model::ModelForCausalLM,
    tool_call_parser: Qwen3ToolCallParser,
}

impl Qwen3Backend {
    /// Load Qwen3 weights from sharded safetensors files.
    pub fn from_safetensors(vb: VarBuilder, model_config_file: &PathBuf) -> Result<Self, Error> {
        let mut config_file = File::open(model_config_file)?;
        let qwen3_config: Qwen3Config = serde_json::from_reader(&mut config_file)?;

        let model = qwen3_model::ModelForCausalLM::new(&qwen3_config, vb)
            .map_err(|e| Error::ExecutionError(e.to_string()))?;

        Ok(Self {
            model,
            tool_call_parser: Qwen3ToolCallParser,
        })
    }
}

impl ModelBackend for Qwen3Backend {
    fn forward(&mut self, input: &Tensor, index: usize) -> Result<Tensor, Error> {
        let logits = self
            .model
            .forward(input, index)
            .map_err(|e| Error::ExecutionError(e.to_string()))?;
        extract_last_token_logits(logits)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn tool_call_parser(&self) -> Option<&dyn ToolCallParser> {
        Some(&self.tool_call_parser)
    }
}
