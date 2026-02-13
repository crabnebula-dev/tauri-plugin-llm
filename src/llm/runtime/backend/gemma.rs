use std::fs::File;
use std::path::PathBuf;

use candle_core::Tensor;
use candle_nn::{Activation, VarBuilder};

use candle_transformers::models::gemma3::{self as gemma3_model, Config as Gemma3Config};
use serde_json::{Number, Value};
// use candle_transformers::models::qwen3::{self as qwen3_model, Config as Qwen3Config};

use crate::error::Error;
use crate::runtime::backend;
use crate::runtime::tool_call::{GemmaToolCallParser, Qwen3ToolCallParser, ToolCallParser};

use super::{extract_last_token_logits, ModelBackend};

pub struct Gemma3Backend {
    model: gemma3_model::Model,
    tool_call_parser: GemmaToolCallParser,
}

impl Gemma3Backend {
    /// Load Qwen3 weights from sharded safetensors files.
    pub fn from_safetensors(
        vb: VarBuilder,
        model_config_file: &PathBuf,
        use_flash: bool,
    ) -> Result<Self, Error> {
        let mut config_file = File::open(model_config_file)?;
        tracing::debug!("Deserialize Gemma3Config: {model_config_file:?}");
        let mut json_value: Value = serde_json::from_reader(&mut config_file)?;

        if let Some(obj) = json_value.as_object_mut() {}

        let gemma3_config: Gemma3Config = match serde_json::from_value(json_value) {
            Ok(inner) => inner,
            Err(err) => {
                tracing::error!("Error reading {model_config_file:?} : {err:?}");
                return Err(err.into());
            }
        };

        let model = gemma3_model::Model::new(use_flash, &gemma3_config, vb)
            .map_err(|e| Error::ExecutionError(e.to_string()))?;

        Ok(Self {
            model,
            tool_call_parser: GemmaToolCallParser,
        })
    }
}

impl ModelBackend for Gemma3Backend {
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
