//! Model backend abstraction.
//!
//! Each supported model family implements [`ModelBackend`] to encapsulate
//! model-specific forward pass, KV cache management, and weight loading.

pub mod gemma;
pub mod llama;
pub mod qwen3;

use std::path::PathBuf;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::error::Error;
use crate::loaders::IndexFile;

use super::tool_call::ToolCallParser;

/// Abstraction over different model weight backends.
///
/// Implementors encapsulate model weights, forward pass logic,
/// and KV cache management. Each backend also provides an optional
/// [`ToolCallParser`] for model-specific tool call detection.
pub trait ModelBackend: Send + Sync {
    /// Run a forward pass through the model.
    ///
    /// `input` is the token tensor [batch, seq_len].
    /// `index` is the position index for KV cache offset.
    /// Returns logits for the last token, squeezed to [vocab_size].
    fn forward(&mut self, input: &Tensor, index: usize) -> Result<Tensor, Error>;

    /// Clear the KV cache for a fresh generation.
    fn clear_kv_cache(&mut self);

    /// Returns the tool call parser for this model, if tool calling is supported.
    fn tool_call_parser(&self) -> Option<&dyn ToolCallParser>;
}

/// Extracts the last token's logits from model output.
///
/// Handles both 3D `[batch, seq_len, vocab]` and 2D `[batch, vocab]` outputs.
/// This is model-agnostic and shared by all backends.
pub fn extract_last_token_logits(logits: Tensor) -> Result<Tensor, Error> {
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

/// Creates the appropriate backend based on model name and config.
///
/// Dispatches to the correct backend implementation by matching on `model_name`.
/// Falls back to `LlamaBackend` for unknown models.
pub fn create_backend_by_model_index_file(
    model_name: &str,
    device: &Device,
    model_index_file: &PathBuf,
    model_dir: &PathBuf,
    model_config_file: &PathBuf,
) -> Result<Box<dyn ModelBackend>, Error> {
    let mut index_file = IndexFile::from_path(model_index_file)?;
    let paths = index_file.files(model_dir);

    if model_name.contains("Qwen") {
        tracing::info!("Loading Qwen3 safetensors model");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::BF16, device)
                .map_err(|e| Error::ExecutionError(e.to_string()))?
        };
        return Ok(Box::new(qwen3::Qwen3Backend::from_safetensors(
            vb,
            model_config_file,
        )?));
    } else if model_name.contains("Llama") {
        tracing::info!("Loading Llama safetensors model");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::BF16, device)
                .map_err(|e| Error::ExecutionError(e.to_string()))?
        };
        return Ok(Box::new(llama::LlamaBackend::from_safetensors(
            vb,
            model_config_file,
            device,
        )?));
    } else if model_name.contains("gemma") {
        tracing::info!("Loading Gemma safetensors model");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::BF16, device)
                .map_err(|e| Error::ExecutionError(e.to_string()))?
        };
        return Ok(Box::new(gemma::Gemma3Backend::from_safetensors(
            vb,
            model_config_file,
            false,
        )?));
    }

    Err(Error::UnsupportedModelType(model_name.to_string()))
}

/// Creates the appropriate backend based on model name and a `model.safetensors` file.
///
/// Dispatches to the correct backend implementation by matching on `model_name`.
/// Falls back to `LlamaBackend` for unknown models.
pub fn create_backend_by_model_file(
    model_name: &str,
    device: &Device,
    model_file: &PathBuf,
    model_config_file: &PathBuf,
) -> Result<Box<dyn ModelBackend>, Error> {
    // let mut index_file = IndexFile::from_path(model_index_file)?;
    // let paths = index_file.files(model_dir);

    // we handle gemma only for now
    if model_name.contains("gemma") {
        tracing::info!("Loading Gemma safetensors model");

        // load from single safetensor file
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], candle_core::DType::BF16, device)
                .map_err(|e| Error::ExecutionError(e.to_string()))?
        };
        return Ok(Box::new(gemma::Gemma3Backend::from_safetensors(
            vb,
            model_config_file,
            false,
        )?));
    }

    Err(Error::UnsupportedModelType(model_name.to_string()))
}
