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
use tokenizers::Model;

use crate::error::Error;
use crate::llm::backend::qwen3::Qwen3Backend;
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

/// Flattens a JSON object by merging all nested objects one level deep into
/// the top level. Non-object values at the top level are kept as-is. Nested
/// objects are removed and their entries are promoted.
///
/// Returns an error if any key appears more than once across the top level
/// and all nested objects.
///
/// ```text
/// { "a": 1, "sub": { "b": 2, "c": 3 } }  →  { "a": 1, "b": 2, "c": 3 }
/// ```
pub fn flatten_json_config(value: serde_json::Value) -> Result<serde_json::Value, Error> {
    let obj = value
        .as_object()
        .ok_or_else(|| Error::ExecutionError("Expected a JSON object to flatten".into()))?;

    let mut result = serde_json::Map::new();

    // First pass: collect all nested object keys so we can skip them at the top level
    let nested_keys: Vec<String> = obj
        .iter()
        .filter(|(_, v)| v.is_object())
        .map(|(k, _)| k.clone())
        .collect();

    // Insert top-level non-object entries
    for (k, v) in obj {
        if nested_keys.contains(k) {
            continue;
        }
        result.insert(k.clone(), v.clone());
    }

    // Merge entries from each nested object
    for key in &nested_keys {
        let nested_obj = obj[key].as_object().unwrap();
        for (k, v) in nested_obj {
            if result.contains_key(k) {
                tracing::warn!(
                    "Duplicate key '{k}' when flattening config: \
                     already present at top level or in another sub-object, \
                     also found in '{key}'"
                );
                // return Err(Error::ExecutionError(format!(
                //     "Duplicate key '{k}' when flattening config: \
                //      already present at top level or in another sub-object, \
                //      also found in '{key}'"
                // )));
                continue;
            }
            result.insert(k.clone(), v.clone());
        }
    }

    Ok(serde_json::Value::Object(result))
}

/// Like [`flatten_json_config`], but allows duplicates when one of the sources
/// is the `priority_key` sub-object. When a key exists in both the priority
/// sub-object and another location (top-level or other sub-object), the
/// priority value wins. Duplicates between non-priority sources still error.
///
/// ```text
/// // priority_key = "text_config"
/// { "model_type": "gemma3",
///   "text_config":   { "hidden_size": 2560, "model_type": "gemma3_text" },
///   "vision_config": { "hidden_size": 1152 } }
/// →
/// { "model_type": "gemma3_text", "hidden_size": 2560 }
/// ```
pub fn flatten_json_config_with_priority(
    value: serde_json::Value,
    priority_key: &str,
) -> Result<serde_json::Value, Error> {
    let obj = value
        .as_object()
        .ok_or_else(|| Error::ExecutionError("Expected a JSON object to flatten".into()))?;

    let mut result = serde_json::Map::new();

    let nested_keys: Vec<String> = obj
        .iter()
        .filter(|(_, v)| v.is_object())
        .map(|(k, _)| k.clone())
        .collect();

    // Insert top-level non-object entries
    for (k, v) in obj {
        if nested_keys.contains(k) {
            continue;
        }
        result.insert(k.clone(), v.clone());
    }

    // Merge non-priority nested objects first
    for key in &nested_keys {
        if key == priority_key {
            continue;
        }
        let nested_obj = obj[key].as_object().unwrap();
        for (k, v) in nested_obj {
            if result.contains_key(k) {
                tracing::warn!(
                    "Duplicate key '{k}' when flattening config: \
                     already present at top level or in another sub-object, \
                     also found in '{key}'"
                );
                // return Err(Error::ExecutionError(format!(
                //     "Duplicate key '{k}' when flattening config: \
                //      already present at top level or in another sub-object, \
                //      also found in '{key}'"
                // )));
                //continue;
            }
            result.insert(k.clone(), v.clone());
        }
    }

    // Merge priority sub-object last — overwrites any existing entries
    if let Some(priority_obj) = obj.get(priority_key).and_then(|v| v.as_object()) {
        for (k, v) in priority_obj {
            if result.contains_key(k) {
                tracing::debug!(
                    "Priority key '{priority_key}' overwrites '{k}' (was {:?}, now {:?})",
                    result.get(k),
                    v
                );
            }
            result.insert(k.clone(), v.clone());
        }
    }

    Ok(serde_json::Value::Object(result))
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
    } else if model_name.contains("Llama3") {
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

pub enum Backend {
    Qwen(Box<dyn ModelBackend>),
    QwenV2(Box<dyn ModelBackend>),
    QwenV3(Box<dyn ModelBackend>),
    Llama(Box<dyn ModelBackend>),
    Llama2(Box<dyn ModelBackend>),
    Llama3(Box<dyn ModelBackend>),
    Gemma(Box<dyn ModelBackend>),
    Gemma2(Box<dyn ModelBackend>),
    Gemma3(Box<dyn ModelBackend>),
}

#[derive(Default)]
struct MockBackend;

impl ModelBackend for MockBackend {
    fn forward(&mut self, input: &Tensor, index: usize) -> Result<Tensor, Error> {
        todo!()
    }

    fn clear_kv_cache(&mut self) {
        todo!()
    }

    fn tool_call_parser(&self) -> Option<&dyn ToolCallParser> {
        todo!()
    }
}

fn unfold() {
    let mut backend = Backend::QwenV3(Box::new(MockBackend::default()));

    match backend {
        Backend::QwenV3(ref mut inner) | Backend::Gemma(ref mut inner) => inner.clear_kv_cache(),
        _ => {}
    }
}
