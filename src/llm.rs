//! LLM
//!
//! This module provides access to load and run inference on most common models and
//! their available formats. For now the LLM loader supports `*.safetensors`  files
//! and text generation models.

pub mod llmconfig;
pub mod loaders;
pub mod runtime;
