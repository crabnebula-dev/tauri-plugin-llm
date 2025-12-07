//! LLM Inference

use crate::llm::{error::Tauri_LLM_PluginError, llmconfig::LLMRuntimeConfig};
use candle_core::Device;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{Receiver, Sender};

use anyhow::{Error as E, Result};
use candle_core::{quantized::gguf_file, DType, Tensor};
use candle_nn::VarBuilder;

/// This needs to be adapted
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use std::{fs::File, io::Write, path::PathBuf};

const CHANNEL_BUFFER_SIZE: usize = 10;

pub struct LLMRuntime<M>
where
    M: LLMRuntimeModel,
{
    /// Enable streaming model responses
    streaming: bool,

    /// The execution device
    device: Device,

    tokenizer: Option<Tokenizer>,

    top_k: usize,

    top_p: f32,

    temperature: f32,

    /// Enable thinking mode
    thinking: bool,

    /// The actual model to execute
    model: Option<M>,
}

pub struct Qwen3Model {}

pub trait LLMRuntimeModel: Send + Sync + 'static {
    fn execute<M>(&self, message: &M)
    where
        M: Send + Sync + 'static;
}

pub trait LLMMessage {}

impl<M> LLMRuntime<M>
where
    M: LLMRuntimeModel,
{
    /// Creates a new LLM
    pub fn from_config(config: LLMRuntimeConfig) -> Self {
        let device = {
            if cfg!(target_os = "macos") {
                match Device::new_metal(0) {
                    Ok(device) => device,
                    Err(error) => {
                        // log error

                        Device::Cpu
                    }
                }
            } else if cfg!(not(target_os = "macos")) {
                match Device::new_cuda(0) {
                    Ok(device) => device,
                    Err(error) => {
                        // log error
                        Device::Cpu
                    }
                }
            } else {
                Device::Cpu
            }
        };

        Self {
            device,
            streaming: true,
            top_k: 20,
            top_p: 0.8,
            temperature: 0.9,
            tokenizer: None,
            thinking: false,
            model: None,
        }
    }

    /// ## Description
    ///
    /// Executes the LLM and returns a [`Sender`] to interact with the Model.
    ///
    /// ## Parameters
    /// - `response` Provide a [`Sender`] where the Model response should be send
    pub async fn run<T>(&mut self, response: Sender<T>) -> Result<Sender<T>, Tauri_LLM_PluginError>
    where
        T: Send + Sync + 'static,
    {
        let (tx, mut rx): (Sender<T>, Receiver<T>) =
            tokio::sync::mpsc::channel(CHANNEL_BUFFER_SIZE);

        let model = self.model.take().unwrap();

        tokio::task::spawn_blocking(async move || {
            loop {
                if let Some(message) = rx.recv().await {
                    // decide if message is control message or just a prompt

                    model.execute(&message);

                    if let Err(error) = response.send(message).await {
                        // log error
                    }
                }
            }
        });

        Ok(tx)
    }
}
