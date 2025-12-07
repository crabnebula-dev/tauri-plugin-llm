//! LLM Inference
mod qwen3;

use crate::error::Error;
use crate::{llm::llmconfig::LLMRuntimeConfig, llmconfig::ModelConfig, runtime::qwen3::Qwen3Model};
use candle_core::Device;
use serde::Deserialize;
use std::sync::mpsc::{Receiver, Sender};
use tokenizers::Tokenizer;

use anyhow::{Error as E, Result};
use candle_core::{quantized::gguf_file, DType, Tensor};
use candle_nn::VarBuilder;

/// This needs to be adapted
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_qwen3::ModelWeights as Qwen3,
};
use std::future::Future;
use std::{fs::File, io::Write, path::PathBuf};
use tokio::task::JoinHandle;
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer, Registry};

const CHANNEL_BUFFER_SIZE: usize = 10;

pub struct LLMRuntime {
    model: Option<Box<dyn LLMRuntimeModel>>,
}

pub trait LLMRuntimeModel: Send + Sync {
    fn execute(&self, message: &LLM_Message);
}

#[derive(Debug)]
pub enum LLM_Message {
    Prompt { system: String, message: String },
    Exit,
}

/// Enable streaming model responses

impl LLMRuntime {
    /// Creates a new LLM
    pub fn from_config(config: &LLMRuntimeConfig) -> Result<Self, Error> {
        let device = Self::load_default_device();
        let model = Self::detect_model(&config, device)?;

        // logging
        let verbose = tracing_subscriber::fmt::layer().with_filter(filter::LevelFilter::DEBUG);
        Registry::default().with(verbose).init();

        Ok(Self {
            model: Some(Box::new(model)),
        })
    }

    fn detect_model(
        config: &LLMRuntimeConfig,
        device: Device,
    ) -> Result<impl LLMRuntimeModel, Error> {
        let LLMRuntimeConfig { model_config, .. } = config.clone();

        let ModelConfig {
            top_k,
            top_p,
            temperature,
            name,
            thinking,
            streaming,
            ..
        } = model_config;

        match &name {
            _ if name.contains("Qwen3") => Ok(Qwen3Model {
                streaming,
                device: Some(device),
                tokenizer: None,
                top_k,
                top_p,
                temperature,
                thinking,
            }),
            _ => Err(Error::ExecutionError),
        }
    }

    /// Loads the best default device that can be detected
    fn load_default_device() -> Device {
        if cfg!(target_os = "macos") {
            match Device::new_metal(0) {
                Ok(device) => device,
                Err(error) => {
                    tracing::error!("Could not detect Metal device. Fall back to CPU: {}", error);
                    Device::Cpu
                }
            }
        } else if cfg!(not(target_os = "macos")) {
            match Device::new_cuda(0) {
                Ok(device) => device,
                Err(error) => {
                    tracing::error!("Could not detect Cuda device. Fall back to CPU: {}", error);
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        }
    }

    /// ## Description
    ///
    /// Executes the LLM and returns a [`Sender`] to interact with the Model.
    ///
    /// ## Parameters
    /// - `response` Provide a [`Sender`] where the Model response should be send
    pub async fn run(
        &mut self,
        response: Sender<LLM_Message>,
    ) -> Result<(Sender<LLM_Message>, JoinHandle<()>), Error> {
        let (tx, rx): (Sender<LLM_Message>, Receiver<LLM_Message>) = std::sync::mpsc::channel();

        let model = self.model.take().unwrap();
        tracing::debug!("Spawing Model in separate thread");

        let worker = tokio::task::spawn_blocking(move || loop {
            tracing::debug!("Awaiting prompt...");
            if let Ok(message) = rx.recv() {
                tracing::debug!("Sending message to model");

                match message {
                    LLM_Message::Prompt { .. } => model.execute(&message),
                    LLM_Message::Exit => break,
                }

                if let Err(error) = response.send(message) {
                    tracing::error!("Error sending model response: {}", error)
                }
            }
        });
        Ok((tx, worker))
    }
}
