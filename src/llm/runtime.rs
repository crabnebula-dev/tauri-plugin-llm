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
    config: LLMRuntimeConfig,
}

pub trait LLMRuntimeModel: Send + Sync {
    /// Sends a [`LlmMessage`] to the loaded model
    fn execute(&self, message: LlmMessage) -> Result<LlmMessage, Error>;

    /// Initializes the model
    ///
    /// This is a heavy process and needs to be run in a dedicated thread
    fn init(&mut self, config: &LLMRuntimeConfig) -> Result<(), Error>;
}

#[derive(Debug)]
pub enum LlmMessage {
    Prompt { system: String, message: String },
    Response { error: String, message: String },
    Exit,
}

/// Enable streaming model responses

impl LLMRuntime {
    /// Creates a new LLM
    pub fn from_config(config: LLMRuntimeConfig) -> Result<Self, Error> {
        if config.verbose {
            let verbose = tracing_subscriber::fmt::layer().with_filter(filter::LevelFilter::DEBUG);
            Registry::default().with(verbose).init();
        }
        let device = Self::load_default_device();
        let model = Self::detect_model(&config, device)?;

        Ok(Self {
            model: Some(Box::new(model)),
            config,
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
                weights: None,
            }),
            _ => Err(Error::ExecutionError),
        }
    }

    /// Loads the best default device that can be detected
    fn load_default_device() -> Device {
        if cfg!(target_os = "macos") {
            match Device::new_metal(0) {
                Ok(device) => {
                    tracing::debug!("Select Metal Device (0)");
                    device
                }
                Err(error) => {
                    tracing::error!("Could not detect Metal device. Fall back to CPU: {}", error);
                    Device::Cpu
                }
            }
        } else if cfg!(not(target_os = "macos")) {
            match Device::new_cuda(0) {
                Ok(device) => {
                    tracing::debug!("Select Cuda Device (0)");
                    device
                }
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
        response: Sender<LlmMessage>,
    ) -> Result<(Sender<LlmMessage>, JoinHandle<()>), Error> {
        let (tx, rx): (Sender<LlmMessage>, Receiver<LlmMessage>) = std::sync::mpsc::channel();

        let mut model = self.model.take().unwrap();
        let config = self.config.clone();

        tracing::debug!("Spawing Model in separate thread");

        let worker = tokio::task::spawn_blocking(move || {
            tracing::debug!("Initializing Model");

            if let Err(error) = model.init(&config) {
                tracing::error!("Error initializing model: {}", error);

                // exit, because model failed to initialize
                return;
            }

            loop {
                if let Ok(message) = rx.recv() {
                    tracing::debug!("Sending message to model");

                    let model_response_message = match message {
                        LlmMessage::Prompt { .. } => model.execute(message),
                        LlmMessage::Exit => break,
                        LlmMessage::Response { .. } => Err(Error::UnexpectedMessage),
                    };

                    match model_response_message {
                        Ok(message) => {
                            if let Err(error) = response.send(message) {
                                tracing::error!("Error sending model response: {}", error)
                            }
                        }
                        Err(error) => {
                            tracing::error!("Message error: {}", error)
                        }
                    }
                }
            }
        });
        Ok((tx, worker))
    }
}
