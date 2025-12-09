//! LLM Inference
mod qwen3;

use crate::error::Error;
use crate::LlmMessage;
use crate::{llm::llmconfig::LLMRuntimeConfig, llmconfig::ModelConfig, runtime::qwen3::Qwen3Model};
use candle_core::Device;
use serde::Deserialize;
use std::sync::mpsc::{Receiver, Sender};
use tokenizers::Tokenizer;
use tracing::trace;

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

    worker: Option<JoinHandle<()>>,
    control: (Sender<LlmMessage>, Option<Receiver<LlmMessage>>),
    response: (Option<Sender<LlmMessage>>, Receiver<LlmMessage>),
    exit: (Sender<()>, Option<Receiver<()>>),
}

pub trait LLMRuntimeModel: Send + Sync {
    /// Sends a [`LlmMessage`] to the loaded model and start sampling
    fn execute(&mut self, message: LlmMessage) -> Result<LlmMessage, Error>;

    /// Initializes the model
    ///
    /// This is a heavy process and needs to be run in a dedicated thread
    fn init(&mut self, config: &LLMRuntimeConfig) -> Result<(), Error>;

    /// Apply a chat template
    fn apply_chat_template(&mut self, template: String);
}

impl Drop for LLMRuntime {
    fn drop(&mut self) {
        if let Err(error) = self.exit.0.send(()) {
            tracing::error!("Unable to send Exit signal to LLMRuntime: {}", error);
        }
    }
}

impl LLMRuntime {
    /// Creates a new LLM
    pub fn from_config(config: LLMRuntimeConfig) -> Result<Self, Error> {
        if config.verbose {
            let verbose = tracing_subscriber::fmt::layer().with_filter(filter::LevelFilter::DEBUG);
            Registry::default().with(verbose).init();
        }
        let device = Self::load_default_device();
        let model = Self::detect_model(&config.clone(), device)?;

        let (ctrl_tx, ctrl_rx) = std::sync::mpsc::channel();
        let (response_tx, response_rx) = std::sync::mpsc::channel();
        let (exit_tx, exit_rx) = std::sync::mpsc::channel();

        Ok(Self {
            model: Some(Box::new(model)),
            config,

            worker: None,
            control: (ctrl_tx, Some(ctrl_rx)),
            response: (Some(response_tx), response_rx),
            exit: (exit_tx, Some(exit_rx)),
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
                logits_processor: None,
            }),
            _ => Err(Error::ExecutionError("".to_string())),
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
    pub async fn run(&mut self) {
        let mut model = self.model.take().unwrap();
        let config = self.config.clone();

        let control_rx = self.control.1.take().unwrap();
        let response_tx = self.response.0.take().unwrap();
        let exit_rx = self.exit.1.take().unwrap();

        tracing::debug!("Spawning Model in separate thread");

        let worker = tokio::task::spawn_blocking(move || {
            tracing::debug!("Initializing Model");

            if let Err(error) = model.init(&config) {
                tracing::error!("Error initializing model: {}", error);

                // exit, because model failed to initialize
                return;
            }

            loop {
                if let Ok(message) = control_rx.try_recv() {
                    tracing::debug!("Sending message to model");

                    let model_response_message = match message {
                        LlmMessage::Prompt { .. } => model.execute(message),
                        LlmMessage::Exit => break,
                        LlmMessage::Response { .. } => Err(Error::UnexpectedMessage),
                        LlmMessage::Status => Err(Error::UnexpectedMessage),
                    };

                    match model_response_message {
                        Ok(message) => {
                            if let Err(error) = response_tx.send(message) {
                                tracing::error!("Error sending model response: {}", error)
                            }
                        }
                        Err(error) => {
                            tracing::error!("Message error: {}", error)
                        }
                    }
                }

                if let Ok(_) = exit_rx.try_recv() {
                    break;
                }
            }
        });

        self.worker = Some(worker);
    }

    pub fn send(&self, msg: LlmMessage) -> Result<LlmMessage, Error> {
        self.control.0.send(msg).expect("Failure to send message");
        Ok(self.response.1.try_recv()?)
    }

    pub fn retry_recv(&self) -> Result<LlmMessage, Error> {
        Ok(self.response.1.try_recv()?)
    }

    pub fn shutdown(&self) {
        self.exit
            .0
            .send(())
            .expect("Error sending termination signal")
    }
}
