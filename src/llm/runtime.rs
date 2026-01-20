//! LLM Inference
mod llama3;
mod mock;
mod qwen3;

use crate::error::Error;
use crate::runtime::llama3::LLama3Model;
use crate::runtime::mock::Mock;
use crate::Query;
use crate::{runtime::qwen3::Qwen3Model, LLMRuntimeConfig, ModelConfig};
use anyhow::Result;
use candle_core::Device;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer, Registry};

pub struct LLMRuntime {
    model: Option<Box<dyn LLMRuntimeModel>>,
    config: LLMRuntimeConfig,

    worker: Option<tauri::async_runtime::JoinHandle<()>>,
    control: (Sender<Query>, Option<Receiver<Query>>),
    response: (Arc<Sender<Query>>, Receiver<Query>),
}

pub trait LLMRuntimeModel: Send + Sync {
    /// Initializes the model
    ///
    /// This is a heavy process and needs to be run in a dedicated thread
    fn init(&mut self, config: &LLMRuntimeConfig) -> Result<(), Error>;

    /// Sends a [`Query`] to the loaded model and accepts a response sender to send chunked messages
    ///
    /// This message addtionally handles sending the termination of the chunk stream
    fn execute(&mut self, _: Query, _: Arc<Sender<Query>>) -> Result<(), Error>;

    /// Sends a [`Query`] to the loaded model and accepts a response sender to send chunked messages
    fn inference(
        &mut self,
        q: crate::Query,
        response_tx: Arc<std::sync::mpsc::Sender<crate::Query>>,
    ) -> Result<(), crate::Error>;

    /// Returns an arbitrary default chunk size.
    ///
    /// The actual chunk size can be configured inside a [`Query`]
    fn default_chunksize(&self) -> usize {
        32
    }
}

impl Drop for LLMRuntime {
    fn drop(&mut self) {
        if let Err(err) = self.control.0.send(Query::Exit) {
            tracing::error!("Unable to send Exit signal to LLMRuntime: {}", err);
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

        let (response_stream_tx, response_stream_rx) = std::sync::mpsc::channel();

        Ok(Self {
            model: Some(model),
            config,

            worker: None,
            control: (ctrl_tx, Some(ctrl_rx)),
            response: (Arc::new(response_stream_tx), response_stream_rx),
        })
    }

    fn detect_model(
        config: &LLMRuntimeConfig,
        device: Device,
    ) -> Result<Box<dyn LLMRuntimeModel>, Error> {
        let LLMRuntimeConfig { model_config, .. } = config.clone();

        let ModelConfig {
            top_k,
            top_p,
            temperature,
            name,
            thinking,
            streaming,
            penalty,
            ..
        } = model_config;

        // TODO:
        // - move initializers to individual constructor functions
        match &name {
            _ if name.contains("Qwen3") => Ok(Box::new(Qwen3Model {
                _streaming: streaming,
                device: Some(device),
                tokenizer: None,
                top_k,
                top_p,
                temperature,
                _thinking: thinking,
                weights: None,
                logits_processor: None,
                template: None,
                template_proc: None,
            })),
            _ if name.contains("Mock") => Ok(Box::new(Mock)),
            _ if name.contains("Llama") => Ok(Box::new(LLama3Model {
                _streaming: streaming,
                device: Some(device),
                tokenizer: None,
                top_k,
                top_p,
                temperature,
                _thinking: thinking,
                weights: None,
                logits_processor: None,
                cache: None,
                penalty,
                template: None,
                template_proc: None,
            })),
            _ => Err(Error::ExecutionError(format!("Unknown Model Name: {name}"))),
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

    pub fn shutdown(&self) {
        self.control
            .0
            .send(Query::Exit)
            .expect("Error sending exit message")
    }

    #[cfg(feature = "mcpurify")]
    pub async fn setup_mcpurify(config: &mcpurify::Config) {
        use mcpurify::converter::Converter;
        use mcpurify::filter::FilterServer;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let mcpurify::Config {
            aiprovider,
            repl,
            timeout,
            telemetry,
            proxy,
        } = config.clone();

        let (mut filter_server, request_tx, response_rx) = FilterServer::new(None);

        // load filterserver
        filter_server
            .insert_all_filters(&config.aiprovider)
            .expect("Add all AI-Providers");

        // split execution of the FilterServer
        tokio::spawn(async move { filter_server.process().await });

        let proxy = proxy.unwrap();

        let response_converter = proxy.converter.clone();

        // This operation is blocking and lets the service
        // proxy run until it is terminated manually.
        proxy
            .start(
                request_tx,
                Arc::new(Mutex::new(response_rx)),
                move |body_str| {
                    // impl of the function requires calling a converter type,
                    // to effectively convert a model response which may potentially
                    // contain a tool call into a filter request. the input will always
                    // be a simple utf8 string.

                    response_converter.convert(body_str.to_owned())
                },
            )
            .await;
    }
}

/// Streaming impl
impl LLMRuntime {
    /// Send a message to the llm backend and receive a stream of messages
    pub fn run_stream(&mut self) -> Result<(), Error> {
        let mut model = self.model.take().unwrap();
        let config = self.config.clone();

        let control_rx = self.control.1.take().unwrap();

        let response_tx = self.response.0.clone();

        tracing::debug!("Spawning Model in separate thread");

        let worker = tauri::async_runtime::spawn_blocking(move || {
            tracing::debug!("Initializing Model");

            if let Err(error) = model.init(&config) {
                tracing::error!("Error initializing model: {}", error);

                return;
            }

            loop {
                // TODO blocking wait for message
                match control_rx.recv() {
                    Ok(message) => {
                        tracing::debug!("Sending message to model");

                        match message {
                            Query::Prompt { .. } => {
                                if let Err(error) = model.execute(message, response_tx.clone()) {
                                    tracing::error!("Error execute streaming: {error}");
                                }
                            }
                            Query::Exit => break,
                            _ => {}
                        }
                    }
                    Err(error) => {
                        tracing::error!("Error! Couldn't receive control message: {error}. Waiting for next message");
                    }
                }
            }
        });

        self.worker = Some(worker);

        Ok(())
    }

    pub fn send_stream(&self, msg: Query) -> Result<(), Error> {
        self.control
            .0
            .send(msg)
            .map_err(|e| Error::ExecutionError(e.to_string()))?;

        Ok(())
    }

    pub fn recv_stream(&self) -> Result<Query, Error> {
        self.response
            .1
            .recv()
            .map_err(|e| Error::StreamError(e.to_string()))
    }
}
