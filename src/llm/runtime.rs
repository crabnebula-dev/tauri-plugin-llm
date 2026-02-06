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
use std::sync::{Arc, Mutex, RwLock};
use tauri::{AppHandle, Runtime};

#[allow(clippy::type_complexity)]
pub struct LLMRuntime {
    config: LLMRuntimeConfig,

    worker: Arc<RwLock<Option<tauri::async_runtime::JoinHandle<()>>>>,
    control: (
        Arc<RwLock<Sender<Query>>>,
        Arc<Mutex<Option<Receiver<Query>>>>,
    ),
    response: (Arc<Sender<Query>>, Arc<Mutex<Receiver<Query>>>),
}

impl LLMRuntime {
    /// This
    pub fn handle_app_events<R: Runtime>(&self, _app: &AppHandle<R>) {}
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
    ) -> Result<Option<crate::TokenUsage>, crate::Error>;

    /// Returns an arbitrary default chunk size.
    ///
    /// The actual chunk size can be configured inside a [`Query`]
    fn default_chunksize(&self) -> usize {
        32
    }
}

impl Drop for LLMRuntime {
    fn drop(&mut self) {
        if let Ok(ctrl_tx) = self.control.0.read() {
            if let Err(err) = ctrl_tx.send(Query::Exit) {
                tracing::error!("Unable to send Exit signal to LLMRuntime: {}", err);
            }
        }
    }
}

impl LLMRuntime {
    /// Creates a new LLM Runtime without loading any model.
    /// The model will be created lazily when the first Query::Prompt is received.
    pub fn from_config(config: LLMRuntimeConfig) -> Result<Self, Error> {
        let (ctrl_tx, ctrl_rx) = std::sync::mpsc::channel();
        let (response_stream_tx, response_stream_rx) = std::sync::mpsc::channel();

        Ok(Self {
            config,

            worker: Arc::new(RwLock::new(None)),
            control: (
                Arc::new(RwLock::new(ctrl_tx)),
                Arc::new(Mutex::new(Some(ctrl_rx))),
            ),
            response: (
                Arc::new(response_stream_tx),
                Arc::new(Mutex::new(response_stream_rx)),
            ),
        })
    }

    /// Creates a model instance based on the model name.
    /// Called lazily when the first Query::Prompt is received.
    fn create_model(
        model_name: &str,
        config: &LLMRuntimeConfig,
        device: Device,
    ) -> Result<Box<dyn LLMRuntimeModel>, Error> {
        let ModelConfig {
            penalty,
            seed,
            sampling_config,
            ..
        } = config.model_config.clone();

        match model_name {
            name if name.contains("Qwen3") => Ok(Box::new(Qwen3Model {
                device: Some(device),
                tokenizer: None,
                weights: None,
                template: None,
                template_proc: None,
                seed,
                sampling_config,
            })),
            name if name.contains("Mock") => Ok(Box::new(Mock)),
            name if name.contains("Llama") => Ok(Box::new(LLama3Model {
                device: Some(device),
                tokenizer: None,
                weights: None,
                cache: None,
                penalty,
                template: None,
                template_proc: None,
                seed,
                sampling_config,
            })),
            _ => Err(Error::ExecutionError(format!(
                "Unknown Model Name: {model_name}"
            ))),
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
            .read()
            .expect("Failed to acquire read lock")
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
    /// Executes the runtime and keeps the executor in a background thread.
    /// The model is created lazily when the first Query::Prompt is received.
    pub fn run_stream(&mut self) -> Result<(), Error> {
        let config = self.config.clone();

        let control_rx = self
            .control
            .1
            .lock()
            .expect("Failed to acquire lock")
            .take()
            .unwrap();

        let response_tx = self.response.0.clone();

        tracing::debug!("Spawning worker thread (model will be loaded on first prompt)");

        let worker = tauri::async_runtime::spawn_blocking(move || {
            let mut model: Option<Box<dyn LLMRuntimeModel>> = None;

            loop {
                match control_rx.recv() {
                    Ok(message) => {
                        match message {
                            Query::Prompt { model: ref query_model, .. } => {
                                // Lazily create and initialize model on first prompt
                                if model.is_none() {
                                    let model_name = query_model
                                        .as_ref()
                                        .map(|s| s.as_str())
                                        .unwrap_or(&config.model_config.name);

                                    tracing::debug!("Creating model: {}", model_name);

                                    let device = LLMRuntime::load_default_device();
                                    match LLMRuntime::create_model(model_name, &config, device) {
                                        Ok(mut m) => {
                                            tracing::debug!("Initializing model");
                                            if let Err(error) = m.init(&config) {
                                                tracing::error!("Error initializing model: {}", error);
                                                continue;
                                            }
                                            model = Some(m);
                                        }
                                        Err(error) => {
                                            tracing::error!("Error creating model: {}", error);
                                            continue;
                                        }
                                    }
                                }

                                if let Some(ref mut m) = model {
                                    tracing::debug!("Sending message to model");
                                    if let Err(error) = m.execute(message, response_tx.clone()) {
                                        tracing::error!("Error execute streaming: {error}");
                                    }
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

        *self.worker.write().expect("Failed to acquire write lock") = Some(worker);

        Ok(())
    }

    pub fn send_stream(&self, msg: Query) -> Result<(), Error> {
        self.control
            .0
            .read()
            .map_err(|e| Error::ExecutionError(e.to_string()))?
            .send(msg)
            .map_err(|e| Error::ExecutionError(e.to_string()))?;

        Ok(())
    }

    pub fn recv_stream(&self) -> Result<Query, Error> {
        self.response
            .1
            .lock()
            .map_err(|e| Error::StreamError(e.to_string()))?
            .recv()
            .map_err(|e| Error::StreamError(e.to_string()))
    }
}
