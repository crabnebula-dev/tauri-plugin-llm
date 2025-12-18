mod commands;
#[cfg(desktop)]
mod desktop;
mod error;
mod llm;
#[cfg(mobile)]
mod mobile;
mod models;

mod templates;

pub use templates::*;

use std::sync::Arc;
use std::sync::Mutex;

use crate::llm::runtime::LLMRuntime;
#[cfg(desktop)]
use desktop::TauriPluginLlm;
pub use error::{Error, Result};
pub use llm::loaders;
pub use llm::runtime;
#[cfg(mobile)]
use mobile::TauriPluginLlm;
pub use models::*;
use serde::Deserialize;
use serde::Serialize;
use tauri::{
    plugin::{Builder as PluginBuilder, TauriPlugin},
    Manager, Runtime,
};

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`] to access the tauri-plugin-llm APIs.
pub trait TauriPluginLlmExt<R: Runtime> {
    fn tauri_plugin_llm(&self) -> &TauriPluginLlm<R>;
}

impl<R: Runtime, T: Manager<R>> crate::TauriPluginLlmExt<R> for T {
    fn tauri_plugin_llm(&self) -> &TauriPluginLlm<R> {
        self.state::<TauriPluginLlm<R>>().inner()
    }
}

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub struct LLMPluginConfig {
    #[cfg(feature = "mcpurify")]
    pub mcpurify_config: Option<mcpurify::Config>,

    pub llmconfig: LLMRuntimeConfig,
}

#[derive(Default)]
pub struct Builder {
    plugin_config: Option<LLMPluginConfig>,
}

pub struct PluginState {
    runtime: Arc<Mutex<LLMRuntime>>,
}

impl Builder {
    /// Create a new plugin builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Programmatically provide a [`LLMPluginConfig`] config. Overwrites tauri.conf.json.
    pub fn config(mut self, config: LLMPluginConfig) -> Self {
        self.plugin_config = Some(config);
        self
    }

    pub fn build<R: Runtime>(self) -> TauriPlugin<R, LLMPluginConfig> {
        PluginBuilder::<R, LLMPluginConfig>::new("llm")
            .invoke_handler(tauri::generate_handler![
                commands::send_message,
                commands::retry_recv
            ])
            .setup(|app, api| {
                let config = self
                    .plugin_config
                    .or(Some((*api.config()).clone()))
                    .ok_or(Error::MissingConfig)?;

                // manage llm runtime ?
                app.manage({
                    // initialize runtime by config
                    let mut runtime = LLMRuntime::from_config(config.llmconfig.clone())?;

                    // start background thread
                    runtime.run();

                    PluginState {
                        runtime: Arc::new(Mutex::new(runtime)),
                    }
                });

                #[cfg(mobile)]
                let tauri_plugin_llm = mobile::init(app, api, config)?;

                #[cfg(desktop)]
                let tauri_plugin_llm = desktop::init(app, api, config)?;
                app.manage(tauri_plugin_llm);

                Ok(())
            })
            .build()
    }
}

/// Initializes the plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R, LLMPluginConfig> {
    Builder::default().build()
}
