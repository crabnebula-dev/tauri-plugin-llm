use tauri::{
    plugin::{Builder as PluginBuilder, TauriPlugin},
    Manager, Runtime,
};

pub use models::*;

#[cfg(desktop)]
mod desktop;
#[cfg(mobile)]
mod mobile;

mod commands;
mod error;
mod models;

mod llm;

pub use llm::llmconfig;

pub use error::{Error, Result};

#[cfg(desktop)]
use desktop::TauriPluginLlm;
#[cfg(mobile)]
use mobile::TauriPluginLlm;

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`] to access the tauri-plugin-llm APIs.
pub trait TauriPluginLlmExt<R: Runtime> {
    fn tauri_plugin_llm(&self) -> &TauriPluginLlm<R>;
}

impl<R: Runtime, T: Manager<R>> crate::TauriPluginLlmExt<R> for T {
    fn tauri_plugin_llm(&self) -> &TauriPluginLlm<R> {
        self.state::<TauriPluginLlm<R>>().inner()
    }
}

#[derive(Default)]
pub struct Builder {
    mcpurify_config: Option<mcpurify::Config>,
}

impl Builder {
    /// Create a new plugin builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Programmatically provide a MCPurify config. Overwrites tauri.conf.json.
    pub fn mcpurify_config(mut self, config: mcpurify::Config) -> Self {
        self.mcpurify_config = Some(config);
        self
    }

    pub fn build<R: Runtime>(self) -> TauriPlugin<R, Option<mcpurify::Config>> {
        PluginBuilder::<R, Option<mcpurify::Config>>::new("tauri-plugin-llm")
            .invoke_handler(tauri::generate_handler![commands::ping])
            .setup(|app, api| {
                let config = self
                    .mcpurify_config
                    .or((*api.config()).clone())
                    .ok_or(Error::MissingConfig)?;

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
pub fn init<R: Runtime>() -> TauriPlugin<R, Option<mcpurify::Config>> {
    Builder::default().build()
}
