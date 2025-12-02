use tauri::{
  plugin::{Builder, TauriPlugin},
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

/// Initializes the plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
  Builder::new("tauri-plugin-llm")
    .invoke_handler(tauri::generate_handler![commands::ping])
    .setup(|app, api| {
      #[cfg(mobile)]
      let tauri_plugin_llm = mobile::init(app, api)?;
      #[cfg(desktop)]
      let tauri_plugin_llm = desktop::init(app, api)?;
      app.manage(tauri_plugin_llm);
      Ok(())
    })
    .build()
}
