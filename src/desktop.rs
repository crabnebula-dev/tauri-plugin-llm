use serde::de::DeserializeOwned;
use tauri::{plugin::PluginApi, AppHandle, Runtime};

use crate::LLMPluginConfig;

pub fn init<R: Runtime, C: DeserializeOwned>(
    app: &AppHandle<R>,
    _api: PluginApi<R, C>,
    config: LLMPluginConfig,
) -> crate::Result<TauriPluginLlm<R>> {
    Ok(TauriPluginLlm {
        _handle: app.clone(),
        _config: config,
    })
}

/// Access to the tauri-plugin-llm APIs.
pub struct TauriPluginLlm<R: Runtime> {
    _handle: AppHandle<R>,
    _config: LLMPluginConfig,
}

impl<R: Runtime> TauriPluginLlm<R> {}
