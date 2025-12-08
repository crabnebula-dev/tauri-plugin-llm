use serde::de::DeserializeOwned;
use tauri::{plugin::PluginApi, AppHandle, Runtime};

use crate::models::*;

pub fn init<R: Runtime, C: DeserializeOwned>(
    app: &AppHandle<R>,
    _api: PluginApi<R, C>,
    config: mcpurify::Config,
) -> crate::Result<TauriPluginLlm<R>> {
    Ok(TauriPluginLlm {
        handle: app.clone(),
        mcpurify_config: config,
    })
}

/// Access to the tauri-plugin-llm APIs.
pub struct TauriPluginLlm<R: Runtime> {
    handle: AppHandle<R>,
    mcpurify_config: mcpurify::Config,
}

impl<R: Runtime> TauriPluginLlm<R> {
    pub fn ping(&self, payload: PingRequest) -> crate::Result<PingResponse> {
        Ok(PingResponse {
            value: payload.value,
        })
    }
}
