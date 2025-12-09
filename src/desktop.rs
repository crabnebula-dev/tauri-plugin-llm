use serde::de::DeserializeOwned;
use tauri::{plugin::PluginApi, AppHandle, Runtime};

use crate::{models::*, LLMPluginConfig};

pub fn init<R: Runtime, C: DeserializeOwned>(
    app: &AppHandle<R>,
    _api: PluginApi<R, C>,
    config: LLMPluginConfig,
) -> crate::Result<TauriPluginLlm<R>> {
    Ok(TauriPluginLlm {
        handle: app.clone(),
        config,
    })
}

/// Access to the tauri-plugin-llm APIs.
pub struct TauriPluginLlm<R: Runtime> {
    handle: AppHandle<R>,
    config: LLMPluginConfig,
}

impl<R: Runtime> TauriPluginLlm<R> {
    // pub fn ping(&self, payload: PingRequest) -> crate::Result<PingResponse> {
    //     Ok(PingResponse {
    //         value: payload.value,
    //     })
    // }
    // TODO: it probably makes sense to have send and try_recv here as well
    // for those that want to use the plugin in rust
}
