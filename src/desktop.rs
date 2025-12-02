use serde::de::DeserializeOwned;
use tauri::{plugin::PluginApi, AppHandle, Runtime};

use crate::models::*;

pub fn init<R: Runtime, C: DeserializeOwned>(
  app: &AppHandle<R>,
  _api: PluginApi<R, C>,
) -> crate::Result<TauriPluginLlm<R>> {
  Ok(TauriPluginLlm(app.clone()))
}

/// Access to the tauri-plugin-llm APIs.
pub struct TauriPluginLlm<R: Runtime>(AppHandle<R>);

impl<R: Runtime> TauriPluginLlm<R> {
  pub fn ping(&self, payload: PingRequest) -> crate::Result<PingResponse> {
    Ok(PingResponse {
      value: payload.value,
    })
  }
}
