use tauri::{command, AppHandle, Runtime};

use crate::models::*;
use crate::Result;
use crate::TauriPluginLlmExt;

// #[command]
// pub(crate) async fn ping<R: Runtime>(
//     app: AppHandle<R>,
//     payload: PingRequest,
// ) -> Result<PingResponse> {
//     app.tauri_plugin_llm().ping(payload)
// }

#[command]
pub(crate) async fn send_message<R: Runtime>(app: AppHandle<R>) {}

#[command]
pub(crate) async fn check_status<R: Runtime>(app: AppHandle<R>) {}
