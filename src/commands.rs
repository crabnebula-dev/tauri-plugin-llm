use crate::Result;
use crate::{models::*, PluginState};
use tauri::command;
use tauri::State;

#[command]
pub(crate) async fn send_message(
    app: State<'_, PluginState>,
    payload: LlmMessage,
) -> Result<LlmMessage> {
    let runtime = app.runtime.lock().unwrap();

    runtime.send(payload)
}

#[command]
pub(crate) async fn retry_recv(app: State<'_, PluginState>) -> Result<LlmMessage> {
    let runtime = app.runtime.lock().unwrap();
    runtime.retry_recv()
}
