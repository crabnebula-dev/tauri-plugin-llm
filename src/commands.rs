use crate::Result;
use crate::{models::*, PluginState};
use tauri::command;
use tauri::State;

#[command]
pub(crate) async fn send_message(
    state: State<'_, PluginState>,
    message: LlmMessage,
) -> Result<LlmMessage> {
    let runtime = state.runtime.lock().unwrap();
    runtime.send(message)
}

#[command]
pub(crate) async fn retry_recv(state: State<'_, PluginState>) -> Result<LlmMessage> {
    let runtime = state.runtime.lock().unwrap();
    runtime.retry_recv()
}
