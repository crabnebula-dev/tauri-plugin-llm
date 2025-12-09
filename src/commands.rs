use crate::Result;
use crate::{models::*, PluginState};
use tauri::State;
use tauri::{command, AppHandle, Runtime};

#[command]
pub(crate) async fn send_message(
    app: State<'_, PluginState>,
    payload: LlmMessage,
) -> Result<LlmMessage> {
    // access inner state
    // app.inner().run(response)

    let runtime = app.runtime.lock().unwrap();

    todo!()
}

#[command]
pub(crate) async fn check_status<R: Runtime>(app: AppHandle<R>) -> Result<LlmMessage> {
    todo!()
}
