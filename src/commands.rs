use crate::Result;
use crate::{models::*, PluginState};
use base64::prelude::BASE64_STANDARD as base64;
use base64::Engine;
use tauri::{command, AppHandle, Runtime};
use tauri::{Emitter, State};

#[command]
pub(crate) async fn send_message(state: State<'_, PluginState>, message: Query) -> Result<Query> {
    let runtime = state.runtime.lock().unwrap();
    runtime.send(message)
}

#[command]
pub(crate) async fn retry_recv(state: State<'_, PluginState>) -> Result<Query> {
    let runtime = state.runtime.lock().unwrap();
    runtime.retry_recv()
}

#[command]
pub(crate) async fn stream<R>(
    state: State<'_, PluginState>,
    message: Query,
    app: AppHandle<R>,
) -> Result<()>
where
    R: Runtime,
{
    let mut runtime = state.runtime.lock().unwrap();
    runtime.run_stream()?;

    // fix this
    runtime.send(message);

    loop {
        if let Ok(query_stream) = runtime.retry_recv() {
            // match &query_stream {
            //     QueryStream::Chunk { .. } => {
            //         app.emit(format!("{}", query_stream).as_str(), query_stream)
            //             .map_err(|e| crate::Error::StreamError(e.to_string()))?;
            //     }
            //     QueryStream::Error { msg } => {
            //         app.emit(format!("{}", query_stream).as_str(), msg)
            //             .map_err(|e| crate::Error::StreamError(e.to_string()))?;
            //         break;
            //     }
            //     QueryStream::End => break,
            // }
        }
    }

    Ok(())
}
