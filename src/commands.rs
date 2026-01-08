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
    let runtime = state.runtime.lock().unwrap();
    let rx = runtime.stream(message)?;

    loop {
        if let Ok(query_stream) = rx.try_recv() {
            // TODO: query stream is Serializable and can be send directly.
            //       However, error handling should be done here in the backend.

            match &query_stream {
                QueryStream::Internal { id, data } => {
                    // encode as base64
                    let data = base64.encode(data);

                    app.emit(
                        format!("{}", query_stream).as_str(),
                        QueryStream::Chunk { id: *id, data },
                    )
                    .map_err(|e| crate::Error::StreamError(e.to_string()))?;
                }
                QueryStream::Chunk { id, data } => {
                    // ?
                }
                QueryStream::Error { msg } => {
                    app.emit(format!("{}", query_stream).as_str(), msg)
                        .map_err(|e| crate::Error::StreamError(e.to_string()))?;
                    break;
                }
                QueryStream::End => break,
            }
        }
    }

    Ok(())
}
