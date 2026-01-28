use crate::{models::*, Error, PluginState};
use crate::{LLMService, Result};
use tauri::{command, AppHandle, Runtime};
use tauri::{Emitter, State};

#[command]
pub(crate) async fn switch_model<R>(state: State<'_, PluginState>, id: String) -> Result<()>
where
    R: Runtime,
{
    // TODO:
    // 1. terminate currently executed runtime
    // 2. load new runtime
    // 3. execute the runtime
    // 4. replace the runtime inside the app handle

    let mut service = state.runtime.lock().unwrap();

    // *runtime = service.switch(id);

    todo!()
}

#[command]
pub(crate) async fn list_available_models<R>(state: State<'_, PluginState>) -> Result<Vec<()>>
where
    R: Runtime,
{
    todo!()
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
    let mut service = state.runtime.lock().unwrap();
    let runtime = service.runtime().ok_or(Error::MissingActiveRuntime)?;

    tracing::debug!("Send query to runtime: {:?}", message);
    runtime.send_stream(message)?;

    loop {
        match runtime.recv_stream() {
            Ok(query) => match &query {
                Query::Chunk { .. } => {
                    tracing::debug!("Got data chunk");
                    let event = query.try_render_as_event_name()?;
                    app.emit(&event, query)
                        .map_err(|e| crate::Error::StreamError(e.to_string()))?;
                }
                Query::Status { msg } => {
                    let event = query.try_render_as_event_name()?;
                    app.emit(&event, msg)
                        .map_err(|e| crate::Error::StreamError(e.to_string()))?;
                }
                Query::End => {
                    tracing::debug!("Reached end of stream");
                    let event = query.try_render_as_event_name()?;
                    app.emit(&event, "")
                        .map_err(|e| crate::Error::StreamError(e.to_string()))?;

                    break;
                }
                _ => {
                    tracing::error!("Unknown response received")
                }
            },
            Err(error) => {
                app.emit("query-stream-error", error.to_string())
                    .map_err(|e| crate::Error::StreamError(e.to_string()))?;

                tracing::error!("receiving stream returned an error: {error}. Exiting");

                break;
            }
        }
    }

    Ok(())
}
