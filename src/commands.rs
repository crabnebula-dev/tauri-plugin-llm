use crate::Result;
use crate::{models::*, PluginState};
use tauri::{command, AppHandle, Runtime};
use tauri::{Emitter, State};

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

    runtime.send_stream(message)?;

    loop {
        if let Ok(query) = runtime.try_recv_stream() {
            match &query {
                Query::Chunk { .. } => {
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
                    let event = query.try_render_as_event_name()?;
                    app.emit(&event, ())
                        .map_err(|e| crate::Error::StreamError(e.to_string()))?;

                    break;
                }
                _ => {
                    tracing::error!("Unknown responsed received")
                }
            }
        }
    }

    Ok(())
}
