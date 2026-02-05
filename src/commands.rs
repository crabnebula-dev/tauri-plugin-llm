use crate::Result;
use crate::{models::*, Error, PluginState};
use tauri::{command, AppHandle, Runtime};
use tauri::{Emitter, State};

#[command]
pub(crate) async fn add_configuration(state: State<'_, PluginState>, config: String) -> Result<()> {
    let mut service = state.runtime.lock().unwrap();
    tracing::debug!("Adding config to runtime service: {}", config);

    service.add_config(config)?;

    Ok(())
}

#[command]
pub(crate) async fn switch_model(state: State<'_, PluginState>, id: String) -> Result<()> {
    let mut service = state.runtime.lock().unwrap();

    tracing::debug!("Switching to model: {}", id);

    service.activate(id)?;

    Ok(())
}

#[command]
pub(crate) async fn list_available_models(state: State<'_, PluginState>) -> Result<Vec<String>> {
    tracing::debug!("list_available_models command called");

    let service = match state.runtime.lock() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to lock runtime mutex: {}", e);
            return Err(Error::StreamError(format!("Failed to lock runtime: {}", e)));
        }
    };

    let models = service.list_models();
    tracing::debug!("Available models: {:?}", models);

    Ok(models)
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
                Query::End { .. } => {
                    tracing::debug!("Reached end of stream");
                    let event = query.try_render_as_event_name()?;
                    app.emit(&event, query)
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
