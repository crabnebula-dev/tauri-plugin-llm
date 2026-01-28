use serde::{ser::Serializer, Serialize};
use std::sync::mpsc::TryRecvError;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[cfg(mobile)]
    #[error(transparent)]
    PluginInvoke(#[from] tauri::plugin::mobile::PluginInvokeError),
    #[error("No MCPurify config promided in tauri.conf.json or plugin Builder")]
    MissingConfig,

    #[error("Error during execution ({0})")]
    ExecutionError(String),

    #[error("Missing config ({0})")]
    MissingConfigLLM(String),

    #[error("Device is missing")]
    MissingDevice,

    #[error("Unexpected Message Variant")]
    UnexpectedMessage,

    #[error("Error Loading File ({0})")]
    LoadingFile(String, String),

    #[error("Error Encoding Message ({0})")]
    MessageEncodingError(String),

    #[error("Error receiving message from sync channel ({0})")]
    ChannelReceiveError(#[from] TryRecvError),

    #[error("Error deserialization from JSON")]
    JsonSerdeError(#[from] serde_json::Error),

    #[error("Error calling foreign function: ({0})")]
    Ffi(String),

    #[error("Error processing template: ({0})")]
    TemplateError(String),

    #[error("Error streaming response: ({0})")]
    StreamError(String),

    #[error("Client event is not defined.")]
    UndefinedClientEvent(String),

    #[error("Run into timeout. ({0})")]
    TimeoutError(String),

    #[error("No active runtime present")]
    MissingActiveRuntime,
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}
