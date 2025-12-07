#[derive(Debug, thiserror::Error)]
pub enum Tauri_LLM_PluginError {
    #[error("Error during execution")]
    ExecutionError,
}
