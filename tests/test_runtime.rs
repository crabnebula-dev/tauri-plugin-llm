use tauri_plugin_llm::{
    error::Error,
    llmconfig::LLMRuntimeConfig,
    runtime::{LLMRuntime, LlmMessage},
};

#[tokio::test]
async fn test_runtime_qwen3_4b_gguf() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;

    let mut runtime = LLMRuntime::from_config(config)?;
    let (response_tx, message_rx) = std::sync::mpsc::channel();
    let (sender_tx, worker) = runtime.run(response_tx).await?;

    if let Err(e) = sender_tx
        .send(LlmMessage::Prompt {
            system: "You are a helpful assistent. Your task is to echo the incoming message. Do not describe anything. ".to_string(),
            message: "Hello, World".to_string(),
        })
        
    {
        tracing::error!("Error sending message to model {}", e);
    }

    loop {
        if let Ok(incoming) = message_rx.recv() {
            tracing::info!("Received Message : {:?}", incoming);
            sender_tx.send(LlmMessage::Exit).map_err(|_|Error::ExecutionError)?;
            break;
        }
    }

    // We need to await the runtime 
    worker.await.map_err(|_| Error::ExecutionError)?;

    Ok(())
}
