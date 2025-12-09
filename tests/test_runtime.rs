use std::time::Duration;
use tauri_plugin_llm::{llmconfig::LLMRuntimeConfig, runtime::LLMRuntime, Error, LlmMessage};

#[tokio::test]
async fn test_runtime_qwen3_4b_gguf() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run().await;

    if let Err(_) = runtime.send(LlmMessage::Prompt {
            system:"You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string(),
            message:"Hello, World".to_string(), 
            num_samples: 200
    }) {

        loop {

            if let Ok(message) = runtime.retry_recv()  {
                tracing::info!("Received Message : {:?}", message);
                break;
            }
        }
    }

    Ok(())
}
