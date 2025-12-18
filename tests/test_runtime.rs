use std::vec;

use tauri_plugin_llm::{
    runtime::LLMRuntime, Error, LLMRuntimeConfig, Query, QueryConfig, QueryMessage,
};

#[tokio::test]
async fn test_runtime_qwen3_4b_gguf() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run();

    if let Err(_) = runtime.send(Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string() },
            QueryMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string()},
        ],
        tools: vec![],
        config: Some(QueryConfig::default()),
    }) {
        loop {
            if let Ok(message) = runtime.retry_recv() {
                tracing::info!("Received Message : {:?}", message);
                break;
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_runtime_llama_3_2_3b_instruct() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_llama3.config.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run();

    if let Err(_) = runtime.send(Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string() },
            QueryMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string()},
        ],
        tools: vec![],
        config: Some(QueryConfig::default()),
    }) {
        loop {
            if let Ok(message) = runtime.retry_recv() {
                tracing::info!("Received Message : {:?}", message);
                break;
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_runtime_mock() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_mock.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run();

    if let Err(_) = runtime.send(Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string() },
            QueryMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string()},
        ],
        tools: vec![],
        config: Some(QueryConfig::default()),
    }) {
        loop {
            if let Ok(message) = runtime.retry_recv() {
                tracing::info!("Received Message : {:?}", message);
                break;
            }
        }
    }

    Ok(())
}
