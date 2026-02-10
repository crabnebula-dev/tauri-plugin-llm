use std::vec;

use proptest::prelude::*;
use tauri_plugin_llm::{
    runtime::LLMRuntime, Error, LLMRuntimeConfig, LLMService, Query, QueryMessage,
};
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer, Registry};

#[allow(dead_code)]
fn enable_logging() {
    let verbose = tracing_subscriber::fmt::layer().with_filter(filter::LevelFilter::DEBUG);
    Registry::default().with(verbose).init();
}

#[tokio::test]
#[ignore = "Load the Qwen3 model first, then run the test manually. This test FAILS, because minijinja does not support all jinja2 functions in the template"]
async fn test_runtime_local_qwen3_safetensors_toolcall() -> Result<(), Error> {
    enable_logging();

    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    let result = runtime.send_stream(Query::Prompt {
        messages: vec![
            QueryMessage {
                role: "user".to_string(),
                content: "Hello, World".to_string(),
            },
            QueryMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything.".to_string(),
            },
        ],
        tools: vec![
            serde_json::json!({
                "type" : "function",
                "function" : {
                    "name" : "get_files_in_directory",
                    "description" : "List all files for a directory path as parameter",
                    "parameters" : {
                        "type" : "object",
                        "properties" : {
                            "path": {
                                "type" : "string",
                                "description" : "The path of the directory to get a listing of"
                            }
                        }
                    }
                }

            }).to_string()
        ],
        max_tokens: Some(50),
        temperature: None,
        top_k: None,
        top_p: None,
        think: false,
        stream: true,
        model: None,
        penalty: None,
        seed: None,
        sampling_config: None,
        chunk_size: None,
        timestamp: None,
    });

    assert!(result.is_ok(), "{result:?}");

    let mut result = vec![];
    tracing::debug!("Assembling message");

    while let Ok(message) = runtime.recv_stream() {
        assert!(matches!(message, Query::Chunk { .. } | Query::End { .. }));

        match message {
            Query::Chunk { data, .. } => result.extend(data),
            _ => break,
        }
    }

    let result_str = String::from_utf8(result);
    assert!(result_str.is_ok());

    let s = result_str.unwrap();
    tracing::debug!("Received LocalRuntime Response: {s}");

    Ok(())
}
