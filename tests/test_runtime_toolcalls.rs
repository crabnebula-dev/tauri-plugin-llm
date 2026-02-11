use std::vec;

use tauri_plugin_llm::{
    runtime::LLMRuntime, Error, GenerationSeed, LLMRuntimeConfig, Query, QueryMessage,
    SamplingConfig,
};
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer, Registry};

#[allow(dead_code)]
fn enable_logging() {
    let verbose = tracing_subscriber::fmt::layer().with_filter(filter::LevelFilter::DEBUG);
    Registry::default().with(verbose).init();
}
#[tokio::test]
#[ignore = "Load the Qwen3 model first, then run the test manually."]
async fn test_runtime_local_qwen3_safetensors_toolcall() -> Result<(), Error> {
    test_runtime_toolcall("tests/fixtures/test_runtime_qwen3.config.json").await
}

#[tokio::test]
#[ignore = "Load the Llama3 model first, then run the test manually."]
async fn test_runtime_local_llama3_safetensors_toolcall() -> Result<(), Error> {
    test_runtime_toolcall("tests/fixtures/test_runtime_llama3.config.json").await
}

async fn test_runtime_toolcall(model_config: &str) -> Result<(), Error> {
    enable_logging();
    let config = LLMRuntimeConfig::from_path(model_config)?;
    let mut runtime = LLMRuntime::from_config(config.clone())?;

    runtime.run_stream()?;

    let result = runtime.send_stream(Query::Prompt {
        messages: vec![
            QueryMessage {
                role: "user".to_string(),
                content: "Write 'Hello, World' and call the tool to list all the files in the home directory of the user".to_string(),
            },
            QueryMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. Call a tool to solve the request.".to_string(),
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
                        },
                        "required" : "path"
                    }
                }

            }).to_string()
        ],
        max_tokens: Some(500),
        temperature: None,
        top_k: None,
        top_p: None,
        think: false,
        stream: true,
        model: None,
        penalty: None,
        seed: Some(GenerationSeed::Fixed(42)),
        sampling_config: Some(SamplingConfig::ArgMax),
        chunk_size: None,
        timestamp: None,
    });

    assert!(result.is_ok(), "{result:?}");

    let mut result = vec![];
    let mut tool_call = None;
    tracing::debug!("Assembling message");

    while let Ok(message) = runtime.recv_stream() {
        assert!(matches!(message, Query::Chunk { .. } | Query::End { .. }));

        match message {
            Query::Chunk { data, kind, .. } => match kind {
                tauri_plugin_llm::QueryChunkType::ToolCall => {
                    tool_call = Some(
                        String::from_utf8(data).map_err(|e| Error::StreamError(e.to_string()))?,
                    )
                }
                _ => result.extend(data),
            },
            _ => break,
        }
    }

    let result_str = String::from_utf8(result);

    assert!(result_str.is_ok());

    if let Some(tool_call) = tool_call.clone() {
        let toolcall_json: serde_json::Value =
            serde_json::from_str(&tool_call).map_err(|e| Error::JsonSerdeError(e))?;

        tracing::debug!("{:#?}", toolcall_json);
    }

    tracing::debug!("Full result: {result_str:?}");
    assert!(tool_call.is_some());

    Ok(())
}
