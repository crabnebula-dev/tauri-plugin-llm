mod common;

use tauri_plugin_llm::{
    runtime::LLMRuntime, GenerationSeed, LLMRuntimeConfig, Query, QueryMessage, SamplingConfig,
};
use tauri_plugin_llm_macros::hf_test;

// TODO: set cache_dir via env
#[hf_test(
    model = "google/gemma-3-1b-it",
    cleanup = false,
    cache_dir = "/Volumes/MLM/huggingface"
)]
fn test_runtime_local_gemma3_safetensors_toolcall(config: LLMRuntimeConfig) {
    let tool_call = serde_json::json!({
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

    })
    .to_string();

    let prelude = "You have access to functions. If you decide to invoke any of the function(s), \
    you MUST put it in the format of\n {{\"name\": function name, \"parameters\": dictionary of argument name and \
    its value}}\n\nYou SHOULD NOT include any other text in the response if you call a function\n\n";

    let prompt = format!("{}{}While \
    browsing the product catalog, I came across a product that piqued my interest. The product ID is 807ZPKBL9V. \
    Can you help me find the name of this product?
    ", prelude, tool_call
    );

    test_runtime_toolcall(config, Some(&prompt))
}

#[hf_test(
    model = "Qwen/Qwen3-4B-Instruct-2507",
    cleanup = false,
    cache_dir = "/Volumes/MLM/huggingface"
)]
fn test_runtime_local_qwen3_safetensors_toolcall(config: LLMRuntimeConfig) {
    test_runtime_toolcall(config, None)
}

#[hf_test(
    model = "meta-llama/Llama-3.2-3B-Instruct",
    cleanup = false,
    cache_dir = "/Volumes/MLM/huggingface"
)]
fn test_runtime_local_llama3_safetensors_toolcall(config: LLMRuntimeConfig) {
    test_runtime_toolcall(config, None)
}

fn test_runtime_toolcall(
    config: LLMRuntimeConfig,
    content: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    let result = runtime.send_stream(Query::Prompt {
        messages: vec![
            QueryMessage {
                role: "user".to_string(),
                content: content.unwrap_or("Write 'Hello, World' and call the tool to list all the files in the home directory of the user").to_string(),
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
                    tool_call = Some(String::from_utf8(data)?)
                }
                _ => result.extend(data),
            },
            _ => break,
        }
    }

    let result_str = String::from_utf8(result)?;

    if let Some(tool_call) = tool_call.clone() {
        let toolcall_json: serde_json::Value = serde_json::from_str(&tool_call)?;
        tracing::debug!("{:#?}", toolcall_json);
    }

    tracing::debug!("Full result: {result_str:?}");

    Ok(())
}
