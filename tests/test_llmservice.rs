use serde_json::ser;
use tauri_plugin_llm::{
    runtime::LLMRuntime, Error, LLMRuntimeConfig, LLMService, Query, QueryConfig, QueryMessage,
};

#[tokio::test]
#[ignore = "Run this test explicitly to avoid using real model weights"]
async fn test_switching_runtimes() -> Result<(), Error> {
    let runtime_config_paths = [
        "tests/fixtures/test_runtime_mock.json",
        "tests/fixtures/test_runtime_llama3.config.json",
        "tests/fixtures/test_runtime_qwen3.config.json",
    ]
    .to_vec();

    let mut configs = vec![];

    for p in runtime_config_paths {
        let config = LLMRuntimeConfig::from_path(p).expect("Loading config failed");

        configs.push(config);
    }

    let service = LLMService::from_runtime_configs(&configs);

    let mut runtime = LLMRuntime::from_config(configs.first().unwrap().clone())
        .expect("Loading runtime via config failed");

    // for _ in service. {
    // runtime.run_stream()?;

    let _ = Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string(),
        }],
        tools: vec![],
        config: Some(QueryConfig::default()),
        chunk_size: Some(25),
        timestamp: None,
    };
    // }

    Ok(())
}
