use std::vec;

use tauri_plugin_llm::{
    runtime::LLMRuntime, Error, LLMRuntimeConfig, LLMService, Query, QueryConfig, QueryMessage,
};
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer, Registry};

#[allow(dead_code)]
fn enable_logging() {
    let verbose = tracing_subscriber::fmt::layer().with_filter(filter::LevelFilter::DEBUG);
    Registry::default().with(verbose).init();
}

#[tokio::test]
#[ignore = "Load the Qwen3 model first, then run this test manually"]
async fn test_runtime_qwen3_4b_gguf() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    tracing::info!("Sending Message");

    let result = runtime.send_stream(Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string(), },
            QueryMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string()
        },
        ],
        tools: vec![],
        config: Some(QueryConfig::default()),
        chunk_size : None,
        timestamp : None
    });

    assert!(result.is_ok(), "{result:?}");

    while let Ok(message) = runtime.recv_stream() {
        assert!(matches!(message, Query::Chunk { .. }));
        break;
    }

    Ok(())
}

#[tokio::test]
#[ignore = "Load the LLama3.2 model first, then run the test manually"]
async fn test_runtime_llama_3_2_3b_instruct() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_llama3.config.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    let result = runtime.send_stream(Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string(), },
            QueryMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string()
        },
        ],
        tools: vec![],
        config: Some(QueryConfig::default()),
        chunk_size : None,
        timestamp : None
    });

    assert!(result.is_ok(), "{result:?}");

    while let Ok(message) = runtime.recv_stream() {
        assert!(matches!(message, Query::Chunk { .. } | Query::End));
        break;
    }

    Ok(())
}

#[tokio::test]
async fn test_runtime_mock() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_mock.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    if let Err(_) = runtime.send_stream(Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string(), },
            QueryMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string(), },
        ],
        tools: vec![],
        config: Some(QueryConfig::default()),
        chunk_size : None, timestamp : None
    }) {
        while let Ok(message) = runtime.recv_stream() {
            assert!(matches!(message, Query::Chunk { ..} | Query::End));
            break;
        }
    }
    Ok(())
}

#[tokio::test]
async fn test_runtime_mock_streaming() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_mock.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    let queries = vec![
        Query::Prompt {
            messages: vec![QueryMessage {
                role: "user".to_string(),
                content: "Message #1".to_string(),
            }],
            tools: vec![],
            config: Some(QueryConfig::default()),
            chunk_size: Some(25),
            timestamp: None,
        },
        Query::Prompt {
            messages: vec![QueryMessage {
                role: "user".to_string(),
                content: "Message #2".to_string(),
            }],
            tools: vec![],
            config: Some(QueryConfig::default()),
            chunk_size: Some(25),
            timestamp: None,
        },
    ];

    for query in queries {
        let _ = runtime.send_stream(query);

        while let Ok(message) = runtime.recv_stream() {
            // we check for Query::Chunk and Query::End, because Query::End will be send after the
            // the first query has been responded.
            assert!(
                matches!(message, Query::Chunk { .. } | Query::End),
                "{message:?}"
            );
            break;
        }
    }

    Ok(())
}

#[tokio::test]
#[ignore = "Load the Qwen3 model first, then run the test manually"]
async fn test_runtime_qwen3_streaming() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    let queries = vec![
        Query::Prompt {
            messages: vec![QueryMessage {
                role: "user".to_string(),
                content: "Just echo This Message 1".to_string(),
            }],
            tools: vec![],
            config: Some(QueryConfig::default()),
            chunk_size: Some(25),
            timestamp: None,
        },
        Query::Prompt {
            messages: vec![QueryMessage {
                role: "user".to_string(),
                content: "Just echo This Message 2".to_string(),
            }],
            tools: vec![],
            config: Some(QueryConfig::default()),
            chunk_size: Some(25),
            timestamp: None,
        },
    ];

    for query in queries {
        let _ = runtime.send_stream(query);

        while let Ok(message) = runtime.recv_stream() {
            assert!(matches!(message, Query::Chunk { .. }));
            break;
        }
    }

    Ok(())
}

#[tokio::test]
#[ignore = "Run this test explicitly to avoid using real model weights"]
async fn test_switching_runtimes() -> Result<(), Error> {
    let runtime_config_paths = [
        "tests/fixtures/test_runtime_mock.json",
        "tests/fixtures/test_runtime_qwen3.config.json",
        "tests/fixtures/test_runtime_llama3.config.json",
    ];

    let configs: Vec<LLMRuntimeConfig> = runtime_config_paths
        .iter()
        .map(|path| LLMRuntimeConfig::from_path(path))
        .collect::<Result<Vec<_>, _>>()?;

    let mut service = LLMService::from_runtime_configs(&configs);

    for config in configs {
        let model_name = config.model_config.name.clone();

        tracing::info!("Activating model: {}", model_name);

        service.activate(model_name.clone())?;

        let runtime = service.runtime().ok_or(Error::MissingActiveRuntime)?;

        let query = Query::Prompt {
            messages: vec![QueryMessage {
                role: "user".to_string(),
                content: format!("Hello from {}. Please echo just this message.", model_name),
            }],
            tools: vec![],
            config: Some(QueryConfig::default()),
            chunk_size: Some(25),
            timestamp: None,
        };

        runtime.send_stream(query)?;

        while let Ok(message) = runtime.recv_stream() {
            assert!(matches!(message, Query::Chunk { .. }));
            break;
        }
    }

    Ok(())
}
