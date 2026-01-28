use std::{cmp::Ordering, usize, vec};

use tauri_plugin_llm::{
    runtime::LLMRuntime, Error, LLMRuntimeConfig, LLMService, Query, QueryConfig, QueryMessage,
};
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer, Registry};

#[tokio::test]
// #[ignore = "Load the Qwen3 model first, then run this test manually"]
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

    match result {
        Ok(_) => {
            while let Ok(message) = runtime.recv_stream() {
                if let Query::Chunk { data, .. } = &message {
                    tracing::info!("Data: {:?}", data)
                }
                if let Query::End = &message {
                    break;
                }
            }
        }
        Err(_) => {
            tracing::error!("Failed sending message")
        }
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

    match result {
        Ok(_) => {
            while let Ok(message) = runtime.recv_stream() {
                if let Query::Chunk { data, .. } = &message {
                    tracing::debug!("Data: {:?}", &data[0..32])
                }
                if let Query::End = &message {
                    break;
                }
            }
        }
        Err(_) => {
            tracing::error!("Failed sending message")
        }
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
            tracing::info!("Received Message : {:?}", message);
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

        let mut full_message = vec![];

        loop {
            match runtime.recv_stream() {
                Ok(message) => {
                    match message {
                        tauri_plugin_llm::Query::Chunk { .. } => {
                            full_message.push(message);
                        }
                        tauri_plugin_llm::Query::End => {
                            //  reassemble the whole message
                            full_message.sort_by(|a, b| {
                                let id_a = match a {
                                    tauri_plugin_llm::Query::Chunk { id, .. } => *id,
                                    _ => usize::MAX,
                                };

                                let id_b = match b {
                                    tauri_plugin_llm::Query::Chunk { id, .. } => *id,
                                    _ => usize::MAX,
                                };

                                match (id_a, id_b) {
                                    _ if id_a > id_b => Ordering::Greater,
                                    _ if id_a < id_b => Ordering::Less,
                                    _ => Ordering::Equal,
                                }
                            });

                            let result = full_message
                                .into_iter()
                                .filter_map(|q| match q {
                                    tauri_plugin_llm::Query::Chunk { data, .. } => Some(data),
                                    _ => None,
                                })
                                .flatten()
                                .collect::<Vec<u8>>();

                            let result_message_string = String::from_utf8(result)
                                .expect("Failed to construct a UTF-8 String from raw bytes");

                            tracing::info!("Result: {result_message_string}");

                            break;
                        }
                        tauri_plugin_llm::Query::Status { msg } => {
                            panic!("Error during receiving stream message. {msg}");
                        }

                        _ => {}
                    }
                }
                Err(error) => {
                    panic!("Error trying to get message: {error}");
                }
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_runtime_qwen3_streaming() -> Result<(), Error> {
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;
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

        let mut full_message = vec![];

        loop {
            match runtime.recv_stream() {
                Ok(message) => {
                    match message {
                        tauri_plugin_llm::Query::Chunk { .. } => {
                            full_message.push(message);
                        }
                        tauri_plugin_llm::Query::End => {
                            //  reassemble the whole message
                            full_message.sort_by(|a, b| {
                                let id_a = match a {
                                    tauri_plugin_llm::Query::Chunk { id, .. } => *id,
                                    _ => usize::MAX,
                                };

                                let id_b = match b {
                                    tauri_plugin_llm::Query::Chunk { id, .. } => *id,
                                    _ => usize::MAX,
                                };

                                match (id_a, id_b) {
                                    _ if id_a > id_b => Ordering::Greater,
                                    _ if id_a < id_b => Ordering::Less,
                                    _ => Ordering::Equal,
                                }
                            });

                            let result = full_message
                                .into_iter()
                                .filter_map(|q| match q {
                                    tauri_plugin_llm::Query::Chunk { data, .. } => Some(data),
                                    _ => None,
                                })
                                .flatten()
                                .collect::<Vec<u8>>();

                            let result_message_string = String::from_utf8(result)
                                .expect("Failed to construct a UTF-8 String from raw bytes");

                            tracing::info!("Result: {result_message_string}");

                            break;
                        }
                        tauri_plugin_llm::Query::Status { msg } => {
                            panic!("Error during receiving stream message. {msg}");
                        }

                        _ => {}
                    }
                }
                Err(error) => {
                    panic!("Error trying to get message: {error}");
                }
            }
        }
    }

    Ok(())
}

#[tokio::test]
#[ignore = "Run this test explicitly to avoid using real model weights"]
async fn test_switching_runtimes() -> Result<(), Error> {
    // handle tracing explicitly here
    {
        let verbose = tracing_subscriber::fmt::layer().with_filter(filter::LevelFilter::DEBUG);
        Registry::default().with(verbose).init();
    }

    let runtime_config_paths = [
        "tests/fixtures/test_runtime_mock.json",
        "tests/fixtures/test_runtime_qwen3.config.json",
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
            if let Query::Chunk { data, .. } = &message {
                let message_str =
                    String::from_utf8(data.to_vec()).expect("Canot read data as UTF-8 String");

                tracing::info!("Received response from {}: {:?}", model_name, message_str);
            }

            if let Query::End = message {
                break;
            }
        }
    }

    Ok(())
}
