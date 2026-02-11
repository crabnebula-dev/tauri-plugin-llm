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
#[ignore = "Load the Qwen3 model first, then run the test manually"]
async fn test_runtime_local_qwen3_safetensors() -> Result<(), Error> {
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
        tools: vec![],
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

#[tokio::test]
#[ignore = "Load the LLama3.2 model first, then run the test manually"]
async fn test_runtime_llama_3_2_3b_instruct() -> Result<(), Error> {
    enable_logging();

    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_llama3.config.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    let result = runtime.send_stream(Query::Prompt {
        messages: vec![
            QueryMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant. Answer questions concisely.".to_string(),
            },
            QueryMessage {
                role: "user".to_string(),
                content: "Just echo 'hello, World!'".to_string(),
            },
        ],
        tools: vec![],
        max_tokens: Some(50),
        temperature: Some(0.0),
        top_k: None,
        top_p: None,
        think: false,
        stream: true,
        model: None,
        penalty: Some(1.5), // Stronger repetition penalty
        seed: None,
        sampling_config: Some(tauri_plugin_llm::SamplingConfig::ArgMax), // Greedy sampling
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
    tracing::debug!("Received LLama3.2 Response: {s}");

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
        max_tokens: None,
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
    }) {
        while let Ok(message) = runtime.recv_stream() {
            assert!(matches!(message, Query::Chunk { .. } | Query::End { .. }));
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
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            think: false,
            stream: true,
            model: None,
            penalty: None,
            seed: None,
            sampling_config: None,
            chunk_size: Some(25),
            timestamp: None,
        },
        Query::Prompt {
            messages: vec![QueryMessage {
                role: "user".to_string(),
                content: "Message #2".to_string(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            think: false,
            stream: true,
            model: None,
            penalty: None,
            seed: None,
            sampling_config: None,
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
                matches!(message, Query::Chunk { .. } | Query::End { .. }),
                "{message:?}"
            );

            // Wait for Query::End before processing the next query to avoid
            // dropping the channel while the worker is still sending
            if matches!(message, Query::End { .. }) {
                break;
            }
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
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            think: false,
            stream: true,
            model: None,
            penalty: None,
            seed: None,
            sampling_config: None,
            chunk_size: Some(25),
            timestamp: None,
        },
        Query::Prompt {
            messages: vec![QueryMessage {
                role: "user".to_string(),
                content: "Just echo This Message 2".to_string(),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            think: false,
            stream: true,
            model: None,
            penalty: None,
            seed: None,
            sampling_config: None,
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
        let model_name = config.name.clone();

        tracing::info!("Activating model: {}", model_name);

        service.activate(model_name.clone())?;

        let runtime = service.runtime().ok_or(Error::MissingActiveRuntime)?;

        let query = Query::Prompt {
            messages: vec![QueryMessage {
                role: "user".to_string(),
                content: format!("Hello from {}. Please echo just this message.", model_name),
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            think: false,
            stream: true,
            model: None,
            penalty: None,
            seed: None,
            sampling_config: None,
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

fn random_query_message() -> impl Strategy<Value = QueryMessage> {
    (prop_oneof![Just("system"), Just("assistant")], ".{1,1000}").prop_map(|(role, content)| {
        QueryMessage {
            role: role.to_string(),
            content,
        }
    })
}

fn random_prompt_with_user_message() -> impl Strategy<Value = (QueryMessage, Vec<QueryMessage>)> {
    (
        // guarantee exactly one "user" message
        ".{1,1000}".prop_map(|content| QueryMessage {
            role: "user".to_string(),
            content,
        }),
        // 0–4 additional messages with any role
        prop::collection::vec(random_query_message(), 0..4),
    )
        .prop_map(|(user_msg, mut others)| {
            others.insert(0, user_msg.clone());
            (user_msg, others)
        })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::WithSource("proptest-regressions"))),
        .. ProptestConfig::default()
    })]
    #[test]
    fn test_mock_token_usage(messages in random_prompt_with_user_message()) {
        let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_mock.json").unwrap();
        let mut runtime = LLMRuntime::from_config(config).unwrap();
        runtime.run_stream().unwrap();

        let (user_msg, messages) = messages;
        let expected_prompt_tokens = serde_json::to_vec(&messages).unwrap().len();
        let user_content = user_msg.content.clone();
        let expected_completion_tokens = user_content.len();

        runtime
            .send_stream(Query::Prompt {
                messages,
                tools: vec![],
                max_tokens: None,
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
            })
            .unwrap();

        while let Ok(message) = runtime.recv_stream() {
            match message {
                Query::Chunk { .. } => {}
                Query::End { usage } => {

                    let usage = usage.expect("No token usage generated");
                    prop_assert_eq!(usage.prompt_tokens, expected_prompt_tokens);
                    prop_assert_eq!(usage.completion_tokens, expected_completion_tokens);
                    prop_assert_eq!(usage.total_tokens, expected_prompt_tokens + expected_completion_tokens);

                    break;
                }
                other => panic!("Unexpected message: {other:?}"),
            }
        }

    }
}

#[test]
fn test_option_none() {
    let mut tools: Option<Vec<u8>> = Some(vec![]);

    let tools_inner = tools.filter(|v| !v.is_empty());
    assert!(tools_inner.is_none())
}
