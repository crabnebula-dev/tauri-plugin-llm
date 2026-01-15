use std::{cmp::Ordering, usize, vec};

use tauri_plugin_llm::{
    runtime::LLMRuntime, Error, LLMRuntimeConfig, Query, QueryConfig, QueryMessage,
};

#[tokio::test]
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
            content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string(),
            
        },
        ],
        tools: vec![],
        config: Some(QueryConfig::default()),
        chunk_size : None,
        timestamp : None
    });

    match result {
        Ok(_) => {
            while let Ok(message) = runtime.try_recv_stream() {
                if let Query::Chunk { id, data, kind } = &message {
                    tracing::info!("Data: {:?}", data)
                }
                if let Query::End = &message { break }
            }
              
        },
        Err(_) => {
            tracing::error!("Failed sending message")
        },
    }
   
    Ok(())
}

#[tokio::test]
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
            content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ".to_string(),
            
        },
        ],
        tools: vec![],
        config: Some(QueryConfig::default()),
        chunk_size : None,
        timestamp : None
    });

    match result {
        Ok(_) => {
              
        loop {
            if let Ok(message) = runtime.try_recv_stream() {
                if let Query::Chunk { id, data, kind } = &message {
                    tracing::debug!("Data: {:?}", &data[0..32])
                }
                if let Query::End = &message { break }
            }
        }
        },
        Err(_) => {
            tracing::error!("Failed sending message")
        },
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
        loop {
            if let Ok(message) = runtime.try_recv_stream() {
                tracing::info!("Received Message : {:?}", message);
                break;
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_runtime_mock_streaming() -> Result<(), Error>{
    let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_mock.json")?;
    let mut runtime = LLMRuntime::from_config(config)?;

    runtime.run_stream()?;

    let query = Query::Prompt { messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string()}], tools: vec![], config: Some(QueryConfig::default()), chunk_size: Some(25), timestamp : None};

    let _ = runtime.send_stream(query); 

    let mut full_message = vec![];

    loop { 
        
        if let Ok(message) = runtime.try_recv_stream() {
            match message {
                tauri_plugin_llm::Query::Chunk { .. } =>  {
                    full_message.push(message);
                },
                tauri_plugin_llm::Query::End => {
                    //  reassemble the whole message
                    full_message.sort_by(|a, b| {
                        let id_a = match a {
                            tauri_plugin_llm::Query::Chunk { id, .. } => *id,
                           _ => usize::MAX
                        };

                         let id_b = match b {
                            tauri_plugin_llm::Query::Chunk { id, .. } => *id,
                           _ => usize::MAX
                        };

                        match (id_a, id_b) {
                            _ if id_a > id_b => Ordering::Greater,
                            _ if id_a < id_b => Ordering::Less,
                            _ => Ordering::Equal
                        }

                    });

                    let result = full_message.into_iter().filter_map(|q| match q {
                        tauri_plugin_llm::Query::Chunk {  data, ..} => Some(data),
                        _ => None
                    }).flatten().collect::<Vec<u8>>();

                    let result_message_string = String::from_utf8(result).expect("Failed to construct a UTF-8 String from raw bytes");

                    println!("{result_message_string}");
                    tracing::info!("Result: {result_message_string}");

                    break
                },
                tauri_plugin_llm::Query::Status { msg } => {
                    tracing::error!("Error during receiving stream message. {msg}");
                },
                
                _ => {
                    // not covered
                }
            }
        }

    }


    Ok(())
}