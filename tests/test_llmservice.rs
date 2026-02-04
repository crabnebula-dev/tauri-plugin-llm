use tauri_plugin_llm::{Error, LLMRuntimeConfig, LLMService, Query, QueryConfig, QueryMessage};

#[tokio::test]
async fn test_add_config_at_runtime() -> Result<(), Error> {
    // Load initial mock config
    let mock_config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_mock.json")?;

    // Create service with one config
    let mut service = LLMService::from_runtime_configs(&[mock_config.clone()]);

    // Verify initial model is available
    let models = service.list_models();
    assert_eq!(models.len(), 1);
    assert!(models.contains(&"Mock".to_string()));

    // Clone the config and modify the name to "Mock2"
    let mut mock2_config = mock_config.clone();
    mock2_config.model_config.name = "Mock2".to_string();

    // Serialize the new config to JSON string
    let mock2_config_json = serde_json::to_string(&mock2_config)?;

    // Add the new config at runtime
    service.add_config(mock2_config_json)?;

    // Verify both models are now available
    let models = service.list_models();
    assert_eq!(models.len(), 2);
    assert!(models.contains(&"Mock".to_string()));
    assert!(models.contains(&"Mock2".to_string()));

    // Activate the original Mock model
    service.activate("Mock".to_string())?;

    let runtime = service.runtime().ok_or(Error::MissingActiveRuntime)?;

    let query = Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello from Mock".to_string(),
        }],
        tools: vec![],
        config: Some(QueryConfig {
            generate_num_samples: 100,
            temperature: None,
            model: None,
        }),
        chunk_size: Some(10),
        timestamp: None,
    };

    runtime.send_stream(query)?;

    // Receive at least one chunk to verify it works
    let message = runtime.recv_stream()?;
    assert!(matches!(message, Query::Chunk { .. }));

    // Switch to Mock2
    service.activate("Mock2".to_string())?;

    let runtime = service.runtime().ok_or(Error::MissingActiveRuntime)?;

    let query2 = Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello from Mock2".to_string(),
        }],
        tools: vec![],
        config: Some(QueryConfig {
            generate_num_samples: 100,
            temperature: None,
            model: None,
        }),
        chunk_size: Some(10),
        timestamp: None,
    };

    runtime.send_stream(query2)?;

    // Receive at least one chunk to verify Mock2 works
    let message = runtime.recv_stream()?;
    assert!(matches!(message, Query::Chunk { .. }));

    Ok(())
}
