# Tauri Plugin LLM

This Tauri plugin allows loading and running inference on various large language models.

| Platform | Supported |
| -------- | --------- |
| Linux    | ✓        |
| Windows  | ✓        |
| macOS    | ✓         |
| Android  | ?         |
| iOS      | ?         |

# Requirements

- Rust >= 1.77

# Install

# Usage

The plugin is not bundled with any LLM. The LLM must be shipped separately. In order to load an LLM, a specific configuration must be provided for the plugin. The JSON example configuration sets `Qwen3-4B-GGUF` as model to be loaded. 

File `tauri.conf.json` :
```json
{
  "build": { ... },
  "tauri": { ... },
  "plugins": {
    "tauri-plugin-llm": {
        "tokenizer_file": "./models/Qwen3-4B-Instruct-2507-FP8/tokenizer.json",
        "tokenizer_config_file": "./models/Qwen3-4B-Instruct-2507-FP8/tokenizer_config.json",
        "model_config_file": "./models/Qwen3-4B-Instruct-2507-FP8/config.json",
        "model_index_file": "",
        "model_file": "./models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
        "model_dir": "./models/Qwen3-4B-GGUF/",
        "model_config": {
            "top_k": 20,
            "top_p": 0.8,
            "temperature": 0.4,
            "name": "Qwen3-4B-GGUF",
            "file_type": "GGUF",
            "penalty": 1.0,
            "seed": "Random",
            "thinking": false,
            "streaming": true
        },
        "verbose": true
    }
  }
}

```

> **Note**: _The example above shows a model present at some folder called `model`. The model itself will not be shipped with the plugin._


The `LLMRuntime` will try to load an LLM and defer the initialization process to the actual loader. The Tauri Plugin LLM does not support all available models, yet. Internally, the model will be run in a dedicated thread to avoid blocking the main thread. 

```Rust
// For demonstration purposes we load the configuration from the tests. 
// Normally you would load the tauri config
let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;

// Initialize the Runtime and detect available specific runtimes
let mut runtime = LLMRuntime::from_config(config)?;

// Starting the runtime. This will internally initialize the model and 
// then run inference on the model on the incoming message.
runtime.run_stream()?;

// You would run the following code in a loop. This example shows 
// how to get individual chunks of the response. 
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

```



