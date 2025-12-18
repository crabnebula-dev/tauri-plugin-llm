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
- Go >= 1.20 

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
        "tokenizer_config_file": "./models/Qwen3-4B-Instruct-2507-FP8/tokenizer.json",
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



> **Note**: _The example above shows a model present at some folder called`model`. The model itself will not be shipped with the plugin. The configuration is shown for demonstration purposes only._


The `LLMRuntime` will take to load the LLM and defer the initialization process to the actual loader. The Tauri Plugin LLM does not support all available models, yet. Internally, the model will be run in a dedicated thread to avoid blocking the main thread. Message passing eg. textual prompts  is done via channels.

```Rust

// TODO: the config is inside the file `tauri.config.json`
let config = LLMRuntimeConfig::from_path("/path/to/runtime_qwen3.config.json")?;

// Initialize the Runtime and detect available specific runtimes
let mut runtime = LLMRuntime::from_config(config)?;

// Create a channel to send model responses to and read from
let (response_tx, message_rx) = std::sync::mpsc::channel();

// Starting the runtime. This will internally initialize the model and 
// then run inference on the model on the incoming message
// Calling this function will return a sender channel to send
// messages to the model, and the worker thread handle of the runtime.
// The worker thread handle is required to be kept blocking at the end to 
// enable concurrent processing of messages
let (sender_tx, worker) = runtime.run(response_tx).await?;

/// Here we send a prompt to the model
if let Err(e) = sender_tx
    .send(LlmMessage::Prompt {
        system:"You are a helpful assistent. Your task is to echo the incoming message. Do not describe anything. ".to_string(),
        message:"Hello, World".to_string(), 
        num_samples: 200 
    })
    
{
    tracing::error!("Error sending message to model {}", e);
}

// We will keep a busy loop to read from incoming messages of the Model
loop {
    if let Ok(incoming) = message_rx.recv() {
        tracing::info!("Received Message : {:?}", incoming);

        // Here a termination signal is being send, allowing the runtime to shutdown gracefully
        sender_tx.send(LlmMessage::Exit).map_err(|_|Error::ExecutionError("".to_string()))?;
        break;
    }
}

// We need to await the runtime, until a termination signal has been send. 
worker.await.map_err(|_| Error::ExecutionError("".to_string()))?;

Ok(())

```
