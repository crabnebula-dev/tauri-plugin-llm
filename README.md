# Tauri Plugin LLM

This Tauri plugin allows loading and running inference on various large language models.

| Platform | Supported |
| -------- | --------- |
| Linux    | ✓         |
| Windows  | ✓         |
| macOS    | ✓         |
| Android  | ?         |
| iOS      | ?         |

## Requirements

- Rust >= 1.77

## Install

## Supported Models

| Model | GGUF | Safetensors |
| ----- | ---- | ----------- |
| Llama 3.x | ✓ | ✓ |
| Qwen3 | ✓ | ✓ |

and much more to follow...

## Usage

The plugin is not bundled with any LLM. The LLM must be shipped separately. In order to load an LLM, a specific configuration must be provided for the plugin.

### Configuration

Add the plugin configuration to your `tauri.conf.json`. The `llmconfig` section defines the model to load:

```json
{
  "plugins": {
    "llm": {
      "llmconfig": {
        "name": "Qwen3-4B-GGUF",
        "tokenizer_file": "./models/Qwen3-4B-GGUF/tokenizer.json",
        "model_file": "./models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
      }
    }
  }
}
```

For Safetensors models (sharded weights), use `model_index_file` and `model_dir` instead of `model_file`:

```json
{
  "plugins": {
    "llm": {
      "llmconfig": {
        "name": "Local-Qwen--Qwen3-4B-Instruct-2507",
        "tokenizer_file": "./models/Qwen3-4B-Instruct-2507/tokenizer.json",
        "tokenizer_config_file": "./models/Qwen3-4B-Instruct-2507/tokenizer_config.json",
        "model_config_file": "./models/Qwen3-4B-Instruct-2507/config.json",
        "model_index_file": "./models/Qwen3-4B-Instruct-2507/model.safetensors.index.json",
        "model_dir": "./models/Qwen3-4B-Instruct-2507/"
      }
    }
  }
}
```

> **Note**: The model files are not shipped with the plugin. You must download them separately.

#### LLMRuntimeConfig Fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `name` | `string` | Model identifier, used for model selection |
| `tokenizer_file` | `string?` | Path to `tokenizer.json` |
| `tokenizer_config_file` | `string?` | Path to `tokenizer_config.json` |
| `model_config_file` | `string?` | Path to `config.json` |
| `model_index_file` | `string?` | Path to `model.safetensors.index.json` (implies Safetensors format) |
| `model_file` | `string?` | Path to model file, e.g. `.gguf` (implies GGUF format) |
| `model_dir` | `string?` | Path to model directory for sharded Safetensors files |
| `template_file` | `string?` | Path to a custom chat template file |

### Rust API

The `LLMRuntime` loads the model lazily on the first prompt and runs inference in a dedicated thread.

```rust
let config = LLMRuntimeConfig::from_path("tests/fixtures/test_runtime_qwen3.config.json")?;
let mut runtime = LLMRuntime::from_config(config)?;

runtime.run_stream()?;

runtime.send_stream(Query::Prompt {
    messages: vec![
        QueryMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        },
        QueryMessage {
            role: "user".to_string(),
            content: "Hello, World".to_string(),
        },
    ],
    tools: vec![],
    max_tokens: Some(200),
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
})?;

while let Ok(message) = runtime.recv_stream() {
    match message {
        Query::Chunk { data, .. } => {
            print!("{}", String::from_utf8_lossy(&data));
        }
        Query::End { usage } => {
            if let Some(usage) = usage {
                println!("\nTokens: {} prompt, {} completion",
                    usage.prompt_tokens, usage.completion_tokens);
            }
            break;
        }
        _ => break,
    }
}
```

#### Query::Prompt Fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `messages` | `Vec<QueryMessage>` | Chat messages (`role` + `content`) |
| `tools` | `Vec<String>` | Tool definitions (MCP-compatible JSON) |
| `max_tokens` | `usize?` | Maximum tokens to generate |
| `temperature` | `f32?` | Sampling temperature |
| `top_k` | `f32?` | Top-K sampling parameter |
| `top_p` | `f32?` | Top-P (nucleus) sampling parameter |
| `think` | `bool` | Enable thinking/reasoning mode |
| `stream` | `bool` | Enable streaming output |
| `model` | `string?` | Target model name (for multi-model setups) |
| `penalty` | `f32?` | Repetition penalty (defaults to 1.1) |
| `seed` | `GenerationSeed?` | `"Random"` (default) or `{ "Fixed": N }` |
| `sampling_config` | `SamplingConfig?` | Sampling strategy: `"ArgMax"`, `"All"` (default), `"TopK"`, `"TopP"`, `"TopKThenTopP"`, `"GumbelSoftmax"` |
| `chunk_size` | `usize?` | Number of tokens per streamed chunk |
| `timestamp` | `u64?` | Optional timestamp for the request |

### TypeScript / Frontend API

```typescript
import { LLMStreamListener } from "tauri-plugin-llm-api";

const listener = new LLMStreamListener();

await listener.setup({
  onData: (id, data, timestamp) => {
    console.log(new TextDecoder().decode(data));
  },
  onError: (msg) => console.error("Error:", msg),
  onEnd: (usage) => {
    if (usage) {
      console.log(`Tokens: ${usage.prompt_tokens} prompt, ${usage.completion_tokens} completion`);
    }
  },
});

await listener.stream({
  type: "Prompt",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello!" },
  ],
  tools: [],
  max_tokens: 200,
  stream: true,
});

// Switch models at runtime
const models = await listener.listAvailableModels();
await listener.switchModel("Qwen3-4B-GGUF");

// Add a new model configuration dynamically
await listener.addConfiguration(JSON.stringify({
  name: "Llama-3.2-3B",
  tokenizer_file: "/path/to/tokenizer.json",
  model_file: "/path/to/model.gguf",
}));

// Clean up when done
listener.teardown();
```

## License

This software is licensed under the [PolyForm Noncommercial License 1.0.0](./LICENSE_POLYFORM-NONCOM).
