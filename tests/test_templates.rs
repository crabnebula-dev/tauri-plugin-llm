use std::fs::File;

use tauri_plugin_llm::{TemplateProcessor, TokenizerConfig};

#[test]
fn test_qwen3_chat_go_template() {
    let chat_template_file_contents = std::fs::read_to_string("./models/Qwen3-4B-GGUF/template")
        .expect("Failed to read chat template file");

    let input_json = serde_json::json!(
        {   
            "System" : "You are a helpful assistent. Do not explain anything. Return just what the user is asking for",
            "Messages" : [
                { "Role" : "user", "Content" : "Return a question that asks for the current time." },
            ]
        }

    )
    .to_string();

    let tmpl_proc = TemplateProcessor::new(tauri_plugin_llm::TemplateType::Go);
    let result = tmpl_proc.render(&chat_template_file_contents, &input_json);

    assert!(result.is_ok(), "{:?}", result);

    let result = result.unwrap();
}

#[test]
fn test_raw_jinja_template() {
    let chat_template_file_contents = File::open("tests/fixtures/test_jinja_template.json")
        .expect("Failed to read chat template file");

    let tokenizer_config : TokenizerConfig = serde_json::from_reader(&chat_template_file_contents).expect("Failed to deserialize TokenizerConfig");

    let input_json = serde_json::json!(
        {   
            "messages" : [
                { "role" : "user", "content" : "Return a question that asks for the current time." },
                { "role" : "system", "content" : "You are a helpful assistent. Do not explain anything. Return just what the user is asking for",}
            ],
            "tools" : [
                {
                    "name": "get_weather",
                    "title": "Weather Information Provider",
                    "description": "Get current weather information for a location",
                    "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                        "type": "string",
                        "description": "City name or zip code"
                        }
                    },
                    "required": ["location"]
                    }
                },
                {
                    "name": "get_location",
                    "title": "Information about the current location",
                    "description": "Get current information for a location",
                    "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                        "type": "string",
                        "description": "City name or zip code"
                        }
                    },
                    "required": ["location"]
                    }
                }
            ]
        }
    )
    .to_string();

    let tmpl_proc = TemplateProcessor::new(tauri_plugin_llm::TemplateType::Jinja);
    let result = tmpl_proc.render(&tokenizer_config.chat_template.unwrap(), &input_json);

    assert!(result.is_ok(), "{:?}", result);
    println!("Template: \n{}", result.unwrap());
}