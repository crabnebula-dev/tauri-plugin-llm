use std::fs::File;
use tauri_plugin_llm::{TemplateProcessor, TokenizerConfig};

#[test]
fn test_qwen3_chat_go_template() {
    let chat_template_file_contents = std::fs::read_to_string("./models/Qwen3-4B-GGUF/template")
        .expect("Failed to read chat template file");

    let input_json = std::fs::read_to_string("tests/fixtures/test_go_qwen3_input_data.json")
        .expect("Could not read test input data");

    let tmpl_proc = TemplateProcessor::new(tauri_plugin_llm::TemplateType::Go);
    let result = tmpl_proc.render(&chat_template_file_contents, &input_json);

    assert!(result.is_ok(), "{:?}", result);
}

#[test]
fn test_raw_jinja_template() {
    let chat_template_file_contents = File::open("tests/fixtures/test_jinja_template.json")
        .expect("Failed to read chat template file");

    let tokenizer_config: TokenizerConfig = serde_json::from_reader(&chat_template_file_contents)
        .expect("Failed to deserialize TokenizerConfig");

    let input_json = std::fs::read_to_string("tests/fixtures/test_jinja_input_data.json")
        .expect("Failed to read chat template input data");

    let tmpl_proc = TemplateProcessor::new(tauri_plugin_llm::TemplateType::Jinja);
    let result = tmpl_proc.render(&tokenizer_config.chat_template.unwrap(), &input_json);

    assert!(result.is_ok(), "{:?}", result);
}
