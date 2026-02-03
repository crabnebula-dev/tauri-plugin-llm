use std::fs::File;
use tauri_plugin_llm::{TemplateProcessor, TokenizerConfig};

#[test]
#[ignore = "Incomplete test. Removing the Go dependency also removed the ability to render Go templates"]
fn test_qwen3_chat_go_template() {
    let _chat_template_file_contents = std::fs::read_to_string("./models/Qwen3-4B-GGUF/template")
        .expect("Failed to read chat template file");

    let _input_json = std::fs::read_to_string("tests/fixtures/test_go_qwen3_input_data.json")
        .expect("Could not read test input data");
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

#[test]
fn test_raw_jinja_template_tera() -> anyhow::Result<()> {
    use minijinja::Environment;

    let chat_template_file_contents = File::open("tests/fixtures/test_jinja_template.json")
        .expect("Failed to read chat template file");

    let tokenizer_config: TokenizerConfig = serde_json::from_reader(&chat_template_file_contents)
        .expect("Failed to deserialize TokenizerConfig");

    let input_json = std::fs::read_to_string("tests/fixtures/test_jinja_input_data.json")
        .expect("Failed to read chat template input data");

    let real_template = tokenizer_config.chat_template.unwrap();
    let real_template = unescape(&real_template)?;
    let real_template = unescape(&unescape(&real_template)?)?;

    println!("{real_template}");

    // ---

    let mut proc = Environment::new();
    let result = proc.add_template("jinja", &real_template);
    assert!(result.is_ok(), "{result:#?}");

    let template = proc.get_template("jinja")?;

    let result = template.render(input_json);
    assert!(result.is_ok(), "{result:#?}");

    Ok(())
}

fn unescape(input: &str) -> anyhow::Result<String> {
    let mut result = String::new();
    let mut chars = input.chars();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some(c) => {
                    result.push('\\');
                    result.push(c);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(ch);
        }
    }

    Ok(result)
}
