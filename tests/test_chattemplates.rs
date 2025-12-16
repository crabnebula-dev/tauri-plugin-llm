use tauri_plugin_llm::{TemplateProcessor};

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

    let tmpl_proc = TemplateProcessor::new(tauri_plugin_llm::TemplateType::GoTemplate);
    let result = tmpl_proc.render(&chat_template_file_contents, &input_json);

    assert!(result.is_ok(), "{:?}", result);

    let result = result.unwrap();
    println!("{result}");
}
