use std::env::current_dir;
use tauri_plugin_llm::{
    LLMPluginConfig, {LLMRuntimeConfig, ModelConfig},
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut builder = tauri::Builder::default().plugin(tauri_plugin_os::init());

    #[cfg(debug_assertions)]
    {
        builder = builder.plugin(tauri_plugin_automation::init());
    }

    #[cfg(target_os = "macos")]
    {
        let basedir = current_dir()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("models");
        let config = LLMPluginConfig {
            llmconfig: LLMRuntimeConfig {
                tokenizer_config_file: None,
                tokenizer_file: Some(basedir.join("Qwen3-4B-Instruct-2507-FP8/tokenizer.json")),
                model_config_file: Some(basedir.join("Qwen3-4B-Instruct-2507-FP8/config.json")),
                model_index_file: None,
                model_file: Some(basedir.join("Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf")),
                model_dir: Some(basedir.join("Qwen3-4B-GGUF/")),
                model_config: ModelConfig {
                    top_k: 20,
                    top_p: 0.8,
                    temperature: 0.4,
                    name: "Qwen3-4B-GGUF".to_string(),
                    file_type: tauri_plugin_llm::ModelFileType::GGUF,
                    penalty: 1.0,
                    seed: tauri_plugin_llm::GenerationSeed::Random,
                    thinking: false,
                    streaming: true,
                    sampling_config: tauri_plugin_llm::SamplingConfig::All,
                },
                verbose: true,
                template_file: None,
            },
        };
        builder = builder.plugin(tauri_plugin_llm::Builder::new().config(config).build())
    }

    #[cfg(not(target_os = "macos"))]
    {
        builder = builder.plugin(tauri_plugin_llm::init());
    }

    builder
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
