mod common;

use proptest::prelude::*;
use std::path::PathBuf;
use tauri_plugin_llm::{LLMRuntimeConfig, Query};
use tauri_plugin_llm_macros::hf_test;

pub fn random() -> impl Strategy<Value = LLMRuntimeConfig> {
    (
        any::<String>(),
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
    )
        .prop_map(
            |(
                name,
                tokenizer_file,
                tokenizer_config_file,
                model_config_file,
                model_index_file,
                model_file,
                model_dir,
                template,
            )| {
                LLMRuntimeConfig {
                    name,
                    tokenizer_file,
                    tokenizer_config_file,
                    model_config_file,
                    model_index_file,
                    model_file,
                    model_dir,
                    template_file: template,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    #[test]
    fn test_runtime_config(input in random()) {
        let serialized = serde_json::to_string_pretty(&input);
        assert!(serialized.is_ok(), "{:?}", serialized);

        let result = serde_json::from_str::<LLMRuntimeConfig>(&serialized.unwrap());
        assert!(result.is_ok(), "{:?}", result)
    }
}

#[test]
fn test_deserialize_default() {
    let json = serde_json::json!(
        {
        "type": "Prompt",
        "messages": [],
        "tools": [],
        "config": null,
        "chunk_size" : null
        }
    )
    .to_string();

    let result: Result<Query, _> = serde_json::from_str(json.as_str());
    assert!(result.is_ok(), "{:?}", result);
}

#[hf_test(
    model = "meta-llama/Llama-3.2-3B-Instruct",
    cleanup = false,
    cache_dir = "/Volumes/MLM/huggingface"
)]
fn test_load_llama32_config_from_hf_cache(config: LLMRuntimeConfig) {
    assert!(config.tokenizer_file.is_some());
    assert!(config.model_config_file.is_some());
    assert!(config.model_index_file.is_some() || config.model_file.is_some());
    Ok(())
}

#[hf_test(
    model = "Qwen/Qwen3-4B-Instruct-2507",
    cleanup = false,
    cache_dir = "/Volumes/MLM/huggingface"
)]
fn test_load_qwen3_config_from_hf_cache(config: LLMRuntimeConfig) {
    assert!(config.tokenizer_file.is_some());
    assert!(config.model_config_file.is_some());
    Ok(())
}

#[hf_test(
    model = "google/gemma-3-4b-it",
    cleanup = false,
    cache_dir = "/Volumes/MLM/huggingface"
)]
fn test_load_gemma3_4b_from_hf_cache(config: LLMRuntimeConfig) {
    assert!(config.tokenizer_file.is_some());
    assert!(config.model_config_file.is_some());
    Ok(())
}
