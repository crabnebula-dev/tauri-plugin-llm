use proptest::prelude::*;
use std::path::PathBuf;
use tauri_plugin_llm::{GenerationSeed, LLMRuntimeConfig, Query, SamplingConfig};

pub fn random_generation_seed() -> impl Strategy<Value = GenerationSeed> {
    prop_oneof![
        any::<usize>().prop_map(GenerationSeed::Fixed),
        Just(GenerationSeed::Random),
    ]
}

pub fn random_sampling_config() -> impl Strategy<Value = SamplingConfig> {
    prop_oneof![
        Just(SamplingConfig::All),
        Just(SamplingConfig::ArgMax),
        Just(SamplingConfig::GumbelSoftmax),
        Just(SamplingConfig::TopK),
        Just(SamplingConfig::TopKThenTopP),
        Just(SamplingConfig::TopP)
    ]
}

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
        1.0f32..2.0f32,
        random_generation_seed(),
        random_sampling_config(),
        any::<bool>(),
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
                penalty,
                seed,
                sampling_config,
                verbose,
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
                    penalty,
                    seed,
                    sampling_config,
                    verbose,
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
