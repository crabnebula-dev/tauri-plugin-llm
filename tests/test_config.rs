use proptest::prelude::*;
use std::path::PathBuf;
use tauri_plugin_llm::llmconfig::{GenerationSeed, LLMRuntimeConfig, ModelConfig, ModelFileType};

pub fn random_model_file_type() -> impl Strategy<Value = ModelFileType> {
    prop_oneof![
        Just(ModelFileType::GGUF),
        Just(ModelFileType::Safetensors),
        Just(ModelFileType::Pickle),
    ]
}

pub fn random_generation_seed() -> impl Strategy<Value = GenerationSeed> {
    prop_oneof![
        any::<usize>().prop_map(GenerationSeed::Fixed),
        Just(GenerationSeed::Random),
    ]
}

pub fn random_model_config() -> impl Strategy<Value = ModelConfig> {
    (
        1usize..100usize,
        0.0f32..1.0f32,
        0.1f32..2.0f32,
        1.0f32..2.0f32,
        any::<String>(),
        random_model_file_type(),
        random_generation_seed(),
    )
        .prop_map(
            |(top_k, top_p, temperature, penalty, name, file_type, seed)| ModelConfig {
                top_k,
                top_p,
                temperature,
                penalty,
                name,
                file_type,
                seed,
            },
        )
}

pub fn random() -> impl Strategy<Value = LLMRuntimeConfig> {
    (
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
        random_model_config(),
    )
        .prop_map(
            |(
                tokenizer_config_file,
                model_config_file,
                model_index_file,
                model_file,
                model_dir,
                model_config,
            )| {
                LLMRuntimeConfig {
                    tokenizer_config_file,
                    model_config_file,
                    model_index_file,
                    model_file,
                    model_dir,
                    model_config,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    #[test]
    fn test_runtime_config(input in random()) {
        // test roundtrip
        let serialized = serde_json::to_string(&input);
        assert!(serialized.is_ok(), "{:?}", serialized);

        let result = serde_json::from_str::<LLMRuntimeConfig>(&serialized.unwrap());
        assert!(result.is_ok(), "{:?}", result)
    }
}
