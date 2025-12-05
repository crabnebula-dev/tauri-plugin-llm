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
        0.0f64..1.0f64,
        0.1f64..2.0f64,
        1.0f32..2.0f32,
        any::<String>(),
        random_model_file_type(),
        random_generation_seed(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(top_k, top_p, temperature, penalty, name, file_type, seed, thinking, streaming)| {
                ModelConfig {
                    top_k,
                    top_p,
                    temperature,
                    penalty,
                    name,
                    file_type,
                    seed,
                    thinking,
                    streaming,
                }
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
        any::<bool>(),
    )
        .prop_map(
            |(
                tokenizer_config_file,
                model_config_file,
                model_index_file,
                model_file,
                model_dir,
                model_config,
                verbose,
            )| {
                LLMRuntimeConfig {
                    tokenizer_config_file,
                    model_config_file,
                    model_index_file,
                    model_file,
                    model_dir,
                    model_config,
                    verbose,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1))]
    #[test]
    fn test_runtime_config(input in random()) {
        // test roundtrip
        let serialized = serde_json::to_string_pretty(&input);
        assert!(serialized.is_ok(), "{:?}", serialized);

        println!("{}", serialized.as_ref().unwrap());

        let result = serde_json::from_str::<LLMRuntimeConfig>(&serialized.unwrap());
        assert!(result.is_ok(), "{:?}", result)
    }
}
