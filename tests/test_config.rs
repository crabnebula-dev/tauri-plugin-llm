use proptest::prelude::*;
use std::path::PathBuf;
use tauri_plugin_llm::{
    GenerationSeed, LLMRuntimeConfig, ModelConfig, ModelFileType, Query, QueryConfig, QueryMessage,
    SamplingConfig,
};

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
        random_sampling_config(),
    )
        .prop_map(
            |(
                top_k,
                top_p,
                temperature,
                penalty,
                name,
                file_type,
                seed,
                thinking,
                streaming,
                sampling_config,
            )| {
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
                    sampling_config,
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
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
        random_model_config(),
        any::<bool>(),
        "[a-z]{3,10}/[a-z]{3,10}"
            .prop_map(PathBuf::from)
            .prop_map(Some),
    )
        .prop_map(
            |(
                tokenizer_file,
                tokenizer_config_file,
                model_config_file,
                model_index_file,
                model_file,
                model_dir,
                model_config,
                verbose,
                template,
            )| {
                LLMRuntimeConfig {
                    tokenizer_file,
                    tokenizer_config_file,
                    model_config_file,
                    model_index_file,
                    model_file,
                    model_dir,
                    model_config,
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
    let query = Query::Prompt {
        messages: vec![QueryMessage {
            role: "user".to_string(),
            content: "Hello, World!".to_string(),
        }],
        tools: vec![],
        config: Some(QueryConfig::default()),
    };

    let json = serde_json::to_string(&query).unwrap();
    println!("{json}");

    let json = serde_json::json!(
        {
        "type": "Prompt",
        "messages": [],
        "tools": [],
        "config": null
        }
    )
    .to_string();

    let result: Result<Query, _> = serde_json::from_str(json.as_str());
    assert!(result.is_ok(), "{:?}", result);

    println!("{:?}", result.unwrap())
}
