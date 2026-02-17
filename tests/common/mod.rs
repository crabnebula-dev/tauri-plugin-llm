use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tracing_subscriber::{filter, layer::SubscriberExt, util::SubscriberInitExt, Layer, Registry};

/// Initializes tracing at DEBUG level.
/// Uses `try_init()` so it doesn't panic when called from multiple tests
/// in the same process.
#[allow(dead_code)]
pub fn enable_logging() {
    let verbose = tracing_subscriber::fmt::layer().with_filter(filter::LevelFilter::DEBUG);
    let _ = Registry::default().with(verbose).try_init();
}

/// Ensures that all required files for a HuggingFace model are present
/// in the local cache, downloading them if necessary.
///
/// Downloads: `config.json`, `tokenizer.json`, `tokenizer_config.json` (optional),
/// and either `model.safetensors` (single file) or sharded weights via
/// `model.safetensors.index.json`.
#[allow(dead_code)]
pub fn ensure_model_downloaded(
    model_id: &str,
    cache_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use hf_hub::api::sync::ApiBuilder;

    // Check if the model is already cached by looking for config.json
    let cache = hf_hub::Cache::new(cache_dir.to_path_buf());
    let cache_repo = cache.model(model_id.to_string());
    if cache_repo.get("config.json").is_some() && cache_repo.get("tokenizer.json").is_some() {
        // Model appears to be cached already
        return Ok(());
    }

    eprintln!(
        "Model '{model_id}' not found in cache at {}, downloading...",
        cache_dir.display()
    );

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()?;
    let repo = api.model(model_id.to_string());

    // Required files
    repo.get("config.json")?;
    repo.get("tokenizer.json")?;

    // Optional but useful
    let _ = repo.get("tokenizer_config.json");

    // Try single-file model first
    if repo.get("model.safetensors").is_err() {
        // Sharded model: download the index and each shard
        let index_path = repo.get("model.safetensors.index.json")?;
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;

        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            let shard_files: HashSet<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();

            for shard in &shard_files {
                eprintln!("  Downloading shard: {shard}");
                repo.get(shard)?;
            }
        }
    }

    eprintln!("Model '{model_id}' download complete.");
    Ok(())
}

/// RAII guard that optionally removes a model's cache directory on drop.
///
/// The HF cache layout is: `<cache_dir>/models--<org>--<model>/`
/// where `org/model` becomes `models--org--model`.
#[allow(dead_code)]
pub struct HfModelGuard {
    model_id: String,
    cache_dir: PathBuf,
    cleanup: bool,
}

impl HfModelGuard {
    #[allow(dead_code)]
    pub fn new(model_id: &str, cache_dir: PathBuf, cleanup: bool) -> Self {
        Self {
            model_id: model_id.to_string(),
            cache_dir,
            cleanup,
        }
    }
}

impl Drop for HfModelGuard {
    fn drop(&mut self) {
        if self.cleanup {
            let dir_name = format!("models--{}", self.model_id.replace('/', "--"));
            let model_dir = self.cache_dir.join(dir_name);
            if model_dir.exists() {
                eprintln!("Cleaning up model cache: {}", model_dir.display());
                if let Err(e) = std::fs::remove_dir_all(&model_dir) {
                    eprintln!(
                        "Warning: failed to clean up model cache at {}: {e}",
                        model_dir.display()
                    );
                }
            }
        }
    }
}
