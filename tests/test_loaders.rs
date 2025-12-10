use tauri_plugin_llm::{loaders::safetensor::IndexFile, Error};

#[test]
fn test_load_safetensors_index_file() -> Result<(), Error> {
    let mut index_file = IndexFile::from_path("tests/fixtures/test_model.safetensors.index.json")?;

    let files = index_file.files("models/Llama-3.2-3B-instruct");

    for f in files {
        let name = f.file_name().unwrap();
        assert!(name.to_str().unwrap().ends_with("safetensors"))
    }

    Ok(())
}
