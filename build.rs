use std::env;
use std::path::PathBuf;

const COMMANDS: &[&str] = &[
    "ping",
    "stream",
    "switch_model",
    "list_available_models",
    "add_configuration",
];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .ios_path("ios")
        .build();

    // Get the project root directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir);
}
