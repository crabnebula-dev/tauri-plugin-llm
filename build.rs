use std::env;
use std::path::PathBuf;
use std::process::Command;

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
}
