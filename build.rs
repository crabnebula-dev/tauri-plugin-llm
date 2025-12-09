const COMMANDS: &[&str] = &["ping", "send_message", "retry_recv"];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .ios_path("ios")
        .build();
}
