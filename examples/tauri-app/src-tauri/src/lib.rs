#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut builder = tauri::Builder::default().plugin(tauri_plugin_os::init());

    #[cfg(debug_assertions)]
    {
        builder = builder.plugin(tauri_plugin_automation::init());
    }

    builder
        .plugin(tauri_plugin_llm::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
