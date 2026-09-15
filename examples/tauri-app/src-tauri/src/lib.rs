#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tauri_plugin_llm=debug".into()),
        )
        .init();

    let mut builder = tauri::Builder::default()
        .runtime(tauri_runtime_wry::Wry::default())
        .plugin(tauri_plugin_os::init());

    // TODO: re-enable once tauri-plugin-automation supports tauri 3.
    // #[cfg(debug_assertions)]
    // {
    //     builder = builder.plugin(tauri_plugin_automation::init());
    // }

    #[cfg(target_os = "macos")]
    {
        builder = builder.plugin(tauri_plugin_llm::Builder::new().build())
    }

    #[cfg(not(target_os = "macos"))]
    {
        builder = builder.plugin(tauri_plugin_llm::init());
    }

    builder
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
