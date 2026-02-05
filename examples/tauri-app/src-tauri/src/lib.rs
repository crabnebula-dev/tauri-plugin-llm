#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tauri_plugin_llm=debug".into()),
        )
        .init();

    let mut builder = tauri::Builder::default().plugin(tauri_plugin_os::init());

    #[cfg(debug_assertions)]
    {
        builder = builder.plugin(tauri_plugin_automation::init());
    }

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
