import { invoke } from '@tauri-apps/api/core';

/// Send a message to the LLM backend
async function send_message(message) {
    return await invoke("plugin:llm|send_message", {
        message,
    });
}
async function retry_recv() {
    return await invoke("plugin:llm|retry_recv", {});
}

export { retry_recv, send_message };
