import { invoke } from '@tauri-apps/api/core';

/// Send a message to the LLM backend
async function sendMessage(message) {
    return await invoke("plugin:llm|send_message", {
        message,
    });
}
async function retryRecv() {
    return await invoke("plugin:llm|retry_recv");
}

export { retryRecv, sendMessage };
