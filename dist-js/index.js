import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

class LLMStreamListener {
    constructor() {
        this.unListeners = [];
        this.isActive = false;
    }
    async setup(callb) {
        if (this.isActive) {
            callb.onError("Listeners have already been initialized");
            return;
        }
        this.isActive = true;
        const unlistenData = await listen('query-stream-chunk', (event) => {
            const message = event.payload;
            if (message?.type == 'Chunk') {
                const { id, data } = message;
                callb.onData(id, data);
            }
        });
        const unlistenError = await listen('query-stream-error', (event) => {
            const message = event.payload;
            if (message?.type == 'Status') {
                const { msg } = message;
                callb.onError(msg);
            }
        });
        const unlistenEnd = await listen('query-stream-end', (event) => {
            const message = event.payload;
            if (message?.type == 'End') {
                callb.onEnd();
            }
        });
        this.unListeners = [unlistenData, unlistenError, unlistenEnd];
    }
    teardown() {
        this.unListeners.forEach(unlisten => unlisten());
        this.unListeners = [];
        this.isActive = false;
    }
    /// Use this function to send a prompt to the backend
    async stream(message) {
        if (!this.isActive) {
            throw new Error('Stream listener not initialized.');
        }
        /// Use this function to send a query to the backend
        await invoke("plugin:llm|stream", { message });
    }
}

export { LLMStreamListener };
