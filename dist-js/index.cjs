'use strict';

var core = require('@tauri-apps/api/core');
var event = require('@tauri-apps/api/event');

/**
 * Manages LLM stream event listeners and handles communication with the backend.
 *
 * This class sets up event listeners for chunk data, errors, and stream end events,
 * and provides a method to send queries to the LLM backend.
 */
class LLMStreamListener {
    constructor() {
        this.unListeners = [];
        this.isActive = false;
    }
    /**
     * Initializes the event listeners to process messages received from the backend.
     *
     * Sets up three event listeners:
     * - `query-stream-chunk`: Receives data chunks from the LLM response
     * - `query-stream-error`: Receives error messages during streaming
     * - `query-stream-end`: Signals the end of the stream
     *
     * @param callb - Callback functions to handle data, errors, and stream completion
     * @returns A promise that resolves when all listeners are set up
     *
     * @example
     * ```typescript
     * const listener = new LLMStreamListener();
     * await listener.setup({
     *   onData: (id, data, timestamp) => {
     *     console.log(`Chunk ${id}:`, new TextDecoder().decode(data));
     *   },
     *   onError: (msg) => console.error("Error:", msg),
     *   onEnd: () => console.log("Stream completed")
     * });
     * ```
     */
    async setup(callb) {
        if (this.isActive) {
            callb.onError("Listeners have already been initialized");
            return;
        }
        this.isActive = true;
        const unlistenData = await event.listen('query-stream-chunk', (event) => {
            const message = event.payload;
            if (message?.type == 'Chunk') {
                const { id, data, timestamp } = message;
                callb.onData(id, data, timestamp);
            }
        });
        const unlistenError = await event.listen('query-stream-error', (event) => {
            const message = event.payload;
            if (message?.type == 'Status') {
                const { msg } = message;
                callb.onError(msg);
            }
        });
        const unlistenEnd = await event.listen('query-stream-end', (event) => {
            callb.onEnd();
        });
        this.unListeners = [unlistenData, unlistenError, unlistenEnd];
    }
    /**
     * Cleans up all event listeners and resets the listener state.
     *
     * Calls the unlisten functions for each registered callback (data, error, and end events)
     * and sets the listener to inactive mode. Should be called when you're done streaming
     * to prevent memory leaks.
     *
     * @example
     * ```typescript
     * const listener = new LLMStreamListener();
     * await listener.setup(callbacks);
     * // ... use the listener ...
     * listener.teardown(); // Clean up when done
     * ```
     */
    teardown() {
        this.unListeners.forEach(unlisten => unlisten());
        this.unListeners = [];
        this.isActive = false;
    }
    /**
     * Sends a query to the backend LLM runtime.
     *
     * The query will be processed by the active LLM model, and responses will be
     * received through the callbacks registered in `setup()`.
     *
     * @param message - The query to send to the backend (typically a Prompt query)
     * @returns A promise that resolves when the query has been sent
     * @throws Error if the listener has not been initialized via `setup()`
     *
     * @example
     * ```typescript
     * const listener = new LLMStreamListener();
     * await listener.setup(callbacks);
     *
     * await listener.stream({
     *   type: "Prompt",
     *   messages: [
     *     { role: "user", content: "Hello, how are you?" }
     *   ],
     *   tools: [],
     *   config: { generate_num_samples: 500 },
     *   chunk_size: 32
     * });
     * ```
     */
    async stream(message) {
        if (!this.isActive) {
            throw new Error('Stream listener not initialized.');
        }
        await core.invoke("plugin:llm|stream", { message });
    }
    /**
     * Switches the active LLM runtime to the specified model.
     *
     * @param id - The model identifier/name to switch to
     * @returns A promise that resolves when the model has been switched
     * @throws Error if the model ID is not found or switching fails
     *
     * @example
     * ```typescript
     * await switchModel("Qwen3-4B-GGUF");
     * ```
     */
    async switchModel(id) {
        await core.invoke("plugin:llm|switch_model", { id });
    }
    /**
     * Returns a list of available model names that can be activated.
     *
     * @returns A promise that resolves to an array of model name strings
     *
     * @example
     * ```typescript
     * const models = await listAvailableModels();
     * console.log("Available models:", models);
     * // Output: ["Mock", "Llama-3.2-3B-Instruct", "Qwen3-4B-GGUF"]
     * ```
     */
    async listAvailableModels() {
        return await core.invoke("plugin:llm|list_available_models");
    }
    /**
     * Adds a new LLMRuntimeConfig to the runtime service at runtime.
     *
     * This method allows you to dynamically add new model configurations without restarting
     * the application. The configuration should be a JSON string representing a valid
     * LLMRuntimeConfig.
     *
     * @param config - The LLM runtime configuration as a JSON string
     * @returns A promise that resolves when the configuration has been added
     * @throws Error if the configuration is invalid or cannot be parsed
     *
     * @example
     * ```typescript
     * const listener = new LLMStreamListener();
     *
     * const newConfig = {
     *   model_config: {
     *     name: "Llama-3.2-3B-Custom",
     *     sampling_config: "TopKThenTopP",
     *     seed: { type: "Random" },
     *     penalty: 1.1
     *   },
     *   tokenizer_file: "/path/to/tokenizer.json",
     *   model_file: "/path/to/model.gguf"
     * };
     *
     * await listener.addConfiguration(JSON.stringify(newConfig));
     * console.log("Configuration added successfully");
     *
     * // Now you can switch to the newly added model
     * await listener.switchModel("Llama-3.2-3B-Custom");
     * ```
     */
    async addConfiguration(config) {
        await core.invoke("plugin:llm|add_configuration", { config });
    }
    async healthCheck() {
        return await core.invoke("plugin:llm|health_check");
    }
}

exports.LLMStreamListener = LLMStreamListener;
