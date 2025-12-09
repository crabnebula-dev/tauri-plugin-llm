'use strict';

var core = require('@tauri-apps/api/core');

/// Send a message to the LLM backend
async function send_message(message) {
    return await core.invoke("plugin:llm|send_message", {
        message,
    });
}
async function retry_recv() {
    return await core.invoke("plugin:llm|retry_recv", {});
}

exports.retry_recv = retry_recv;
exports.send_message = send_message;
