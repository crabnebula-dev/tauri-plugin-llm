'use strict';

var core = require('@tauri-apps/api/core');

/// Send a message to the LLM backend
async function sendMessage(message) {
    return await core.invoke("plugin:llm|send_message", {
        message,
    });
}
async function retryRecv() {
    return await core.invoke("plugin:llm|retry_recv");
}

exports.retryRecv = retryRecv;
exports.sendMessage = sendMessage;
