/**
 * End-to-end tests for tauri-plugin-llm.
 *
 * These tests call the Tauri backend directly via window.__TAURI_INTERNALS__
 * running inside the webview via browser.executeAsync(). No DOM elements,
 * no Svelte components, no UI interaction — just IPC.
 */

// ─── Helpers ────────────────────────────────────────────────────────────────

/**
 * Invoke a Tauri plugin command directly inside the webview.
 *
 * @param {string} cmd   - The fully-qualified command (e.g. "plugin:llm|stream")
 * @param {object} args  - Arguments to pass to the command
 * @returns {Promise<*>} - The command's return value
 * @throws {Error}       - If the command rejects
 */
async function invoke(cmd, args = {}) {
  const [err, value] = await browser.executeAsync(
    async (cmd, args, done) => {
      try {
        done([null, await window.__TAURI_INTERNALS__.invoke(cmd, args)]);
      } catch (e) {
        done([e.message || String(e), null]);
      }
    },
    cmd,
    args,
  );
  if (err !== null) {
    throw new Error(`invoke("${cmd}") failed: ${err}`);
  }
  return value;
}

/**
 * Send a streaming prompt and collect all response chunks until the stream ends.
 *
 * Sets up Tauri event listeners for chunk/error/end, fires the stream command,
 * and resolves once the backend signals completion. Everything runs inside the
 * webview — no DOM interaction.
 *
 * @param {Array<{role: string, content: string}>} messages  - Chat messages
 * @param {object}  [opts]
 * @param {number}  [opts.chunkSize=200]   - Bytes per chunk
 * @param {number}  [opts.timeoutMs=30000] - Max wait before giving up
 * @returns {Promise<{error: string|null, chunks: Array<{id: number, text: string}>, tokenUsage: object|null}>}
 */
async function streamPrompt(messages, opts = {}) {
  const { chunkSize = 200, timeoutMs = 30000 } = opts;

  return browser.executeAsync(
    async (messages, chunkSize, timeoutMs, done) => {
      const { invoke, transformCallback } = window.__TAURI_INTERNALS__;
      const chunks = [];
      let tokenUsage = null;
      const listeners = [];

      const timeout = setTimeout(() => {
        cleanup().then(() =>
          done({
            error: "Timed out waiting for stream end",
            chunks,
            tokenUsage,
          }),
        );
      }, timeoutMs);

      // Register a Tauri event listener via the internal plugin API
      async function on(event, handler) {
        const id = transformCallback(handler);
        await invoke("plugin:event|listen", {
          event,
          target: { kind: "Any" },
          handler: id,
        });
        listeners.push({ event, eventId: id });
      }

      // Tear down all listeners we registered
      async function cleanup() {
        for (const listener of listeners) {
          try {
            await invoke("plugin:event|unlisten", listener);
          } catch (_) {
            /* best effort */
          }
        }
      }

      // Wire up listeners BEFORE starting the stream so we don't miss events
      await on("query-stream-chunk", (e) => {
        if (e.payload?.type === "Chunk") {
          const bytes = e.payload.data;
          const text = new TextDecoder().decode(
            bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes),
          );
          chunks.push({ id: e.payload.id, text });
        }
      });

      await on("query-stream-error", (e) => {
        if (e.payload?.type === "Status") {
          clearTimeout(timeout);
          cleanup().then(() =>
            done({ error: e.payload.msg, chunks, tokenUsage }),
          );
        }
      });

      await on("query-stream-end", (e) => {
        if (e.payload?.type === "End") {
          tokenUsage = e.payload.usage;
        }
        clearTimeout(timeout);
        cleanup().then(() => done({ error: null, chunks, tokenUsage }));
      });

      // Fire the stream command
      try {
        await invoke("plugin:llm|stream", {
          message: {
            type: "Prompt",
            messages,
            tools: [],
            chunk_size: chunkSize,
          },
        });
      } catch (e) {
        clearTimeout(timeout);
        cleanup().then(() =>
          done({ error: e.message || String(e), chunks, tokenUsage }),
        );
      }
    },
    messages,
    chunkSize,
    timeoutMs,
  );
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe("End to end tests", () => {
  it("should list available models via IPC", async () => {
    const models = await invoke("plugin:llm|list_available_models");

    expect(models).toBeInstanceOf(Array);
    expect(models.length).toBeGreaterThan(0);
    expect(models).toContain("Mock");
  });

  it("should switch model via IPC", async () => {
    const models = await invoke("plugin:llm|list_available_models");
    expect(models.length).toBeGreaterThan(0);

    // switchModel resolves void on success, rejects on failure
    await invoke("plugin:llm|switch_model", { id: models[0] });
  });

  it("should receive streamed response chunks via events", async () => {
    const { error, chunks } = await streamPrompt([
      {
        role: "system",
        content:
          "You are a helpful assistant. You repeat the incoming message in your own words.",
      },
      { role: "user", content: "Hello from e2e test" },
    ]);

    expect(error).toBeNull();
    expect(chunks.length).toBeGreaterThan(0);

    // Mock runtime echoes the user message content back
    const fullResponse = chunks.map((c) => c.text).join("");
    expect(fullResponse).toContain("Hello from e2e test");

    // Chunk IDs should be sequential (iterator enumerate)
    chunks.forEach((chunk, i) => {
      expect(chunk.id).toBe(i);
    });
  });

  it("should report token usage after stream ends", async () => {
    const { error, tokenUsage } = await streamPrompt([
      { role: "system", content: "Echo the message." },
      { role: "user", content: "Token usage test" },
    ]);

    expect(error).toBeNull();
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.prompt_tokens).toBeGreaterThanOrEqual(0);
    expect(tokenUsage.completion_tokens).toBeGreaterThanOrEqual(0);
    expect(tokenUsage.total_tokens).toBeGreaterThanOrEqual(0);
  });

  it("should stream successfully after model switch", async () => {
    const models = await invoke("plugin:llm|list_available_models");
    await invoke("plugin:llm|switch_model", { id: models[0] });

    const { error, chunks } = await streamPrompt([
      { role: "user", content: "Message after model switch" },
    ]);

    expect(error).toBeNull();
    expect(chunks.length).toBeGreaterThan(0);

    const fullResponse = chunks.map((c) => c.text).join("");
    expect(fullResponse.length).toBeGreaterThan(0);
  });
});
