<script>
  import { LLMStreamListener } from "tauri-plugin-llm-api";

  let promptMsg = $state("");
  let promptRes = $state("");
  let modelsList = $state("");
  let switchModelResult = $state("");
  let llm = new LLMStreamListener();

  async function initialize() {
    console.log("Initializing LLMStreamListener...");

    await llm.setup({
      onData: (_id, data) => {
        promptRes =
          [...data].map((byte) => String.fromCharCode(byte)).join("") +
          "\n\n" +
          promptRes;
      },
      onEnd: () => console.log("end"),
      onError: (msg) => console.error("error", msg),
    });
    console.log("LLMStreamListener setup complete");
  }

  async function send() {
    const healthResult = await llm.healthCheck().catch((e) => {
      console.error(e);
    });
    console.log("Health check result:", healthResult);

    await llm
      .stream({
        type: "Prompt",
        messages: [
          {
            role: "system",
            content:
              "You are a helpful assistant. You repeat the incoming message in your own words. ",
          },
          {
            role: "user",
            content: promptMsg || "Hello World! Echo this message",
          },
        ],
        tools: [],
        chunk_size: 200,
      })
      .catch((e) => {
        console.error(e);
      });
  }

  async function listModels() {
    console.log("Fetching available models...");
    try {
      const models = await llm.listAvailableModels();
      console.log("Available models:", models);
      modelsList = models.join(", ");
    } catch (e) {
      console.error("Failed to list models:", e);
      modelsList = "Error loading models:" + e;
    }
  }

  async function switchModel() {
    console.log("Switching model to Mock...");
    try {
      const models = await llm.listAvailableModels();
      console.log("Available models for switch:", models);
      if (models.length > 0) {
        await llm.switchModel(models[0]);
        switchModelResult = "Switched to: " + models[0];
      } else {
        switchModelResult = "No models available";
      }
    } catch (e) {
      console.error("Failed to switch model:", e);
      switchModelResult = "Error: " + e;
    }
  }

  initialize();
</script>

<div>
  <div class="row">
    <input
      id="prompt-input"
      placeholder="Enter prompt to echo..."
      bind:value={promptMsg}
    />
    <button id="prompt-send-btn" onclick={send}> Prompt </button>
  </div>
  <p id="prompt-response">{promptRes}</p>

  <div class="row">
    <button id="list-models-btn" onclick={listModels}> List Models </button>
  </div>
  <div id="models-list">{modelsList}</div>

  <div class="row">
    <button id="switch-model-btn" onclick={switchModel}> Switch Model </button>
  </div>
  <div id="switch-model-result">{switchModelResult}</div>
</div>
