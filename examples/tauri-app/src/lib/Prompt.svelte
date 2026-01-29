<script>
  import { LLMStreamListener } from "tauri-plugin-llm-api";

  let promptMsg = $state("");
  let promptRes = $state("");
  let availableModels = $state([]);
  let currentModel = $state("");
  let modelsLoading = $state(true);

  let llm = new LLMStreamListener();

  // Initialize the listener and load models
  async function initialize() {
    console.log("Initializing LLMStreamListener...");

    // First, test if Tauri invoke is working
    // try {
    //   console.log("Testing Tauri health check...");
    //   // const { invoke } = await import("@tauri-apps/api/core");
    // } catch (e) {
    //   console.error("Health check failed:", e);
    //   console.error("This indicates Tauri invoke is not working properly");
    // }

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

    // Now load models
    await loadModels();
  }

  // Load available models on component mount
  async function loadModels() {
    try {
      modelsLoading = true;
      console.log("Loading available models...");
      console.log("Calling llm.listAvailableModels()...");

      const models = await llm.listAvailableModels();

      console.log("listAvailableModels() returned:", models);
      console.log("Models type:", typeof models);
      console.log("Is array?", Array.isArray(models));

      availableModels = models;
      console.log("Available models loaded:", availableModels);

      if (availableModels.length === 0) {
        console.warn("No models available!");
      }
    } catch (e) {
      console.error("Failed to load models:", e);
      console.error("Error type:", typeof e);
      console.error("Error name:", e?.name);
      console.error("Error message:", e?.message);
      console.error("Error stack:", e?.stack);
      console.error("Error details:", JSON.stringify(e));
    } finally {
      console.log("Setting modelsLoading to false");
      modelsLoading = false;
    }
  }

  // Switch to selected model
  async function switchModel() {
    if (!currentModel) return;

    try {
      await llm.switchModel(currentModel);
      console.log("Switched to model:", currentModel);
    } catch (e) {
      console.error("Failed to switch model:", e);
    }
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

  // Initialize listener and load models when component mounts
  initialize();
</script>

<div>
  <div class="row">
    <select
      id="model-select"
      bind:value={currentModel}
      data-loaded={!modelsLoading}
    >
      <option value=""
        >{modelsLoading ? "Loading models..." : "Select a model..."}</option
      >
      {#each availableModels as model}
        <option value={model}>{model}</option>
      {/each}
    </select>
    <button
      id="model-switch-btn"
      onclick={switchModel}
      disabled={!currentModel || modelsLoading}
    >
      Switch Model
    </button>
    <span id="models-status"
      >{modelsLoading ? "Loading..." : `${availableModels.length} models`}</span
    >
  </div>
  <div class="row">
    <input
      id="prompt-input"
      placeholder="Enter prompt to echo..."
      bind:value={promptMsg}
    />
    <button id="prompt-send-btn" onclick={send}> Prompt </button>
  </div>
  <p id="prompt-response">{promptRes}</p>
</div>
