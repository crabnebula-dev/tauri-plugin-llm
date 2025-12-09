<script>
  import { retryRecv, sendMessage } from "tauri-plugin-llm-api";

  let promptMsg = $state("");
  let promptRes = $state("");

  async function send() {
    await sendMessage({
      type: "Prompt",
      system:
        "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ",
      message: promptMsg || "Hello, World!",
      num_samples: 200,
    });

    promptRes = JSON.stringify(await retryRecv());
  }
</script>

<div>
  <div class="row">
    <input
      id="greet-input"
      placeholder="Enter prompt to echo..."
      bind:value={promptMsg}
    />
    <button onclick={send}> Greet </button>
  </div>
  <p>{promptRes}</p>
</div>
