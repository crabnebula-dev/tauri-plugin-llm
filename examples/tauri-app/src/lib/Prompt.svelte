<script>
  import { onMount } from "svelte";
  import { platform } from "@tauri-apps/plugin-os";
  import { LLMStreamListener } from "tauri-plugin-llm-api";

  let promptMsg = $state("");
  let promptRes = $state("");

  let llm = new LLMStreamListener();
  llm.setup({
    onData: (id, data) => console.log(id, data),
    onEnd: () => console.log("end"),
    onError: (msg) => console.error("error", msg),
  });

  async function send() {
    await llm.stream({
      type: "Prompt",
      messages: [
        { role: "system", content: "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. " },
        { role: "user", content: promptMsg || "Hello World!" },
      ],
      tools: [],
      chunk_size: 200,
    }).catch((e) => {
      // TODO: sendMessage should probably not throw an empty channel error
      console.error(e);
    });
  }

  onMount(() => {
    if (platform() === "linux") {
      // WebKitWebDriver doesn't seem to actually work well so we do the interaction here.
      promptMsg = "Hello from Linux!";
      send();
    }
  });
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
</div>
