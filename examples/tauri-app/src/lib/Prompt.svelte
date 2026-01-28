<script>
  import { LLMStreamListener } from "tauri-plugin-llm-api";

  let promptMsg = $state("");
  let promptRes = $state("");

  let llm = new LLMStreamListener();


  llm.setup({
    onData: (_id, data) => {
      promptRes = [...data].map((byte) => String.fromCharCode(byte)).join('') + 
        '\n\n' + promptRes;
    },
    onEnd: () => console.log("end"),
    onError: (msg) => console.error("error", msg),
  });

  async function send() {
    await llm.stream({
      type: "Prompt",
      messages: [
        { role: "system", content: "You are a helpful assistant. You repeat the incoming message in your own words. " },
        { role: "user", content: promptMsg || "Hello World! Echo this message" },
      ],
      tools: [],
      chunk_size: 200,
    }).catch((e) => {
      console.error(e);
    });
  }
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
