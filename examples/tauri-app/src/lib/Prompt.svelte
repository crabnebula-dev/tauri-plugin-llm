<script>
  import { onMount } from "svelte";
  import { platform } from "@tauri-apps/plugin-os";
  import { retryRecv, sendMessage } from "tauri-plugin-llm-api";

  let promptMsg = $state("");
  let promptRes = $state("");

  function recv() {
    retryRecv()
      .then((msg) => {
        console.log(msg);
        promptRes = JSON.stringify(msg);
      })
      .catch((e) => {
        console.error(e);
        setTimeout(recv, 1000);
      });
  }

  async function send() {
    await sendMessage({
      type: "Prompt",
      system:
        "You are a helpful assistant. Your task is to echo the incoming message. Do not describe anything. ",
      message: promptMsg || "Hello World!",
      num_samples: 200,
    }).catch((e) => {
      // TODO: sendMessage should probably not throw an empty channel error
      console.error(e);
    });

    setTimeout(recv, 1000);
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
