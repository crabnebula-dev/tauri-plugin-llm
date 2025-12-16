import { invoke } from "@tauri-apps/api/core";

type LlmMessage =
  | {
      type: "Prompt";
      system: string;
      message: string;
      num_samples: number;
    }
  | {
      type: "Response";
      error?: string;
      message: string;
    }
  | {
      type: "Exit";
    }
  | {
      type: "Status";
    };

/// Send a message to the LLM backend
export async function sendMessage(
  message: LlmMessage
): Promise<LlmMessage | null> {
  return await invoke("plugin:llm|send_message", {
    message,
  });
}

export async function retryRecv(): Promise<LlmMessage | null> {
  return await invoke("plugin:llm|retry_recv");
}
