import { invoke } from '@tauri-apps/api/core'

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
export async function send_message(value: LlmMessage): Promise<LlmMessage | null> {
  return await invoke<{ value?: LlmMessage }>('plugin:llm|send_message', {
    payload: {
      value,
    },
  }).then((r) => (r.value ? r.value : null));
}

export async function retry_recv(value: LlmMessage): Promise<LlmMessage | null> {
  return await invoke<{ value?: LlmMessage }>('plugin:llm|retry_recv', {
    payload: {
      value,
    },
  }).then((r) => (r.value ? r.value : null));
}
