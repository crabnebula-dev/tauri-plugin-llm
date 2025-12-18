import { invoke } from "@tauri-apps/api/core";

type Query =
  | {
    type: "Prompt";
    messages: QueryMessage[];

    // We keep the tools info as generic as possible.
    // This may change in the future. For now a model can be
    // informed about available tools by a json encoded message
    // as defined by the MCP standard
    tools: string[];

    // Optional config for the query.
    // If no value has been set, the default is assumed
    config?: QueryConfig | null;
  }
  | {
    type: "Binary";
  }
  | {
    type: "Response";
    error?: string | null;
    messages: QueryMessage[];
    tools: string[];
  }
  | {
    type: "Exit";
  }
  | {
    type: "Status";
  };

interface QueryConfig {
  generate_num_samples: number;
}

interface QueryMessage {
  role: string;
  content: string;
}

/// Send a message to the LLM backend
export async function sendMessage(
  message: Query
): Promise<Query | null> {
  return await invoke("plugin:llm|send_message", {
    message,
  });
}

export async function retryRecv(): Promise<Query | null> {
  return await invoke("plugin:llm|retry_recv");
}
