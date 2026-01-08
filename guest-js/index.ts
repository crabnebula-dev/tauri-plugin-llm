import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export type QueryStream =
  | {
    type: "Chunk",
    id: number,
    data: string,
  }
  | {
    type: "End"
  }
  | {
    type: "Error",
    msg: string
  }

export type ChunkStreamDataCallback = (id: number, data: string) => void;
export type ChunkStreamEndCallback = () => void;
export type ChunkStreamErrorCallback = (msg: string) => void;

export type Query =
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

export interface QueryConfig {
  generate_num_samples: number;
}

export interface QueryMessage {
  role: string;
  content: string;
  timestamp?: number
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


/// Use this interface to define the callbacks to control the uery response messages
export interface CallBacks {
  onChunk: (id: number, data: string) => void,
  onError: (msg: string) => void,
  onEnd: () => void
}

export class LLMStreamListener {
  private unListeners: UnlistenFn[] = [];
  private isActive = false;

  async setup(callb: CallBacks): Promise<void> {
    if (this.isActive) {
      // TODO: where to display errors? fix this
      console.log("Listeners have already been initialized");
      return;
    }

    this.isActive = true;

    const chunkListener = await listen('query-stream-chunk', (event) => {
      const message = event.payload as QueryStream;
      if (message.type == 'Chunk') {
        const { id, data } = message;
        callb.onChunk(id, data);
      }
    });

    const errorListener = await listen('query-stream-error', (event) => {
      const message = event.payload as QueryStream;
      if (message.type == 'Error') {
        const { msg } = message;
        callb.onError(msg);
      }
    });

    const endListener = await listen('query-stream-end', (event) => {
      const message = event.payload as QueryStream;
      if (message.type == 'End') {
        callb.onEnd();
      }
    });

    this.unListeners = [chunkListener, errorListener, endListener];
  }

  teardown(): void {
    this.unListeners.forEach(unlisten => unlisten());
    this.unListeners = [];
    this.isActive = false;
  }

  async stream(message: Query, window: Window): Promise<void> {
    if (!this.isActive) {
      throw new Error('Stream listener not initialized.');
    }

    /// Use this function to send a query to the backend
    await invoke("plugin:llm|stream", { message, window })
  }

}

// export async function stream(message: Query, window: Window): Promise<void> {
//   return await invoke("plugin:llm|stream", { message, window });
// }