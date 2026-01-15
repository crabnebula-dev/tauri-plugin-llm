import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export type ChunkStreamDataCallback = (id: number, data: string) => void;
export type ChunkStreamEndCallback = () => void;
export type ChunkStreamErrorCallback = (msg: string) => void;

export type Query =
  | {
    type: "Prompt";
    messages: QueryMessage[];
    tools: string[];
    config?: QueryConfig;
    chunk_size?: number;
    timestamp?: number;
  }
  | {
    type: "Response";
    error?: string;
    messages: QueryMessage[];
    tools: string[];
  }
  | {
    type: "Chunk";
    id: number;
    data: Uint8Array;
    kind: "string" | "bytes";
  }
  | {
    type: "End";
  }
  | {
    type: "Exit";
  }
  | {
    type: "Status";
    msg: string;
  };

export interface QueryConfig {
  generate_num_samples: number;
}

export interface QueryMessage {
  role: string;
  content: string;
}

/// Use this interface to define the callbacks to control the response messages
export interface CallBacks {
  onData: (id: number, data: Uint8Array) => void,
  onError: (msg: string) => void,
  onEnd: () => void
}

export class LLMStreamListener {
  private unListeners: UnlistenFn[] = [];
  private isActive = false;

  async setup(callb: CallBacks): Promise<void> {
    if (this.isActive) {
      callb.onError("Listeners have already been initialized");
      return;
    }

    this.isActive = true;

    const chunkListener = await listen('query-stream-chunk', (event) => {
      const message = event.payload as Query;
      if (message.type == 'Chunk') {
        const { id, data } = message;
        callb.onData(id, data);
      }
    });

    const errorListener = await listen('query-stream-error', (event) => {
      const message = event.payload as Query;
      if (message.type == 'Status') {
        const { msg } = message;
        callb.onError(msg);
      }
    });

    const endListener = await listen('query-stream-end', (event) => {
      const message = event.payload as Query;
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

  /// Use this function to send a prompt to the backend
  async stream(message: Query, window: Window): Promise<void> {
    if (!this.isActive) {
      throw new Error('Stream listener not initialized.');
    }

    /// Use this function to send a query to the backend
    await invoke("plugin:llm|stream", { message, window })
  }

}
