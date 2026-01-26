export type ChunkStreamDataCallback = (id: number, data: string) => void;
export type ChunkStreamEndCallback = () => void;
export type ChunkStreamErrorCallback = (msg: string) => void;
export type Query = {
    type: "Prompt";
    messages: QueryMessage[];
    tools: string[];
    config?: QueryConfig;
    chunk_size?: number;
    timestamp?: number;
} | {
    type: "Response";
    error?: string;
    messages: QueryMessage[];
    tools: string[];
} | {
    type: "Chunk";
    id: number;
    data: Uint8Array;
    kind: "string" | "bytes";
    timestamp?: number;
} | {
    type: "End";
} | {
    type: "Exit";
} | {
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
export interface CallBacks {
    onData: (id: number, data: Uint8Array, timestamp?: number) => void;
    onError: (msg: string) => void;
    onEnd: () => void;
}
export declare class LLMStreamListener {
    private unListeners;
    private isActive;
    setup(callb: CallBacks): Promise<void>;
    teardown(): void;
    stream(message: Query): Promise<void>;
}
