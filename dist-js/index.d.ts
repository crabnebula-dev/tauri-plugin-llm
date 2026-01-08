export type Query = {
    type: "Prompt";
    messages: QueryMessage[];
    tools: string[];
    config?: QueryConfig | null;
} | {
    type: "Binary";
} | {
    type: "Response";
    error?: string | null;
    messages: QueryMessage[];
    tools: string[];
} | {
    type: "Exit";
} | {
    type: "Status";
};
export interface QueryConfig {
    generate_num_samples: number;
}
export interface QueryMessage {
    role: string;
    content: string;
    timestamp?: number;
}
export declare function sendMessage(message: Query): Promise<Query | null>;
export declare function retryRecv(): Promise<Query | null>;
