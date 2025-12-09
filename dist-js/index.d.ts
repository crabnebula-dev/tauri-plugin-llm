type LlmMessage = {
    type: "Prompt";
    system: string;
    message: string;
    num_samples: number;
} | {
    type: "Response";
    error?: string;
    message: string;
} | {
    type: "Exit";
} | {
    type: "Status";
};
export declare function send_message(message: LlmMessage): Promise<LlmMessage | null>;
export declare function retry_recv(): Promise<LlmMessage | null>;
export {};
