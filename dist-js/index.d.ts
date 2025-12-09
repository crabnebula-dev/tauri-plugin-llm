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
export declare function sendMessage(message: LlmMessage): Promise<LlmMessage | null>;
export declare function retryRecv(): Promise<LlmMessage | null>;
export {};
