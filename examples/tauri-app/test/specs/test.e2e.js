describe("A basic end to end test to call a model from the frontend and display the response", () => {
  it("should receive streamed response via LLMStreamListener events", async () => {
    // This test verifies the complete event-driven flow:
    // 1. Button click triggers llm.stream() call to backend
    // 2. Backend processes query and sends events back (query-stream-chunk, query-stream-end)
    // 3. LLMStreamListener callbacks (onData, onEnd) receive and process events
    // 4. onData callback updates the UI with received chunks
    // 5. UI displays the accumulated response

    const promptInput = await $("input#prompt-input");
    const promptButton = await $("button#prompt-send-btn");

    // Optionally set a custom message (or use default "Hello World! Echo this message")
    if (process.platform !== "linux") {
      await promptInput.setValue("Test message for e2e");
    }

    // Click the button to trigger the stream call
    await promptButton.click();

    // Wait for the response to appear in the UI
    // The Mock runtime echoes back the message content in chunks
    // The onData callback receives these chunks and updates promptRes
    const promptResponse = await $("p#prompt-response");
    await promptResponse.waitUntil(
      async function () {
        return (await this.getText()) !== "";
      },
      {
        timeout: 60000,
        timeoutMsg: "expected response from backend events after 60s",
        interval: 1000,
      }
    );

    // Verify the response contains the echoed message
    // Mock runtime returns the user message content in chunks
    if (process.platform !== "linux") {
      await expect(promptResponse).toHaveText(
        expect.stringContaining("Test message for e2e")
      );
    } else {
      // On Linux, default message is used
      await expect(promptResponse).toHaveText(
        expect.stringContaining("Hello World! Echo this message")
      );
    }
  });
});
