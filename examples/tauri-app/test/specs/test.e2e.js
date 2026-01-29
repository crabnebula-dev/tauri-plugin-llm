describe("End to end tests", () => {
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

  it("should list available models via listAvailableModels()", async () => {
    // This test verifies the listAvailableModels backend function:
    // 1. Button click triggers llm.listAvailableModels() call
    // 2. Backend returns list of available model names
    // 3. UI displays the list in a div
    // 4. Verify the list is not empty and contains expected models

    const listModelsButton = await $("#list-models-btn");
    const modelsList = await $("#models-list");

    // Click the button to fetch available models
    console.log("Clicking list models button...");
    await listModelsButton.click();

    // Wait for the models list to appear
    console.log("Waiting for models list to populate...");
    await modelsList.waitUntil(
      async function () {
        const text = await this.getText();
        console.log("Models list text:", text);
        return text !== "";
      },
      {
        timeout: 10000,
        timeoutMsg: "expected models list to populate after 10s",
        interval: 500,
      }
    );

    // Verify the list contains at least one model
    const modelsText = await modelsList.getText();
    console.log("Final models list:", modelsText);

    // Check that we got a non-empty response
    await expect(modelsList).toHaveText(expect.not.stringMatching(/^$/));

    // Optionally verify that "Mock" model is in the list (should always be available in test env)
    await expect(modelsList).toHaveText(expect.stringContaining("Mock"));
  });

  it("should switch model and send a message successfully", async () => {
    // This test verifies the switchModel flow:
    // 1. List available models to find one to switch to
    // 2. Switch to that model (reusing "Mock" since it's the only one configured)
    // 3. Send a message after switching and verify we get a response

    const switchModelButton = await $("#switch-model-btn");
    const switchModelResult = await $("#switch-model-result");

    // Click the button to switch model
    console.log("Clicking switch model button...");
    await switchModelButton.click();

    // Wait for the switch result to appear
    console.log("Waiting for switch model result...");
    await switchModelResult.waitUntil(
      async function () {
        const text = await this.getText();
        console.log("Switch model result text:", text);
        return text !== "";
      },
      {
        timeout: 10000,
        timeoutMsg: "expected switch model result after 10s",
        interval: 500,
      }
    );

    // Verify the switch was successful
    const resultText = await switchModelResult.getText();
    console.log("Switch model result:", resultText);
    await expect(switchModelResult).toHaveText(
      expect.stringContaining("Switched to:")
    );

    // Now send a message to verify the switched model works
    const promptInput = await $("input#prompt-input");
    const promptButton = await $("button#prompt-send-btn");

    if (process.platform !== "linux") {
      await promptInput.setValue("Message after model switch");
    }

    await promptButton.click();

    const promptResponse = await $("p#prompt-response");
    await promptResponse.waitUntil(
      async function () {
        return (await this.getText()) !== "";
      },
      {
        timeout: 60000,
        timeoutMsg:
          "expected response from backend after model switch within 60s",
        interval: 1000,
      }
    );

    // Verify we got a non-empty response after switching
    const responseText = await promptResponse.getText();
    console.log("Response after model switch:", responseText);
    expect(responseText.length).toBeGreaterThan(0);
  });

});
