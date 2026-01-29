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

  it("should switch runtime and receive response from switched model", async () => {
    // This test verifies runtime switching functionality via UI:
    // 1. Wait for available models to load in the select dropdown
    // 2. Select "Mock" from the dropdown
    // 3. Click "Switch Model" button
    // 4. Send a query after switching
    // 5. Verify we receive a response, confirming the switch worked

    // Note: We reuse Mock runtime for simplicity, but this tests the switching mechanism

    const modelSelect = await $("#model-select");
    const modelsStatus = await $("#models-status");
    const switchButton = await $("#model-switch-btn");
    const promptInput = await $("input#prompt-input");
    const promptButton = await $("button#prompt-send-btn");
    const promptResponse = await $("p#prompt-response");

    // First, check if models are loading
    console.log("Waiting for models to load...");
    const statusText = await modelsStatus.getText();
    console.log("Initial models status:", statusText);

    // Wait for models to be loaded (check data-loaded attribute)
    await modelSelect.waitUntil(
      async function () {
        const loaded = await this.getAttribute("data-loaded");
        console.log("data-loaded attribute:", loaded);
        return loaded === "true";
      },
      {
        timeout: 10000,
        timeoutMsg: "expected models to finish loading after 10s",
        interval: 500,
      }
    );

    // Verify models are actually available
    const finalStatus = await modelsStatus.getText();
    console.log("Final models status:", finalStatus);

    // Wait for models to be populated in the select dropdown
    await modelSelect.waitUntil(
      async function () {
        const options = await this.$$("option");
        //console.log("Number of options:", options.length);
        return options.length > 1; // More than just the "Select a model..." option
      },
      {
        timeout: 5000,
        timeoutMsg: "expected models to be populated in dropdown after 5s",
        interval: 500,
      }
    );

    // Select "Mock" model from dropdown
    console.log("Selecting Mock model...");
    await modelSelect.selectByVisibleText("Mock");

    // Click switch button to activate the selected model
    console.log("Clicking switch button...");
    await switchButton.click();

    // Wait for the switch to complete
    await browser.pause(2000);

    // Clear any previous response text
    await browser.execute(() => {
      const responseElement = document.getElementById("prompt-response");
      if (responseElement) {
        responseElement.textContent = "";
      }
    });

    // Set a message indicating this is after switch
    console.log("Setting prompt message...");
    if (process.platform !== "linux") {
      await promptInput.setValue("Message after runtime switch");
    }

    // Click the button to send query through the switched runtime
    console.log("Clicking prompt button...");
    await promptButton.click();

    // Wait for response to confirm the switched runtime is working
    console.log("Waiting for response...");
    await promptResponse.waitUntil(
      async function () {
        const text = await this.getText();
        console.log("Response text length:", text.length);
        return text !== "";
      },
      {
        timeout: 60000,
        timeoutMsg: "expected response after runtime switch after 60s",
        interval: 1000,
      }
    );

    console.log("Response received!");
    const responseText = await promptResponse.getText();
    console.log("Response text:", responseText.substring(0, 100));

    // Verify we got a response after switching
    if (process.platform !== "linux") {
      await expect(promptResponse).toHaveText(
        expect.stringContaining("Message after runtime switch")
      );
    } else {
      // On Linux, default message is used
      await expect(promptResponse).toHaveText(
        expect.stringContaining("Hello World! Echo this message")
      );
    }
  });


});
