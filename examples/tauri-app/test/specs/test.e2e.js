describe("A basic end to end test to call a model from the frontend and display the response", () => {
  it("should get some response for simple prompts", async () => {
    // Linux' WebKitWebDriver can't send anything to the webview via setValue/click
    // Use browser.execute() as workaround to set input and trigger send
    if (process.platform === "linux") {
      await browser.execute(() => {
        const input = document.querySelector("input#prompt-input");
        if (input) {
          input.value = "Hello, World!";
          input.dispatchEvent(new Event('input', { bubbles: true }));
        }
        const button = document.querySelector("button#prompt-send-btn");
        if (button) {
          button.click();
        }
      });
    } else {
      const promptInput = await $("input#prompt-input");
      await promptInput.setValue('{"type":"Prompt","messages":[{"role":"user","content":"Hello, World!"}],"tools":[],"config":{"generate_num_samples":100}}');
      const promptButton = await $("button#prompt-send-btn");
      await promptButton.click();
    }

    const promptResponse = await $("p#prompt-response");
    await promptResponse.waitUntil(
      async function () {
        return (await this.getText()) !== "";
      },
      {
        timeout: 60000,
        timeoutMsg: "expected response after 60s",
        interval: 1000,
      }
    );
    // Verify we got a response (works for both Mock and real runtime)
    const responseText = await promptResponse.getText();
    expect(responseText.length).toBeGreaterThan(0);
  });
});
