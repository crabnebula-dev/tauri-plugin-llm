describe("some description lol", () => {
  it("should get some response for simple prompts", async () => {
    const promptInput = await $("input#prompt-input");
    await promptInput.setValue("Hello from e2e Test.");
    const promptButton = await $("button#prompt-send-btn");
    await promptButton.click();
    const promptResponse = await $("p#prompt-response");
    await promptResponse.waitUntil(
      async function () {
        return (await this.getText()) !== "";
      },
      {
        timeout: 600000,
        timeoutMsg: "expected response after 60s",
        interval: 1000,
      }
    );
    await expect(promptResponse).toHaveText(
      expect.stringContaining('"type":"Response"')
    );
  });
});
