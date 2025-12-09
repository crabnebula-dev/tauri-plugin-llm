import os from "os";
import path from "path";

describe("some description lol", () => {
  it("shold open a new window and should change focus from one to other", async () => {
    const interactiveTab = await $(
      '[role="tab"][aria-label="Interactive tab"]'
    );
    await interactiveTab.click();
    const newWindow = await $('[aria-label="New window button"]');
    await newWindow.click();
    const handles = await browser.getWindowHandles();
    const secondary = handles.find((handle) => handle === "secondary");
    expect(secondary).toBe("secondary");
    await browser.switchWindow(secondary);
    await browser.switchWindow("main");
    const other = await $('[role="tab"][aria-label="Interactive tab"]');
  });
});
