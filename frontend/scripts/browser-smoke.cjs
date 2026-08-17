const { chromium } = require("playwright-core");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const baseUrl = (process.env.HOLDEM_UI_URL || "http://127.0.0.1:5173").replace(/\/$/, "");
const apiUrl = (process.env.HOLDEM_API_URL || baseUrl).replace(/\/$/, "");
const chromeCandidates = [
  process.env.CHROME_EXECUTABLE,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  ...["LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"]
    .filter((key) => process.env[key])
    .map((key) => path.join(process.env[key], "Google", "Chrome", "Application", "chrome.exe")),
].filter(Boolean);
const chromeExecutable = chromeCandidates.find((candidate) => fs.existsSync(candidate));
if (!chromeExecutable) {
  throw new Error("Set CHROME_EXECUTABLE to an installed Chrome or Chromium executable.");
}

(async () => {
  const browser = await chromium.launch({ executablePath: chromeExecutable, headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    await page.goto(baseUrl);
    await page.getByText("Server connected", { exact: true }).waitFor();
    const response = await context.request.get(`${apiUrl}/api/v1/checkpoints`);
    assert.equal(response.status(), 200);
    const registry = await response.json();
    const defaultId = registry.recommended_checkpoint?.id || registry.checkpoints[0]?.id;
    assert.ok(defaultId, "A saved policy must be available");
    assert.equal(await page.getByLabel("Model").inputValue(), defaultId);

    const ready = () => page.waitForFunction(() => {
      const button = document.querySelector(".lab-setup .lab-primary");
      return button && !button.disabled;
    });
    async function changeSession(button, endpoint, status = 200) {
      const pending = page.waitForResponse((result) =>
        result.url().endsWith(endpoint) && result.request().method() === "POST");
      await button.click();
      const result = await pending;
      assert.equal(result.status(), status);
      const data = await result.json();
      await ready();
      assert.equal(data.checkpoint.id, defaultId);
      return data;
    }

    let session = await changeSession(
      page.getByRole("button", { name: "Start a new session", exact: true }),
      "/holdem/sessions", 201,
    );
    const sessionId = session.session_id;
    for (let i = 0; i < 40 && !session.state.terminal; i++) {
      session = await changeSession(
        page.locator(".lab-actions button").filter({ hasText: /^(call|check)$/i }), "/act",
      );
      assert.equal(await page.getByLabel("Model").inputValue(), defaultId);
    }
    assert.equal(session.state.terminal, true, "The hand should finish");
    const botEvents = session.events.filter((event) => event.probabilities);
    assert.ok(botEvents.length > 0, "The bot should act during the hand");
    for (const event of botEvents) {
      assert.ok(Math.abs(event.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
      assert.ok(event.action_menu.some((action) => action.slot === event.slot));
      assert.ok(event.probabilities.every((value) => Number.isFinite(value) && value >= 0));
      event.probabilities.forEach((probability, slot) => {
        if (!event.action_menu.some((action) => action.slot === slot)) assert.equal(probability, 0);
      });
    }
    const lastBot = botEvents.at(-1);
    assert.deepEqual(await page.locator(".lab-probability strong").allTextContents(),
      lastBot.action_menu.map((action) => `${(100 * lastBot.probabilities[action.slot]).toFixed(1)}%`));

    await page.getByLabel("Select hand replay").selectOption({ index: 1 });
    assert.equal(await page.locator(".lab-board .card-empty").count(), 5);
    assert.equal(await page.locator(".lab-bot .card-back").count(), 2);
    const slider = page.getByRole("slider", { name: "Replay decision" });
    await slider.fill(await slider.getAttribute("max"));
    assert.deepEqual(await page.locator(".lab-board .playing-card:not(.card-empty)").evaluateAll(
      (cards) => cards.map((card) => card.getAttribute("aria-label"))), session.state.board);
    assert.equal(await page.locator(".lab-bot .card-back").count(), 2 - session.state.opponent_cards.length);
    await page.getByLabel("Select hand replay").selectOption("");

    await page.setViewportSize({ width: 390, height: 844 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(overflow, false, "The mobile layout should fit the viewport");
    if (process.env.HOLDEM_SCREENSHOT_PATH) {
      await page.screenshot({ path: process.env.HOLDEM_SCREENSHOT_PATH, fullPage: true });
    }
    await page.setViewportSize({ width: 1440, height: 1100 });
    session = await changeSession(page.getByRole("button", { name: "Deal next hand" }), "/new-hand");
    assert.equal(session.hand_number, 2);

    await page.reload();
    await page.getByText("Server connected", { exact: true }).waitFor();
    await page.getByLabel("Saved sessions").selectOption(sessionId);
    await ready();
    await page.locator(".lab-table-toolbar").getByText("LIVE · HAND 2", { exact: true }).waitFor();
    assert.deepEqual(errors, []);
    console.log("Browser smoke passed: saved policy, complete hand, probabilities, replay, next hand, saved session, mobile layout.");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
