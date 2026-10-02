import { test, expect } from "@playwright/test";

async function mock(page: any) {
  await page.route("**/api/models", async (route: any) =>
    route.fulfill({
      json: {
        default: "supervised",
        models: [
          {
            id: "supervised",
            name: "Supervised model",
            kind: "supervised",
            evidence: "pilot",
            checkpoint_id: "s".repeat(64),
          },
          {
            id: "rl",
            name: "Self-play model",
            kind: "rl",
            evidence: "pilot",
            checkpoint_id: "r".repeat(64),
          },
          {
            id: "handcrafted",
            name: "Classical baseline",
            kind: "handcrafted",
            evidence: "baseline",
            checkpoint_id: "handcrafted-v1",
          },
        ],
      },
    }),
  );
  await page.route("**/api/analyze", async (route: any) => {
    const b = route.request().postDataJSON();
    await route.fulfill({
      json: {
        best_move: b.moves.length === 1 ? "e7e5" : "g8f6",
        value: 0.12,
        depth: 2,
        nodes: 1000,
        elapsed_ms: 120,
        pv: ["e7e5", "g1f3"],
        checkpoint_id: "s".repeat(64),
        terminal: null,
        mate_score: false,
      },
    });
  });
}
test("play, reply, undo, restart, export and compare", async ({ page }) => {
  await mock(page);
  await page.goto("/");
  await expect(page.getByText("Engine online", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e4", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await expect(page.locator(".move-list")).toContainText("e4");
  await expect(page.locator(".move-list")).toContainText("e5");
  const dl = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export PGN", exact: true }).click();
  expect((await dl).suggestedFilename()).toBe("chessengine-game.pgn");
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(page.getByText("0 plies", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Compare", exact: true }).click();
  await page
    .getByRole("button", { name: "Compare models", exact: true })
    .click();
  await expect(page.locator("aside")).toContainText("1000 nodes");
});
test("offline research and mobile layout", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route("**/api/models", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByText("The live engine is unavailable")).toBeVisible();
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await expect(page.getByText("Does the model learn?")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});
test("busy response is visible and retry remains possible", async ({
  page,
}) => {
  await mock(page);
  await page.unroute("**/api/analyze");
  await page.route("**/api/analyze", (route) =>
    route.fulfill({ status: 429, json: { detail: "busy" } }),
  );
  await page.goto("/");
  await expect(page.getByText("Engine online", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e4", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("engine is busy");
});
