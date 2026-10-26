import { test, expect } from "@playwright/test";

test.describe("production containers", () => {
  test.skip(!process.env.PLAYWRIGHT_BASE_URL, "Requires production containers");

  test("real models, gameplay, comparison and export", async ({
    page,
  }, testInfo) => {
    await page.goto("/");
    await expect(
      page.getByText("Engine online", { exact: true }),
    ).toBeVisible();
    for (const name of [
      /^Supervised A neural network learns/,
      /^Classical Handcrafted/,
    ]) {
      await page.getByRole("button", { name }).click();
      await page
        .getByRole("button", { name: "e2 white pawn", exact: true })
        .click();
      await page.getByRole("button", { name: "e4", exact: true }).click();
      await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
      await expect(page.locator(".analysis-card .hash")).toContainText(
        "Checkpoint",
      );
      const comparison = page.getByRole("region", {
        name: "Stockfish comparison",
        exact: true,
      });
      await comparison
        .getByRole("button", { name: "Compare with Stockfish", exact: true })
        .click();
      await expect(comparison.getByRole("table")).toBeVisible({
        timeout: 20000,
      });
      await expect(comparison.getByRole("alert")).toHaveCount(0);
      await expect(
        comparison.getByRole("row", { name: /Position evaluation/ }),
      ).toBeVisible();
      await expect(
        comparison.getByRole("row", { name: /After search/ }),
      ).toBeVisible();
      await comparison
        .getByText("Search details and model version", { exact: true })
        .click();
      await expect(comparison).toContainText("Stockfish 19");
      await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
      const download = page.waitForEvent("download");
      await page
        .getByRole("button", { name: "Export PGN", exact: true })
        .click();
      expect((await download).suggestedFilename()).toBe("chessengine-game.pgn");
      await page.getByRole("button", { name: "New game", exact: true }).click();
    }
    await page.getByRole("button", { name: "Compare", exact: true }).click();
    await page
      .getByRole("button", { name: "Compare models", exact: true })
      .click();
    await expect(page.locator("aside")).toContainText("nodes");
    await page.screenshot({
      path: testInfo.outputPath("production.png"),
      fullPage: true,
    });
  });

  test("mobile research survives API outage and reconnects", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    const api = process.env.DEPLOYMENT_API_URL!;
    await page.route(`${api}/**`, (route) => route.abort());
    await page.goto("/");
    await expect(
      page.getByText("The live engine is unavailable"),
    ).toBeVisible();
    await page.getByRole("button", { name: "Research", exact: true }).click();
    await page
      .getByText("Original study: charts and saved games", { exact: true })
      .click();
    await expect(page.locator(".seed-results tbody tr")).toHaveCount(3);
    const pdf = await page.request.get("/research/report.pdf");
    expect(pdf.headers()["content-type"]).toContain("pdf");
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBeTruthy();
    await page.unroute(`${api}/**`);
    await page.getByRole("button", { name: "Reconnect", exact: true }).click();
    await expect(page.locator(".connection")).toHaveClass("connection ready");
    await expect(
      page.getByText("The live engine is unavailable"),
    ).not.toBeVisible();
  });
});
