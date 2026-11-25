import { test, expect } from "@playwright/test";

test("real current-position comparison preserves play for each released evaluator", async ({
  page,
}, info) => {
  test.skip(
    !process.env.RUN_LIVE_TESTS,
    "Requires the local Python server and Stockfish 19.",
  );
  await page.goto("/");
  const installed = await (await page.request.get("/api/models")).json();
  for (const model of installed.models) {
    await page.goto(`/?model=${model.id}`);
    const comparison = page.getByRole("region", {
      name: "Stockfish comparison",
      exact: true,
    });
    await expect(
      comparison.getByRole("button", {
        name: "Compare with Stockfish",
        exact: true,
      }),
    ).toBeEnabled();
    const response = page.waitForResponse("**/evaluate-comparison");
    await comparison
      .getByRole("button", { name: "Compare with Stockfish", exact: true })
      .click();
    const data = await (await response).json();
    expect(data.stockfish.status).toBe("ok");
    expect(data.model_id).toBe(model.id);
    expect(data.checkpoint_id).toBe(model.checkpoint_id);
    await expect(
      comparison.getByRole("row", { name: /Position evaluation/ }),
    ).toBeVisible();
    if (model.strategy === "direct-policy-v1") {
      expect(data.search).toBeNull();
      expect(data.position_value).toBeNull();
      expect(data.policy.root_rankings.length).toBeGreaterThan(0);
      await expect(comparison).toContainText("No search");
    } else {
      await expect(
        comparison.getByRole("row", { name: /After search/ }),
      ).toBeVisible();
    }
    await expect(page.getByText("0 plies", { exact: true })).toBeVisible();
    await comparison
      .getByText("Search details and model version", { exact: true })
      .click();
    await expect(comparison).toContainText("Stockfish 19");
    await expect(comparison.getByRole("alert")).toHaveCount(0);
  }
  // Playing as Black produces a Black-to-move position after the engine's
  // opening move. Comparison must retain White's perspective and not reply.
  await page.getByRole("button", { name: "Black", exact: true }).click();
  await expect(page.getByText("1 plies", { exact: true })).toBeVisible({
    timeout: 10000,
  });
  const comparison = page.getByRole("region", {
    name: "Stockfish comparison",
    exact: true,
  });
  const response = page.waitForResponse("**/evaluate-comparison");
  await comparison
    .getByRole("button", { name: "Compare with Stockfish", exact: true })
    .click();
  const data = await (await response).json();
  expect(data.fen.split(" ")[1]).toBe("b");
  expect(data.perspective).toBe("white");
  expect(data.stockfish.status).toBe("ok");
  await expect(comparison.getByRole("table")).toBeVisible();
  await expect(page.getByText("1 plies", { exact: true })).toBeVisible();
  await page.screenshot({
    path: info.outputPath("position-comparison-real-desktop.png"),
    fullPage: true,
  });
});

test("live backend and released study render together", async ({
  page,
}, testInfo) => {
  test.skip(
    !process.env.RUN_LIVE_TESTS,
    "Requires the local Python server and released research artifacts.",
  );
  await page.goto("/");
  await expect(page.getByText("Engine online", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: /^Augmented Supervised A neural/,
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e4", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible({
    timeout: 10000,
  });
  await expect(page.locator(".analysis-card .hash")).toContainText(
    "Checkpoint",
  );
  await page.screenshot({
    path: testInfo.outputPath("demo-preview.jpg"),
    fullPage: true,
    type: "jpeg",
    quality: 85,
  });
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await page
    .locator(".research-family > summary")
    .filter({ hasText: "Self-play" })
    .click();
  await page
    .getByText("Original study: charts and saved games", { exact: true })
    .click();
  await expect(
    page.getByText("One result is not the whole experiment."),
  ).toBeVisible();
  await expect(page.locator(".seed-results tbody tr")).toHaveCount(3);
  const report = await page.request.get("/research/report.pdf");
  expect(report.status()).toBe(200);
  expect(report.headers()["content-type"]).toContain("pdf");
});
