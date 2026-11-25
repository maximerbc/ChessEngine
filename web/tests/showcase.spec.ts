import { test, expect } from "@playwright/test";

test("local research preview pages explain and analyze the same position", async ({
  page,
}, info) => {
  test.skip(
    !process.env.RUN_SHOWCASE_TESTS,
    "Requires the four-model preview backend and assets",
  );
  for (const route of ["classical", "supervised", "move-guided"]) {
    await page.goto(`/lab/${route}?moves=e2e4,e7e5`);
    await expect(
      page.getByRole("button", { name: "Analyze this position", exact: true }),
    ).toBeEnabled();
    await page
      .getByRole("button", { name: "Analyze this position", exact: true })
      .click();
    await expect(
      page.getByText("Last analysis:", { exact: false }),
    ).toBeVisible({ timeout: 10000 });
    if (route === "move-guided")
      await expect(
        page.getByRole("list", { name: "Ranked root moves" }).locator("li"),
      ).toHaveCount(5);
    await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
    await page.screenshot({
      path: info.outputPath(`${route}.png`),
      fullPage: true,
    });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: info.outputPath("mobile.png"),
    fullPage: true,
  });
});

test("local preview retains manual analysis with matching identities (mocked)", async ({
  page,
}) => {
  test.skip(!process.env.RUN_SHOWCASE_TESTS, "Requires VITE_SHOWCASE=1");
  await page.route("**/api/models", (route) =>
    route.fulfill({
      json: {
        default: "supervised-1m",
        models: [
          {
            id: "handcrafted",
            name: "Classical",
            kind: "handcrafted",
            checkpoint_id: "handcrafted-v1",
          },
        ],
      },
    }),
  );
  await page.route("**/research/model-catalog.json", (route) =>
    route.fulfill({
      json: {
        models: {
          handcrafted: {
            checkpoint_id: "handcrafted-v1",
            name: "Classical",
            facts: [],
            experimental: false,
          },
        },
      },
    }),
  );
  const requests: any[] = [];
  await page.route("**/api/analyze", (route) => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({
      json: {
        best_move: "g1f3",
        value: 0.12,
        depth: 2,
        nodes: 10000,
        elapsed_ms: 100,
        pv: ["g1f3"],
        checkpoint_id: "handcrafted-v1",
        terminal: null,
        mate_score: false,
        explanation: {
          raw_value: 0.12,
          side_to_move: "white",
          feature_version: null,
          piece_counts: { own: { p: 8 }, opponent: { p: 8 } },
          handcrafted: { material: 0, position: 0.1, scale: 6 },
        },
      },
    });
  });
  await page.goto("/lab/classical?moves=e2e4,e7e5");
  await page
    .getByRole("button", { name: "Analyze this position", exact: true })
    .click();
  await expect(
    page.getByText("Last analysis:", { exact: false }),
  ).toBeVisible();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  expect(requests).toHaveLength(1);
  expect(requests[0].moves).toEqual(["e2e4", "e7e5"]);
  expect(requests[0].model_id).toBe("handcrafted");
  await expect(
    page.getByRole("button", { name: "A quiet opening", exact: true }),
  ).toBeVisible();
});
