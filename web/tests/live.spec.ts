import { test, expect } from "@playwright/test";

test("live backend and released study render together", async ({ page }, testInfo) => {
  test.skip(
    !process.env.RUN_LIVE_TESTS,
    "Requires the local Python server and released research artifacts.",
  );
  await page.goto("/");
  await expect(page.getByText("Engine online", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: /Supervised model Learned from Stockfish/,
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e4", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible({
    timeout: 10000,
  });
  await expect(page.locator(".hash")).toContainText("Checkpoint");
  await page.screenshot({
    path: testInfo.outputPath("demo-preview.jpg"),
    fullPage: true,
    type: "jpeg",
    quality: 85,
  });
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await expect(
    page.getByText("One result is not the whole experiment."),
  ).toBeVisible();
  await expect(page.locator(".seed-results tbody tr")).toHaveCount(3);
  const report = await page.request.get("/research/report.pdf");
  expect(report.status()).toBe(200);
  expect(report.headers()["content-type"]).toContain("pdf");
});
