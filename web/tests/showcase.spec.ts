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
