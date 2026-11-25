import { test, expect } from "@playwright/test";

async function offline(page: any) {
  await page.route("**/api/models", (route: any) => route.abort());
}

test("Research leads with three approaches and four collapsed families", async ({
  page,
}, info) => {
  await offline(page);
  await page.goto("/research");
  await expect(
    page.getByRole("heading", { name: "Three ways to choose a move." }),
  ).toBeVisible();
  await expect(page.locator(".approach-overview h3")).toHaveText([
    "Classical",
    "Augmented Supervised",
    "Direct-Move CNN",
  ]);
  await expect(page.locator(".approach-overview")).toContainText(
    "0.21476 across three seeds",
  );
  await expect(page.locator(".research-family")).toHaveCount(4);
  await expect(page.locator(".research-family[open]")).toHaveCount(0);
  await expect(
    page
      .locator(".research-family > summary")
      .filter({ hasText: "Move prediction" }),
  ).toContainText("collapsed seed");
  await page.screenshot({
    path: info.outputPath("research-desktop.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: info.outputPath("research-mobile.png"),
    fullPage: true,
  });
});

test("study permalinks expand the correct family and preserve replay", async ({
  page,
}) => {
  await offline(page);
  await page.goto("/research?study=stockfish-move-policy-1m-20261124");
  await expect(
    page.locator("#study-stockfish-move-policy-1m-20261124"),
  ).toBeVisible();
  await expect(page.locator(".research-family[open]")).toHaveCount(1);
  await expect(
    page.locator("#study-stockfish-move-policy-1m-20261124"),
  ).toContainText("failed strict export parity");
  const report = await page
    .locator("#study-stockfish-move-policy-1m-20261124 a")
    .first()
    .getAttribute("href");
  expect((await page.request.get(report!)).status()).toBe(200);
  await page
    .getByRole("button", { name: "Explore the evidence", exact: true })
    .click();
  await expect(page).toHaveURL(/study=gpu-value/);
  await expect(page.locator("#study-gpu-value")).toBeInViewport();
  await page.goto("/research?study=original-study");
  await expect(
    page.locator("#study-original-study .archive-dashboard"),
  ).toBeVisible();
  await expect(
    page.getByText("Does the model learn?", { exact: true }),
  ).toBeVisible();
});

for (const id of [
  "supervised",
  "supervised-relative",
  "supervised-averaged",
  "stockfish-policy",
  "evolutionary",
  "rl",
  "move-guided",
]) {
  test(`retired ${id} preserves history offline and on browser navigation`, async ({
    page,
  }) => {
    await offline(page);
    await page.goto(`/?model=${id}&mode=compare&moves=e2e4,e7e5`);
    await expect(page).toHaveURL(/model=supervised-1m/);
    await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
    await expect(
      page.getByRole("status").filter({ hasText: "now in Research" }),
    ).toBeVisible();
    await expect(
      page.locator(".model-option[aria-pressed=true]"),
    ).toContainText("Augmented Supervised");
    await page.evaluate(() => {
      history.pushState(null, "", "/?model=supervised&moves=d2d4,d7d5");
      dispatchEvent(new PopStateEvent("popstate"));
    });
    await expect(page).toHaveURL(/model=supervised-1m/);
    await expect(page.locator(".move-list")).toContainText("d4");
    await page.goBack();
    await expect(page.locator(".move-list")).toContainText("e4");
  });
}
