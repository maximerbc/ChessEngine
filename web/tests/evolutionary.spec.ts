import { test, expect } from "@playwright/test";

async function models(page: any) {
  await page.route("**/api/models", (route: any) =>
    route.fulfill({
      json: {
        default: "supervised",
        models: [
          {
            id: "handcrafted",
            name: "Classical",
            kind: "handcrafted",
            checkpoint_id: "handcrafted-v1",
          },
          {
            id: "supervised",
            name: "Supervised",
            kind: "supervised",
            checkpoint_id: "s".repeat(64),
          },
        ],
      },
    }),
  );
}

for (const url of ["/lab/rl?moves=e2e4,e7e5", "/?model=rl&moves=e2e4,e7e5"]) {
  test(`retired self-play redirects without making a move: ${url}`, async ({
    page,
  }) => {
    await models(page);
    const calls: string[] = [];
    page.on("request", (r) => {
      if (r.url().endsWith("/api/analyze")) calls.push(r.url());
    });
    await page.goto(url);
    await expect(page).toHaveURL(/research.*study=self-play-archive/);
    await expect(page.locator("#study-self-play-archive")).toContainText(
      "larger follow-up regressed",
    );
    expect(calls).toHaveLength(0);
    await page.getByRole("button", { name: "Demo", exact: true }).click();
    await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
    await expect(page.locator(".model-option strong")).toHaveText([
      "Classical",
      "Supervised",
      "Augmented Supervised",
      "GPU-trained Evolutionary",
    ]);
    await expect(
      page.getByRole("link", { name: "RL weights", exact: true }),
    ).toHaveCount(0);
  });
}

for (const [status, label] of [
  ["planned", "IN RESEARCH"],
  ["pilot_failed", "IN RESEARCH"],
]) {
  test(`unqualified evolutionary card stays disabled: ${status}`, async ({
    page,
  }) => {
    await models(page);
    await page.route("**/research/evolutionary/evidence.json", (route) =>
      route.fulfill({ json: { status, qualified: false } }),
    );
    await page.goto("/");
    const card = page.getByRole("button", { name: /^GPU-trained Evolutionary A new/ });
    await expect(card).toBeDisabled();
    await expect(card).toContainText(label);
    await page
      .getByRole("button", { name: "Explore the evidence", exact: true })
      .click();
    await expect(page).toHaveURL(/research.*study=gpu-value/);
    await expect(page.locator("#study-gpu-value")).toBeVisible();
  });
}
