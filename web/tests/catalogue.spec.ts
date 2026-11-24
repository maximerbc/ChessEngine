import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { Chess } from "chess.js";

const manifest = JSON.parse(
  readFileSync("../artifacts/models/manifest.json", "utf8"),
);
const models = [
  {
    id: "handcrafted",
    name: "Classical",
    kind: "handcrafted",
    group: "Classical",
    description: "Handcrafted evaluation",
    checkpoint_id: "handcrafted-v1",
    search: { strategy: "alpha-beta" },
  },
  ...manifest.models.map((row: any) => ({
    ...row,
    checkpoint_id: row.sha256,
    strategy: row.search.strategy,
  })),
];

async function mockCatalogue(page: any) {
  await page.route("**/api/models", (route: any) =>
    route.fulfill({ json: { default: "supervised", models } }),
  );
  await page.route("**/api/analyze", (route: any) => {
    const body = route.request().postDataJSON();
    const model = models.find((row: any) => row.id === body.model_id);
    const board = new Chess(body.initial_fen);
    for (const move of body.moves)
      board.move({
        from: move.slice(0, 2),
        to: move.slice(2, 4),
        promotion: move[4],
      });
    const move = board.moves({ verbose: true })[0];
    const uci = move.from + move.to + (move.promotion || "");
    const policy = model.search.strategy === "direct-policy-v1";
    return route.fulfill({
      json: {
        model_id: model.id,
        checkpoint_id: model.checkpoint_id,
        best_move: uci,
        terminal: null,
        value: policy ? null : 0.1,
        depth: policy ? null : 1,
        nodes: policy ? null : 32,
        elapsed_ms: 10,
        pv: policy ? [] : [uci],
        strategy: model.search.strategy,
        mate_score: false,
        root_rankings: policy ? [{ move: uci, probability: 1 }] : undefined,
      },
    });
  });
}

for (const model of models) {
  test(`${model.id} is selectable and preserves its identity after play and refresh`, async ({
    page,
  }) => {
    await mockCatalogue(page);
    await page.goto("/");
    await expect(
      page.getByText("20 of 20 opponents", { exact: true }),
    ).toBeVisible();
    const search = page.getByRole("searchbox", { name: "Search opponents" });
    await search.fill(model.name);
    await page
      .locator(".model-option")
      .filter({ has: page.locator("strong", { hasText: model.name }) })
      .filter({ hasText: model.description })
      .first()
      .click();
    await expect(page).toHaveURL(new RegExp(`model=${model.id}(?:&|$)`));
    await expect(page.locator(".selected-opponent")).toContainText(model.name);
    if (model.search.strategy === "direct-policy-v1") {
      await expect(page.locator(".analysis-card .eval-track")).toHaveCount(0);
      await expect(page.locator(".analysis-card")).toContainText(
        "NO POSITION VALUE",
      );
      await expect(page.locator(".player-strip").first()).toContainText(
        model.id === "stockfish-policy"
          ? "Stockfish move prediction"
          : "Human-move prediction",
      );
    }
    await page.getByRole("button", { name: "Black", exact: true }).click();
    await expect(page.getByText("1 plies", { exact: true })).toBeVisible();
    if (model.search.strategy === "direct-policy-v1") {
      await expect(page.locator(".analysis-card .eval-track")).toHaveCount(0);
      await expect(
        page.getByRole("list", { name: "Predicted legal moves", exact: true }),
      ).toBeVisible();
    }
    await page.reload();
    await expect(page.getByText("1 plies", { exact: true })).toBeVisible();
    await expect(page).toHaveURL(new RegExp(`model=${model.id}(?:&|$)`));
    await expect(
      page.locator(".model-option[aria-pressed=true]"),
    ).toBeDisabled();
  });
}

test("catalogue searches groups, keeps the selected summary and fits mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockCatalogue(page);
  await page.goto("/?model=supervised-relative");
  const search = page.getByRole("searchbox", { name: "Search opponents" });
  await search.fill("Regularization");
  await expect(page.locator(".model-option")).toHaveCount(4);
  await expect(page.locator(".selected-opponent")).toContainText(
    "relative features",
  );
  await search.fill("no matching model");
  await expect(page.getByText("No opponents match your search.")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await search.fill("averaged weights");
  await page.locator(".model-option").focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/model=supervised-averaged/);
});

test("unknown opponent URL never silently substitutes a model", async ({
  page,
}) => {
  await mockCatalogue(page);
  await page.goto("/?model=missing-variant&moves=e2e4,e7e5");
  await expect(
    page.getByText(
      "The requested opponent is unavailable. Choose an available opponent to play.",
    ),
  ).toBeVisible();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await expect(page).toHaveURL(/model=missing-variant/);
});
