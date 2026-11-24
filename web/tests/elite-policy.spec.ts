import { test, expect } from "@playwright/test";
import { Chess } from "chess.js";

async function mockPolicy(page: any, available = true) {
  const models = [
    { id: "handcrafted", name: "Classical", kind: "handcrafted", checkpoint_id: "handcrafted-v1", strategy: "alpha-beta" },
    { id: "supervised", name: "Supervised", kind: "supervised", checkpoint_id: "s".repeat(64), strategy: "alpha-beta" },
    { id: "supervised-1m", name: "Augmented Supervised", kind: "supervised", checkpoint_id: "a".repeat(64), strategy: "alpha-beta" },
    ...(available ? [{ id: "elite-policy", name: "Move Prediction", kind: "human-move-policy", checkpoint_id: "e".repeat(64), strategy: "direct-policy-v1" }] : []),
  ];
  await page.route("**/api/models", (route: any) => route.fulfill({ json: { default: "supervised", models } }));
  await page.route("**/api/analyze", (route: any) => {
    const body = route.request().postDataJSON();
    const game = new Chess(body.initial_fen);
    for (const move of body.moves) game.move({ from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4] });
    const move = game.moves({ verbose: true }).sort((a, b) => (a.from + a.to).localeCompare(b.from + b.to))[0];
    const uci = move.from + move.to + (move.promotion || "");
    game.move(move);
    return route.fulfill({ json: {
      best_move: uci, resulting_fen: game.fen(), terminal: null, value: null,
      depth: null, nodes: null, simulations: null, search: null, pv: [],
      elapsed_ms: 2, checkpoint_id: "e".repeat(64), model_id: "elite-policy",
      strategy: "direct-policy-v1", mate_score: false,
      root_rankings: [{ move: uci, probability: 1 }],
    } });
  });
  await page.route("**/api/evaluate-comparison", (route: any) => {
    const body = route.request().postDataJSON();
    return route.fulfill({ json: {
      ...body, perspective: "white", checkpoint_id: "e".repeat(64), terminal: null,
      position_value: null, search: null,
      policy: { root_rankings: [{ move: "g1f3", probability: .25 }], elapsed_ms: 2 },
      stockfish: { status: "ok", value: .1, pawn_score: .2, mate: null, mate_winner: null,
        terminal: null, version: "Stockfish 19", depth: 5, nodes: 10000, elapsed_ms: 30,
        limits: { nodes: 10000, time_ms: 1500 }, binary_sha256: "f".repeat(64), error: null },
    } });
  });
}

for (const color of ["White", "Black"]) {
  test(`human policy plays as opponent to ${color} without fabricated values`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", e => errors.push(e.message));
    await mockPolicy(page);
    await page.goto("/?model=elite-policy");
    await expect(page.locator(".model-option strong")).toHaveText([
      "Classical", "Supervised", "Augmented Supervised", "Move Prediction",
    ]);
    if (color === "Black") {
      await page.getByRole("button", { name: "Black", exact: true }).click();
      await expect(page.getByText("1 plies", { exact: true })).toBeVisible();
    } else {
      await page.getByRole("button", { name: "e2 white pawn", exact: true }).click();
      await page.getByRole("button", { name: "e4", exact: true }).click();
      await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
    }
    await expect(page.locator(".analysis-card")).toContainText("NO POSITION VALUE");
    await expect(page.locator(".analysis-card")).toContainText("INFERENCE TIME");
    await expect(page.getByRole("list", { name: "Predicted legal moves", exact: true })).toBeVisible();
    await expect(page.locator(".analysis-card .eval-track")).toHaveCount(0);
    await expect(page.locator(".player-strip").first()).toContainText("no search");
    expect(errors).toEqual([]);
  });
}

test("human policy comparison keeps Stockfish separate and accepts null search", async ({ page }) => {
  await mockPolicy(page);
  await page.goto("/?model=elite-policy");
  const comparison = page.getByRole("region", { name: "Stockfish comparison", exact: true });
  await comparison.getByRole("button", { name: "Compare with Stockfish", exact: true }).click();
  await expect(comparison).toContainText("Not provided by this move predictor");
  await expect(comparison).toContainText("No search");
  await expect(comparison.getByRole("list", { name: "Current-position predicted legal moves" })).toBeVisible();
  await comparison.getByText("Search details and model version", { exact: true }).click();
  await expect(comparison).toContainText("direct inference");
  await expect(comparison).toContainText("Stockfish 19");
  await expect(page.getByText("0 plies", { exact: true })).toBeVisible();
});

test("untrained policy stays unavailable and preserves opponent in saved URLs", async ({ page }) => {
  await mockPolicy(page, false);
  await page.goto("/?model=elite-policy&moves=e2e4,e7e5");
  const card = page.getByRole("button", { name: /^Move Prediction A convolutional/ });
  await expect(card).toBeDisabled();
  await expect(card).toContainText("IN RESEARCH");
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await page.reload();
  await expect(card).toBeDisabled();
  await expect(page).toHaveURL(/model=elite-policy/);
});

test("elite baseline report supports mobile reading and real result rows", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const evidence = await (await page.request.get('/research/elite-policy/evidence.json')).json();
  await page.goto('/research/elite-policy/index.html');
  await expect(page.getByRole('heading', { name: 'Learning moves from elite human games' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  if (evidence.status === 'complete') {
    await expect(page.getByRole('region', { name: 'Seed results' }).locator('tbody tr')).toHaveCount(3);
    const curves = page.getByRole('img', { name: 'Actual training and validation curves for every seed' });
    await expect(curves).toBeVisible();
    await expect.poll(() => curves.evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth > 0)).toBe(true);
  }
  await page.getByRole('link', { name: 'Research', exact: true }).click();
  await expect(page).toHaveURL(/\/research$/);
});
