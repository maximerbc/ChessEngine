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

test("reconnect recovers after an unavailable engine", async ({ page }) => {
  let available = false;
  await page.route("**/api/models", async (route) => {
    if (!available) return route.abort();
    return route.fulfill({
      json: {
        default: "handcrafted",
        models: [
          {
            id: "handcrafted",
            name: "Classical baseline",
            kind: "handcrafted",
            evidence: "baseline",
            checkpoint_id: "handcrafted-v1",
          },
        ],
      },
    });
  });
  await page.goto("/");
  await expect(page.getByText("The live engine is unavailable")).toBeVisible();
  available = true;
  await page.getByRole("button", { name: "Reconnect", exact: true }).click();
  await expect(page.getByText("Engine online", { exact: true })).toBeVisible();
});

test("model choice, drag move, and restart are coherent", async ({ page }) => {
  await mock(page);
  await page.goto("/");
  await expect(page.getByText("Engine online", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: /Self-play model Learned from self-play/ })
    .click();
  await expect(page.locator(".player-strip").first()).toContainText(
    "Self-play model",
  );
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .dragTo(page.getByRole("button", { name: "e4", exact: true }));
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "New game", exact: true }).click();
  await expect(page.getByText("0 plies", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "e2 white pawn", exact: true }),
  ).toBeVisible();
});

test("saved-game replay works without the API", async ({ page }) => {
  await page.route("**/api/models", (route) => route.abort());
  await page.route("**/research/results.json", (route) =>
    route.fulfill({
      json: {
        status: "Test fixture",
        games: [
          {
            opening_id: "test",
            candidate_color: "white",
            score: 1,
            capped: false,
            pgn: '[Result "1-0"]\n\n1. e4 e5 2. Nf3 1-0',
          },
        ],
        limitations: ["Test fixture only"],
      },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "e4 white pawn", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Back", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "e2 white pawn", exact: true }),
  ).toBeVisible();
});

test("model pages preserve the position and enforce checkpoint identity", async ({ page }) => {
  await mock(page);
  await page.route("**/research/model-catalog.json", route => route.fulfill({json:{models:{
    supervised:{checkpoint_id:"s".repeat(64),name:"Supervised",facts:["Verified seed 42"],experimental:false},
    rl:{checkpoint_id:"different",name:"RL",facts:["Negative live-self-play result"],experimental:true}
  }}}));
  await page.goto("/lab/supervised?moves=e2e4,e7e5");
  await expect(page.getByRole("heading", {name:/Supervised evaluator/,level:1})).toBeVisible();
  await expect(page.getByText("2 plies", {exact:true})).toBeVisible();
  await expect(page.getByRole("button",{name:"Analyze this position",exact:true})).toBeEnabled();
  await page.locator('.model-tile').filter({hasText:'Self-play experiment'}).click();
  await expect(page).toHaveURL(/lab\/rl.*moves=e2e4%2Ce7e5/);
  await expect(page.getByText("2 plies", {exact:true})).toBeVisible();
  await expect(page.getByRole("button",{name:"Analyze this position",exact:true})).toBeDisabled();
  await expect(page.getByText("The live engine is serving a different checkpoint.", {exact:false})).toBeVisible();
});

test("model explanation remains readable offline on mobile", async ({ page }) => {
  await page.setViewportSize({width:390,height:844});
  await page.route("**/api/models", route=>route.abort());
  await page.goto('/lab/move-guided');
  await expect(page.getByRole('heading',{name:/Move-guided engine/,level:1})).toBeVisible();
  await expect(page.getByText('Does learned ordering make search more efficient?',{exact:true}).last()).toBeVisible();
  await expect(page.getByRole('button',{name:'Analyze this position',exact:true})).toBeDisabled();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
});
