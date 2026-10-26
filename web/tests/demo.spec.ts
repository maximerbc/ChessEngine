import { test, expect } from "@playwright/test";

async function mock(page: any) {
  await page.route("**/research/studies.json", async (route: any) => {
    const response = await route.fetch();
    const data = await response.json();
    data.studies.find((s: any) => s.id === "original-study").checkpoint_ids = {
      supervised: "s".repeat(64),
      rl: "r".repeat(64),
    };
    await route.fulfill({ json: data });
  });
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
        best_move:
          b.moves.length === 0
            ? "e2e4"
            : b.moves.length === 1
              ? "e7e5"
              : "g8f6",
        value: 0.12,
        depth: 2,
        nodes: 1000,
        elapsed_ms: 120,
        pv: ["e7e5", "g1f3"],
        checkpoint_id:
          b.model_id === "rl"
            ? "r".repeat(64)
            : b.model_id === "handcrafted"
              ? "handcrafted-v1"
              : "s".repeat(64),
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
  await page
    .getByText("Original study: charts and saved games", { exact: true })
    .click();
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
    .getByRole("button", {
      name: /^Self-play Starting from the supervised/,
    })
    .click();
  await expect(page.locator(".player-strip").first()).toContainText(
    "Self-play",
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
  await page
    .getByText("Original study: charts and saved games", { exact: true })
    .click();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "e4 white pawn", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Back", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "e2 white pawn", exact: true }),
  ).toBeVisible();
});

test("legacy preview links redirect to research without substituting a released model", async ({
  page,
}) => {
  await mock(page);
  await page.goto("/lab/move-guided?moves=e2e4,e7e5");
  await expect(page).toHaveURL(/research.*study=move-ranking/);
  await expect(page.locator("#study-move-ranking")).toContainText("48.75%");
  await page.getByRole("button", { name: "Demo", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await expect(page.locator(".model-tile")).toHaveCount(0);
});

test("research deep links, compare refresh and browser history preserve the position", async ({
  page,
}) => {
  await mock(page);
  await page.goto("/?moves=e2e4,e7e5&mode=compare");
  await expect(
    page.getByRole("button", { name: "Compare models", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await page
    .getByText("Original study: charts and saved games", { exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "The research journal" }),
  ).toBeVisible();
  await page.goBack();
  await expect(
    page.getByRole("button", { name: "Compare models", exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Compare models", exact: true }),
  ).toBeVisible();
});

test("the main navigation contains only Demo and Research and reports work offline", async ({
  page,
}) => {
  await page.route("**/api/models", (route) => route.abort());
  await page.goto("/research?study=rl-update-diagnosis");
  await expect(
    page
      .getByRole("navigation", { name: "Main navigation" })
      .getByRole("button"),
  ).toHaveCount(2);
  await expect(page.locator("#study-rl-update-diagnosis")).toContainText(
    "Off-policy",
  );
  for (const path of [
    "/research/move-ranking/index.html",
    "/research/rl-improvement/index.html",
    "/research/policy-value/index.html",
    "/research/policy-value/release-scope.json",
    "/research/report.pdf",
  ]) {
    expect((await page.request.get(path)).ok()).toBeTruthy();
  }
  await expect(page.locator("#study-policy-value")).toContainText(
    "Completed feasibility pilot",
  );
  await expect(page.locator("#study-policy-value")).toContainText("deferred");
  const evidence = await page.request.get(
    "/research/policy-value/evidence.json",
  );
  expect((await evidence.json()).status.status).toBe("feasibility_complete");
});

test("promotion choices preserve underpromotions", async ({ page }) => {
  await mock(page);
  await page.unroute("**/api/analyze");
  await page.route("**/api/analyze", (route) =>
    route.fulfill({
      json: {
        best_move: null,
        value: 0,
        depth: 1,
        nodes: 1,
        elapsed_ms: 1,
        pv: [],
        checkpoint_id: "s".repeat(64),
        terminal: null,
        mate_score: false,
      },
    }),
  );
  const query = new URLSearchParams({ fen: "k7/4P3/8/8/8/8/8/7K w - - 0 1" });
  await page.goto("/?" + query);
  await page
    .getByRole("button", { name: "e7 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e8", exact: true }).click();
  await expect(
    page.getByRole("dialog", { name: "Choose promotion" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Promote to knight", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "e8 white knight", exact: true }),
  ).toBeVisible();
});

test("released checkpoint mismatch blocks a move", async ({ page }) => {
  await mock(page);
  await page.unroute("**/api/analyze");
  await page.route("**/api/analyze", (route) =>
    route.fulfill({
      json: {
        best_move: "e7e5",
        value: 0,
        depth: 1,
        nodes: 1,
        elapsed_ms: 1,
        pv: [],
        checkpoint_id: "wrong-release",
        terminal: null,
        mate_score: false,
      },
    }),
  );
  await page.goto("/");
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e4", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("checkpoint changed");
  await expect(page.getByText("1 plies", { exact: true })).toBeVisible();
});

test("compare uses the actual shared position and Research has a compact archive", async ({
  page,
}, info) => {
  await mock(page);
  const positions: any[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/api/analyze"))
      positions.push(request.postDataJSON());
  });
  await page.goto("/?moves=e2e4,e7e5");
  await page.getByRole("button", { name: "Compare", exact: true }).click();
  await page
    .getByRole("button", { name: "Compare models", exact: true })
    .click();
  await expect(page.locator(".comparison-card")).toHaveCount(3);
  expect(positions.map((p) => p.model_id)).toEqual([
    "handcrafted",
    "supervised",
    "rl",
  ]);
  expect(
    positions.every(
      (p) =>
        p.moves.join(",") === "e2e4,e7e5" &&
        p.nodes === 1000 &&
        p.time_ms === 3000,
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await expect(
    page.getByText("Does the model learn?", { exact: true }),
  ).not.toBeVisible();
  await page.screenshot({
    path: info.outputPath("research.png"),
    fullPage: true,
  });
});

test("new game and move history survive a refresh", async ({ page }) => {
  await mock(page);
  await page.goto("/?moves=e2e4,e7e5");
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "New game", exact: true }).click();
  await expect(page.getByText("0 plies", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByText("0 plies", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e4", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
});

for (const [id, name] of [
  ["handcrafted", "Classical"],
  ["supervised", "Supervised"],
  ["rl", "Self-play"],
]) {
  for (const color of ["White", "Black"]) {
    test(`${name} plays with the user as ${color} and keeps its identity`, async ({
      page,
    }) => {
      await mock(page);
      const requests: any[] = [];
      page.on("request", (request) => {
        if (request.url().endsWith("/api/analyze"))
          requests.push(request.postDataJSON());
      });
      await page.goto("/");
      const cards = page.locator(".model-option");
      await expect(cards).toHaveCount(3);
      await expect(cards.locator("strong")).toHaveText([
        "Classical",
        "Supervised",
        "Self-play",
      ]);
      await page.getByRole("button", { name: new RegExp(`^${name} `) }).click();
      await expect(
        page.getByRole("region", { name: `${name} explained` }),
      ).toContainText("Playing here does not train the models.");
      if (color === "Black") {
        await page.getByRole("button", { name: "Black", exact: true }).click();
        await expect(page.getByText("1 plies", { exact: true })).toBeVisible();
      } else {
        await page
          .getByRole("button", { name: "e2 white pawn", exact: true })
          .click();
        await page.getByRole("button", { name: "e4", exact: true }).click();
        await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
      }
      expect(requests).toHaveLength(1);
      expect(requests[0].model_id).toBe(id);
      await expect(page.locator(".analysis-card .hash")).toContainText(
        id === "handcrafted"
          ? "handcrafted-"
          : (id === "rl" ? "r" : "s").repeat(12),
      );
      await expect(
        page.getByText("Start a new game to change opponents."),
      ).toBeVisible();
      for (const card of await cards.all()) await expect(card).toBeDisabled();
      // White avoids an immediate opening move after resetting as Black.
      await page.getByRole("button", { name: "White", exact: true }).click();
      for (const card of await cards.all()) await expect(card).toBeEnabled();
    });
  }
}

test("released evidence is matched to checkpoint identity", async ({
  page,
}) => {
  await mock(page);
  await page.goto("/?model=rl");
  const explanation = page.getByRole("region", { name: "Self-play explained" });
  await expect(explanation).toContainText(
    "inconclusive evidence of improvement",
  );
  await expect(
    explanation.getByRole("link", { name: /Original local-study/ }),
  ).toHaveAttribute("href", "/research?study=original-study");
  await expect(explanation).toContainText("different checkpoints");
  await expect(explanation).toContainText("full comparison deferred");
  await page.getByRole("button", { name: /^Supervised A neural/ }).click();
  await expect(
    page.getByRole("region", { name: "Supervised explained" }),
  ).toContainText("not Stockfish moves");
  await page.route("**/research/studies.json", (route) =>
    route.fulfill({
      json: {
        studies: [
          { id: "original-study", checkpoint_ids: { supervised: "other" } },
        ],
      },
    }),
  );
  await page.reload();
  await expect(
    page.getByRole("region", { name: "Supervised explained" }),
  ).toContainText("Evidence unavailable for this version");
  await expect(
    page.getByRole("link", { name: /Original local-study methods/ }),
  ).toHaveCount(0);
});

test("unavailable requested opponent stays selected without fallback; two opponents can compare", async ({
  page,
}) => {
  await mock(page);
  await page.route("**/api/models", (route) =>
    route.fulfill({
      json: {
        default: "handcrafted",
        models: [
          {
            id: "handcrafted",
            name: "Baseline",
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
  const requests: any[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/api/analyze"))
      requests.push(request.postDataJSON());
  });
  await page.goto("/?model=rl");
  await expect(
    page.getByText("The requested opponent is unavailable.", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /^Self-play Starting/ }),
  ).toBeDisabled();
  await expect(page.locator(".model-option[aria-pressed=true]")).toContainText(
    "Self-play",
  );
  await expect(
    page.getByRole("button", { name: "e2 white pawn", exact: true }),
  ).toBeDisabled();
  expect(requests).toHaveLength(0);
  await page.getByRole("button", { name: "Compare", exact: true }).click();
  await page
    .getByRole("button", { name: "Compare models", exact: true })
    .click();
  await expect(page.locator(".comparison-card")).toHaveCount(2);
  expect(requests.map((r) => r.model_id)).toEqual([
    "handcrafted",
    "supervised",
  ]);
  expect(
    requests.every(
      (r) => r.nodes === 1000 && r.time_ms === 3000 && r.moves.length === 0,
    ),
  ).toBeTruthy();
});

test("loading and offline cards stay visible and keyboard selection works on mobile", async ({
  page,
}, info) => {
  await mock(page);
  let releaseModels!: () => void;
  const gate = new Promise<void>((resolve) => {
    releaseModels = resolve;
  });
  await page.route("**/api/models", async (route) => {
    await gate;
    await route.fallback();
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator(".model-option")).toHaveCount(3);
  await expect(page.locator(".model-options")).toContainText("LOADING");
  for (const card of await page.locator(".model-option").all())
    await expect(card).toBeDisabled();
  releaseModels();
  const classical = page.getByRole("button", {
    name: /^Classical Handcrafted/,
  });
  await expect(classical).toBeEnabled();
  await classical.focus();
  await page.keyboard.press("Enter");
  await expect(classical).toHaveAttribute("aria-pressed", "true");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: info.outputPath("three-approaches-mobile.png"),
    fullPage: true,
  });
  await page.route("**/api/models", (route) => route.abort());
  await page.reload();
  await expect(page.locator(".model-option")).toHaveCount(3);
  for (const card of await page.locator(".model-option").all()) {
    await expect(card).toBeDisabled();
    await expect(card).toContainText("UNAVAILABLE");
  }
});

test("explanation research links preserve the game and selected opponent", async ({
  page,
}, info) => {
  await mock(page);
  await page.goto("/?model=rl&moves=e2e4,e7e5");
  const explanation = page.getByRole("region", { name: "Self-play explained" });
  await expect(explanation).toContainText(
    "inconclusive evidence of improvement",
  );
  await page.screenshot({
    path: info.outputPath("three-approaches-desktop.png"),
    fullPage: true,
  });
  await explanation
    .getByRole("link", { name: /Original local-study methods/ })
    .click();
  await expect(page).toHaveURL(/research.*study=original-study/);
  await page.getByRole("button", { name: "Demo", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await expect(page.locator(".model-option[aria-pressed=true]")).toContainText(
    "Self-play",
  );
});
