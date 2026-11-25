import { test, expect } from "@playwright/test";

async function mock(page: any) {
  await page.route("**/research/studies.json", async (route: any) => {
    const response = await route.fetch();
    const data = await response.json();
    data.studies.find((s: any) => s.id === "original-study").checkpoint_ids = {
      supervised: "s".repeat(64),
      rl: "r".repeat(64),
    };
    data.studies.find((s: any) => s.id === "gpu-value").checkpoint_ids = {
      "evolutionary-gpu": "e".repeat(64),
      "supervised-1m": "a".repeat(64),
    };
    await route.fulfill({ json: data });
  });
  await page.route("**/research/evolutionary/evidence.json", (route: any) =>
    route.fulfill({ json: { qualified: true, status: "complete" } }),
  );
  await page.route("**/api/models", async (route: any) =>
    route.fulfill({
      json: {
        default: "supervised-1m",
        models: [
          {
            id: "supervised",
            name: "Supervised model",
            kind: "supervised",
            evidence: "pilot",
            checkpoint_id: "s".repeat(64),
          },
          {
            id: "evolutionary-gpu",
            strategy: "puct-uniform-v1",
            name: "GPU-trained Evolutionary",
            kind: "evolutionary",
            evidence: "pilot",
            checkpoint_id: "e".repeat(64),
          },
          {
            id: "supervised-1m",
            name: "Augmented Supervised",
            kind: "supervised",
            evidence: "test-qualified",
            checkpoint_id: "a".repeat(64),
            strategy: "alpha-beta",
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
    expect(b.nodes).toBe(30000);
    expect(b.time_ms).toBe(3000);
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
        nodes: b.model_id === "evolutionary-gpu" ? 0 : 10000,
        strategy:
          b.model_id === "evolutionary-gpu" ? "puct-uniform-v1" : "alpha-beta",
        simulations: b.model_id === "evolutionary-gpu" ? 64 : undefined,
        elapsed_ms: 120,
        pv: ["e7e5", "g1f3"],
        checkpoint_id:
          b.model_id === "evolutionary-gpu"
            ? "e".repeat(64)
            : b.model_id === "handcrafted"
              ? "handcrafted-v1"
              : b.model_id === "supervised-1m"
                ? "a".repeat(64)
                : "s".repeat(64),
        terminal: null,
        mate_score: false,
      },
    });
  });
}
test("play, reply, undo, restart and export", async ({ page }) => {
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
  await page.getByRole("button", { name: "New game", exact: true }).click();
  await expect(page.getByText("0 plies", { exact: true })).toBeVisible();
});
test("offline research and mobile layout", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route("**/api/models", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByText("The live engine is unavailable")).toBeVisible();
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await page
    .locator(".research-family > summary")
    .filter({ hasText: "Self-play" })
    .click();
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
      name: /^Augmented Supervised A neural/,
    })
    .click();
  await expect(page.locator(".player-strip").first()).toContainText(
    "Augmented Supervised",
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
    .locator(".research-family > summary")
    .filter({ hasText: "Self-play" })
    .click();
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

for (const offline of [false, true]) {
  test(`legacy Compare links preserve position and opponent through refresh and history (${offline ? "offline" : "online"})`, async ({
    page,
  }) => {
    await mock(page);
    if (offline) await page.route("**/api/models", (route) => route.abort());
    const fen = "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2";
    const moves = "g1f3,g8f6";
    const query = new URLSearchParams({
      fen,
      moves,
      model: "handcrafted",
      mode: "compare",
    });
    const requests: string[] = [];
    page.on("request", (request) => {
      if (request.url().endsWith("/api/analyze")) requests.push(request.url());
    });
    await page.goto(`/?${query}`);
    const preserved = async () => {
      await expect(page).toHaveURL(/mode=play/);
      const current = new URL(page.url()).searchParams;
      expect(current.get("fen")).toBe(fen);
      expect(current.get("moves")).toBe(moves);
      expect(current.get("model")).toBe("handcrafted");
      await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
      await expect(
        page.locator(".model-option[aria-pressed=true]"),
      ).toContainText("Classical");
      await expect(
        page.getByRole("button", { name: "f3 white knight", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Compare", exact: true }),
      ).toHaveCount(0);
      await expect(
        page.getByRole("button", { name: "Compare models", exact: true }),
      ).toHaveCount(0);
    };
    await preserved();
    if (offline)
      await expect(
        page.getByText("The live engine is unavailable"),
      ).toBeVisible();
    else
      await expect(
        page.getByText("Engine online", { exact: true }),
      ).toBeVisible();
    await page.reload();
    await preserved();
    // Simulate a legacy entry remaining in the browser's session history.
    await page.evaluate(() => {
      const url = new URL(location.href);
      url.searchParams.set("mode", "compare");
      history.replaceState(null, "", url);
    });
    await page.getByRole("button", { name: "Research", exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "Three ways to choose a move." }),
    ).toBeVisible();
    await page.goBack();
    await preserved();
    await page.goForward();
    await expect(
      page.getByRole("heading", { name: "Three ways to choose a move." }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Demo", exact: true }).click();
    await preserved();
    expect(requests).toHaveLength(0);
  });
}

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

test("Research preserves the shared position and has a compact archive", async ({
  page,
}, info) => {
  await mock(page);
  await page.goto("/?moves=e2e4,e7e5");
  await page.getByRole("button", { name: "Research", exact: true }).click();
  await expect(
    page.getByText("Does the model learn?", { exact: true }),
  ).not.toBeVisible();
  await page.screenshot({
    path: info.outputPath("research.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Demo", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
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
  ["supervised-1m", "Augmented Supervised"],
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
        "Augmented Supervised",
        "Direct-Move CNN",
      ]);
      await page.getByRole("button", { name: new RegExp(`^${name} `) }).click();
      await expect(
        page.getByRole("region", { name: `${name} research summary` }),
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
      await expect(page.locator(".player-strip").first()).toContainText(
        "Shared alpha-beta search",
      );
      await expect(page.locator(".player-strip").first()).not.toContainText(
        "PUCT",
      );
      await expect(page.locator(".analysis-card .hash")).toContainText(
        id === "handcrafted"
          ? "handcrafted-"
          : (id === "supervised-1m" ? "a" : "s").repeat(12),
      );
      await expect(
        page.getByText("Start a new game to change opponents."),
      ).toBeVisible();
      for (const card of await cards.all()) await expect(card).toBeDisabled();
      // White avoids an immediate opening move after resetting as Black.
      await page.getByRole("button", { name: "White", exact: true }).click();
      for (const card of await cards.all()) {
        if ((await card.innerText()).includes("Direct-Move CNN"))
          await expect(card).toBeDisabled();
        else await expect(card).toBeEnabled();
      }
    });
  }
}

test("released evidence is matched to checkpoint identity", async ({
  page,
}) => {
  await mock(page);
  await page.goto("/?model=supervised-1m");
  const explanation = page.getByRole("region", {
    name: "Augmented Supervised research summary",
  });
  await expect(explanation).toContainText(
    "teacher agreement does not establish stronger play",
  );
  await expect(
    explanation.getByRole("link", { name: "Read the research ↗", exact: true }),
  ).toHaveAttribute("href", "/research?study=gpu-value");
  await page.route("**/research/studies.json", (route) =>
    route.fulfill({
      json: {
        studies: [
          { id: "gpu-value", checkpoint_ids: { "supervised-1m": "other" } },
        ],
      },
    }),
  );
  await page.reload();
  await expect(
    page.getByRole("region", { name: "Augmented Supervised research summary" }),
  ).toContainText("Evidence unavailable for this version");
  await expect(
    page
      .getByRole("region", { name: "Augmented Supervised research summary" })
      .getByRole("link", { name: "Read the research ↗", exact: true }),
  ).toHaveAttribute("href", "/research");
});

test("unavailable requested opponent stays selected without fallback and requires explicit selection", async ({
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
  await page.goto("/?model=supervised-1m");
  await expect(
    page.getByText("The requested opponent is unavailable.", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: /^Augmented Supervised A neural/,
    }),
  ).toBeDisabled();
  await expect(page.locator(".model-option[aria-pressed=true]")).toContainText(
    "Augmented Supervised",
  );
  await expect(
    page.getByRole("button", { name: "e2 white pawn", exact: true }),
  ).toBeDisabled();
  expect(requests).toHaveLength(0);
  await expect(
    page.getByRole("button", { name: "Compare with Stockfish", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: /^Classical Handcrafted/ }).click();
  await expect(page.locator(".model-option[aria-pressed=true]")).toContainText(
    "Classical",
  );
  await expect(
    page.getByRole("button", { name: "Compare with Stockfish", exact: true }),
  ).toBeEnabled();
  expect(requests).toHaveLength(0);
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
  await page.goto("/?model=supervised-1m&moves=e2e4,e7e5");
  const explanation = page.getByRole("region", {
    name: "Augmented Supervised research summary",
  });
  await expect(explanation).toContainText(
    "teacher agreement does not establish stronger play",
  );
  await page.screenshot({
    path: info.outputPath("three-approaches-desktop.png"),
    fullPage: true,
  });
  await explanation
    .getByRole("link", { name: "Read the research ↗", exact: true })
    .click();
  await expect(page).toHaveURL(/research.*study=gpu-value/);
  await page.getByRole("button", { name: "Demo", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await expect(page.locator(".model-option[aria-pressed=true]")).toContainText(
    "Augmented Supervised",
  );
});

test("footer Research link preserves the current game and selected opponent", async ({
  page,
}) => {
  await mock(page);
  await page.goto("/?model=handcrafted&moves=e2e4,e7e5,g1f3,b8c6");
  await expect(page.getByText("4 plies", { exact: true })).toBeVisible();
  const position = new URL(page.url()).searchParams.get("fen");
  await page
    .getByRole("contentinfo")
    .getByRole("link", { name: "Research", exact: true })
    .click();
  await expect(page).toHaveURL(/\/research\?/);
  const query = new URL(page.url()).searchParams;
  expect(query.get("fen")).toBe(position);
  expect(query.get("moves")).toBe("e2e4,e7e5,g1f3,b8c6");
  expect(query.get("model")).toBe("handcrafted");
  await page.getByRole("button", { name: "Demo", exact: true }).click();
  await expect(page.getByText("4 plies", { exact: true })).toBeVisible();
  await expect(page.locator(".model-option[aria-pressed=true]")).toContainText(
    "Classical",
  );
});

for (const width of [1366, 890, 734, 390]) {
  test(`annotated demo and Research stay clear at ${width}px`, async ({
    page,
  }, info) => {
    await mock(page);
    await page.setViewportSize({ width, height: 973 });
    await page.goto("/");
    await expect(page.locator(".brand")).toHaveText("CHESSENGINE");
    await expect(page.locator(".brand-mark, .brand-sub")).toHaveCount(0);
    await expect(page.locator(".eyebrow")).toHaveCount(0);
    await expect(
      page.getByRole("heading", { name: "Play chess", exact: true }),
    ).toHaveClass("sr-only");
    await expect(page.getByText("Learning, one", { exact: false })).toHaveCount(
      0,
    );
    await expect(page.locator(".demo-intro")).toContainText(
      "Play Classical, Augmented Supervised, or Direct-Move CNN. Explore how each approach chooses a move.",
    );
    await expect(
      page.getByRole("button", { name: "Play", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("region", { name: "Stockfish comparison" }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Analyze this position", exact: true }),
    ).toHaveCount(0);
    await expect(
      page.getByText("Last analysis:", { exact: false }),
    ).toHaveCount(0);
    await expect(
      page.getByText(
        "One search engine. Different ways to understand the board.",
        { exact: true },
      ),
    ).toHaveCount(0);
    await expect(
      page.getByRole("link", {
        name: "Self-play: results and lessons ↗",
        exact: true,
      }),
    ).toHaveCount(0);
    await expect(page.locator(".model-option")).toHaveCount(3);
    await expect(
      page.getByRole("region", {
        name: "Augmented Supervised research summary",
      }),
    ).toContainText("Research summary");
    await expect(page.locator(".analysis-card")).toContainText(
      "ENGINE INSIGHT",
    );
    await expect(page.locator(".moves-card")).toBeVisible();
    await expect(page.locator(".method-strip")).toHaveCount(0);
    await expect(
      page.getByText("Test the difference", { exact: true }),
    ).toHaveCount(0);
    const summary = page.getByRole("region", {
      name: "Augmented Supervised research summary",
    });
    await expect(summary.getByRole("heading")).toHaveText("Research summary");
    await expect(summary.getByRole("link")).toHaveCount(1);
    await expect(summary.getByRole("link")).toHaveText("Read the research ↗");
    await expect(
      page.getByText("Mathematics and checkpoint", { exact: true }),
    ).toHaveCount(0);
    const footer = page.getByRole("contentinfo");
    await expect(footer).toHaveText("CHESSENGINEResearch");
    await expect(footer.getByRole("link")).toHaveCount(1);
    await expect(
      footer.getByRole("link", { name: "Research", exact: true }),
    ).toHaveAttribute("href", "/research");
    await expect(page.locator(".intro-note")).toContainText(
      "How do handcrafted evaluation, learned position assessment, and direct move prediction differ?",
    );
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBeTruthy();
    await page.screenshot({
      path: info.outputPath(`demo-${width}.png`),
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Explore the evidence", exact: true })
      .click();
    await expect(page).toHaveURL(/research.*study=gpu-value/);
    await expect(page.locator("#study-gpu-value")).toBeVisible();
    await expect(page.locator(".eyebrow")).toContainText(
      "METHODS AND EVIDENCE",
    );
    await expect(
      page.getByRole("heading", { name: "Evidence over assumptions." }),
    ).toBeVisible();
    await expect(page.locator(".intro > div:first-child > p")).toHaveCount(0);
    await expect(page.locator(".method-strip")).toHaveCount(0);
    await expect(page.locator("#study-self-play-archive")).toHaveCount(1);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBeTruthy();
    await page
      .getByRole("heading", { name: "Evidence over assumptions." })
      .scrollIntoViewIfNeeded();
    await page.screenshot({
      path: info.outputPath(`research-${width}.png`),
      fullPage: true,
    });
  });
}
