import { test, expect, type Page } from "@playwright/test";
import { Chess } from "chess.js";

const identities: Record<string, string> = {
  handcrafted: "handcrafted-v1",
  supervised: "s".repeat(64),
  "supervised-1m": "a".repeat(64),
  evolutionary: "e".repeat(64),
};

function comparison(body: any) {
  const board = new Chess(body.initial_fen);
  for (const uci of body.moves)
    board.move({
      from: uci.slice(0, 2),
      to: uci.slice(2, 4),
      promotion: uci[4],
    });
  const search = {
    strategy: "alpha-beta",
    value: 0.3,
    mate_winner: null,
    terminal: null,
    depth: 3,
    nodes: 900,
    elapsed_ms: 100,
    limits: { nodes: 30000, time_ms: 3000 },
  };
  return {
    ...body,
    fen: board.fen(),
    perspective: "white",
    checkpoint_id: identities[body.model_id],
    position_value: 0.22,
    terminal: null,
    search,
    stockfish: {
      ...search,
      status: "ok",
      value: 0.15,
      pawn_score: 0.45,
      mate: null,
      version: "Stockfish 19",
      binary_sha256: "f".repeat(64),
      error: null,
    },
  };
}

async function setup(page: Page) {
  await page.route("**/api/models", (route) =>
    route.fulfill({
      json: {
        default: "supervised-1m",
        models: Object.entries(identities).map(([id, checkpoint_id]) => ({
          id,
          checkpoint_id,
          name: id,
          kind: id,
          strategy: "alpha-beta",
          evidence: "test",
        })),
      },
    }),
  );
  await page.route("**/api/analyze", (route) => {
    const body = route.request().postDataJSON();
    const board = new Chess(body.initial_fen);
    for (const uci of body.moves)
      board.move({
        from: uci.slice(0, 2),
        to: uci.slice(2, 4),
        promotion: uci[4],
      });
    const move =
      board
        .moves({ verbose: true })
        .find((m) => m.from === "e7" && m.to === "e5") ||
      board.moves({ verbose: true })[0];
    return route.fulfill({
      json: {
        best_move: move ? move.from + move.to + (move.promotion || "") : null,
        value: 0.1,
        depth: 2,
        nodes: 1000,
        elapsed_ms: 120,
        pv: [],
        checkpoint_id: identities[body.model_id],
        terminal: null,
        mate_score: false,
      },
    });
  });
  await page.route("**/api/evaluate-comparison", (route) =>
    route.fulfill({ json: comparison(route.request().postDataJSON()) }),
  );
}

const panel = (page: Page) =>
  page.getByRole("region", { name: "Stockfish comparison", exact: true });

for (const [id, name] of [
  ["handcrafted", "Classical"],
  ["supervised-1m", "Augmented Supervised"],
]) {
  test(`${name} displays adjacent position and search comparisons without moving`, async ({
    page,
  }) => {
    await setup(page);
    await page.goto(`/?model=${id}`);
    const section = panel(page);
    const button = section.getByRole("button", {
      name: "Compare with Stockfish",
      exact: true,
    });
    await expect(button).toBeEnabled();
    await button.focus();
    await page.keyboard.press("Enter");
    await expect(
      section.getByRole("row", { name: /Position evaluation/ }),
    ).toContainText("+0.220+0.150");
    await expect(
      section.getByRole("row", { name: /After search/ }),
    ).toContainText("+0.300+0.150");
    await expect(
      section.getByRole("row", { name: /Absolute difference/ }).first(),
    ).toContainText("0.070");
    await expect(
      section.getByRole("row", { name: /Absolute difference/ }).last(),
    ).toContainText("0.150");
    await expect(section).toContainText("Stockfish pawn score: +0.45");
    await expect(section).toContainText("White’s perspective");
    await expect(page.getByText("0 plies", { exact: true })).toBeVisible();
    if (id === "handcrafted")
      await expect(section).toContainText("Classical is an untrained baseline");
    await section
      .getByText("Search details and model version", { exact: true })
      .click();
    await expect(section).toContainText("30,000 nodes / 3.0s");
    await expect(section).toContainText(identities[id]);
    await section
      .getByRole("button", { name: "Refresh comparison", exact: true })
      .click();
    await expect(section.getByRole("table")).toBeVisible();
    await section
      .getByRole("button", { name: "Hide comparison", exact: true })
      .click();
    await expect(section.getByRole("table")).toHaveCount(0);
    await expect(
      section.getByRole("button", {
        name: "Compare with Stockfish",
        exact: true,
      }),
    ).toBeVisible();
  });
}

test("scores clear on model changes, moves, undo and restart", async ({
  page,
}) => {
  await setup(page);
  await page.goto("/");
  const section = panel(page);
  const compare = () =>
    section
      .getByRole("button", { name: "Compare with Stockfish", exact: true })
      .click();
  await compare();
  await expect(section.getByRole("table")).toBeVisible();
  await page.getByRole("button", { name: /^Classical Handcrafted/ }).click();
  await expect(section.getByRole("table")).toHaveCount(0);
  await compare();
  await expect(section.getByRole("table")).toBeVisible();
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e4", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  await expect(section.getByRole("table")).toHaveCount(0);
  const request = page.waitForRequest("**/api/evaluate-comparison");
  await compare();
  expect((await request).postDataJSON().moves).toEqual(["e2e4", "e7e5"]);
  await expect(section.getByRole("table")).toBeVisible();
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(section.getByRole("table")).toHaveCount(0);
  await compare();
  await expect(section.getByRole("table")).toBeVisible();
  await page.getByRole("button", { name: "New game", exact: true }).click();
  await expect(section.getByRole("table")).toHaveCount(0);
});

test("reference failure preserves model assessments, allows retry and does not block gameplay", async ({
  page,
}) => {
  await setup(page);
  let attempts = 0;
  await page.route("**/api/evaluate-comparison", (route) => {
    const data = comparison(route.request().postDataJSON());
    if (++attempts === 1)
      Object.assign(data.stockfish, {
        status: "unavailable",
        value: null,
        pawn_score: null,
        error: "Stockfish analysis is unavailable. Please retry shortly.",
      });
    return route.fulfill({ json: data });
  });
  await page.goto("/");
  const section = panel(page);
  await section
    .getByRole("button", { name: "Compare with Stockfish", exact: true })
    .click();
  await expect(section.getByRole("alert")).toContainText(
    "Stockfish analysis is unavailable",
  );
  await expect(
    section.getByRole("row", { name: /Position evaluation/ }),
  ).toContainText("+0.220Unavailable");
  await expect(
    section.getByRole("row", { name: /After search/ }),
  ).toContainText("+0.300Unavailable");
  await section
    .getByRole("button", { name: "Retry comparison", exact: true })
    .click();
  await expect(section.getByRole("alert")).toHaveCount(0);
  await expect(
    section.getByRole("row", { name: /Position evaluation/ }),
  ).toContainText("+0.220+0.150");
  await page
    .getByRole("button", { name: "e2 white pawn", exact: true })
    .click();
  await page.getByRole("button", { name: "e4", exact: true }).click();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
});

test("loading prevents overlapping requests and hide discards a late response", async ({
  page,
}) => {
  await setup(page);
  let finish!: () => void;
  const gate = new Promise<void>((resolve) => {
    finish = resolve;
  });
  await page.route("**/api/evaluate-comparison", async (route) => {
    await gate;
    await route.fulfill({ json: comparison(route.request().postDataJSON()) });
  });
  await page.goto("/");
  const section = panel(page);
  await section
    .getByRole("button", { name: "Compare with Stockfish", exact: true })
    .click();
  await expect(section.getByRole("status")).toContainText(
    "Evaluating the current board",
  );
  await expect(
    section.getByRole("button", { name: "Comparing position…", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "New game", exact: true }),
  ).toBeDisabled();
  await section
    .getByRole("button", { name: "Hide comparison", exact: true })
    .click();
  finish();
  await expect(
    section.getByRole("button", {
      name: "Compare with Stockfish",
      exact: true,
    }),
  ).toBeEnabled();
  await expect(section.getByRole("table")).toHaveCount(0);
});

test("browser navigation rejects a pending result for the previous position", async ({
  page,
}) => {
  await setup(page);
  await page.goto("/?moves=e2e4,e7e5");
  // Create a real history entry with a different position; popstate is the same
  // handler used by the browser Back button during a request.
  await page.evaluate(() => {
    history.pushState(null, "", "/?model=supervised");
    dispatchEvent(new PopStateEvent("popstate"));
  });
  let finish!: () => void;
  const gate = new Promise<void>((resolve) => {
    finish = resolve;
  });
  await page.route("**/api/evaluate-comparison", async (route) => {
    await gate;
    await route.fulfill({ json: comparison(route.request().postDataJSON()) });
  });
  const section = panel(page);
  await section
    .getByRole("button", { name: "Compare with Stockfish", exact: true })
    .click();
  await expect(section.getByRole("status")).toBeVisible();
  await page.goBack();
  await expect(page.getByText("2 plies", { exact: true })).toBeVisible();
  finish();
  await expect(
    section.getByRole("button", {
      name: "Compare with Stockfish",
      exact: true,
    }),
  ).toBeEnabled();
  await expect(section.getByRole("table")).toHaveCount(0);
});

test("busy errors and checkpoint mismatches stay retryable", async ({
  page,
}) => {
  await setup(page);
  let attempts = 0;
  await page.route("**/api/evaluate-comparison", (route) => {
    if (++attempts === 1)
      return route.fulfill({ status: 429, json: { detail: "busy" } });
    const data = comparison(route.request().postDataJSON());
    data.checkpoint_id = "changed";
    return route.fulfill({ json: data });
  });
  await page.goto("/");
  const section = panel(page);
  await section
    .getByRole("button", { name: "Compare with Stockfish", exact: true })
    .click();
  await expect(section.getByRole("alert")).toContainText("engine is busy");
  await section
    .getByRole("button", { name: "Retry comparison", exact: true })
    .click();
  await expect(section.getByRole("alert")).toContainText(
    "engine or position changed",
  );
  await expect(section.getByRole("table")).toHaveCount(0);
});

test("mates stay explicit and both comparisons fit mobile", async ({
  page,
}, info) => {
  await setup(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route("**/api/evaluate-comparison", (route) => {
    const data = comparison(route.request().postDataJSON());
    data.terminal = "1-0" as any;
    Object.assign(data.search, {
      value: null,
      mate_winner: "white",
      terminal: "1-0",
    });
    Object.assign(data.stockfish, {
      status: "terminal",
      value: null,
      pawn_score: null,
      mate_winner: "white",
      terminal: "1-0",
    });
    return route.fulfill({ json: data });
  });
  await page.goto(
    "/?fen=" + encodeURIComponent("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
  );
  const section = panel(page);
  await section
    .getByRole("button", { name: "Compare with Stockfish", exact: true })
    .click();
  await expect(
    section.getByRole("row", { name: /After search/ }),
  ).toContainText("White matesWhite mates");
  await expect(section).toContainText("game outcome takes precedence");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  const columns = section.getByRole("columnheader");
  const selected = await columns.nth(1).boundingBox();
  const reference = await columns.nth(2).boundingBox();
  expect(reference!.x).toBeGreaterThan(selected!.x);
  await section.scrollIntoViewIfNeeded();
  await page.screenshot({
    path: info.outputPath("position-comparison-mobile.png"),
    fullPage: true,
  });
});
