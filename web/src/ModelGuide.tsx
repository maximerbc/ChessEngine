import { ArrowUpRight, ChevronRight } from "lucide-react";

export const approaches = [
  {
    id: "handcrafted",
    path: "/lab/classical",
    title: "Classical baseline",
    tag: "01 / DESIGNED",
    subtitle: "Chess knowledge, written as rules.",
    description:
      "Material values and small positional bonuses turn a board into an advantage estimate. There is no training data and no neural network.",
    input: "Board + side to move",
    middle: "Material + positional rules",
    output: "Position value → search",
    formula: "v(s) = tanh((material + position) / 6)",
    limitation:
      "Simple rules miss long-term plans and tactical consequences. Search helps, but the evaluator itself cannot learn.",
    question: "What can a few human-written rules achieve?",
  },
  {
    id: "supervised",
    path: "/lab/supervised",
    title: "Supervised evaluator",
    tag: "02 / LEARNED VALUE",
    subtitle: "Learn to evaluate a position.",
    description:
      "A compact neural network learns Stockfish-derived position values. Pieces are represented relative to the player to move, with material summaries and game-state features.",
    input: "853 board features",
    middle: "128 → 64 hidden neurons",
    output: "One value in [−1, 1] → search",
    formula: "L = mean((vθ(s) − yStockfish)²)",
    limitation:
      "Agreement with the teacher is not proof of stronger play. The scalar output is an advantage estimate, not a calibrated win probability.",
    question: "Can a small network learn useful chess estimates?",
  },
  {
    id: "move-guided",
    path: "/lab/move-guided",
    title: "Move-guided engine",
    tag: "03 / LEARNED ORDER",
    subtitle: "Learn where to look first.",
    description:
      "A separate network ranks legal root moves. The supervised value evaluator still scores positions. Search keeps every legal move and uses the previous iteration’s preferred move first.",
    input: "Board + each legal move",
    middle: "Board encoder + move scorer",
    output: "Move order → unchanged value evaluator",
    formula: "π(a|s) = softmax(scoreθ(s,a)); L = −log π(aTeacher|s)",
    limitation:
      "Ranking probabilities are not win probabilities. Faster node search may be offset by inference overhead; equal-time matches test the combined effect.",
    question: "Does learned ordering make search more efficient?",
  },
] as const;
export type Approach = (typeof approaches)[number];
export type Catalog = {
  models: Record<
    string,
    {
      checkpoint_id: string;
      name: string;
      facts: string[];
      report?: string;
      experimental: boolean;
      training?: { label: string; points: number[] };
    }
  >;
};
export type GuideAnalysis = {
  root_rankings?: { move: string; probability: number }[];
  explanation?: {
    raw_value: number;
    side_to_move: string;
    feature_version: number | null;
    piece_counts: Record<string, Record<string, number>>;
    handcrafted?: { material: number; position: number; scale: number } | null;
  };
};

export function ModelGallery({
  navigate,
  disabled,
}: {
  navigate: (id: string) => void;
  disabled: boolean;
}) {
  return (
    <section
      className="model-gallery"
      aria-label="Explore local research models"
    >
      {approaches.map((a) => (
        <button
          key={a.id}
          disabled={disabled}
          onClick={() => navigate(a.id)}
          className="model-tile"
        >
          <span className="tiny-label">{a.tag}</span>
          <h2>{a.title}</h2>
          <p>{a.question}</p>
          <span className="tile-link">
            Explore the model <ArrowUpRight size={16} />
          </span>
        </button>
      ))}
    </section>
  );
}

export function ModelGuide({
  approach,
  catalog,
  checkpoint,
  analysis,
  analyze,
  busy,
  available,
}: {
  approach: Approach;
  catalog: Catalog | null;
  checkpoint?: string;
  analysis: GuideAnalysis | null;
  analyze: () => void;
  busy: boolean;
  available: boolean;
}) {
  const evidence = catalog?.models[approach.id];
  const mismatch =
    !!evidence && !!checkpoint && checkpoint !== evidence.checkpoint_id;
  const values = evidence?.training?.points || [];
  const min = Math.min(...values),
    max = Math.max(...values);
  return (
    <section className="model-guide" aria-label={`${approach.title} explained`}>
      <div className="guide-heading">
        <div>
          <span className="tiny-label">HOW IT WORKS</span>
          <h2>{approach.question}</h2>
        </div>
        <span className="guide-badge">
          {approach.id === "handcrafted"
            ? "Designed baseline"
            : evidence?.experimental
              ? "Experimental model"
              : "Trained model"}
        </span>
      </div>
      <p>{approach.description}</p>
      <div className="model-flow">
        <span>{approach.input}</span>
        <ChevronRight aria-hidden />
        <span>{approach.middle}</span>
        <ChevronRight aria-hidden />
        <span>{approach.output}</span>
      </div>
      <details>
        <summary>The mathematics</summary>
        <p className="model-formula">{approach.formula}</p>
        <p>
          {approach.id === "move-guided"
            ? "Illegal moves are excluded from the probability distribution. Teacher targets are Stockfish’s first principal-variation move at a fixed search budget. Other moves can be equally good, so top-one accuracy has limitations."
            : "Positive values favour the player to move. Search negates the child’s value once when moving between players."}
        </p>
      </details>
      <div className="guide-columns">
        <div>
          <h3>Evidence for this version</h3>
          {evidence ? (
            <>
              <ul>
                {evidence.facts.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
              {evidence.report && (
                <a href={evidence.report}>
                  Full methods and results <ArrowUpRight size={14} />
                </a>
              )}
            </>
          ) : (
            <p>
              Version-specific research is not installed in this preview. No
              metrics are inferred from another checkpoint.
            </p>
          )}
          {values.length > 1 && (
            <figure className="mini-curve">
              <svg
                viewBox="0 0 400 110"
                role="img"
                aria-label={evidence?.training?.label}
              >
                <polyline
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  points={values
                    .map(
                      (v, i) =>
                        `${10 + (i * 380) / (values.length - 1)},${100 - ((v - min) * 90) / (max - min || 1)}`,
                    )
                    .join(" ")}
                />
              </svg>
              <figcaption>
                {evidence?.training?.label} · first {values[0].toFixed(3)}, last{" "}
                {values[values.length - 1].toFixed(3)}
              </figcaption>
            </figure>
          )}
        </div>
        <div>
          <h3>Inspect a position</h3>
          <p>
            Use the board below, then inspect what this model returns. Analysis
            does not play a move.
          </p>
          <button
            className="primary"
            onClick={analyze}
            disabled={busy || !available || mismatch}
          >
            {busy ? "Analyzing…" : "Analyze this position"}
          </button>
          {analysis?.explanation && (
            <div className="live-explanation">
              <p>
                Last analysis:{" "}
                <strong>{analysis.explanation.raw_value.toFixed(3)}</strong>,
                from {analysis.explanation.side_to_move}’s perspective.
              </p>
              {analysis.explanation.handcrafted && (
                <p>
                  Material:{" "}
                  {analysis.explanation.handcrafted.material.toFixed(2)} ·
                  Position:{" "}
                  {analysis.explanation.handcrafted.position.toFixed(2)} · tanh
                  scale: 6
                </p>
              )}
              <p>
                Own pieces:{" "}
                {Object.entries(analysis.explanation.piece_counts.own || {})
                  .map(([k, v]) => `${v} ${k}`)
                  .join(", ")}
              </p>
            </div>
          )}
          {approach.id === "move-guided" && analysis?.root_rankings && (
            <ol className="move-rankings" aria-label="Ranked root moves">
              {analysis.root_rankings.slice(0, 5).map((r) => (
                <li key={r.move}>
                  <code>{r.move}</code>
                  <meter min="0" max="1" value={r.probability} />
                  <span>{(100 * r.probability).toFixed(1)}%</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>
      <aside className="guide-limit">
        <strong>What this does not establish</strong>
        <p>{approach.limitation}</p>
      </aside>
      <p className="checkpoint-label">
        Research checkpoint: {evidence?.checkpoint_id || "not installed"}
        <br />
        Live checkpoint: {checkpoint || "engine unavailable"}
      </p>
      {mismatch && (
        <p role="alert" className="notice">
          The live engine is serving a different checkpoint. Play and analysis
          on this page are disabled until the matching release is installed.
        </p>
      )}
    </section>
  );
}
