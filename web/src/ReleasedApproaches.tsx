import { useEffect, useState } from "react";

export const releasedApproaches = [
  {
    id: "handcrafted",
    name: "Classical",
    description:
      "Handcrafted material and positional rules evaluate the board. Alpha-beta search looks ahead to choose a move.",
    evaluation:
      "Human-written material values and piece-square bonuses score the position. This evaluator has no training data or neural network.",
    evidence:
      "A designed baseline for comparing handcrafted and learned evaluation. It carries no claim of measured playing strength.",
    formula: "v(s) = tanh((material + position) / 6)",
  },
  {
    id: "supervised",
    name: "Supervised",
    description:
      "A neural network learns position assessments from Stockfish. The same alpha-beta search uses those assessments to choose a move.",
    evaluation:
      "A compact neural network predicts position value from board and game-state features. It learns Stockfish assessments, not Stockfish moves.",
    evidence:
      "The original local study trained this evaluator using Stockfish position assessments. Teacher agreement does not establish stronger play or a calibrated win probability.",
    formula: "L = mean((vθ(s) − yStockfish)²)",
  },
  {
    id: "evolutionary",
    name: "Evolutionary",
    description:
      "A small neural evaluator evolves through tournament results. All evaluation weights start randomly; alpha-beta search chooses its moves.",
    evaluation:
      "CMA-ES tests competing neural evaluators against fixed opponents and earlier generations. Human-designed features describe the board; every evaluation weight is learned without Stockfish labels or a fixed material score.",
    evidence:
      "Availability requires a completed three-seed study and passing qualification and fresh-final gates. See the evolutionary research entry for the measured results and limitations.",
    formula:
      "vθ(s) = tanh(u · ReLU(Wx + b) + w · x + c); 324 learned parameters",
  },
] as const;

export type ReleasedId = (typeof releasedApproaches)[number]["id"];
export function releasedApproach(id: string) {
  return releasedApproaches.find((approach) => approach.id === id);
}

type StudyIndex = {
  studies: { id: string; checkpoint_ids?: Record<string, string> }[];
};

export function ReleasedExplanation({
  id,
  checkpoint,
  openResearch,
}: {
  id: string;
  checkpoint?: string;
  openResearch: (study: string) => void;
}) {
  const [index, setIndex] = useState<StudyIndex | null>(null);
  useEffect(() => {
    fetch("/research/studies.json")
      .then((response) => (response.ok ? response.json() : null))
      .then(setIndex)
      .catch(() => setIndex(null));
  }, []);
  const approach = releasedApproach(id);
  if (!approach) return null;
  const original = index?.studies?.find(
    (study) =>
      study.id === (id === "evolutionary" ? "evolutionary" : "original-study"),
  );
  const matches =
    id === "handcrafted"
      ? checkpoint === "handcrafted-v1"
      : !!checkpoint && original?.checkpoint_ids?.[id] === checkpoint;
  return (
    <section
      className="released-explanation"
      aria-label={`${approach.name} explained`}
    >
      <h3>How it evaluates</h3>
      <p>{approach.evaluation}</p>
      <h3>How it chooses moves</h3>
      <p>
        All three opponents use the same alpha-beta search to look ahead through
        legal moves. The evaluator scores the positions it reaches. Playing here
        does not train the models.
      </p>
      <h3>What the evidence shows</h3>
      <p>
        {matches ? approach.evidence : "Evidence unavailable for this version"}
      </p>
      {matches && id !== "handcrafted" && (
        <a
          href={`/research?study=${id === "evolutionary" ? "evolutionary" : "original-study"}`}
          onClick={(event) => {
            if (
              event.button === 0 &&
              !event.metaKey &&
              !event.ctrlKey &&
              !event.shiftKey &&
              !event.altKey
            ) {
              event.preventDefault();
              openResearch(
                id === "evolutionary" ? "evolutionary" : "original-study",
              );
            }
          }}
        >
          {id === "evolutionary"
            ? "Evolutionary methods and results ↗"
            : "Original local-study methods and results ↗"}
        </a>
      )}
      {id === "supervised" && (
        <p className="research-followups">
          Separate research:{" "}
          <a
            href="/research?study=rl-improvement"
            onClick={(event) => {
              if (
                event.button === 0 &&
                !event.metaKey &&
                !event.ctrlKey &&
                !event.shiftKey &&
                !event.altKey
              ) {
                event.preventDefault();
                openResearch("rl-improvement");
              }
            }}
          >
            later RL study (regression; different checkpoints)
          </a>{" "}
          ·{" "}
          <a
            href="/research?study=policy-value"
            onClick={(event) => {
              if (
                event.button === 0 &&
                !event.metaKey &&
                !event.ctrlKey &&
                !event.shiftKey &&
                !event.altKey
              ) {
                event.preventDefault();
                openResearch("policy-value");
              }
            }}
          >
            policy/value feasibility pilot (full comparison deferred)
          </a>
          .
        </p>
      )}
      <details>
        <summary>Mathematics and checkpoint</summary>
        <p>{approach.formula}</p>
        <p>
          Values favour the player to move; search negates child values once per
          ply. Values are not calibrated win probabilities. Terminal mate search
          scores use a separate range.
        </p>
        <p className="hash">Checkpoint: {checkpoint || "unavailable"}</p>
      </details>
    </section>
  );
}
