import { useEffect, useState } from "react";

export const releasedApproaches = [
  {
    id: "handcrafted",
    name: "Classical",
    description:
      "Handcrafted material and positional rules evaluate the board. Alpha-beta search looks ahead to choose a move.",
    evaluation:
      "Human-written material and positional rules score the board; no training is involved.",
    evidence: "A baseline for comparison, with no measured-strength claim.",
  },
  {
    id: "supervised-1m",
    name: "Augmented Supervised",
    description:
      "A neural evaluator trained on 800,000 positions learns Stockfish position assessments. Shared alpha-beta search chooses moves.",
    evaluation:
      "The dataset contains 800,000 training, 100,000 validation and 100,000 test positions. The seed-42 value network learned Stockfish assessments from scratch on a GPU and uses CPU alpha-beta during play.",
    evidence:
      "Mean validation MSE was 0.21476 across three seeds on this study’s own validation dataset. The demo uses seed 42; teacher agreement does not establish stronger play.",
  },
  {
    id: "elite-policy",
    name: "Direct-Move CNN",
    description:
      "Predicts a preferred legal move learned from elite human games. It chooses directly, without search. Playing strength is not established.",
    evaluation:
      "A 13-plane CNN learns from 800,000 elite-game positions, with 100,000 validation and 100,000 held-out test examples. Its output ranks moves, not position values.",
    evidence:
      "Trained on 800,000 elite-game positions. All three seeds passed export checks; the demo uses fixed seed 42, selected epoch 8. Human-move agreement does not establish playing strength.",
  },
] as const;

export type ReleasedId = (typeof releasedApproaches)[number]["id"];
export function releasedApproach(id: string) {
  return releasedApproaches.find((approach) => approach.id === id);
}

type StudyIndex = {
  studies: { id: string; checkpoint_ids?: Record<string, string> }[];
};

export function ResearchSummary({
  id,
  checkpoint,
  openResearch,
}: {
  id: string;
  checkpoint?: string;
  openResearch: (study?: string) => void;
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
      study.id ===
      (id === "elite-policy"
        ? "elite-pytorch-baseline-20261124"
        : id === "supervised-1m"
          ? "gpu-value"
          : "original-study"),
  );
  const matches =
    id === "handcrafted"
      ? checkpoint === "handcrafted-v1"
      : !!checkpoint && original?.checkpoint_ids?.[id] === checkpoint;
  const study = matches
    ? id === "elite-policy"
      ? "elite-pytorch-baseline-20261124"
      : id === "supervised-1m"
        ? "gpu-value"
        : "original-study"
    : undefined;
  return (
    <section
      className="research-summary"
      aria-label={`${approach.name} research summary`}
    >
      <h3>Research summary</h3>
      <p>{approach.evaluation}</p>
      <p>
        {id === "elite-policy"
          ? "This model chooses a legal move directly, without search. "
          : "This model uses CPU alpha-beta search to choose moves. "}
        Playing here does not train the models.
      </p>
      <p>
        {matches ? approach.evidence : "Evidence unavailable for this version"}
      </p>
      <a
        href={study ? `/research?study=${study}` : "/research"}
        onClick={(event) => {
          if (
            event.button === 0 &&
            !event.metaKey &&
            !event.ctrlKey &&
            !event.shiftKey &&
            !event.altKey
          ) {
            event.preventDefault();
            openResearch(study);
          }
        }}
      >
        Read the research ↗
      </a>
    </section>
  );
}
