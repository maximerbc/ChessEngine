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
    id: "supervised",
    name: "Supervised",
    description:
      "A neural network trained on about 80,000 positions learns assessments from Stockfish. Alpha-beta search uses those assessments to choose a move.",
    evaluation:
      "A neural network learns Stockfish position assessments, not Stockfish moves.",
    evidence:
      "Agreement with Stockfish does not establish stronger play or a calibrated win probability.",
  },
  {
    id: "supervised-1m",
    name: "Augmented Supervised",
    description:
      "Trained on a one-million-position dataset, with 800,000 positions used for training. A different search method chooses moves. Playing strength is not yet established.",
    evaluation:
      "The dataset contains 800,000 training, 100,000 validation and 100,000 test positions. The seed-42 value network learned Stockfish assessments from scratch on a GPU and uses CPU PUCT during play.",
    evidence:
      "Experimental playable model. Validation MSE was 0.21454; playing strength is not yet established. Different validation pools prevent a direct error comparison with the 80k model.",
  },
  {
    id: "evolutionary-gpu",
    name: "GPU-trained Evolutionary",
    description:
      "A new value network evolves through GPU tournaments from random initialization. CPU PUCT chooses moves after qualification.",
    evaluation:
      "Separable CMA-ES evolves the same value-network architecture through tournament results, without teacher labels, imported supervised weights or gradients. The planned main study contains 368,640 training games.",
    evidence:
      "Play requires a completed three-seed study and passing qualification and fresh-final gates.",
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
      (id === "supervised-1m" || id === "evolutionary-gpu"
        ? "gpu-value"
        : "original-study"),
  );
  const matches =
    id === "handcrafted"
      ? checkpoint === "handcrafted-v1"
      : !!checkpoint && original?.checkpoint_ids?.[id] === checkpoint;
  const study = matches
    ? id === "supervised-1m" || id === "evolutionary-gpu"
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
        {id === "supervised-1m" || id === "evolutionary-gpu"
          ? "This approach uses a value network with CPU PUCT search. "
          : "Classical and current Supervised use the same alpha-beta search. "}
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
