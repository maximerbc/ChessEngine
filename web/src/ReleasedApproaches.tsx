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
      "A neural network learns position assessments from Stockfish. The same alpha-beta search uses those assessments to choose a move.",
    evaluation:
      "A neural network learns Stockfish position assessments, not Stockfish moves.",
    evidence:
      "Agreement with Stockfish does not establish stronger play or a calibrated win probability.",
  },
  {
    id: "evolutionary",
    name: "Evolutionary",
    description:
      "A small neural evaluator evolves through tournament results. All evaluation weights start randomly; alpha-beta search chooses its moves.",
    evaluation:
      "CMA-ES evolves all 324 evaluation weights from random initialization through tournaments, using human-designed features without Stockfish labels or a fixed material score.",
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
  const matchingStudy = checkpoint
    ? index?.studies?.find(
        (study) =>
          (study.id === "original-study" ||
            study.id === "evolutionary" ||
            study.id.startsWith("supervised-")) &&
          study.checkpoint_ids?.[id] === checkpoint,
      )
    : undefined;
  const matches =
    id === "handcrafted"
      ? checkpoint === "handcrafted-v1"
      : !!matchingStudy;
  const study = matches
    ? matchingStudy?.id || "original-study"
    : undefined;
  return (
    <section
      className="research-summary"
      aria-label={`${approach.name} research summary`}
    >
      <h3>Research summary</h3>
      <p>{approach.evaluation}</p>
      <p>
        All opponents use the same alpha-beta search. Playing here does not
        train the models.
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
