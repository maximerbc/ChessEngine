import { useEffect, useState, type ReactNode } from "react";
import { releasedApproaches } from "./ReleasedApproaches";

type Study = {
  id: string;
  title: string;
  level: string;
  result: string;
  availability: string;
  question: string;
  takeaway: string;
  report: string | null;
  group?: string;
  checkpoint_ids?: Record<string, string>;
};
type Group = { id: string; title: string; summary: string };
type Approach = {
  id: string;
  study_id: string | null;
  checkpoint_id: string;
  representative: string;
};
type Index = { studies: Study[]; groups: Group[]; approaches: Approach[] };

export function ResearchIndex({
  selected,
  children,
}: {
  selected?: string;
  children?: ReactNode;
}) {
  const [index, setIndex] = useState<Index | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    fetch("/research/studies.json")
      .then((r) => {
        if (!r.ok) throw Error();
        return r.json();
      })
      .then(setIndex)
      .catch(() => setFailed(true));
  }, []);
  useEffect(() => {
    if (!selected || !index) return;
    const entry = document.getElementById(`study-${selected}`);
    if (!entry) return;
    let parent = entry.parentElement;
    while (parent) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
      parent = parent.parentElement;
    }
    entry.scrollIntoView({ block: "start" });
  }, [selected, index]);
  const featured = new Set(
    index?.approaches?.map((a) => a.study_id).filter(Boolean),
  );
  return (
    <section className="study-index" aria-label="Research studies">
      <div className="section-heading">
        <h2>Three ways to choose a move.</h2>
        <span className="tiny-label">METHODS AND EVIDENCE</span>
      </div>
      <p className="muted">
        Classical and Augmented Supervised share alpha-beta search; Direct-Move
        CNN chooses directly. Playing here does not train the models.
      </p>
      <div className="approach-overview">
        {releasedApproaches.map((approach) => {
          const identity = index?.approaches?.find((a) => a.id === approach.id);
          const study = index?.studies.find((s) => s.id === identity?.study_id);
          const matches =
            approach.id === "handcrafted"
              ? identity?.checkpoint_id === "handcrafted-v1"
              : !!identity &&
                study?.checkpoint_ids?.[approach.id] === identity.checkpoint_id;
          return (
            <article
              className="card"
              key={approach.id}
              id={study ? `study-${study.id}` : undefined}
            >
              <p className="tiny-label">
                {approach.id === "elite-policy"
                  ? "MOVE PREDICTION"
                  : "POSITION EVALUATION + SEARCH"}
              </p>
              <h3>{approach.name}</h3>
              <p>{approach.description}</p>
              <p>
                {matches
                  ? approach.evidence
                  : "Evidence unavailable for this version"}
              </p>
              {identity && <p className="micro">{identity.representative}</p>}
              <div className="research-links">
                <a href={`/?model=${approach.id}`}>Play {approach.name} ↗</a>
                {study?.report && (
                  <a href={study.report}>Methods and results ↗</a>
                )}
                {approach.id === "handcrafted" && (
                  <a href="/research?study=original-study">
                    Baseline methods ↗
                  </a>
                )}
              </div>
            </article>
          );
        })}
      </div>
      {failed && (
        <p role="status">
          The study index is unavailable.{" "}
          <a href="/research/report.html">Read the original report</a>.
        </p>
      )}
      <div className="section-heading archive-heading">
        <h2>Experiments and lessons.</h2>
        <span className="tiny-label">EXPAND A RESEARCH FAMILY</span>
      </div>
      {index?.groups?.map((group) => (
        <details className="research-family" key={group.id}>
          <summary>
            <strong>{group.title}</strong>
            <span>{group.summary}</span>
          </summary>
          {index.studies
            .filter(
              (study) => study.group === group.id && !featured.has(study.id),
            )
            .map((study) => (
              <article
                className={`study-row ${selected === study.id ? "study-selected" : ""}`}
                id={`study-${study.id}`}
                key={study.id}
              >
                <div className="study-badges">
                  <span className="tiny-label">{study.level}</span>
                  <span className={`result-label result-${study.result}`}>
                    {study.result}
                  </span>
                </div>
                <h3>{study.title}</h3>
                <p>
                  <strong>{study.question}</strong>
                </p>
                <p>{study.takeaway}</p>
                <div className="research-links">
                  {study.report ? (
                    <a href={study.report}>Methods and results ↗</a>
                  ) : (
                    <span className="muted">
                      Results have not been published.
                    </span>
                  )}
                  <a href={`/research?study=${study.id}`}>Link to this study</a>
                </div>
                {study.id === "original-study" && children}
              </article>
            ))}
        </details>
      ))}
    </section>
  );
}
