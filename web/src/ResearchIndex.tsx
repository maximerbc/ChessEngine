import { useEffect, useState } from "react";

type Study = {
  id: string;
  title: string;
  level: string;
  result: string;
  availability: string;
  question: string;
  takeaway: string;
  report: string | null;
};
export function ResearchIndex({ selected }: { selected?: string }) {
  const [studies, setStudies] = useState<Study[]>([]);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    fetch("/research/studies.json")
      .then((r) => {
        if (!r.ok) throw Error();
        return r.json();
      })
      .then((data) => setStudies(data.studies))
      .catch(() => setFailed(true));
  }, []);
  useEffect(() => {
    if (selected && studies.length)
      document
        .getElementById(`study-${selected}`)
        ?.scrollIntoView({ block: "start" });
  }, [selected, studies]);
  return (
    <section className="study-index" aria-label="Research studies">
      <div className="section-heading">
        <h2>The research journal</h2>
        <span className="tiny-label">ALL OUTCOMES, ALL STUDIES</span>
      </div>
      <p className="muted">
        A sequence of questions and measured results. Availability and evidence
        are separate: a released model is not a claim of proven improvement.
      </p>
      {failed && (
        <p role="status">
          The study index is unavailable. The original report remains available
          below.
        </p>
      )}
      <div className="study-grid">
        {studies.map((study) => (
          <article
            className={`card study-card ${selected === study.id ? "study-selected" : ""}`}
            id={`study-${study.id}`}
            key={study.id}
          >
            <div className="study-badges">
              <span className="chip">{study.availability}</span>
              <span className={`result-label result-${study.result}`}>
                {study.result}
              </span>
            </div>
            <p className="tiny-label">{study.level}</p>
            <h3>{study.title}</h3>
            <p>
              <strong>{study.question}</strong>
            </p>
            <p>{study.takeaway}</p>
            {study.report ? (
              <a href={study.report}>Methods and results ↗</a>
            ) : (
              <span className="muted">
                Results will appear after the protocol is completed.
              </span>
            )}
            <a className="study-permalink" href={`/research?study=${study.id}`}>
              Link to this study
            </a>
          </article>
        ))}
      </div>
    </section>
  );
}
