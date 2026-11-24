import { useState } from "react";

export type Opponent = {
  id: string;
  name: string;
  kind: string;
  description?: string;
  group?: string;
  experimental?: boolean;
  evidence_id?: string;
  documentation_path?: string;
  checkpoint_id?: string;
  strategy?: string;
  capabilities?: {
    position_value: boolean;
    search: boolean;
    move_probabilities: boolean;
  };
};

export function OpponentChooser({
  options,
  selected,
  installed,
  connection,
  disabled,
  choose,
}: {
  options: Opponent[];
  selected: string;
  installed: string[];
  connection: "ready" | "loading" | "offline";
  disabled: boolean;
  choose: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const filtered = options.filter((model) =>
    `${model.name} ${model.group || ""} ${model.description || ""}`
      .toLowerCase()
      .includes(query.toLowerCase().trim()),
  );
  const groups = [
    ...new Set(filtered.map((model) => model.group || "Supervised models")),
  ];
  const current = options.find((model) => model.id === selected);
  return (
    <>
      <label className="opponent-search">
        <span>Search opponents</span>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Name, training method or dataset"
        />
      </label>
      <p className="micro" role="status">
        {filtered.length} of {options.length} opponents
      </p>
      <div className="model-options" aria-label="Opponent catalogue">
        {groups.map((group) => (
          <section key={group} className="opponent-group" aria-label={group}>
            <h3>{group}</h3>
            {filtered
              .filter((model) => (model.group || "Supervised models") === group)
              .map((model) => {
                const available =
                  connection === "ready" && installed.includes(model.id);
                return (
                  <button
                    key={model.id}
                    className={`model-option ${selected === model.id ? "chosen" : ""}`}
                    disabled={disabled || !available}
                    aria-pressed={selected === model.id}
                    onClick={() => choose(model.id)}
                  >
                    <span className="radio" />
                    <span>
                      <strong>{model.name}</strong>
                      <small>{model.description}</small>
                    </span>
                    <span className="selected-label">
                      {connection === "loading"
                        ? "LOADING"
                        : !available
                          ? connection === "ready" &&
                            model.id === "elite-policy"
                            ? "IN RESEARCH"
                            : "UNAVAILABLE"
                          : selected === model.id
                            ? "SELECTED"
                            : "AVAILABLE"}
                    </span>
                  </button>
                );
              })}
          </section>
        ))}
        {filtered.length === 0 && <p>No opponents match your search.</p>}
      </div>
      {current && (
        <div
          className="selected-opponent"
          aria-label="Selected opponent summary"
        >
          <span className="tiny-label">SELECTED OPPONENT</span>
          <h3>{current.name}</h3>
          <p>{current.description}</p>
          <p className="micro">
            {current.strategy === "direct-policy-v1"
              ? "Direct legal-move prediction · no search"
              : "Shared alpha-beta search"}
            {current.experimental
              ? " · Experimental · strength not established"
              : ""}
          </p>
        </div>
      )}
    </>
  );
}
