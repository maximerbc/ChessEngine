import { useEffect, useRef, useState } from "react";

type SearchAssessment = {
  value: number | null;
  mate_winner: "white" | "black" | null;
  terminal: string | null;
  depth: number;
  nodes: number;
  strategy?: string;
  simulations?: number;
  elapsed_ms: number;
  limits: { nodes?: number; simulations?: number; time_ms: number };
};
export type ComparisonResult = {
  initial_fen: string;
  moves: string[];
  fen: string;
  perspective: "white";
  model_id: string;
  checkpoint_id: string;
  position_value: number | null;
  terminal: string | null;
  search: SearchAssessment | null;
  policy?: {
    root_rankings: { move: string; probability: number }[];
    elapsed_ms: number;
  };
  stockfish: SearchAssessment & {
    status: "ok" | "unavailable" | "terminal";
    pawn_score: number | null;
    mate: number | null;
    version: string | null;
    binary_sha256: string | null;
    error: string | null;
  };
};
type ComparisonRequest = {
  initial_fen: string;
  moves: string[];
  model_id: string;
};

function signed(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function assessment(score: SearchAssessment) {
  if (score.terminal === "1/2-1/2") return "Draw";
  if (score.mate_winner)
    return `${score.mate_winner === "white" ? "White" : "Black"} mates`;
  return score.value === null ? "Unavailable" : signed(score.value);
}

export function PositionComparison({
  initialFen,
  moves,
  modelId,
  modelName,
  checkpoint,
  revision,
  available,
  busy,
  evaluate,
  onBusy,
}: {
  initialFen: string;
  moves: string[];
  modelId: string;
  modelName: string;
  checkpoint?: string;
  revision: number;
  available: boolean;
  busy: boolean;
  evaluate: (body: ComparisonRequest) => Promise<ComparisonResult>;
  onBusy: (busy: boolean) => void;
}) {
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState("");
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const sequence = useRef(0);
  const inFlight = useRef(false);
  const positionKey = JSON.stringify([
    initialFen,
    moves,
    modelId,
    checkpoint,
    revision,
  ]);
  const currentKey = useRef(positionKey);
  currentKey.current = positionKey;
  useEffect(() => {
    sequence.current++;
    setResult(null);
    setError("");
    setVisible(false);
    return () => {
      sequence.current++;
    };
  }, [positionKey]);

  async function compare() {
    if (busy || inFlight.current || !available) return;
    inFlight.current = true;
    const token = ++sequence.current;
    const requestedKey = positionKey;
    const isCurrent = () =>
      token === sequence.current && requestedKey === currentKey.current;
    setVisible(true);
    setLoading(true);
    setError("");
    onBusy(true);
    try {
      const data = await evaluate({
        initial_fen: initialFen,
        moves,
        model_id: modelId,
      });
      if (!isCurrent()) return;
      if (
        data.checkpoint_id !== checkpoint ||
        data.model_id !== modelId ||
        data.perspective !== "white" ||
        data.initial_fen !== initialFen ||
        JSON.stringify(data.moves) !== JSON.stringify(moves)
      ) {
        throw new Error(
          "The engine or position changed. Reconnect before comparing.",
        );
      }
      setResult(data);
    } catch (e) {
      if (isCurrent()) setError((e as Error).message);
    } finally {
      inFlight.current = false;
      setLoading(false);
      onBusy(false);
    }
  }
  const reference = result?.stockfish;
  function gap(value: number | null) {
    return value !== null &&
      reference?.value !== null &&
      reference?.value !== undefined
      ? Math.abs(value - reference.value).toFixed(3)
      : "—";
  }
  return (
    <section
      className="card position-comparison"
      aria-label="Stockfish comparison"
    >
      <span className="tiny-label">CURRENT POSITION</span>
      <h2>What does each engine think?</h2>
      <div className="actions">
        <button
          disabled={busy || !available}
          onClick={() => void compare()}
          aria-expanded={visible}
          aria-controls="position-comparison-results"
        >
          {loading
            ? "Comparing position…"
            : visible
              ? error || reference?.status === "unavailable"
                ? "Retry comparison"
                : "Refresh comparison"
              : "Compare with Stockfish"}
        </button>
        {visible && (
          <button
            onClick={() => {
              sequence.current++;
              setVisible(false);
              setResult(null);
              setError("");
            }}
          >
            Hide comparison
          </button>
        )}
      </div>
      {visible && (
        <div id="position-comparison-results" aria-busy={loading}>
          {loading && (
            <p role="status">
              Evaluating the current board with {modelName} and Stockfish…
            </p>
          )}
          {error && <p role="alert">{error}</p>}
          {result && !loading && (
            <>
              <p className="micro">
                White’s perspective · −1 to +1 · Positive favors White; negative
                favors Black.
              </p>
              <table>
                <caption className="sr-only">
                  Current position: selected engine and Stockfish assessments
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Assessment</th>
                    <th scope="col">{modelName}</th>
                    <th scope="col">
                      Stockfish
                      <br />
                      <small>searched reference</small>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">Position evaluation</th>
                    <td>
                      {result.position_value === null
                        ? "Not provided by this move predictor"
                        : signed(result.position_value)}
                    </td>
                    <td>{assessment(result.stockfish)}</td>
                  </tr>
                  <tr className="comparison-gap">
                    <th scope="row">Absolute difference</th>
                    <td colSpan={2}>{gap(result.position_value)}</td>
                  </tr>
                  <tr>
                    <th scope="row">After search</th>
                    <td>
                      {result.search ? assessment(result.search) : "No search"}
                    </td>
                    <td>{assessment(result.stockfish)}</td>
                  </tr>
                  <tr className="comparison-gap">
                    <th scope="row">Absolute difference</th>
                    <td colSpan={2}>{gap(result.search?.value ?? null)}</td>
                  </tr>
                </tbody>
              </table>
              {reference?.status === "unavailable" && (
                <p role="alert">
                  {reference.error} The selected engine’s assessments are still
                  shown.
                </p>
              )}
              {reference?.pawn_score !== null &&
                reference?.pawn_score !== undefined && (
                  <p className="micro">
                    Stockfish pawn score: {reference.pawn_score >= 0 ? "+" : ""}
                    {reference.pawn_score.toFixed(2)}
                  </p>
                )}
              {reference?.mate !== null && reference?.mate !== undefined && (
                <p className="micro">
                  Stockfish finds mate in {Math.abs(reference.mate)} for{" "}
                  {reference.mate_winner === "white" ? "White" : "Black"}.
                </p>
              )}
              {result.terminal && (
                <p className="micro">
                  Game complete: {result.terminal}. The position evaluation is
                  the evaluator’s raw output; the game outcome takes precedence.
                </p>
              )}
              <p className="micro">
                {result.policy
                  ? "This network predicts legal moves and has no position-value head or search. Stockfish provides a separate position assessment."
                  : `Position evaluation is ${modelName}’s own board assessment. After search adds looking ahead through possible moves. Both rows use the same Stockfish searched reference.`}
              </p>
              {result.policy && (
                <ol aria-label="Current-position predicted legal moves">
                  {result.policy.root_rankings.slice(0, 5).map((row) => (
                    <li key={row.move}>
                      {row.move} · {(row.probability * 100).toFixed(1)}%
                    </li>
                  ))}
                </ol>
              )}
              <p className="micro">
                Agreement on one position does not measure training efficiency
                or playing strength. These values are not calibrated win
                probabilities.
                {modelId === "handcrafted" &&
                  " Classical is an untrained baseline."}
              </p>
              <details>
                <summary>Search details and model version</summary>
                {result.search && (
                  <>
                    <p>
                      {modelName}: {result.search.depth} depth ·{" "}
                      {result.search.strategy === "puct-uniform-v1"
                        ? `${result.search.simulations?.toLocaleString()} simulations`
                        : `${result.search.nodes.toLocaleString()} nodes`}{" "}
                      · {(result.search.elapsed_ms / 1000).toFixed(2)}s
                    </p>
                    <p>
                      Selected engine limits:{" "}
                      {result.search.strategy === "puct-uniform-v1"
                        ? `${result.search.limits.simulations?.toLocaleString()} simulations`
                        : `${result.search.limits.nodes?.toLocaleString()} nodes`}{" "}
                      / {(result.search.limits.time_ms / 1000).toFixed(1)}s.
                    </p>
                  </>
                )}
                {result.policy && (
                  <p>
                    {modelName}: direct inference ·{" "}
                    {(result.policy.elapsed_ms / 1000).toFixed(3)}s · no search.
                  </p>
                )}
                <p>
                  {reference?.version || "Stockfish reference"}:{" "}
                  {reference?.depth} depth · {reference?.nodes.toLocaleString()}{" "}
                  nodes · {((reference?.elapsed_ms || 0) / 1000).toFixed(2)}s
                </p>
                <p>
                  Stockfish limits: {reference?.limits.nodes?.toLocaleString()}{" "}
                  nodes / {((reference?.limits.time_ms || 0) / 1000).toFixed(1)}
                  s. Node counts reflect each engine’s own search; this is not a
                  strength test.
                </p>
                <p>
                  Stockfish value = (wins − losses) / total from its WDL
                  estimates.
                </p>
                <p className="hash">Checkpoint: {result.checkpoint_id}</p>
                {reference?.binary_sha256 && (
                  <p className="hash">
                    Stockfish SHA-256: {reference.binary_sha256}
                  </p>
                )}
              </details>
            </>
          )}
        </div>
      )}
    </section>
  );
}
