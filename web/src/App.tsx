import { useEffect, useRef, useState } from "react";
import { Chess, type Square, DEFAULT_POSITION } from "chess.js";
import {
  ArrowUpRight,
  ArrowLeft,
  ArrowRight,
  ChevronRight,
  RotateCcw,
  Download,
  FlaskConical,
  Activity,
  Cpu,
  BookOpen,
  Play,
  RefreshCw,
} from "lucide-react";

import { approaches, ModelGallery, ModelGuide, type Catalog, type GuideAnalysis } from "./ModelGuide";

const API = import.meta.env.VITE_API_URL || "/api";
type Model = {
  id: string;
  name: string;
  evidence: string;
  checkpoint_id: string;
  kind: string;
};
type Analysis = GuideAnalysis & {
  best_move: string | null;
  value: number;
  depth: number;
  nodes: number;
  elapsed_ms: number;
  pv: string[];
  checkpoint_id: string;
  terminal: string | null;
  mate_score: boolean;
};
type Summary = {
  score: number;
  ci95: number[];
  games: number;
  conclusion: string;
  capped_games: number;
};
type Research = {
  status: string;
  summary?: Summary;
  supervised?: {
    history: { epoch: number; train_loss: number; validation_loss: number }[];
  };
  rl?: {
    history: { batch: number; loss: number | null; games_total: number }[];
  };
  games?: {
    pgn: string;
    opening_id: string;
    candidate_color: string;
    score: number;
    capped: boolean;
  }[];
  limitations?: string[];
  aggregate?: {
    representative: number;
    mean_score: number;
    seed_score_std: number;
    runs: { seed: number; summary: Summary }[];
  };
};
const PIECES: Record<string, string> = {
  wk: "♚",
  wq: "♛",
  wr: "♜",
  wb: "♝",
  wn: "♞",
  wp: "♟",
  bk: "♚",
  bq: "♛",
  br: "♜",
  bb: "♝",
  bn: "♞",
  bp: "♟",
};
const examples = [
  {
    name: "A quiet opening",
    fen: DEFAULT_POSITION,
    moves: ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
  },
  { name: "Find the finish", fen: "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", moves: [] },
  {
    name: "An open endgame",
    fen: "6k1/5pp1/7p/8/3R4/5P2/6PP/6K1 w - - 0 1",
    moves: [],
  },
];
async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(API + path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(75000),
  });
  if (!response.ok) {
    const text = await response
      .json()
      .catch(() => ({ detail: "Service unavailable" }));
    throw new Error(
      response.status === 429
        ? "The engine is busy. Try again in a moment."
        : typeof text.detail === "string"
          ? text.detail
          : "The position could not be analyzed.",
    );
  }
  return response.json();
}
function boardFrom(fen: string, moves: string[]) {
  const b = new Chess(fen);
  for (const uci of moves)
    b.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
  return b;
}
function sanLine(board: Chess, moves: string[]) {
  const b = new Chess(board.fen());
  const line = [];
  for (const uci of moves) {
    try {
      line.push(
        b.move({
          from: uci.slice(0, 2),
          to: uci.slice(2, 4),
          promotion: uci[4],
        }).san,
      );
    } catch {
      break;
    }
  }
  return line.join("  ");
}
function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function Board({
  game,
  flip = false,
  selected,
  onSquare,
  disabled = false,
  lastMove,
}: {
  game: Chess;
  flip?: boolean;
  selected?: string | null;
  onSquare?: (s: Square) => void;
  disabled?: boolean;
  lastMove?: string;
}) {
  const ranks = flip ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];
  const files = flip ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];
  const legal = selected
    ? game.moves({ square: selected as Square, verbose: true }).map((m) => m.to)
    : [];
  return (
    <div className="board" role="group" aria-label="Chessboard">
      {ranks.flatMap((rank, ri) =>
        files.map((file, fi) => {
          const sq = (String.fromCharCode(97 + file) + (rank + 1)) as Square;
          const piece = game.get(sq);
          const highlighted =
            lastMove?.slice(0, 2) === sq || lastMove?.slice(2, 4) === sq;
          return (
            <button
              key={sq}
              aria-label={`${sq}${piece ? " " + (piece.color === "w" ? "white" : "black") + " " + { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" }[piece.type] : ""}`}
              className={`square ${(rank + file) % 2 ? "light" : "dark"} ${selected === sq ? "selected" : ""} ${highlighted ? "last" : ""}`}
              onClick={() => !disabled && onSquare?.(sq)}
              draggable={!disabled && !!piece}
              onDragStart={(e) => {
                e.dataTransfer.setData("text/plain", sq);
                onSquare?.(sq);
              }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (!disabled) onSquare?.(sq);
              }}
            >
              {fi === 0 && <span className="rank">{rank + 1}</span>}
              {ri === 7 && (
                <span className="file">{String.fromCharCode(97 + file)}</span>
              )}
              {piece && (
                <span
                  className={`piece ${piece.color === "w" ? "white-piece" : "black-piece"}`}
                >
                  {PIECES[piece.color + piece.type]}
                </span>
              )}
              {legal.includes(sq) && (
                <span className={piece ? "capture-ring" : "legal-dot"} />
              )}
            </button>
          );
        }),
      )}
    </div>
  );
}
function Chart({
  data,
}: {
  data: NonNullable<Research["supervised"]>["history"];
}) {
  if (!data.length) return null;
  const max = Math.max(
    ...data.flatMap((d) => [d.train_loss, d.validation_loss]),
    0.001,
  );
  const line = (key: "train_loss" | "validation_loss") =>
    data
      .map(
        (d, i) =>
          `${35 + (i / Math.max(data.length - 1, 1)) * 520},${145 - (d[key] / max) * 120}`,
      )
      .join(" ");
  return (
    <div className="chart">
      <svg
        viewBox="0 0 580 185"
        role="img"
        aria-label="Training and validation mean squared error by epoch"
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="35"
              x2="555"
              y1={145 - t * 120}
              y2={145 - t * 120}
              stroke="#35433e"
            />
            <text x="0" y={149 - t * 120}>
              {(max * t).toFixed(2)}
            </text>
          </g>
        ))}
        <polyline
          fill="none"
          stroke="#bce3aa"
          strokeWidth="2.5"
          points={line("train_loss")}
        />
        <polyline
          fill="none"
          stroke="#b4a2ee"
          strokeWidth="2.5"
          points={line("validation_loss")}
        />
        <text x="35" y="175">
          Epoch 1
        </text>
        <text x="480" y="175">
          Epoch {data.length}
        </text>
      </svg>
      <div className="legend">
        <span>● Training</span>
        <span>● Validation</span>
      </div>
    </div>
  );
}

function routePosition() {
  try {
    const params = new URLSearchParams(window.location.search);
    const fen = params.get("fen") || DEFAULT_POSITION;
    const moves = (params.get("moves") || "").split(",").filter(Boolean);
    if (moves.length > 1024 || fen.length > 100) throw new Error("Invalid position");
    boardFrom(fen, moves);
    return { fen, moves };
  } catch { return { fen: DEFAULT_POSITION, moves: [] as string[] }; }
}
const showcaseEnabled = import.meta.env.DEV || import.meta.env.VITE_SHOWCASE === "1";
export default function App() {
  const [path, setPath] = useState(window.location.pathname);
  const page = showcaseEnabled ? approaches.find(a => a.path === path) : undefined;
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [initialFen, setInitialFen] = useState(() => routePosition().fen);
  const [tab, setTab] = useState<"play" | "compare" | "research">("play");
  const [models, setModels] = useState<Model[]>([]);
  const [model, setModel] = useState("handcrafted");
  const [connection, setConnection] = useState<"loading" | "ready" | "offline">(
    "loading",
  );
  const [history, setHistory] = useState<string[]>(() => routePosition().moves);
  const [color, setColor] = useState("w");
  const [selected, setSelected] = useState<Square | null>(null);
  const [promotion, setPromotion] = useState<{
    from: Square;
    to: Square;
  } | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<string[]>([]);
  const [terminal, setTerminal] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [research, setResearch] = useState<Research | null>(null);
  const [example, setExample] = useState(0);
  const [comparisons, setComparisons] = useState<
    { model: Model; result: Analysis }[]
  >([]);
  const [replay, setReplay] = useState(0);
  const [replayPly, setReplayPly] = useState(0);
  const generation = useRef(0);
  const activeRequest = useRef(false);
  const game = boardFrom(initialFen, history);
  const currentModel = models.find((m) => m.id === model);
  const expectedCheckpoint = page && catalog?.models[page.id]?.checkpoint_id;
  const identityMatches = !page || (!!currentModel && currentModel.id === page.id && (!!expectedCheckpoint && expectedCheckpoint === currentModel.checkpoint_id));
  const done = !!terminal || game.isGameOver();
  function navigate(id: string) {
    if (busy) return;
    const target = approaches.find(a => a.id === id);
    const query = new URLSearchParams({ fen: initialFen, moves: history.join(",") });
    window.history.pushState(null, "", (target?.path || "/") + "?" + query);
    setPath(window.location.pathname);
    if (target) setModel(target.id);
    generation.current++; setAnalysis(null); setSelected(null); setPromotion(null); setError(""); setTab("play");
  }
  useEffect(() => {
    const back = () => { const position = routePosition(); setInitialFen(position.fen); setHistory(position.moves); setPath(window.location.pathname); const target = approaches.find(a => a.path === window.location.pathname); if (target) setModel(target.id); generation.current++; setAnalysis(null); setSelected(null); setPromotion(null); setError(""); setTerminal(null); };
    window.addEventListener("popstate", back); return () => window.removeEventListener("popstate", back);
  }, []);
  async function inspectPosition() {
    if (busy || !identityMatches || connection !== "ready") return;
    setBusy(true); setError(""); const token = generation.current;
    try { const result = await request<Analysis>("/analyze", { initial_fen: initialFen, moves: history, model_id: model, nodes: 2000, time_ms: 1500 });
      if (page && result.checkpoint_id !== expectedCheckpoint) throw new Error("The engine checkpoint changed. Reconnect before continuing.");
      if (token === generation.current) { setAnalysis(result); setAnalysisHistory(history); }
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  function loadExample(index: number) {
    if (busy) return;
    const position = examples[index]; generation.current++; setInitialFen(position.fen); setHistory(position.moves || []); setColor(boardFrom(position.fen, position.moves || []).turn()); setSelected(null); setAnalysis(null); setTerminal(null); setError("");
  }
  async function connect() {
    setConnection("loading");
    try {
      const data = await request<{ models: Model[]; default: string }>(
        "/models",
      );
      setModels(data.models);
      setModel((showcaseEnabled && approaches.find(a => a.path === window.location.pathname)?.id) || data.default);
      setConnection("ready");
    } catch {
      setConnection("offline");
    }
  }
  useEffect(() => {
    void connect();
    fetch("/research/model-catalog.json").then(r => r.ok ? r.json() : null).then(setCatalog).catch(() => setCatalog(null));
    fetch("/research/results.json")
      .then((r) => (r.ok ? r.json() : null))
      .then(setResearch)
      .catch(() => setResearch(null));
  }, []);
  function reset(newColor = color) {
    generation.current++;
    setHistory([]);
    setInitialFen(DEFAULT_POSITION);
    setSelected(null);
    setPromotion(null);
    setAnalysis(null);
    setTerminal(null);
    setError("");
    setColor(newColor);
  }
  function move(from: Square, to: Square, piece = "q") {
    if (!identityMatches) return;
    try {
      const copy = boardFrom(initialFen, history);
      const m = copy.move({ from, to, promotion: piece });
      setHistory([...history, m.from + m.to + (m.promotion || "")]);
      setSelected(null);
      setPromotion(null);
      setError("");
    } catch {
      setSelected(null);
    }
  }
  function clickSquare(sq: Square) {
    if (busy || done || !identityMatches || game.turn() !== color || connection !== "ready") return;
    if (selected) {
      const options = game
        .moves({ square: selected, verbose: true })
        .filter((m) => m.to === sq);
      if (options.length) {
        if (options.some((m) => m.promotion))
          setPromotion({ from: selected, to: sq });
        else move(selected, sq);
        return;
      }
    }
    setSelected(game.get(sq)?.color === color ? sq : null);
  }
  async function engineMove() {
    if (activeRequest.current) return;
    activeRequest.current = true;
    setBusy(true);
    setError("");
    const token = generation.current;
    try {
      const result = await request<Analysis>("/analyze", {
        initial_fen: initialFen,
        moves: history,
        model_id: model,
        nodes: 2000,
        time_ms: 1500,
      });
      if (token !== generation.current) return;
      if (page && result.checkpoint_id !== expectedCheckpoint) throw new Error("The engine checkpoint changed. Reconnect before continuing.");
      setAnalysis(result);
      setAnalysisHistory(history);
      setTerminal(result.terminal);
      if (result.best_move) setHistory((h) => [...h, result.best_move!]);
    } catch (e) {
      if (token === generation.current) setError((e as Error).message);
    } finally {
      activeRequest.current = false;
      setBusy(false);
    }
  }
  useEffect(() => {
    if (
      tab === "play" &&
      identityMatches &&
      game.turn() !== color &&
      !done &&
      !busy &&
      !error &&
      connection === "ready"
    )
      void engineMove();
  }, [history, initialFen, color, tab, connection, busy, error, model, identityMatches]);
  function undo() {
    generation.current++;
    setHistory((h) =>
      h.slice(0, Math.max(0, h.length - (game.turn() === color ? 2 : 1))),
    );
    setAnalysis(null);
    setTerminal(null);
    setSelected(null);
    setError("");
  }
  async function compare() {
    setBusy(true);
    setError("");
    setComparisons([]);
    try {
      const values = [];
      for (const m of models.filter((m) => m.id !== "handcrafted")) {
        const result = await request<Analysis>("/analyze", {
          initial_fen: examples[example].fen,
          moves: examples[example].moves,
          model_id: m.id,
          nodes: 1000,
          time_ms: 3000,
        });
        values.push({ model: m, result });
      }
      setComparisons(values);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const exampleBoard = boardFrom(
    examples[example].fen,
    examples[example].moves,
  );
  const replayGame = new Chess();
  const selectedReplay = research?.games?.[replay];
  if (selectedReplay) {
    try {
      replayGame.loadPgn(selectedReplay.pgn);
      const moves = replayGame.history();
      const initial = replayGame.getHeaders().FEN || DEFAULT_POSITION;
      replayGame.load(initial);
      for (const move of moves.slice(0, replayPly)) replayGame.move(move);
    } catch {
      /* Display empty board if artifact is malformed. */
    }
  }
  const historySAN = game.history();
  const status = terminal
    ? `Game complete · ${terminal}`
    : game.isCheckmate()
      ? "Checkmate"
      : game.isDraw()
        ? "Draw"
        : busy
          ? "Engine is thinking"
          : game.turn() === color
            ? "Your move"
            : "Engine to move";
  return (
    <>
      <header>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("overview");
          }}
        >
          <span className="brand-mark">♞</span>
          <span>
            CHESSENGINE<span className="brand-sub">A MACHINE LEARNING LAB</span>
          </span>
        </a>
        <nav aria-label="Main navigation">
          {(["play", "compare", "research"] as const).map((t) => (
            <button
              key={t}
              className={tab === t ? "active" : ""}
              onClick={() => setTab(t)}
            >
              {t === "play" ? (
                <Play size={14} />
              ) : t === "compare" ? (
                <FlaskConical size={15} />
              ) : (
                <BookOpen size={15} />
              )}{" "}
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <div className={`connection ${connection}`}>
          <i />
          {connection === "ready"
            ? "Engine online"
            : connection === "loading"
              ? "Waking engine"
              : "Engine offline"}
        </div>
      </header>
      <main>
        <div className="eyebrow">
          <span /> FROM SUPERVISION TO SELF-PLAY{" "}
          <span className="version">EXPERIMENT 01</span>
        </div>
        <section className={`intro ${page ? "model-intro" : ""}`}>
          <div>
            <h1>
              {page ? <>{page.title}<br/><em>{page.subtitle}</em></> : tab === "play" ? (
                <>
                  Learning, one
                  <br />
                  <em>position at a time.</em>
                </>
              ) : tab === "compare" ? (
                <>
                  Same position.
                  <br />
                  <em>Different experience.</em>
                </>
              ) : (
                <>
                  Evidence over
                  <br />
                  <em>assumptions.</em>
                </>
              )}
            </h1>
            <p>
              {page ? page.description : tab === "play"
                ? "Play against a neural chess evaluator. Explore what changes when a model learns from its own games."
                : tab === "compare"
                  ? "Look inside the experiment. Compare the moves and evaluations of the supervised and self-play models."
                  : "A reproducible study of supervised learning and reinforcement learning, including the results that do not improve."}
            </p>
          </div>
          <div className="intro-note">
            <span className="tiny-label">THE RESEARCH QUESTION</span>
            <p>
              {page ? page.question : <>Does learning through self-play<br />make a chess evaluator better?</>}
            </p>
            <button className="text-button" onClick={() => setTab("research")}>
              Explore the evidence <ArrowUpRight size={16} />
            </button>
          </div>
        </section>
        {showcaseEnabled && <ModelGallery navigate={navigate} disabled={busy} />}
        {page && <><ModelGuide approach={page} catalog={catalog} checkpoint={currentModel?.checkpoint_id} analysis={analysis} analyze={inspectPosition} busy={busy} available={connection === "ready" && identityMatches}/><div className="curated-positions"><span className="tiny-label">TRY THE SAME POSITION ACROSS MODELS</span>{examples.map((e,i)=><button key={e.name} disabled={busy} onClick={()=>loadExample(i)}>{e.name}</button>)}</div></>}
        {connection !== "ready" && (
          <div className="notice">
            <Activity size={18} />
            <div>
              <strong>
                {connection === "loading"
                  ? "Connecting to the engine"
                  : "The live engine is unavailable"}
              </strong>
              <p>
                The engine may be starting or temporarily unavailable. Retry shortly.
                Research and saved games remain available.
              </p>
            </div>
            <button
              onClick={() => void connect()}
              disabled={connection === "loading"}
            >
              Reconnect
            </button>
          </div>
        )}
        {error && (
          <div className="notice error" role="alert">
            {error}
            <button onClick={() => setError("")}>Try again</button>
          </div>
        )}
        {tab === "play" && (
          <div className="workspace">
            <section className="board-panel">
              <div className="player-strip">
                <div className="avatar">
                  <Cpu size={18} />
                </div>
                <div>
                  <strong>{currentModel?.name || "ChessEngine"}</strong>
                  <span>
                    {currentModel?.evidence === "pilot"
                      ? "Pilot checkpoint · research in progress"
                      : "Shared search engine"}
                  </span>
                </div>
                <span className="chip">
                  {color === "w" ? "BLACK" : "WHITE"}
                </span>
              </div>
              <Board
                game={game}
                flip={color === "b"}
                selected={selected}
                onSquare={clickSquare}
                disabled={busy || done || !identityMatches}
                lastMove={history.at(-1)}
              />
              <div className="player-strip bottom">
                <div className="avatar you">YOU</div>
                <div>
                  <strong>{status}</strong>
                  <span>
                    {game.isCheck() && !done
                      ? "Your king is in check"
                      : "No clock. Take your time."}
                  </span>
                </div>
                <span className="chip">
                  {color === "w" ? "WHITE" : "BLACK"}
                </span>
              </div>
            </section>
            <aside>
              <section className="card">
                <div className="section-heading">
                  <span className="tiny-label">01 / YOUR OPPONENT</span>
                  <Cpu size={17} />
                </div>
                <h2>Choose an evaluator</h2>
                <p className="muted">
                  One search engine. Different ways to understand the board.
                </p>
                <div className="model-options">
                  {models.map((m) => (
                    <button
                      disabled={busy || history.length > 0}
                      key={m.id}
                      className={`model-option ${model === m.id ? "chosen" : ""}`}
                      onClick={() => {
                        if (page) navigate(m.id); else { setModel(m.id); setAnalysis(null); }
                      }}
                    >
                      <span className="radio" />
                      <span>
                        <strong>{m.name}</strong>
                        <small>
                          {m.kind === "move-guided" ? "Learned root ordering + supervised value" : m.kind === "rl"
                            ? "Learned from self-play outcomes"
                            : m.kind === "supervised"
                              ? "Learned from Stockfish evaluations"
                              : "Material and piece-square heuristics"}
                        </small>
                      </span>
                      {model === m.id && (
                        <span className="selected-label">SELECTED</span>
                      )}
                    </button>
                  ))}
                </div>
                <div className="color-row">
                  <span>Play as</span>
                  <div className="segmented">
                    <button
                      disabled={busy}
                      className={color === "w" ? "chosen" : ""}
                      onClick={() => reset("w")}
                    >
                      White
                    </button>
                    <button
                      disabled={busy}
                      className={color === "b" ? "chosen" : ""}
                      onClick={() => reset("b")}
                    >
                      Black
                    </button>
                  </div>
                </div>
                <div className="actions">
                  <button onClick={() => reset()} disabled={busy}>
                    <RefreshCw size={14} /> New game
                  </button>
                  <button onClick={undo} disabled={busy || !history.length}>
                    <RotateCcw size={14} /> Undo
                  </button>
                  <button
                    title="Export PGN"
                    aria-label="Export PGN"
                    onClick={() => download("chessengine-game.pgn", game.pgn())}
                  >
                    <Download size={15} />
                  </button>
                </div>
              </section>
              <section className="card analysis-card">
                <div className="section-heading">
                  <span className="tiny-label">02 / ENGINE INSIGHT</span>
                  <Activity size={16} />
                </div>
                <div className="eval-value">
                  {analysis
                    ? analysis.mate_score
                      ? "Mate found"
                      : `${analysis.value >= 0 ? "+" : ""}${analysis.value.toFixed(2)}`
                    : "—"}
                  <span>MODEL ADVANTAGE</span>
                </div>
                <div className="eval-track">
                  <span
                    style={{
                      width: `${analysis ? (analysis.value + 1) * 50 : 50}%`,
                    }}
                  />
                </div>
                <p className="micro">
                  From the engine’s perspective before its last move. Not a win
                  probability.
                </p>
                <div className="metrics">
                  <div>
                    <strong>{analysis?.depth ?? "—"}</strong>
                    <span>DEPTH</span>
                  </div>
                  <div>
                    <strong>{analysis?.nodes.toLocaleString() ?? "—"}</strong>
                    <span>NODES</span>
                  </div>
                  <div>
                    <strong>
                      {analysis
                        ? `${(analysis.elapsed_ms / 1000).toFixed(2)}s`
                        : "—"}
                    </strong>
                    <span>SEARCH TIME</span>
                  </div>
                </div>
                <span className="tiny-label">PRINCIPAL VARIATION</span>
                <p className="pv">
                  {analysis
                    ? sanLine(
                        boardFrom(initialFen, analysisHistory),
                        analysis.pv,
                      ) || "No continuation"
                    : "Make a move to begin the analysis."}
                </p>
                {analysis && (
                  <code className="hash">
                    Checkpoint {analysis.checkpoint_id.slice(0, 12)}
                  </code>
                )}
              </section>
              <section className="card moves-card">
                <div className="section-heading">
                  <span className="tiny-label">03 / MOVE HISTORY</span>
                  <span className="micro">{history.length} plies</span>
                </div>
                <div className="move-list">
                  {historySAN.length ? (
                    Array.from(
                      { length: Math.ceil(historySAN.length / 2) },
                      (_, i) => (
                        <div key={i}>
                          <span>{i + 1}.</span>
                          <strong>{historySAN[i * 2]}</strong>
                          <strong>{historySAN[i * 2 + 1] || "—"}</strong>
                        </div>
                      ),
                    )
                  ) : (
                    <p className="muted">Your first move starts the story.</p>
                  )}
                </div>
              </section>
            </aside>
          </div>
        )}
        {tab === "compare" && (
          <div className="workspace">
            <section className="board-panel">
              <div className="player-strip">
                <FlaskConical size={20} />
                <strong>{examples[example].name}</strong>
              </div>
              <Board game={exampleBoard} disabled />
            </section>
            <aside>
              <section className="card">
                <span className="tiny-label">A CONTROLLED COMPARISON</span>
                <h2>
                  Change the model,
                  <br />
                  keep the position.
                </h2>
                <p className="muted">
                  Both searches use a 1,000-node ceiling. Actual completed depth
                  and node counts are shown.
                </p>
                <label className="field">
                  Position
                  <select
                    disabled={busy}
                    value={example}
                    onChange={(e) => {
                      setExample(Number(e.target.value));
                      setComparisons([]);
                    }}
                  >
                    {examples.map((p, i) => (
                      <option key={p.name} value={i}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="primary"
                  disabled={
                    busy ||
                    connection !== "ready" ||
                    models.filter((m) => m.id !== "handcrafted").length < 2
                  }
                  onClick={() => void compare()}
                >
                  {busy ? "Analyzing…" : "Compare models"}
                  <ArrowRight size={16} />
                </button>
                {models.filter((m) => m.id !== "handcrafted").length < 2 && (
                  <p className="micro">
                    Two trained checkpoints must be installed to compare models.
                  </p>
                )}
              </section>
              {comparisons.map(({ model: m, result: r }) => (
                <section className="card" key={m.id}>
                  <span className="tiny-label">{m.name}</span>
                  <h2>
                    {sanLine(exampleBoard, r.pv.slice(0, 1)) ||
                      "Terminal position"}
                  </h2>
                  <p>
                    Value {r.value.toFixed(3)} · Depth {r.depth} · {r.nodes}{" "}
                    nodes
                  </p>
                  <p className="pv">{sanLine(exampleBoard, r.pv)}</p>
                  <code className="hash">{r.checkpoint_id.slice(0, 16)}</code>
                </section>
              ))}
            </aside>
          </div>
        )}
        {tab === "research" && (
          <>
            <div className="research-banner">
              <FlaskConical size={22} />
              <div>
                <strong>
                  {research?.status || "The study is being prepared"}
                </strong>
                <p>
                  {research?.aggregate
                    ? "All three training seeds are reported. Conclusions apply to the stated data and shallow search budget."
                    : "Pilot evidence tests the pipeline. It does not establish that reinforcement learning improves chess strength."}
                </p>
              </div>
              <a href="/research/report.html" target="_blank" rel="noreferrer">
                Read the report <ArrowUpRight size={16} />
              </a>
            </div>
            <div className="research-grid">
              <section className="card">
                <span className="tiny-label">01 / FITTING THE EVALUATOR</span>
                <h2>Does the model learn?</h2><p className="notice">Archive: these charts and saved games describe the original local study. The four model pages show evidence for their own checkpoint versions.</p>
                {research?.supervised ? (
                  <Chart data={research.supervised.history} />
                ) : (
                  <p className="muted">
                    Training curves will appear here after a measured run. No
                    synthetic results are displayed.
                  </p>
                )}
                <p className="micro">
                  Mean squared error against teacher WDL targets. Lower error is
                  not necessarily stronger play.
                </p>
              </section>
              <section className="card">
                <span className="tiny-label">02 / PLAYING STRENGTH</span>
                <h2>
                  {research?.summary
                    ? `${(research.summary.score * 100).toFixed(1)}%`
                    : "Awaiting matches"}
                  <small> {research?.summary ? "match score" : ""}</small>
                </h2>
                {research?.summary && (
                  <>
                    <p>
                      {research.summary.games} games · 95% paired interval{" "}
                      {research.summary.ci95
                        .map((v) => (v * 100).toFixed(1) + "%")
                        .join(" – ")}
                    </p>
                    <span className="result-label">
                      {research.summary.conclusion}
                    </span>
                    <p className="micro">
                      {research.summary.capped_games} games reached the move
                      cap. See the report for sensitivity analysis.
                    </p>
                  </>
                )}
                <p className="muted">
                  Colors are swapped for each opening. Search budgets are held
                  constant.
                </p>
              </section>
            </div>
            {research?.aggregate && (
              <section className="card seed-results">
                <span className="tiny-label">
                  ROBUSTNESS / ALL TRAINING SEEDS
                </span>
                <h2>One result is not the whole experiment.</h2>
                <p className="muted">
                  The headline above is seed {research.aggregate.representative}
                  , selected using development matches. Each row below uses 400
                  held-out games. Training-seed variation is separate from match
                  uncertainty.
                </p>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Seed</th>
                        <th>Score</th>
                        <th>95% paired interval</th>
                        <th>Conclusion</th>
                        <th>Records</th>
                      </tr>
                    </thead>
                    <tbody>
                      {research.aggregate.runs.map((r) => (
                        <tr key={r.seed}>
                          <td>{r.seed}</td>
                          <td>{(r.summary.score * 100).toFixed(1)}%</td>
                          <td>
                            {r.summary.ci95
                              .map((x) => (x * 100).toFixed(1) + "%")
                              .join(" – ")}
                          </td>
                          <td>{r.summary.conclusion}</td>
                          <td>
                            <a
                              href={`/research/games-seed-${r.seed}.json`}
                              download
                            >
                              Download games
                            </a>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <img
                  className="strength-chart"
                  src="/research/strength.png"
                  alt="Development playing performance by training games and compute for each seed"
                />
              </section>
            )}
            <div className="workspace replay">
              <section className="board-panel">
                <div className="player-strip">
                  <BookOpen size={18} />
                  <strong>Inside an experiment game</strong>
                </div>
                <Board game={replayGame} disabled lastMove={undefined} />
              </section>
              <aside>
                <section className="card">
                  <span className="tiny-label">03 / REPLAY THE EVIDENCE</span>
                  <h2>Every result has a game.</h2>
                  <p className="muted">
                    Inspect the decisions behind a score, including draws and
                    regressions.
                  </p>
                  <label className="field">
                    Recorded game
                    <select
                      value={replay}
                      onChange={(e) => {
                        setReplay(Number(e.target.value));
                        setReplayPly(0);
                      }}
                    >
                      {research?.games?.map((g, i) => (
                        <option value={i} key={i}>
                          Game {i + 1} · {g.opening_id} · {g.candidate_color}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="actions">
                    <button
                      disabled={!selectedReplay || replayPly === 0}
                      onClick={() => setReplayPly((p) => Math.max(0, p - 1))}
                    >
                      <ArrowLeft size={16} /> Back
                    </button>
                    <button
                      disabled={
                        !selectedReplay ||
                        (() => {
                          try {
                            const g = new Chess();
                            g.loadPgn(selectedReplay.pgn);
                            return replayPly >= g.history().length;
                          } catch {
                            return true;
                          }
                        })()
                      }
                      onClick={() => setReplayPly((p) => p + 1)}
                    >
                      Next <ArrowRight size={16} />
                    </button>
                    <span className="micro">Ply {replayPly}</span>
                  </div>
                  {selectedReplay && (
                    <button
                      className="text-button"
                      onClick={() =>
                        download("experiment-game.pgn", selectedReplay.pgn)
                      }
                    >
                      Download this game <Download size={14} />
                    </button>
                  )}
                </section>
                <section className="card">
                  <span className="tiny-label">READ THIS RESULT CAREFULLY</span>
                  <ul className="limitations">
                    {(
                      research?.limitations || [
                        "Experiments have not yet been published.",
                      ]
                    ).map((t) => (
                      <li key={t}>{t}</li>
                    ))}
                  </ul>
                </section>
              </aside>
            </div>
          </>
        )}
        <section className="method-strip">
          <div>
            <span>01</span>
            <h3>Learn from a teacher</h3>
            <p>Fit position values to engine-generated targets.</p>
          </div>
          <ChevronRight />
          <div>
            <span>02</span>
            <h3>Learn through play</h3>
            <p>Update the evaluator from self-play outcomes.</p>
          </div>
          <ChevronRight />
          <div>
            <span>03</span>
            <h3>Test the difference</h3>
            <p>Compare frozen models under equal budgets.</p>
          </div>
        </section>
      </main>
      <footer>
        <span>
          CHESSENGINE <span className="footer-dot">/</span> An experiment in
          learning
        </span>
        <span>
          <a href="/models/supervised.npz" download>
            Supervised weights
          </a>{" "}
          ·{" "}
          <a href="/models/rl.npz" download>
            RL weights
          </a>
        </span>
      </footer>
      {promotion && (
        <div className="modal-backdrop">
          <div
            className="card promotion"
            role="dialog"
            aria-modal="true"
            aria-label="Choose promotion"
          >
            <h2>Promote your pawn</h2>
            <div>
              {["q", "r", "b", "n"].map((p) => (
                <button
                  key={p}
                  aria-label={`Promote to ${{ q: "queen", r: "rook", b: "bishop", n: "knight" }[p]}`}
                  onClick={() => move(promotion.from, promotion.to, p)}
                >
                  {PIECES[color + p]}
                </button>
              ))}
            </div>
            <button onClick={() => setPromotion(null)}>Cancel</button>
          </div>
        </div>
      )}
    </>
  );
}
