import { useEffect, useRef, useState } from "react";

const BASE = import.meta.env.VITE_HOLDEM_API_BASE || "/api/v1";
const number = (n) => Number(n || 0).toLocaleString();
const streetName = (n) =>
  typeof n === "string"
    ? n
    : ["Preflop", "Flop", "Turn", "River"][n] || "Showdown";
const actionLabel = (e) =>
  e.kind === "raise_to"
    ? `Raise to ${number(e.amount)}`
    : e.kind === "call"
      ? e.check
        ? "Check"
        : "Call"
      : "Fold";
async function api(path, body) {
  let response;
  try {
    response = await fetch(
      `${BASE}${path}`,
      body === undefined
        ? {}
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          },
    );
  } catch {
    throw new Error(
      "Cannot reach the game server. Start it and try again.",
    );
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      "Cannot reach the game server. Start it and try again.",
    );
  }
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Action failed. Check your move and try again.",
    );
  return data;
}
function Card({ card, hidden, empty }) {
  const suit = card?.slice(-1)?.toLowerCase();
  return (
    <div
      className={`playing-card ${hidden ? "card-back" : ""} ${empty ? "card-empty" : ""} ${["d", "h"].includes(suit) ? "card-red" : ""}`}
      aria-label={hidden ? "Hidden card" : empty ? "Undealt card" : card}
    >
      {hidden ? (
        <span>♠</span>
      ) : (
        !empty && (
          <>
            <strong>{card?.slice(0, -1).replace("T", "10")}</strong>
            <span>{{ c: "♣", d: "♦", h: "♥", s: "♠" }[suit] || suit}</span>
          </>
        )
      )}
    </div>
  );
}
function Cards({ cards = [], count, hidden = false }) {
  return (
    <div className="lab-cards">
      {Array.from({ length: count }, (_, i) => (
        <Card
          key={i}
          card={cards[i]}
          hidden={hidden && !cards[i]}
          empty={!hidden && !cards[i]}
        />
      ))}
    </div>
  );
}

export default function HoldemGame() {
  const [registry, setRegistry] = useState({
    checkpoints: [],
    diagnostics: [],
  });
  const [sessions, setSessions] = useState([]);
  const [selected, setSelected] = useState("diagnostic:random");
  const initialSelectionHandled = useRef(false);
  const [seat, setSeat] = useState(1);
  const [session, setSession] = useState(null);
  const [hands, setHands] = useState([]);
  const [replay, setReplay] = useState("");
  const [step, setStep] = useState(0);
  const [raiseTo, setRaiseTo] = useState(300);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [online, setOnline] = useState(false);
  async function refresh() {
    try {
      const [r, s] = await Promise.all([
        api("/checkpoints"),
        api("/holdem/sessions"),
      ]);
      setRegistry(r);
      if (!initialSelectionHandled.current) {
        initialSelectionHandled.current = true;
        const recommended = r.recommended_checkpoint;
        if (recommended && r.checkpoints.some((checkpoint) => checkpoint.id === recommended.id)) {
          setSelected(recommended.id);
        } else if (r.checkpoints.length) {
          setSelected(r.checkpoints[0].id);
        }
      }
      setSessions(s.sessions);
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 30000);
    return () => clearInterval(timer);
  }, []);
  async function accept(data) {
    initialSelectionHandled.current = true;
    setSession(data);
    setReplay("");
    const raise = data.state.legal_actions.find((a) => a.kind === "raise_to");
    if (raise) setRaiseTo(raise.amount);
    const history = await api(`/holdem/sessions/${data.session_id}/hands`);
    setHands(history.hands.map((h) => ({ ...h, id: h.hand_id })));
    await refresh();
  }
  async function perform(fn) {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  const create = () =>
    perform(async () =>
      accept(
        await api("/holdem/sessions", {
          checkpoint_id: selected,
          human_seat: seat,
        }),
      ),
    );
  const act = (kind, amount) =>
    perform(async () =>
      accept(
        await api(`/holdem/sessions/${session.session_id}/act`, {
          revision: session.revision,
          kind,
          ...(amount == null ? {} : { amount }),
        }),
      ),
    );
  const next = () =>
    perform(async () =>
      accept(
        await api(`/holdem/sessions/${session.session_id}/new-hand`, {
          revision: session.revision,
        }),
      ),
    );
  const replayHand = hands.find((h) => h.id === replay);
  const state = replayHand
    ? step
      ? replayHand.events[step - 1]?.state
      : replayHand.initial
    : session?.state;
  const events = replayHand
    ? replayHand.events.slice(0, step)
    : session?.events || [];
  const human = session?.human_seat ?? seat;
  const bot = 1 - human;
  const lastBot = [...events].reverse().find((e) => e.probabilities);
  const canAct =
    session &&
    !replay &&
    !state?.terminal &&
    state?.current_player === human &&
    !busy;
  const menu = state?.legal_actions || [];
  const checkpoint = session?.checkpoint;
  const policies = [...registry.checkpoints, ...registry.diagnostics];
  const choice = policies.find((c) => c.id === selected);
  const policyName = (id) => {
    if (id === registry.recommended_checkpoint?.id && id) {
      return registry.recommended_checkpoint.label;
    }
    const policy = policies.find((c) => c.id === id);
    return policy?.name || policy?.label || id || "HoldemBot";
  };
  const setupSpec = choice?.game_spec || {
    stack: 20000,
    big_blind: 100,
    small_blind: 50,
  };
  const spec = state?.game_spec || setupSpec;
  const winnings = state?.returns?.[human];
  const actualHands = replayHand
    ? hands.findIndex((h) => h.id === replay) + 1
    : session?.hand_number;

  return (
    <main className="lab-shell">
      <div className="lab-intro">
        <div>
          <p className="lab-eyebrow">
            HEADS-UP NO-LIMIT POKER
          </p>
          <h1>Play Hold’em</h1>
          <p>Play the bot and replay its decisions.</p>
        </div>
        <div className={`lab-service ${online ? "is-online" : ""}`}>
          <i />
          {online ? "Server connected" : "Server offline"}
        </div>
      </div>
      {error && (
        <div className="lab-error" role="alert">
          {error}
          <button onClick={() => setError("")} aria-label="Dismiss error">
            ×
          </button>
        </div>
      )}
      {!online && (
        <div className="lab-offline">
          Start the game server to play.
        </div>
      )}
      <div className="lab-play-layout">
        <aside className="lab-setup">
          <span className="lab-eyebrow">TABLE SETUP</span>
          <h2>Choose an opponent</h2>
          <label htmlFor="checkpoint">Model</label>
          <select
            id="checkpoint"
            value={selected}
            onChange={(e) => {
              initialSelectionHandled.current = true;
              setSelected(e.target.value);
            }}
          >
            <optgroup label="Practice opponents">
              {(registry.diagnostics.length
                ? registry.diagnostics
                : [{ id: "diagnostic:random", label: "Uniform random" }]
              ).map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </optgroup>
            <optgroup label="Saved models">
              {registry.checkpoints.map((c) => (
                <option key={c.id} value={c.id}>
                  {policyName(c.id)}
                </option>
              ))}
            </optgroup>
          </select>
          {selected === registry.recommended_checkpoint?.id && registry.recommended_checkpoint.qualification && (
            <p className="lab-fine" data-testid="recommended-policy">
              {registry.recommended_checkpoint.qualification}
            </p>
          )}
          <label htmlFor="seat">Your seat</label>
          <select
            id="seat"
            value={seat}
            onChange={(e) => setSeat(Number(e.target.value))}
          >
            <option value={1}>Button / small blind</option>
            <option value={0}>Big blind</option>
          </select>
          <div className="lab-spec">
            <div>
              <span>Starting stack</span>
              <strong>
                {number(setupSpec.stack / setupSpec.big_blind)} bb
              </strong>
            </div>
            <div>
              <span>Blinds</span>
              <strong>
                {number(setupSpec.small_blind)} /{" "}
                {number(setupSpec.big_blind)}
              </strong>
            </div>
            <div>
              <span>Game</span>
              <strong>Heads-up · no rake</strong>
            </div>
          </div>
          <button
            className="lab-primary"
            onClick={create}
            disabled={busy || !online}
          >
            {busy ? "Working…" : "Start a new session"}
          </button>
          <p className="lab-fine">
            Stacks reset after each hand.
          </p>
          {sessions.length > 0 && (
            <>
              <label htmlFor="saved-session">Saved sessions</label>
              <select
                id="saved-session"
                value=""
                disabled={busy}
                onChange={(e) =>
                  e.target.value &&
                  perform(async () =>
                    accept(await api(`/holdem/sessions/${e.target.value}`)),
                  )
                }
              >
                <option value="">Select a session…</option>
                {sessions.map((s) => (
                  <option key={s.session_id} value={s.session_id}>
                    {policyName(s.checkpoint_id)} · {s.hands} hands ·{" "}
                    {new Date(s.updated).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </option>
                ))}
              </select>
            </>
          )}
        </aside>
        <section className="lab-table-wrap">
          <div className="lab-table-toolbar">
            <span>
              {session
                ? `${replay ? "REPLAY" : "LIVE"} · HAND ${actualHands}`
                : "START A SESSION"}
            </span>
            <span>
              {state ? streetName(state.street) : "HEADS-UP HOLD’EM"}
            </span>
          </div>
          <div className="lab-felt">
            <div className="lab-seat lab-bot">
              <div className="lab-player-label">
                <span className="lab-avatar">♠</span>
                <div>
                  <strong>{policyName(checkpoint?.id)}</strong>
                  <small>
                    {human === 1 ? "Big blind" : "Button / small blind"} ·{" "}
                    {number(state?.stacks?.[bot] ?? spec.stack)} chips
                  </small>
                </div>
              </div>
              <Cards cards={state?.opponent_cards || []} count={2} hidden />
            </div>
            <div className="lab-board">
              <div className="lab-pot-label">
                {state?.terminal ? "COMPLETED POT" : "POT"}{" "}
                <strong>
                  {number(state?.terminal ? state.last_pot : state?.pot)}
                </strong>
              </div>
              <Cards cards={state?.board || []} count={5} />
              <span className="lab-felt-mark">
                HOLDEMBOT <i>♠</i> NO-LIMIT HOLD’EM
              </span>
            </div>
            <div className="lab-seat lab-human">
              <Cards
                cards={state?.hole_cards || []}
                count={2}
                hidden={!state}
              />
              <div className="lab-player-label">
                <span className="lab-avatar">YOU</span>
                <div>
                  <strong>Your seat</strong>
                  <small>
                    {human === 1 ? "Button / small blind" : "Big blind"} ·{" "}
                    {number(state?.stacks?.[human] ?? spec.stack)} chips
                  </small>
                </div>
              </div>
            </div>
          </div>
          <div className="lab-controls">
            <div className="lab-turn">
              <i className={canAct ? "ready" : ""} />
              <span>
                {!session
                  ? "Start a session to play."
                  : replay
                    ? `Replay · decision ${step} of ${replayHand.events.length}`
                    : state?.terminal
                      ? `Hand complete · ${winnings > 0 ? "+" : ""}${number(winnings)} chips net`
                      : "Your action"}
              </span>
              {state?.terminal && !replay && (
                <button
                  className="lab-primary"
                  onClick={next}
                  disabled={busy}
                >
                  Deal next hand
                </button>
              )}
            </div>
            {session && !state?.terminal && !replay && (
              <>
                <div className="lab-actions">
                  {menu.map((a) => (
                    <button
                      key={a.slot}
                      onClick={() => act(a.kind, a.amount)}
                      disabled={!canAct}
                      className={a.kind === "fold" ? "lab-fold" : ""}
                    >
                      {a.kind === "raise_to"
                        ? `${a.label} · ${number(a.amount)}`
                        : a.label || actionLabel(a)}
                    </button>
                  ))}
                </div>
                <form
                  className="lab-custom-raise"
                  onSubmit={(e) => {
                    e.preventDefault();
                    act("raise_to", Number(raiseTo));
                  }}
                >
                  <label htmlFor="raise-total">Custom raise to</label>
                  <input
                    id="raise-total"
                    type="number"
                    step="1"
                    min="1"
                    max={
                      (state?.stacks?.[human] || 0) +
                      (state?.street_contributions?.[human] || 0)
                    }
                    value={raiseTo}
                    onChange={(e) => setRaiseTo(e.target.value)}
                    disabled={
                      !canAct || !menu.some((a) => a.kind === "raise_to")
                    }
                  />
                  <button
                    disabled={
                      !canAct ||
                      !menu.some((a) => a.kind === "raise_to") ||
                      !raiseTo
                    }
                  >
                    Raise
                  </button>
                  <small>Total chips committed on this street</small>
                </form>
              </>
            )}
          </div>
        </section>
      </div>
      <div className="lab-inspection">
        <section className="lab-inspector">
          <div className="lab-section-heading">
            <h2>Action probabilities</h2>
            <span>LAST BOT ACTION</span>
          </div>
          {lastBot ? (
            <>
              <p className="lab-fine">
                Chance of choosing each action, not odds of winning.
              </p>
              <div className="lab-probabilities">
                {lastBot.action_menu.map((a) => (
                  <div className="lab-probability" key={a.slot}>
                    <span>
                      {a.kind === "raise_to"
                        ? `${a.label} · ${number(a.amount)}`
                        : a.label || actionLabel(a)}
                      {a.slot === lastBot.slot && <em>chosen</em>}
                    </span>
                    <div>
                      <i
                        style={{
                          width: `${lastBot.probabilities[a.slot] * 100}%`,
                        }}
                      />
                    </div>
                    <strong>
                      {(100 * lastBot.probabilities[a.slot]).toFixed(1)}%
                    </strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="lab-empty">
              Probabilities appear after the bot acts.
            </p>
          )}
        </section>
        <section className="lab-inspector">
          <div className="lab-section-heading">
            <h2>Hand history</h2>
            <span>{events.length} ACTIONS</span>
          </div>
          {hands.length > 0 && (
            <div className="lab-replay">
              <select
                aria-label="Select hand replay"
                value={replay}
                onChange={(e) => {
                  setReplay(e.target.value);
                  setStep(0);
                }}
              >
                <option value="">Current hand · live</option>
                {hands.map((h, i) => (
                  <option value={h.id} key={h.id}>
                    Replay hand {i + 1}
                    {h.terminal ? " · complete" : " · in progress"}
                  </option>
                ))}
              </select>
              {replayHand && (
                <input
                  aria-label="Replay decision"
                  type="range"
                  min="0"
                  max={replayHand.events.length}
                  value={step}
                  onChange={(e) => setStep(Number(e.target.value))}
                />
              )}
            </div>
          )}
          <ol className="lab-journal">
            {events.map((e, i) => (
              <li key={i}>
                <span>{String(i + 1).padStart(2, "0")}</span>
                <strong>{e.actor === human ? "You" : "Bot"}</strong>
                <span>{actionLabel(e)}</span>
                <small>{streetName(e.street)}</small>
              </li>
            ))}
          </ol>
          {!events.length && (
            <p className="lab-empty">
              Actions appear here as you play.
            </p>
          )}
        </section>
      </div>
      <footer className="lab-footer">
        <span>HOLDEMBOT / {new Date().getFullYear()}</span>
        <span>Local play</span>
      </footer>
    </main>
  );
}
