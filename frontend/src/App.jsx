import { useState, useRef, useEffect } from "react";

const SAMPLES = [
  "What is the recommended timeframe for giving antibiotics in adult sepsis?",
  "How is a COPD diagnosis confirmed?",
  "How is diabetic ketoacidosis managed in adults with type 1 diabetes?",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [meta, setMeta] = useState(null); // {sources, contexts, refused}
  const [stats, setStats] = useState(null); // {latency_ms, cost_usd, grounded, model}
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const answerRef = useRef(null);

  useEffect(() => {
    if (answerRef.current) answerRef.current.scrollTop = answerRef.current.scrollHeight;
  }, [answer]);

  async function ask(q) {
    const query = (q ?? question).trim();
    if (!query || busy) return;
    setBusy(true);
    setError("");
    setAnswer("");
    setMeta(null);
    setStats(null);

    try {
      const res = await fetch("/v1/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query, top_k: 8 }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // keep the trailing partial event
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const evt = JSON.parse(line.slice(6));
          if (evt.type === "meta") {
            setMeta({ sources: evt.sources, contexts: evt.contexts, refused: evt.refused });
          } else if (evt.type === "token") {
            setAnswer((a) => a + evt.text);
          } else if (evt.type === "done") {
            setStats({
              latency_ms: evt.latency_ms,
              cost_usd: evt.cost_usd,
              grounded: evt.grounded,
              groundedness: evt.groundedness,
              model: evt.model,
            });
          } else if (evt.type === "error") {
            setError(evt.message);
          }
        }
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Clinical RAG</h1>
        <p className="sub">Grounded answers from Ireland's NCEC National Clinical Guidelines</p>
      </header>

      <div className="composer">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
          }}
          placeholder="Ask a clinical question (Ctrl/Cmd+Enter to send)…"
          rows={2}
        />
        <button onClick={() => ask()} disabled={busy}>
          {busy ? "…" : "Ask"}
        </button>
      </div>

      <div className="samples">
        {SAMPLES.map((s) => (
          <button key={s} className="chip" onClick={() => { setQuestion(s); ask(s); }} disabled={busy}>
            {s.length > 48 ? s.slice(0, 45) + "…" : s}
          </button>
        ))}
      </div>

      <div className="grid">
        <section className="answer-card">
          <div className="card-head">
            <span>Answer</span>
            {meta?.refused && <span className="badge warn">refused — outside corpus</span>}
            {stats && !meta?.refused && (
              <span className={`badge ${stats.grounded ? "ok" : "warn"}`}>
                {stats.grounded ? "grounded ✓" : "low groundedness"}
              </span>
            )}
          </div>
          <div className="answer" ref={answerRef}>
            {answer || (busy ? "…" : "Your grounded answer will stream here.")}
          </div>
          {error && <div className="error">{error}</div>}
          <p className="disclaimer">
            Engineering demonstration — not a medical device and not clinical advice.
          </p>
        </section>

        <aside className="panel">
          <div className="metrics">
            <Metric label="Latency" value={stats ? `${stats.latency_ms} ms` : "—"} />
            <Metric label="Cost" value={stats ? `$${Number(stats.cost_usd).toFixed(5)}` : "—"} />
            <Metric label="Model" value={stats?.model || "—"} />
          </div>

          <h3>Retrieved sources</h3>
          <div className="sources">
            {meta?.contexts?.length
              ? meta.contexts.map((c, i) => (
                  <div className="source" key={i}>
                    <div className="source-head">
                      <span className="num">[{i + 1}]</span>
                      <a href={c.source_url} target="_blank" rel="noreferrer">
                        {c.source_title || "source"}
                      </a>
                      <span className="score">{c.score.toFixed(2)}</span>
                    </div>
                    <p className="snippet">{c.text.slice(0, 180)}…</p>
                  </div>
                ))
              : <p className="muted">No sources yet.</p>}
          </div>
        </aside>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
