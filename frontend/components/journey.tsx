import type { QueryResult } from "@/lib/types";

export function Journey({ result, pending }: { result: QueryResult | null; pending: boolean }) {
  const hit = result?.cache_hit;
  const state = pending ? "pending" : result?.hit_type ?? "ready";
  const completed = !!result && !pending;
  return <section className={`journey journey-${state}`} aria-label="Request journey">
    <div className="journey-heading"><span className="eyebrow">THE REQUEST CIRCUIT</span><span>{pending ? "Request sent · awaiting backend decision" : result ? "Confirmed route · last completed request" : "One question. Two possible paths."}</span></div>
    <div className="circuit">
      <div className={`station ${completed || pending ? "active" : ""}`}><span className="station-number">01</span><span className="station-symbol">↗</span><strong>Request</strong><small>{pending ? "In flight" : "Your question"}</small></div>
      <div className={`station ${completed ? "active" : ""}`}><span className="station-number">02</span><span className="station-symbol embedding-symbol">⠿</span><strong>Embed</strong><small>Local vectors</small></div>
      <div className={`station ${completed ? "active" : ""}`}><span className="station-number">03</span><span className="station-symbol">⌕</span><strong>Cache search</strong><small>{completed ? result.isolate_by_model ? "Provider + model" : "Shared cache" : "Exact → semantic"}</small></div>
      <div className="branch-group">
        <div className={`branch branch-hit ${completed && hit ? "chosen" : ""}`}><span className="branch-dot" /><div><strong>{completed && hit ? result.hit_type === "exact" ? "Exact hit" : "Semantic hit" : "Cache hit"}</strong><small>Reuse stored response</small></div><span>↗</span></div>
        <div className={`branch branch-miss ${completed && !hit ? "chosen" : ""}`}><span className="branch-dot" /><div><strong>Miss → provider</strong><small>{completed && !hit ? `${result.provider} → cache written` : "Generate → cache write"}</small></div><span>↘</span></div>
      </div>
      <div className={`station destination ${completed ? "active" : ""}`}><span className="station-number">04</span><span className="station-symbol">↳</span><strong>Response</strong><small>{completed ? "Returned to you" : "Ready for reuse"}</small></div>
    </div>
    <div className="circuit-caption"><span className="flow-key"><i /> Request flow</span><span className="flow-key hit-key"><i /> Reuse path</span><span className="flow-key miss-key"><i /> Provider path</span><span className="route-note">Embeddings run on every request, including exact matches.</span></div>
  </section>;
}

export function SimilarityRuler({ result }: { result: QueryResult }) {
  const score = result.similarity;
  const minimum = score !== null && score < 0 ? -1 : 0;
  const position = (value: number) => `${Math.min(100, Math.max(0, (value - minimum) / (1 - minimum) * 100))}%`;
  return <div className="similarity">
    <div className="similarity-heading"><span className="eyebrow">COSINE / CALIBRATED SCALE</span><span className="mono">{score === null ? "NOT MEASURED" : score.toFixed(4)}</span></div>
    <div className="ruler" role="img" aria-label={`Cosine similarity ${score === null ? "unavailable" : score.toFixed(4)}; threshold ${result.threshold.toFixed(2)}`}>
      <div className="ticks">{Array.from({ length: 21 }, (_, i) => <i key={i} className={i % 5 === 0 ? "major" : ""} />)}</div>
      <span className="threshold-pin" style={{ left: position(result.threshold) }} />
      {score !== null && <span className="score-pin" style={{ left: position(score) }} />}
    </div>
    <div className="ruler-labels mono"><span>{minimum.toFixed(2)}</span><span>1.00</span></div>
    <div className="ruler-legend"><span><i className="score-swatch" /> Score {score === null ? "—" : score.toFixed(4)}</span><span><i className="threshold-swatch" /> Threshold {result.threshold.toFixed(2)}</span></div>
    {score === null && <p className="subtle">{result.hit_type === "exact" ? "Same normalized question. Cosine comparison was not needed." : "No cosine score was available for this lookup."}</p>}
  </div>;
}
