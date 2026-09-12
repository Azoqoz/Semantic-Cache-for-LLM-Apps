import type { QueryResult } from "@/lib/types";
import { SimilarityRuler } from "./journey";

export function Decision({ result }: { result: QueryResult }) {
  const labels = { exact: "Exact hit.", semantic: "Semantic hit.", miss: "Cache miss." };
  return <section className={`decision decision-${result.hit_type}`} aria-label="Cache decision">
    <div className="decision-band"><div><p className="eyebrow">DECISION / {result.cache_hit ? "REUSED" : "GENERATED"}</p><h2>{labels[result.hit_type]}</h2></div><div className="decision-time"><strong>{result.latency_ms.toFixed(0)}<small>ms</small></strong><span>Total request latency</span></div><span className="decision-glyph" aria-hidden="true">{result.cache_hit ? "↺" : "↗"}</span></div>
    <div className="decision-body"><div className="answer-block"><p className="eyebrow">{result.cache_hit ? "FROM THE CACHE" : `FROM ${result.provider.toUpperCase()}`}</p><p className="request-echo">{result.question}</p><div className="answer">{result.answer || "The provider returned an empty response."}</div>
      {result.matched_question && <div className="matched"><span className="eyebrow">{result.cache_hit ? "MATCHED QUESTION" : "CLOSEST CACHED QUESTION"}</span><p>{result.matched_question}</p></div>}
      <div className="result-meta"><span className="mono">{result.provider} / {result.model}</span><span>{result.cache_hit ? "Provider call avoided" : "Cache write confirmed"}</span><span>TTL <b className="mono">{result.ttl_hours <= 0 ? "∞" : `${result.ttl_hours}h`}</b></span></div>
    </div><aside className="decision-aside"><SimilarityRuler result={result} />{result.estimated_cost_usd !== null && <div className="result-cost"><span className="eyebrow">ESTIMATED COST {result.cost_kind === "avoided" ? "AVOIDED" : "INCURRED"}</span><strong>${result.estimated_cost_usd.toFixed(4)}</strong><span className="subtle">{result.cost_basis === "local_zero" ? "Local provider · no API charge estimate" : result.cost_basis === "default_estimate" ? "Default illustrative estimate" : "Provider-supplied estimate"}</span></div>}</aside></div>
  </section>;
}
