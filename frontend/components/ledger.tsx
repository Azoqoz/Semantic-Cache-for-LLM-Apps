import { useEffect, useRef, useState } from "react";
import type { CacheSnapshot } from "@/lib/types";

function age(value: string, now: number) {
  const minutes = Math.max(0, Math.floor((now - Date.parse(value)) / 60000));
  return minutes < 1 ? "Just now" : minutes < 60 ? `${minutes}m ago` : minutes < 1440 ? `${Math.floor(minutes / 60)}h ago` : `${Math.floor(minutes / 1440)}d ago`;
}

export function Ledger({ snapshot, loading, clearing, error, disabled, refresh, clear }: {
  snapshot: CacheSnapshot | null; loading: boolean; clearing: boolean; error: string; disabled: boolean;
  refresh: () => void; clear: () => Promise<boolean>;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [filter, setFilter] = useState("");
  const [now, setNow] = useState(0);
  useEffect(() => {
    const tick = () => setNow(Date.now());
    const initial = window.setTimeout(tick, 0);
    const timer = window.setInterval(tick, 30000);
    return () => { clearTimeout(initial); clearInterval(timer); };
  }, []);
  const entries = snapshot?.entries.filter(entry => `${entry.question} ${entry.provider} ${entry.model} ${entry.id}`.toLowerCase().includes(filter.toLowerCase())) ?? [];
  async function confirmClear() { if (await clear()) dialog.current?.close(); }
  return <section id="ledger" className="ledger section-anchor">
    <div className="section-title"><div><p className="eyebrow">02 / PERSISTENT MEMORY</p><h2>Cache ledger<span className="count">{snapshot?.metrics.cache_entries ?? "—"}</span></h2></div><div className="actions"><button className="secondary" onClick={refresh} disabled={disabled || loading || clearing}>{loading ? "Refreshing…" : "Refresh ↻"}</button><button className="text-button danger" disabled={disabled || loading || clearing || !snapshot || (snapshot.metrics.cache_entries === 0 && snapshot.metrics.total_queries === 0)} onClick={() => dialog.current?.showModal()}>Clear cache</button></div></div>
    <div className="ledger-toolbar"><p>SQLite persistence <span>/</span> Entries are scoped by provider + model when isolation is on.</p><label className="search"><span className="sr-only">Filter cache entries</span><input value={filter} onChange={event => setFilter(event.target.value)} placeholder="Find prompt, model or ID" type="search" /></label></div>
    {error && <p className="error-message" role="alert">{error}</p>}
    <div className="table-scroll" tabIndex={0} role="region" aria-label="Cache entries" aria-busy={loading}>
      <table className="ledger-table"><thead><tr><th>ID / PROMPT</th><th>PROVIDER / MODEL</th><th>AGE</th><th>EXPIRY</th><th>REUSES</th></tr></thead><tbody>
        {entries.map(entry => {
          const expired = entry.expires_at !== null && now > 0 && Date.parse(entry.expires_at) <= now;
          return <tr key={entry.id} className={expired ? "expired-row" : ""}><td><details><summary><span className="entry-id mono">{String(entry.id).padStart(3, "0")}</span><span>{entry.question}</span><span className="expand-mark">+</span></summary><div className="entry-answer"><p>{entry.answer || "Empty stored response"}</p><span className="subtle">Last reused: {entry.last_accessed_at ? new Date(entry.last_accessed_at).toLocaleString() : "Not yet"}</span></div></details></td><td><span className="provider-label">{entry.provider}</span><span className="model-label mono">{entry.model}</span></td><td className="mono" title={new Date(entry.created_at).toLocaleString()}>{now ? age(entry.created_at, now) : "—"}</td><td><span className={`expiry ${expired ? "is-expired" : ""}`}>{entry.expires_at === null ? "No expiry" : expired ? "Expired" : "Active"}</span>{entry.expires_at && <small className="expiry-date">{new Date(entry.expires_at).toLocaleString()}</small>}</td><td className="reuse-count mono">{entry.access_count}</td></tr>;
        })}
      </tbody></table>
      {entries.length === 0 && <div className="empty-state"><span className="empty-symbol">{filter ? "⌕" : "∅"}</span><h3>{filter ? "No entries match." : loading ? "Opening the ledger…" : snapshot ? "A clean slate." : "The ledger is not available yet."}</h3><p>{filter ? "Try a different prompt or model." : snapshot ? "Send a question above. A cache miss creates the first record." : "Connect to the API to read persisted entries."}</p></div>}
    </div>
    <div className="ledger-foot"><span>{entries.length} shown{snapshot && snapshot.metrics.cache_entries > snapshot.entries.length ? ` · latest ${snapshot.entries.length} of ${snapshot.metrics.cache_entries} loaded` : ""}</span><span>Expired entries stay in the ledger but cannot match.</span></div>
    <dialog ref={dialog} className="confirm-dialog" onCancel={event => { if (clearing) event.preventDefault(); }} aria-labelledby="clear-title"><p className="eyebrow">RESET / SHARED CACHE</p><h2 id="clear-title">Start from zero?</h2><p>This permanently removes all cached answers and query metrics, across every provider and model. Evaluation results are separate.</p>{error && <p className="error-message" role="alert">{error}</p>}<div className="actions"><button className="secondary" disabled={clearing} onClick={() => dialog.current?.close()} autoFocus>Keep the cache</button><button className="danger-button" disabled={clearing} onClick={confirmClear}>{clearing ? "Clearing…" : "Clear everything"}</button></div></dialog>
  </section>;
}
