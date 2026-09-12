import { useState } from "react";
import { api, errorMessage } from "@/lib/api";
import type { EvaluationReport } from "@/lib/types";

export function Quality({ defaultThreshold, disabled }: { defaultThreshold: number; disabled: boolean }) {
  const [threshold, setThreshold] = useState(defaultThreshold);
  const [usedThreshold, setUsedThreshold] = useState<number | null>(null);
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  async function evaluate() {
    if (pending) return;
    setPending(true); setError("");
    try { const result = await api.evaluate(threshold); setReport(result); setUsedThreshold(threshold); }
    catch (error) { setError(errorMessage(error)); }
    finally { setPending(false); }
  }
  return <section id="quality" className="quality section-anchor">
    <div className="section-title"><div><p className="eyebrow">03 / TEST THE ASSUMPTION</p><h2>Cache quality</h2></div><span className="quality-stamp">MEASURE.<br />THEN REUSE.</span></div>
    <div className="quality-intro"><p>Similar words don’t always mean the same thing.</p><span>Evaluate labeled pairs before choosing a reuse threshold.</span></div>
    <div className="quality-controls"><label htmlFor="quality-threshold">Evaluation threshold <output className="mono">{threshold.toFixed(2)}</output><input id="quality-threshold" type="range" min="0" max="1" step="0.01" value={threshold} disabled={pending} onChange={event => setThreshold(Number(event.target.value))} /></label><span className="subtle">Independent of your query settings.</span><button onClick={evaluate} disabled={disabled || pending}>{pending ? "Evaluating pairs…" : report ? "Run evaluation again ↗" : "Run quality check ↗"}</button></div>
    {error && <p role="alert" className="error-message">{error}</p>}
    <div aria-live="polite" aria-busy={pending}>{pending && <p className="pending-note"><span className="spinner" /> Embedding and scoring the evaluation dataset. Results appear when complete.</p>}
      {!report && !pending && <div className="quality-empty"><span>01 <b>Embed pairs</b></span><span>02 <b>Compare intent</b></span><span>03 <b>Measure precision</b></span></div>}
      {report && <div className={pending ? "stale" : ""}><div className="quality-report-header"><span className="eyebrow">RESULTS AT <b className="mono">{usedThreshold?.toFixed(2)}</b> · {report.selected_rows.length} PAIRS</span><span>Recommended threshold <b className="recommendation mono">{report.recommended_threshold.toFixed(2)}</b></span></div>
        {usedThreshold !== threshold && <p className="subtle">Threshold changed. Run evaluation to update these results.</p>}
        <div className="quality-metrics">{([ ["Accuracy", report.selected_metrics.accuracy], ["Precision", report.selected_metrics.precision], ["Recall", report.selected_metrics.recall], ["F1 score", report.selected_metrics.f1] ] as const).map(([label, value]) => <div key={label}><span>{label}</span><strong>{(value * 100).toFixed(1)}<small>%</small></strong></div>)}</div>
        <div className="quality-detail"><div><p className="eyebrow">ACCURACY BY DIFFICULTY</p>{report.difficulty_rows.map(row => <div className="difficulty" key={row.Difficulty}><span>{row.Difficulty}</span><div className="difficulty-track"><i style={{ width: `${row.Accuracy * 100}%` }} /></div><span className="mono">{row.Correct}/{row.Total}</span></div>)}</div><div><p className="eyebrow">CLASSIFICATION COUNTS</p><div className="confusion"><span>True positive <b>{report.selected_metrics.true_positives}</b></span><span>True negative <b>{report.selected_metrics.true_negatives}</b></span><span>False positive <b>{report.selected_metrics.false_positives}</b></span><span>False negative <b>{report.selected_metrics.false_negatives}</b></span></div></div></div>
        <details className="quality-details"><summary>Compare thresholds <span>↗</span></summary><p className="subtle">Recommendation: highest F1, then precision, then higher threshold.</p><div className="table-scroll" tabIndex={0} role="region" aria-label="Threshold comparisons"><table><thead><tr><th>THRESHOLD</th><th>ACCURACY</th><th>PRECISION</th><th>RECALL</th><th>F1</th></tr></thead><tbody>{report.comparison_rows.map(row => <tr key={row.Threshold} className={row.Threshold === report.recommended_threshold ? "recommended-row" : ""}><td className="mono">{row.Threshold.toFixed(2)} {row.Threshold === report.recommended_threshold && <span aria-label="Recommended">↗</span>}</td>{[row.Accuracy, row.Precision, row.Recall, row["F1 score"]].map((value, index) => <td className="mono" key={index}>{(value * 100).toFixed(1)}%</td>)}</tr>)}</tbody></table></div></details>
        <details className="quality-details"><summary>Inspect all {report.selected_rows.length} pairs <span>↗</span></summary><div className="table-scroll" tabIndex={0} role="region" aria-label="Evaluation pair results"><table className="pair-table"><thead><tr><th>ORIGINAL / COMPARISON</th><th>LEVEL</th><th>COSINE</th><th>EXPECTED</th><th>PREDICTED</th><th>RESULT</th></tr></thead><tbody>{report.selected_rows.map((row, index) => <tr key={index}><td>{row["Original question"]}<small>{row["Similar question"]}</small></td><td>{row.Difficulty}</td><td className="mono">{row.Similarity.toFixed(4)}</td><td>{row.Expected}</td><td>{row.Predicted}</td><td><span className={row.Correct ? "pair-correct" : "pair-incorrect"}>{row.Correct ? "Correct" : "Incorrect"}</span></td></tr>)}</tbody></table></div></details>
      </div>}
    </div>
  </section>;
}
