"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, errorMessage } from "@/lib/api";
import type { Capabilities, CacheSnapshot, Provider, QueryResult } from "@/lib/types";
import { Journey } from "./journey";
import { Decision } from "./result";
import { Ledger } from "./ledger";
import { Quality } from "./quality";
import { ThemeToggle } from "./theme-toggle";

const examples = [
  { label: "Semantic caching", question: "What is semantic caching?" },
  { label: "Try a paraphrase", question: "Can you explain semantic cache?" },
  { label: "Embeddings", question: "What is an embedding?" },
  { label: "Reduce cost", question: "How can semantic caching reduce LLM cost?" },
];

export function CacheFlow() {
  const [connection, setConnection] = useState<"starting" | "online" | "offline">("starting");
  const [engine, setEngine] = useState<"warming" | "ready" | "error">("warming");
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [snapshot, setSnapshot] = useState<CacheSnapshot | null>(null);
  const [prompt, setPrompt] = useState(examples[0].question);
  const [provider, setProvider] = useState<Provider>("Demo");
  const [model, setModel] = useState("");
  const [threshold, setThreshold] = useState(.84);
  const [ttl, setTtl] = useState("168");
  const [isolated, setIsolated] = useState(true);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [pending, setPending] = useState(false);
  const [loadingCache, setLoadingCache] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [queryError, setQueryError] = useState("");
  const [cacheError, setCacheError] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const initialized = useRef(false);
  const cacheVersion = useRef(0);
  const submission = useRef(false);

  const refresh = useCallback(async () => {
    const version = ++cacheVersion.current;
    setLoadingCache(true); setCacheError("");
    try { const data = await api.cache(); if (version === cacheVersion.current) setSnapshot(data); }
    catch (error) {
      if (version === cacheVersion.current) setCacheError(errorMessage(error));
      if (error instanceof ApiError && error.code === "offline") setConnection("offline");
    } finally { if (version === cacheVersion.current) setLoadingCache(false); }
  }, []);

  const connect = useCallback(async () => {
    setConnection("starting");
    setEngine("warming");
    try {
      const [health, caps] = await Promise.all([api.health(), api.capabilities()]);
      if (health.status !== "ok") throw new Error("unavailable");
      setCapabilities(caps); setConnection("online"); if (caps.app_mode === "demo") setPrompt(caps.demo_samples?.[0]?.question ?? "");
      if (!initialized.current || caps.app_mode === "demo") {
        setProvider(caps.defaults.provider);
        setModel(caps.providers.find(item => item.name === caps.defaults.provider)?.default_model ?? "");
      }
      if (!initialized.current || caps.app_mode === "demo") {
        setThreshold(caps.defaults.threshold); setTtl(String(caps.defaults.ttl_hours));
        setIsolated(caps.defaults.isolate_by_model); initialized.current = true;
      }
    } catch { setConnection("offline"); }
  }, []);

  useEffect(() => { const timer = window.setTimeout(() => { void connect(); }, 0); return () => clearTimeout(timer); }, [connect]);

  useEffect(() => {
    if (connection !== "online" || engine !== "warming") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const readiness = await api.ready();
        if (cancelled) return;
        setEngine(readiness.status);
        if (readiness.status === "ready") { void refresh(); }
        else if (readiness.status === "warming") { timer = setTimeout(poll, 2000); }
      } catch (error) {
        if (!cancelled) {
          if (error instanceof ApiError && error.code === "offline") setConnection("offline");
          else setEngine("error");
        }
      }
    }
    timer = setTimeout(poll, 0);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [connection, engine, refresh]);

  function selectProvider(value: Provider) {
    setProvider(value);
    setModel(capabilities?.providers.find(item => item.name === value)?.default_model ?? "");
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submission.current || clearing || connection !== "online" || engine !== "ready") return;
    const hours = Number(ttl);
    if (!prompt.trim() || !ttl.trim() || !Number.isSafeInteger(hours) || (provider !== "Demo" && !model.trim())) {
      setQueryError("Enter a prompt, a model, and a whole number of TTL hours."); return;
    }
    submission.current = true; setPending(true); setQueryError(""); setResult(null); setAnnouncement("");
    try {
      const answer = await api.query({ question: prompt, provider, model: model || undefined, threshold, ttl_hours: hours, isolate_by_model: isolated });
      setResult(answer);
      setAnnouncement(`${answer.hit_type === "miss" ? "Cache miss. Answer generated and cached." : answer.hit_type === "exact" ? "Exact cache hit." : "Semantic cache hit."} ${answer.latency_ms.toFixed(0)} milliseconds.`);
      await refresh();
    } catch (error) {
      setQueryError(errorMessage(error));
      if (error instanceof ApiError && error.code === "service_warming") setEngine("warming");
      if (error instanceof ApiError && error.code === "model_initialization_failed") setEngine("error");
      if (error instanceof ApiError && error.code === "offline") setConnection("offline");
    } finally { setPending(false); submission.current = false; }
  }

  async function clear() {
    if (pending || clearing || capabilities?.controls.clear_cache === false) return false;
    setClearing(true); setCacheError("");
    try {
      const response = await api.clear();
      if (!response.cleared) throw new Error("not cleared");
      setResult(null); setQueryError(""); setSnapshot(null);
      setAnnouncement("All cache entries and query metrics cleared.");
      await refresh(); return true;
    } catch (error) {
      setCacheError(errorMessage(error));
      if (error instanceof ApiError && error.code === "offline") setConnection("offline");
      return false;
    } finally { setClearing(false); }
  }

  const online = connection === "online"; const isDemo = capabilities?.app_mode === "demo"; const requestLibrary = isDemo ? capabilities.demo_samples ?? [] : examples;
  const metrics = snapshot?.metrics;
  return <>
    <a className="skip-link" href="#lab">Skip to query lab</a>
    <header className="site-header shell"><a href="#" className="wordmark" aria-label="Cache Flow home"><span className="brand-mark" aria-hidden="true"><svg viewBox="0 0 34 34" fill="none"><path d="M3 17h8c6 0 3-11 11-11h2a9 9 0 0 1 0 18H14m5-5-5 5 5 5" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /><circle cx="3" cy="17" r="2.5" fill="currentColor" /><circle className="reuse-node" cx="24" cy="6" r="3" /></svg></span>CACHE FLOW</a><nav aria-label="Main navigation"><a href="#lab">Query lab</a><a href="#ledger">Cache ledger</a><a href="#quality">Quality</a></nav><div className="header-tools"><ThemeToggle /><div className={`connection ${connection}`} role="status"><i />{connection === "starting" ? "Connecting" : online ? engine === "ready" ? "API ready" : engine === "error" ? "Engine error" : "API live" : "API offline"}</div></div></header>
    <main className="shell">
      <section className="hero"><div><p className="eyebrow"><span className="tiny-square" /> SEMANTIC CACHE FOR LLM APPS</p><h1>Same intent.<br /><span>Shorter journey.</span></h1><p className="hero-deck">Ask. Match. Reuse.<br /> See what happens between a question and its answer.</p></div><div className="hero-art" aria-hidden="true"><svg viewBox="0 0 330 240"><path d="M20 120H105C150 120 130 35 180 35H295" /><path d="M105 120C150 120 130 205 180 205H295" className="art-miss" /><path d="M295 35V80C295 110 295 120 260 120H175" className="art-return" /><circle cx="20" cy="120" r="8" /><circle cx="105" cy="120" r="8" /><circle cx="295" cy="35" r="14" className="art-hit" /><circle cx="295" cy="205" r="9" className="art-miss-dot" /><path d="m190 105-15 15 15 15" className="art-arrow" /></svg><span className="art-caption">LESS REPETITION. MORE REUSE.</span></div></section>
      {connection === "offline" && <div className="offline-banner" role="alert"><div><strong>The lab is open. The API is offline.</strong><p>Start FastAPI or check the configured backend address. Any visible results are from the last successful request.</p></div><button className="secondary" onClick={() => void connect()}>Reconnect ↻</button></div>}
      <Journey result={result} pending={pending} />
      <section id="lab" className="lab section-anchor"><div className="section-title"><div><p className="eyebrow">01 / SEND A SIGNAL</p><h2>Query lab</h2></div><span className="mode-tag">{capabilities ? capabilities.app_mode === "demo" ? "PUBLIC DEMO" : "LOCAL MODE" : connection === "offline" ? "API OFFLINE" : "CONNECTING TO LAB"}</span></div>
        <form onSubmit={submit}><fieldset disabled={!online || pending || clearing || engine !== "ready"} className="composer"><div className="prompt-side"><label className="eyebrow" htmlFor="prompt">{isDemo ? "CURATED REQUEST LIBRARY" : "YOUR QUESTION"}</label><textarea id="prompt" readOnly={isDemo} aria-describedby={isDemo ? "demo-notice" : undefined} value={prompt} onChange={event => setPrompt(event.target.value)} required placeholder="What would you like to ask?" /><div className="examples"><span>{isDemo ? "SELECT" : "TRY"}</span>{requestLibrary.map(example => <button type="button" className="example" key={example.label} aria-pressed={isDemo ? prompt === example.question : undefined} onClick={() => setPrompt(example.question)}>{example.label} ↗</button>)}</div><div className="composer-bottom"><span className="subtle">{isDemo ? "Curated requests. Real ONNX lookup. No external providers." : provider === "Demo" ? "Offline topic rules. No API key needed." : provider === "Ollama" ? "Uses the Ollama server configured on the backend." : "Credentials are configured on the backend. No keys enter this browser."}</span><button type="submit" className="send-button" disabled={!prompt.trim() || engine !== "ready"}>{pending ? "Request in flight…" : "Trace request"}<span aria-hidden="true"><svg viewBox="0 0 24 24" fill="none"><path d="M2 19h6c5 0 2-14 7-14h6m-5-4 5 4-5 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg></span></button></div></div>
          <div className="settings-side"><div className="setting-row"><label htmlFor="provider">Provider</label><select id="provider" disabled={isDemo} value={provider} onChange={event => selectProvider(event.target.value as Provider)}>{capabilities ? capabilities.providers.map(item => <option key={item.name}>{item.name}</option>) : <option>Demo</option>}</select></div><div className="setting-row"><label htmlFor="model">Model</label><input id="model" className="mono" value={model} readOnly={provider === "Demo"} required={provider !== "Demo"} onChange={event => setModel(event.target.value)} placeholder="Backend default" /></div><div className="threshold-setting"><label htmlFor="threshold">Reuse threshold <output className="mono">{threshold.toFixed(2)}</output></label><input id="threshold" disabled={isDemo} type="range" min="0" max="1" step="0.01" value={threshold} onChange={event => setThreshold(Number(event.target.value))} /><div className="range-caption"><span>More reuse</span><span>Closer match</span></div></div><div className="setting-row"><label htmlFor="ttl">TTL <span className="subtle">hours</span></label><input id="ttl" readOnly={isDemo} type="number" step="1" value={ttl} onChange={event => setTtl(event.target.value)} required className="mono ttl-input" /></div><span className="ttl-note">0 or negative = no expiration</span><label className="isolation"><input type="checkbox" disabled={isDemo} checked={isolated} onChange={event => setIsolated(event.target.checked)} /><span>Isolate by provider + model<small>{isolated ? "Reuse stays within the selected model." : "Shared reuse across every provider and model."}</small></span></label></div>
        </fieldset></form>{isDemo && <p id="demo-notice" className="subtle">{capabilities.demo_notice}</p>}
        {online && engine === "warming" && <p className="pending-note" role="status"><span className="spinner" /> Warming semantic engine… Actions will enable automatically when ready.</p>}
        {online && engine === "error" && <div className="error-message" role="alert">The semantic engine could not initialize. Restart the backend service to retry. <button className="secondary" onClick={() => setEngine("warming")}>Check readiness</button></div>}
        {queryError && <div className="error-message" role="alert"><strong>Request not completed.</strong> {queryError}</div>}
        {pending && <p className="pending-note" role="status"><span className="spinner" /> Waiting for the backend: cache lookup or provider generation. No intermediate stage is reported.</p>}
      </section>
      <div className="sr-only" role="status" aria-live="polite">{announcement}</div>
      {result && !pending ? <Decision result={result} /> : !pending && !queryError && <div className="ready-strip"><span className="ready-cross">+</span><p>{online && engine === "ready" ? "Ready to trace your first request." : "Your next request starts here."}<small>{isDemo ? "Select a curated request to trace real cache behavior." : "Repeat the same question for exact reuse. Change the wording to test semantic matching."}</small></p><span className="mono">REQUEST / {online && engine === "ready" ? "READY" : "WAITING"}</span></div>}
      {metrics && <section className={`value-section ${cacheError ? "stale" : ""}`} aria-label="Measured cache value"><div className="value-heading"><p className="eyebrow">REUSE ADDS UP</p><span>{cacheError ? "Last loaded snapshot" : "All persisted query events"}</span></div><div className="value-metrics"><div><strong>{metrics.cache_hits}<small>/{metrics.total_queries}</small></strong><span>Requests reused</span></div><div><strong>{(metrics.hit_rate * 100).toFixed(1)}<small>%</small></strong><span>Cache hit rate</span></div>{snapshot.latency_reduction_percentage !== null && <div><strong>{snapshot.latency_reduction_percentage.toFixed(1)}<small>%</small></strong><span>Average latency reduction</span></div>}<div><strong><small>$</small>{metrics.avoided_cost_usd.toFixed(4)}</strong><span>Estimated cost avoided</span></div></div><p className="value-note">Illustrative cost estimates, not billing. Legacy totals may include zero fallbacks for unknown costs.</p></section>}
      <Ledger canClear={capabilities?.controls.clear_cache ?? false} snapshot={snapshot} loading={loadingCache} clearing={clearing} error={cacheError} disabled={!online || pending || engine !== "ready"} refresh={() => void refresh()} clear={clear} />
      {capabilities && <Quality fixedThreshold={isDemo} key={capabilities.app_mode} defaultThreshold={capabilities.defaults.threshold} disabled={!online || pending || clearing || engine !== "ready"} />}
      {!capabilities && <section id="quality" className="quality section-anchor"><p className="eyebrow">03 / TEST THE ASSUMPTION</p><h2>Cache quality</h2><p className="subtle">Connect to the API to evaluate semantic matching.</p></section>}
    </main><footer className="site-footer shell"><a href="#" className="wordmark">CACHE FLOW <span>↗</span></a><p>A laboratory for useful repetition.</p><span>LOCAL EMBEDDINGS · PERSISTENT MEMORY</span></footer>
  </>;
}
