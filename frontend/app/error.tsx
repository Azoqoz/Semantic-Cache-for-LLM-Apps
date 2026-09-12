"use client";
export default function ErrorPage({ reset }: { reset: () => void }) {
  return <main className="shell fatal"><p className="eyebrow">CACHE FLOW / CONNECTION INTERRUPTED</p><h1>Let’s try that again.</h1><p>The page could not finish rendering. Your cache lives on the server.</p><button onClick={reset}>Reload the laboratory ↗</button></main>;
}
