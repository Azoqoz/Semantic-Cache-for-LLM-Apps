// Run only with scripts/qa_backend.py; this clears its disposable SQLite cache.
import assert from "node:assert/strict";

const base = "http://127.0.0.1:3000/api/backend";
async function request(path, method = "GET", body) {
  const response = await fetch(base + path, {
    method, headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(185000),
  });
  assert.equal(response.status, 200, `${method} ${path}: expected successful response`);
  return response.json();
}
assert.equal((await request("/health")).status, "ok");
assert.equal((await request("/capabilities")).app_mode, "demo");
assert.equal((await request("/cache", "DELETE")).cleared, true);
const query = { question: "What is semantic caching?", provider: "Demo", threshold: .5, ttl_hours: 0, isolate_by_model: true };
const miss = await request("/query", "POST", query);
assert.equal(miss.hit_type, "miss");
assert.equal(miss.similarity, null);
assert.ok(miss.answer.length > 0);
assert.equal((await request("/query", "POST", query)).hit_type, "exact");
const semantic = await request("/query", "POST", { ...query, question: "Can you explain semantic cache?" });
assert.equal(semantic.hit_type, "semantic");
assert.ok(semantic.similarity >= .5);
const snapshot = await request("/cache");
assert.equal(snapshot.entries.length, 1);
assert.equal(snapshot.entries[0].access_count, 2);
assert.equal(snapshot.metrics.cache_hits, 2);
assert.equal(snapshot.entries[0].expires_at, null);
const denied = await fetch(base + "/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...query, provider: "OpenAI" }) });
assert.equal(denied.status, 400);
assert.equal((await denied.json()).error.code, "provider_configuration");
const quality = await request("/evaluation", "POST", { threshold: .84 });
assert.equal(quality.selected_rows.length, 36);
assert.equal(quality.difficulty_rows.length, 3);
assert.equal(quality.comparison_rows.length, 7);
assert.equal((await request("/cache", "DELETE")).cleared, true);
assert.equal((await request("/cache")).metrics.total_queries, 0);
console.log("PASS: real FastAPI proxy, Demo miss/exact/semantic, isolation setting, TTL=0, ledger/reuses, provider denial, evaluation and clear.");
