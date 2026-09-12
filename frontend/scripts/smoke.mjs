// Run only with scripts/qa_backend.py; this clears its disposable SQLite cache.
import assert from "node:assert/strict";

const base = process.env.SMOKE_BASE_URL || "http://127.0.0.1:3000/api/backend";
let cookie = "";
async function request(path, method = "GET", body, expectedStatus = 200) {
  const response = await fetch(base + path, {
    method, headers: { "Content-Type": "application/json", ...(cookie ? { Cookie: cookie } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(185000),
  });
  if (response.headers.get("set-cookie")) cookie = response.headers.get("set-cookie").split(";")[0];
  assert.equal(response.status, expectedStatus, `${method} ${path}: expected successful response`);
  return response.json();
}
assert.equal((await request("/health")).status, "ok");
const capabilities = await request("/capabilities");
const warmupDeadline = Date.now() + 180000;
while (true) {
  const readiness = await request("/ready");
  if (readiness.status === "ready") break;
  assert.equal(readiness.status, "warming", "The semantic engine failed to initialize");
  assert.ok(Date.now() < warmupDeadline, "Warm-up timed out");
  await new Promise(resolve => setTimeout(resolve, 1000));
}
if (capabilities.app_mode === "demo") {
  assert.deepEqual(capabilities.providers, [{ name: "Demo", default_model: "demo-rule-based" }]);
  assert.equal(capabilities.controls.free_form, false);
  assert.equal(capabilities.controls.clear_cache, false);
  const samples = capabilities.demo_samples;
  assert.equal((await request("/cache")).entries.length, 2);
  const outcomes = [];
  for (const sample of samples) outcomes.push(await request("/query", "POST", { question: sample.question }));
  assert.deepEqual(outcomes.map(result => result.hit_type), ["exact", "semantic", "miss", "semantic"]);
  assert.equal(outcomes[0].similarity, null);
  assert.ok(outcomes[1].similarity > .94);
  assert.ok(outcomes[2].similarity < .84);
  assert.ok(outcomes[3].similarity > .91);
  assert.equal((await request("/query", "POST", { question: samples[2].question })).hit_type, "exact");
  assert.equal((await request("/cache", "DELETE", undefined, 403)).error.code, "demo_read_only");
  await request("/query", "POST", { question: "arbitrary prompt" }, 422);
  await request("/query", "POST", { question: samples[0].question, threshold: .1 }, 403);
  await request("/query", "POST", { question: samples[0].question, provider: "OpenAI" }, 422);
  const quality = await request("/evaluation", "POST", { threshold: .84 });
  assert.equal(quality.selected_rows.length, 36);
  cookie = "";
  assert.equal((await request("/cache")).entries.length, 2);
  assert.equal((await request("/query", "POST", { question: samples[2].question })).hit_type, "miss");
  console.log("PASS: Public Demo proxy, cookies, four real scenarios, visitor isolation, fixed settings, prompt/provider/clear denial and evaluation.");
} else {
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
assert.equal(capabilities.providers.length, 5);
const quality = await request("/evaluation", "POST", { threshold: .84 });
assert.equal(quality.selected_rows.length, 36);
assert.equal(quality.difficulty_rows.length, 3);
assert.equal(quality.comparison_rows.length, 7);
assert.equal((await request("/cache", "DELETE")).cleared, true);
assert.equal((await request("/cache")).metrics.total_queries, 0);
console.log("PASS: real FastAPI proxy, Local Mode arbitrary prompts, miss/exact/semantic, all providers, TTL=0, ledger/reuses, evaluation and clear.");

}
