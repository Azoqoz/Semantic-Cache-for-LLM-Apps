export type Provider = "Demo" | "OpenAI" | "Claude" | "Gemini" | "Ollama";
export type HitType = "exact" | "semantic" | "miss";
export interface Capabilities {
  app_mode: "demo" | "local";
  providers: { name: Provider; default_model: string }[];
  defaults: { provider: Provider; threshold: number; ttl_hours: number; isolate_by_model: boolean };
  evaluation_thresholds: number[];
  credentials: "server_environment_only";
}
export interface QueryRequest {
  question: string;
  provider: Provider;
  model?: string;
  threshold: number;
  ttl_hours: number;
  isolate_by_model: boolean;
}
export interface QueryResult extends QueryRequest {
  answer: string;
  cache_hit: boolean;
  hit_type: HitType;
  similarity: number | null;
  matched_question: string | null;
  model: string;
  latency_ms: number;
  estimated_cost_usd: number | null;
  cost_basis: "provider_estimate" | "default_estimate" | "local_zero" | null;
  cost_kind: "avoided" | "incurred";
}
export interface CacheEntry {
  id: number; question: string; answer: string; provider: string; model: string;
  created_at: string; expires_at: string | null; access_count: number; last_accessed_at: string | null;
}
export interface CacheMetrics {
  total_queries: number; cache_hits: number; cache_misses: number; hit_rate: number;
  average_latency_ms: number; average_hit_latency_ms: number; average_miss_latency_ms: number;
  llm_cost_usd: number; avoided_cost_usd: number; total_cost_without_cache_usd: number;
  savings_percentage: number; cache_entries: number; total_reuses: number;
}
export interface CacheSnapshot {
  entries: CacheEntry[];
  metrics: CacheMetrics;
  latency_reduction_percentage: number | null;
  cost_metrics_basis: "legacy_estimates_not_billing";
}
export interface ClassificationMetrics {
  accuracy: number; precision: number; recall: number; f1: number;
  true_positives: number; true_negatives: number; false_positives: number; false_negatives: number;
}
export interface EvaluationPair {
  "Original question": string; "Similar question": string; Difficulty: "Easy" | "Medium" | "Hard";
  Similarity: number; Predicted: "match" | "no match"; Expected: "match" | "no match"; Correct: boolean;
}
export interface ThresholdResult {
  Threshold: number; Accuracy: number; Precision: number; Recall: number; "F1 score": number;
  "True positives": number; "True negatives": number; "False positives": number; "False negatives": number;
}
export interface EvaluationReport {
  selected_rows: EvaluationPair[];
  selected_metrics: ClassificationMetrics;
  difficulty_rows: { Difficulty: string; Accuracy: number; Correct: number; Total: number }[];
  comparison_rows: ThresholdResult[];
  recommended_threshold: number;
}
