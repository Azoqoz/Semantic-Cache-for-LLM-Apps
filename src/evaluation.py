from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from src.embeddings import EmbeddingService


Difficulty = Literal["Easy", "Medium", "Hard"]
MetricValue = float | int
ScoredPair = tuple["EvaluationPair", float]

COMPARISON_THRESHOLDS: tuple[float, ...] = (
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.84,
    0.90,
)


@dataclass(frozen=True)
class EvaluationPair:
    """A labeled semantic-matching pair with a difficulty category."""

    original_question: str
    similar_question: str
    expected_match: bool
    difficulty: Difficulty


@dataclass(frozen=True)
class EvaluationReport:
    """All results needed to render the Evaluation tab."""

    selected_rows: list[dict[str, object]]
    selected_metrics: dict[str, MetricValue]
    difficulty_rows: list[dict[str, object]]
    comparison_rows: list[dict[str, MetricValue]]
    recommended_threshold: float


EVALUATION_DATASET: tuple[EvaluationPair, ...] = (
    # Positive pairs: same intent, with increasing paraphrase difficulty.
    EvaluationPair("What is semantic caching?", "Explain semantic caching.", True, "Easy"),
    EvaluationPair("What is a cache hit?", "Explain what a cache hit is.", True, "Easy"),
    EvaluationPair("What is a cache miss?", "Explain what a cache miss is.", True, "Easy"),
    EvaluationPair("What is an embedding?", "Can you explain an embedding?", True, "Easy"),
    EvaluationPair("What is cosine similarity?", "Explain cosine similarity.", True, "Easy"),
    EvaluationPair("Why is threshold tuning important?", "Why is tuning the similarity threshold important?", True, "Easy"),
    EvaluationPair("How does semantic caching work?", "Describe how a semantic cache works.", True, "Medium"),
    EvaluationPair("How does semantic caching reduce LLM costs?", "How does a semantic cache lower LLM API costs?", True, "Medium"),
    EvaluationPair("What is a cache hit?", "What does a cache hit mean when reusing a stored response?", True, "Medium"),
    EvaluationPair("What is cosine similarity?", "How does cosine similarity measure similarity between vectors?", True, "Medium"),
    EvaluationPair("Why do cache entries need a TTL?", "Why do semantic cache entries need a time-to-live?", True, "Medium"),
    EvaluationPair("Why isolate a cache by model?", "Why isolate semantic cache entries for each LLM model?", True, "Medium"),
    EvaluationPair("What happens on a semantic cache miss?", "On a semantic cache miss, what does the system do next?", True, "Hard"),
    EvaluationPair("How should I choose a similarity threshold?", "How should I tune the semantic similarity threshold?", True, "Hard"),
    EvaluationPair("Why normalize embedding vectors?", "Why should text embedding vectors be normalized?", True, "Hard"),
    EvaluationPair("How can caching lower LLM latency?", "How does semantic caching reduce LLM response latency?", True, "Hard"),
    EvaluationPair("What should be stored in a semantic cache?", "What information should a semantic cache store for reuse?", True, "Hard"),
    EvaluationPair("How are semantically similar questions detected?", "How does the cache detect questions with similar meaning?", True, "Hard"),
    # Negative pairs: clearly different intents within semantic-cache/LLM topics.
    EvaluationPair("What is semantic caching?", "What is a vector database?", False, "Easy"),
    EvaluationPair("What is a cache hit?", "How do I deploy Streamlit?", False, "Easy"),
    EvaluationPair("What is cosine similarity?", "What is an API key?", False, "Easy"),
    EvaluationPair("What is an embedding?", "What is SQLite?", False, "Easy"),
    EvaluationPair("What is a cache miss?", "How do I install Ollama?", False, "Easy"),
    EvaluationPair("Why is threshold tuning important?", "What is prompt injection?", False, "Easy"),
    EvaluationPair("How does caching reduce LLM costs?", "How do I fine-tune a language model?", False, "Medium"),
    EvaluationPair("Why do cache entries need a TTL?", "How do I calculate token usage?", False, "Medium"),
    EvaluationPair("Why isolate a cache by model?", "How do I create a Python virtual environment?", False, "Medium"),
    EvaluationPair("How does semantic caching work?", "How does retrieval-augmented generation work?", False, "Medium"),
    EvaluationPair("How are embeddings generated?", "How should I rotate an OpenAI API key?", False, "Medium"),
    EvaluationPair("How is cache-hit latency measured?", "How can I quantize a local language model?", False, "Medium"),
    EvaluationPair("How should I choose a similarity threshold?", "Which temperature produces more creative LLM responses?", False, "Hard"),
    EvaluationPair("How can caching lower LLM latency?", "How can model batching increase inference throughput?", False, "Hard"),
    EvaluationPair("What should be stored in a semantic cache?", "What documents should be added to a RAG knowledge base?", False, "Hard"),
    EvaluationPair("How are semantically similar questions detected?", "How can hallucinations in generated answers be detected?", False, "Hard"),
    EvaluationPair("Why normalize embedding vectors?", "Why should input prompts be sanitized?", False, "Hard"),
    EvaluationPair("What happens on a semantic cache miss?", "What happens when an LLM exceeds its context window?", False, "Hard"),
)


def score_pairs(
    embedding_service: EmbeddingService,
    pairs: Sequence[EvaluationPair] = EVALUATION_DATASET,
) -> list[ScoredPair]:
    """Embed every pair once and return its cosine similarity."""
    scored_pairs: list[ScoredPair] = []
    for pair in pairs:
        original_embedding = embedding_service.encode(pair.original_question)
        comparison_embedding = embedding_service.encode(pair.similar_question)
        similarity = embedding_service.cosine_similarity(
            original_embedding,
            comparison_embedding,
        )
        scored_pairs.append((pair, similarity))
    return scored_pairs


def evaluate_scored_pairs(
    scored_pairs: Sequence[ScoredPair],
    threshold: float,
) -> tuple[list[dict[str, object]], dict[str, MetricValue]]:
    """Evaluate pre-scored pairs at one threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    rows: list[dict[str, object]] = []
    tp = tn = fp = fn = 0
    for pair, similarity in scored_pairs:
        predicted = similarity >= threshold
        if predicted and pair.expected_match:
            tp += 1
        elif not predicted and not pair.expected_match:
            tn += 1
        elif predicted:
            fp += 1
        else:
            fn += 1

        rows.append(
            {
                "Original question": pair.original_question,
                "Similar question": pair.similar_question,
                "Difficulty": pair.difficulty,
                "Similarity": round(similarity, 4),
                "Predicted": "match" if predicted else "no match",
                "Expected": "match" if pair.expected_match else "no match",
                "Correct": predicted == pair.expected_match,
            }
        )

    total = len(scored_pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return rows, {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def calculate_difficulty_results(
    evaluation_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Calculate accuracy for each difficulty, including empty groups safely."""
    results: list[dict[str, object]] = []
    for difficulty in ("Easy", "Medium", "Hard"):
        group = [row for row in evaluation_rows if row["Difficulty"] == difficulty]
        correct = sum(bool(row["Correct"]) for row in group)
        results.append(
            {
                "Difficulty": difficulty,
                "Accuracy": correct / len(group) if group else 0.0,
                "Correct": correct,
                "Total": len(group),
            }
        )
    return results


def compare_thresholds(
    scored_pairs: Sequence[ScoredPair],
    thresholds: Sequence[float] = COMPARISON_THRESHOLDS,
) -> list[dict[str, MetricValue]]:
    """Calculate classification metrics across multiple thresholds."""
    comparison: list[dict[str, MetricValue]] = []
    for threshold in thresholds:
        _, metrics = evaluate_scored_pairs(scored_pairs, threshold)
        comparison.append(
            {
                "Threshold": threshold,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 score": metrics["f1"],
                "True positives": metrics["true_positives"],
                "True negatives": metrics["true_negatives"],
                "False positives": metrics["false_positives"],
                "False negatives": metrics["false_negatives"],
            }
        )
    return comparison


def recommend_threshold(comparison_rows: Sequence[dict[str, MetricValue]]) -> float:
    """Choose highest F1, then highest precision, then highest threshold."""
    if not comparison_rows:
        raise ValueError("At least one threshold result is required.")
    best = max(
        comparison_rows,
        key=lambda row: (
            float(row["F1 score"]),
            float(row["Precision"]),
            float(row["Threshold"]),
        ),
    )
    return float(best["Threshold"])


def build_evaluation_report(
    embedding_service: EmbeddingService,
    selected_threshold: float,
    pairs: Sequence[EvaluationPair] = EVALUATION_DATASET,
    comparison_thresholds: Sequence[float] = COMPARISON_THRESHOLDS,
) -> EvaluationReport:
    """Build selected, difficulty, comparison, and recommendation results."""
    scored_pairs = score_pairs(embedding_service, pairs)
    selected_rows, selected_metrics = evaluate_scored_pairs(
        scored_pairs,
        selected_threshold,
    )
    difficulty_rows = calculate_difficulty_results(selected_rows)
    comparison_rows = compare_thresholds(scored_pairs, comparison_thresholds)
    return EvaluationReport(
        selected_rows=selected_rows,
        selected_metrics=selected_metrics,
        difficulty_rows=difficulty_rows,
        comparison_rows=comparison_rows,
        recommended_threshold=recommend_threshold(comparison_rows),
    )


def evaluate_pairs(
    embedding_service: EmbeddingService,
    threshold: float,
    pairs: Sequence[EvaluationPair] = EVALUATION_DATASET,
) -> tuple[list[dict[str, object]], dict[str, MetricValue]]:
    """Evaluate labeled pairs at one threshold for backward compatibility."""
    return evaluate_scored_pairs(score_pairs(embedding_service, pairs), threshold)
