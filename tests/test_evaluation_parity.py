"""Production Evaluation tab calculations, independently of model quality."""

import unittest
from collections import Counter
from unittest.mock import Mock

from src.embeddings import EmbeddingService
from src.evaluation import (
    COMPARISON_THRESHOLDS, EVALUATION_DATASET, EvaluationPair,
    build_evaluation_report, calculate_difficulty_results, compare_thresholds,
    evaluate_pairs, evaluate_scored_pairs, recommend_threshold,
)


class EvaluationParity(unittest.TestCase):
    def setUp(self):
        self.pairs = [EvaluationPair("a", "b", True, "Easy"),
                      EvaluationPair("c", "d", False, "Easy"),
                      EvaluationPair("e", "f", False, "Medium"),
                      EvaluationPair("g", "h", True, "Hard")]
        self.scored = list(zip(self.pairs, [.8, .2, .9, .79999]))

    def test_confusion_metrics_inclusive_boundary_and_rounding_after_decision(self):
        rows, metrics = evaluate_scored_pairs(self.scored, .8)
        self.assertEqual(metrics, {"accuracy": .5, "precision": .5, "recall": .5, "f1": .5,
                                   "true_positives": 1, "true_negatives": 1,
                                   "false_positives": 1, "false_negatives": 1})
        self.assertEqual(rows[0]["Predicted"], "match")
        self.assertEqual(rows[3]["Similarity"], .8)
        self.assertEqual(rows[3]["Predicted"], "no match")
        self.assertEqual(calculate_difficulty_results(rows), [
            {"Difficulty": "Easy", "Accuracy": 1, "Correct": 2, "Total": 2},
            {"Difficulty": "Medium", "Accuracy": 0, "Correct": 0, "Total": 1},
            {"Difficulty": "Hard", "Accuracy": 0, "Correct": 0, "Total": 1}])

    def test_zero_denominators_empty_groups_and_invalid_thresholds(self):
        rows, metrics = evaluate_scored_pairs([], .8)
        self.assertEqual(rows, [])
        self.assertTrue(all(value == 0 for value in metrics.values()))
        self.assertTrue(all(row["Total"] == row["Accuracy"] == 0 for row in calculate_difficulty_results(rows)))
        _, negative = evaluate_scored_pairs([(self.pairs[1], .2)], .8)
        self.assertEqual((negative["accuracy"], negative["precision"], negative["recall"], negative["f1"]), (1, 0, 0, 0))
        for threshold in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                evaluate_scored_pairs([], threshold)
        with self.assertRaises(ValueError):
            recommend_threshold([])

    def test_recommendation_prioritizes_f1_then_precision_then_higher_threshold(self):
        rows = [{"Threshold": .6, "F1 score": .9, "Precision": .8},
                {"Threshold": .7, "F1 score": .8, "Precision": 1}]
        self.assertEqual(recommend_threshold(rows), .6)
        rows.append({"Threshold": .65, "F1 score": .9, "Precision": .9})
        self.assertEqual(recommend_threshold(rows), .65)
        rows.append({"Threshold": .8, "F1 score": .9, "Precision": .9})
        self.assertEqual(recommend_threshold(rows), .8)

    def test_report_scores_once_for_all_thresholds_and_legacy_api_agrees(self):
        embedding = Mock(spec=EmbeddingService)
        embedding.encode.return_value = [1.0, 0.0]
        embedding.cosine_similarity.side_effect = [score for _, score in self.scored]
        report = build_evaluation_report(embedding, .8, self.pairs)
        self.assertEqual(embedding.encode.call_count, 8)
        self.assertEqual([c.args[0] for c in embedding.encode.call_args_list], list("abcdefgh"))
        self.assertEqual(embedding.cosine_similarity.call_count, 4)
        self.assertEqual((report.selected_rows, report.selected_metrics), evaluate_scored_pairs(self.scored, .8))
        self.assertEqual(report.comparison_rows, compare_thresholds(self.scored))
        self.assertEqual(report.recommended_threshold, .75)
        embedding.cosine_similarity.side_effect = [score for _, score in self.scored]
        self.assertEqual(evaluate_pairs(embedding, .8, self.pairs),
                         (report.selected_rows, report.selected_metrics))

    def test_current_dataset_balance_and_threshold_grid(self):
        self.assertEqual(COMPARISON_THRESHOLDS, (.60, .65, .70, .75, .80, .84, .90))
        self.assertEqual(len(EVALUATION_DATASET), 36)
        counts = Counter((pair.difficulty, pair.expected_match) for pair in EVALUATION_DATASET)
        self.assertEqual(counts, {(difficulty, label): 6
                                  for difficulty in ("Easy", "Medium", "Hard") for label in (True, False)})
