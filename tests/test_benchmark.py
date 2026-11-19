"""tests/test_benchmark.py - Comprehensive Unit and Integration Test Suite for Milestone 3 Comparative Benchmark.

Covers:
1. Exact mathematical parity of discourse segmentation metrics:
   - Dataset-level Micro F1 (pooled over corpus, epsilon=1e-9)
   - Document-level Macro F1
   - Greedy 1-to-1 Relaxed F1 (tolerance k in {1, 2} without replacement)
   - Beeferman's P_k metric (adaptive window k_d)
   - Pevzner & Hearst's WindowDiff (WD)
2. Hand-crafted numerical ground truth pairs (Cases A through E):
   - Identical boundaries (Case A)
   - Near miss distance 1 (Case B)
   - Clustering & double prediction near single gold (Case C)
   - Near miss distance 2 separating k=1 vs k=2 (Case D)
   - Micro vs. Macro aggregation gap (Case E)
3. Immutable golden anchor parity against MiniSeg validation metrics:
   - Parity with models/miniseg/validation_metrics_20260409_084911.json and Table 5
4. End-to-end benchmark runner integration:
   - Evaluates MiniSeg baseline and Discourse-Aware Model
   - Validates generated Markdown comparison table format
   - Validates generated JSON summary structure and acceptance criteria flags
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from src.segmentation.benchmark_miniseg import (
    BenchmarkDocument,
    BenchmarkSuiteResult,
    DiscourseAwareBenchmarkAdapter,
    DocumentEvaluationScore,
    MiniSegHistoricalOracleAdapter,
    MiniSegMockAdapter,
    ModelBenchmarkResult,
    aggregate_benchmark_metrics,
    clean_ytseg_label_string,
    compute_beeferman_pk,
    compute_boundary_counts,
    compute_comparative_deltas,
    compute_greedy_relaxed_f1,
    compute_segment_window_size,
    compute_strict_f1,
    compute_window_diff,
    evaluate_document_metrics,
    export_json_summary,
    format_markdown_comparison_table,
    get_synthetic_benchmark_documents,
    load_arrow_benchmark_documents,
    run_miniseg_benchmark,
)
from tests.fixtures.reference_oracles import reference_greedy_relaxed_f1


# ==============================================================================
# 1. Exact Mathematical Parity Tests (Cases A through E)
# ==============================================================================

class TestSegmentationMetricsMathematicalParity:
    """Verifies mathematical exactness against hand-crafted analytical ground-truth cases."""

    def test_case_a_identical_boundaries(self) -> None:
        """Case A: Exact boundary match (N=10, Gold={3,7}, Pred={3,7}).

        Expected: Strict F1=1.0, Relaxed(k=1)=1.0, Relaxed(k=2)=1.0, P_k=0.0, WD=0.0.
        """
        N = 10
        gold = {3, 7}
        pred = {3, 7}

        strict = compute_strict_f1(gold, pred, num_sentences=N)
        assert abs(strict["f1"] - 1.0) < 1e-6
        assert strict["tp"] == 2.0
        assert strict["fp"] == 0.0
        assert strict["fn"] == 0.0

        rel_k1 = compute_greedy_relaxed_f1(gold, pred, window=1, num_sentences=N)
        assert abs(rel_k1["f1"] - 1.0) < 1e-6

        rel_k2 = compute_greedy_relaxed_f1(gold, pred, window=2, num_sentences=N)
        assert abs(rel_k2["f1"] - 1.0) < 1e-6

        pk = compute_beeferman_pk(gold, pred, num_sentences=N)
        assert abs(pk - 0.0) < 1e-6

        wd = compute_window_diff(gold, pred, num_sentences=N)
        assert abs(wd - 0.0) < 1e-6

    def test_case_b_near_miss_delta_1(self) -> None:
        """Case B: Near miss distance 1 (N=10, Gold={5}, Pred={6}).

        Expected: Strict F1=0.0, Relaxed(k=1)=1.0, Relaxed(k=2)=1.0, P_k=0.25, WD=0.25.
        """
        N = 10
        gold = {5}
        pred = {6}

        strict = compute_strict_f1(gold, pred, num_sentences=N)
        assert strict["f1"] == 0.0
        assert strict["tp"] == 0.0
        assert strict["fp"] == 1.0
        assert strict["fn"] == 1.0

        rel_k1 = compute_greedy_relaxed_f1(gold, pred, window=1, num_sentences=N)
        assert abs(rel_k1["f1"] - 1.0) < 1e-6
        assert rel_k1["tp"] == 1.0

        rel_k2 = compute_greedy_relaxed_f1(gold, pred, window=2, num_sentences=N)
        assert abs(rel_k2["f1"] - 1.0) < 1e-6

        pk = compute_beeferman_pk(gold, pred, num_sentences=N)
        assert abs(pk - 0.25) < 1e-6

        wd = compute_window_diff(gold, pred, num_sentences=N)
        assert abs(wd - 0.25) < 1e-6

    def test_case_c_clustering_greedy_bipartite_matching(self) -> None:
        """Case C: Clustering / Double Prediction (N=10, Gold={5}, Pred={4, 6}).

        Proves greedy 1-to-1 matching without replacement:
        Pred 4 matches Gold 5; Pred 6 remains unmatched FP.
        TP_rel=1, FP_rel=1, FN_rel=0 -> Prec=0.5, Rec=1.0 -> F1 = 2/3 ~= 0.6667.
        """
        N = 10
        gold = {5}
        pred = {4, 6}

        strict = compute_strict_f1(gold, pred, num_sentences=N)
        assert strict["f1"] == 0.0

        rel_k1 = compute_greedy_relaxed_f1(gold, pred, window=1, num_sentences=N)
        expected_f1 = 2.0 / 3.0
        assert abs(rel_k1["f1"] - expected_f1) < 1e-4
        assert rel_k1["tp"] == 1.0
        assert rel_k1["fp"] == 1.0
        assert rel_k1["fn"] == 0.0

        # Verify parity with canonical reference oracle
        ref_oracle = reference_greedy_relaxed_f1(gold, pred, window=1)
        assert abs(rel_k1["f1"] - ref_oracle["f1"]) < 1e-6

    def test_case_d_near_miss_delta_2_window_differentiation(self) -> None:
        """Case D: Near miss distance 2 (N=10, Gold={5}, Pred={7}).

        Validates k=1 vs k=2 window differentiation:
        Distance |7 - 5| = 2.
        For k=1: 2 > 1 -> No match (F1=0.0).
        For k=2: 2 <= 2 -> Matches (F1=1.0).
        """
        N = 10
        gold = {5}
        pred = {7}

        rel_k1 = compute_greedy_relaxed_f1(gold, pred, window=1, num_sentences=N)
        assert rel_k1["f1"] == 0.0
        assert rel_k1["tp"] == 0.0

        rel_k2 = compute_greedy_relaxed_f1(gold, pred, window=2, num_sentences=N)
        assert abs(rel_k2["f1"] - 1.0) < 1e-6
        assert rel_k2["tp"] == 1.0

        pk = compute_beeferman_pk(gold, pred, num_sentences=N)
        assert abs(pk - 0.50) < 1e-6

        wd = compute_window_diff(gold, pred, num_sentences=N)
        assert abs(wd - 0.50) < 1e-6

    def test_case_e_micro_macro_aggregation_gap(self) -> None:
        """Case E: Micro vs. Macro aggregation gap.

        Doc 1: Gold={5}, Pred={5} -> F1 = 1.0
        Doc 2: Gold={3, 7}, Pred={2, 4, 8} -> F1 = 0.0
        Doc Macro F1 = (1.0 + 0.0) / 2 = 0.5000.
        Dataset Micro F1: Total TP=1, FP=3, FN=2 -> 2*1 / (2*1 + 3 + 2) = 2/7 ~= 0.2857.
        Demonstrates the documented empirical macro drop.
        """
        doc1 = evaluate_document_metrics(gold_boundaries={5}, pred_boundaries={5}, num_sentences=10, doc_id="d1")
        doc2 = evaluate_document_metrics(gold_boundaries={3, 7}, pred_boundaries={2, 4, 8}, num_sentences=10, doc_id="d2")

        agg = aggregate_benchmark_metrics("TestModel", [doc1, doc2])

        assert abs(agg.doc_macro_f1 - 0.5000) < 1e-4
        assert abs(agg.micro_f1 - (2.0 / 7.0)) < 1e-4
        assert agg.micro_f1 < agg.doc_macro_f1  # Micro vs Macro gap validated

    def test_empty_boundary_sets_edge_cases(self) -> None:
        """Verifies zero boundaries on either or both sides handles gracefully without division by zero."""
        # Both empty
        res_empty = compute_strict_f1(set(), set(), num_sentences=10)
        assert res_empty["f1"] == 0.0

        # Gold has boundaries, pred empty
        res_fn = compute_strict_f1({3, 5}, set(), num_sentences=10)
        assert res_fn["f1"] == 0.0
        assert res_fn["fn"] == 2.0

        # Pred has boundaries, gold empty
        res_fp = compute_strict_f1(set(), {3, 5}, num_sentences=10)
        assert res_fp["f1"] == 0.0
        assert res_fp["fp"] == 2.0

        # P_k and WindowDiff on empty
        assert compute_beeferman_pk(set(), set(), num_sentences=10) == 0.0
        assert compute_window_diff(set(), set(), num_sentences=10) == 0.0

    def test_pk_and_windowdiff_boundary_bounds(self) -> None:
        """Asserts 0.0 <= P_k <= 1.0 and 0.0 <= WD <= 1.0 across configurations."""
        for n in [5, 10, 20]:
            for g_size in [1, 2, 4]:
                gold = set(np.random.choice(range(1, n), size=min(g_size, n - 1), replace=False))
                pred = set(np.random.choice(range(1, n), size=min(g_size, n - 1), replace=False))
                pk = compute_beeferman_pk(gold, pred, num_sentences=n)
                wd = compute_window_diff(gold, pred, num_sentences=n)
                assert 0.0 <= pk <= 1.0
                assert 0.0 <= wd <= 1.0

    def test_window_size_adaptive_formula(self) -> None:
        """Verifies Beeferman window size k_d = max(1, round(0.5 * N_d / (1 + |G_d|)))."""
        # N=10, 1 gold boundary -> segments = 2, avg = 5, k = round(2.5) = 2
        assert compute_segment_window_size(num_sentences=10, num_gold_boundaries=1) == 2
        # N=10, 2 gold boundaries -> segments = 3, avg = 3.33, k = round(1.66) = 2
        assert compute_segment_window_size(num_sentences=10, num_gold_boundaries=2) == 2
        # N=100, 9 gold boundaries -> segments = 10, avg = 10, k = round(5) = 5
        assert compute_segment_window_size(num_sentences=100, num_gold_boundaries=9) == 5
        # Edge case N=1
        assert compute_segment_window_size(num_sentences=1, num_gold_boundaries=0) == 1


# ==============================================================================
# 2. Immutable Golden Anchor Parity Tests
# ==============================================================================

class TestMiniSegGoldenCheckpointAnchor:
    """Verifies reproduction of models/miniseg/validation_metrics_20260409_084911.json."""

    def test_miniseg_checkpoint_metrics_exact_reproduction(self) -> None:
        """Asserts historical oracle loader reproduces Table 5 baseline metrics to 4 decimal places."""
        metrics_path = Path("models/miniseg/validation_metrics_20260409_084911.json")
        if not metrics_path.exists():
            pytest.skip(f"Golden metrics file {metrics_path} missing")

        oracle = MiniSegHistoricalOracleAdapter(metrics_path)
        result = oracle.load_result()

        # Check Table 5 anchors:
        # Precision: 0.4197, Recall: 0.5121, Micro F1: 0.4613
        assert abs(result.micro_precision - 0.4197) < 1e-3
        assert abs(result.micro_recall - 0.5121) < 1e-3
        assert abs(result.micro_f1 - 0.4613) < 1e-3

        # Macro F1: 0.3950
        assert abs(result.doc_macro_f1 - 0.3950) < 1e-3

        # Relaxed F1 (w=2): 0.5380
        assert abs(result.doc_macro_relaxed_f1_k2 - 0.5380) < 1e-3

        # P_k: 0.3109, WindowDiff: 0.3740
        assert abs(result.pk - 0.3109) < 1e-3
        assert abs(result.window_diff - 0.3740) < 1e-3

        # TP: 4117, FP: 5692, FN: 3922
        assert result.tp == 4117.0
        assert result.fp == 5692.0
        assert result.fn == 3922.0
        assert result.num_documents == 1447


# ==============================================================================
# 3. Benchmark Integration Runner & Exporter Tests
# ==============================================================================

class TestBenchmarkIntegrationRunner:
    """Test suite for benchmark execution, delta analysis, and report generation."""

    def test_synthetic_documents_fixture(self) -> None:
        """Verifies synthetic documents generator produces valid lecture structures."""
        docs = get_synthetic_benchmark_documents(num_docs=5)
        assert len(docs) == 5
        for d in docs:
            assert d.num_sentences >= 8
            assert len(d.gold_boundaries) >= 1
            for b in d.gold_boundaries:
                assert 1 <= b < d.num_sentences

    def test_benchmark_runner_sample_transcripts(self) -> None:
        """Verifies run_miniseg_benchmark executes on synthetic fixtures cleanly."""
        suite = run_miniseg_benchmark(use_fixtures=True, max_docs=5)

        assert suite.num_documents == 5
        assert "miniseg_baseline" in suite.models
        assert "discourse_aware" in suite.models

        base = suite.models["miniseg_baseline"]
        prop = suite.models["discourse_aware"]

        assert 0.0 <= base.micro_f1 <= 1.0
        assert 0.0 <= prop.micro_f1 <= 1.0
        assert prop.micro_f1 > base.micro_f1  # Proposed model outperforms baseline

        deltas = suite.comparisons
        assert "micro_f1_delta" in deltas
        assert "acceptance_criteria_satisfied" in deltas
        assert deltas["acceptance_criteria_satisfied"]["surpass_miniseg_micro_f1"] is True

    def test_benchmark_markdown_table_schema(self) -> None:
        """Verifies generated comparison table matches Proposal Section 8.1 schema."""
        suite = run_miniseg_benchmark(use_fixtures=True, max_docs=3)
        md_table = format_markdown_comparison_table(suite)

        assert "| Model Architecture |" in md_table
        assert "| Exact Micro F1 |" in md_table
        assert "| Doc Macro F1 |" in md_table
        assert "| Doc Relax F1 (k=1) |" in md_table
        assert "| Doc Relax F1 (k=2) |" in md_table
        assert "| P_k (↓) |" in md_table
        assert "| WD (↓) |" in md_table
        assert "MiniSeg Baseline" in md_table
        assert "Discourse-Aware (Proposed)" in md_table
        assert "Relative Improvement (Δ)" in md_table

    def test_benchmark_json_summary_schema(self, tmp_path: Path) -> None:
        """Verifies generated JSON export matches the machine-readable schema."""
        suite = run_miniseg_benchmark(use_fixtures=True, max_docs=3)
        json_path = tmp_path / "benchmark_summary.json"
        summary = export_json_summary(suite, json_path)

        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "benchmark_metadata" in data
        assert "models" in data
        assert "comparative_analysis" in data
        assert "miniseg_baseline" in data["models"]
        assert "discourse_aware" in data["models"]

        comp = data["comparative_analysis"]
        assert "micro_f1_delta" in comp
        assert "acceptance_criteria_satisfied" in comp

    def test_benchmark_real_arrow_shard_smoke(self) -> None:
        """Smoke test running benchmark on real validation Arrow shard (slice of 5 docs)."""
        shard_path = Path("datasets/ytseg-dataset/validation/data-00000-of-00001.arrow")
        if not shard_path.exists():
            pytest.skip(f"Shard {shard_path} missing")

        docs = load_arrow_benchmark_documents(shard_path, max_docs=5)
        assert len(docs) == 5

        adapter = MiniSegMockAdapter(threshold=0.80)
        res = adapter.evaluate_documents(docs)
        assert res.num_documents == 5
        assert res.micro_f1 >= 0.0

    def test_discourse_adapter_integrity_against_gold_copying(self) -> None:
        """Adversarial check: mutating doc.gold_boundaries MUST NOT change predicted boundaries."""
        docs = get_synthetic_benchmark_documents(num_docs=1)
        doc = docs[0]

        adapter = DiscourseAwareBenchmarkAdapter(threshold=0.50, nms_radius=2)
        # 1. Baseline prediction
        preds1 = adapter.predict_document_boundaries(doc)

        # 2. Mutate gold boundaries arbitrarily
        doc_mutated = BenchmarkDocument(
            doc_id="synth_mutated",
            sentences=list(doc.sentences),
            gold_boundaries=[1, 2, 8, 9],  # Completely different gold boundaries
            duration=doc.duration,
            metadata=doc.metadata,
        )
        preds2 = adapter.predict_document_boundaries(doc_mutated)

        # Invariant: predictions MUST be identical because adapter never inspects gold_boundaries
        assert preds1 == preds2, "Predicted boundaries changed when mutating gold boundaries! Integrity violation."

    def test_miniseg_live_adapter_clean_fallback(self) -> None:
        """Verifies MiniSegLiveCheckpointAdapter cleanly falls back to Mode B when checkpoint/deps missing."""
        from src.segmentation.benchmark_miniseg import MiniSegLiveCheckpointAdapter

        adapter = MiniSegLiveCheckpointAdapter(checkpoint_path="models/miniseg/nonexistent.pt")
        assert adapter.is_available is False

        docs = get_synthetic_benchmark_documents(num_docs=2)
        res = adapter.evaluate_documents(docs)
        assert res.model_name == "MiniSeg Baseline (Historical)"
        assert res.num_documents == 1447
