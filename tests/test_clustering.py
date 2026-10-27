"""Unit and integration tests for TF-IDF and Sentence Transformers clustering."""

import pandas as pd
import pytest

from src.clustering import (
    ModelResult,
    _adjust_k,
    run_embedding_clustering,
    run_tfidf_clustering,
)
from src.data_processing import FeedbackRecord, load_feedback_csv
from src.evaluation import compare_models, evaluate_model_result


@pytest.fixture
def sample_records() -> list[FeedbackRecord]:
    """Fixture providing validated records from the sample dataset."""
    report = load_feedback_csv("data/sample_feedback.csv")
    assert report.valid_records_count > 0
    return report.records


def test_adjust_k_safety():
    """Verify k is safely clamped when k >= n_samples or k < 2."""
    with pytest.raises(ValueError, match="At least 2 feedback samples"):
        _adjust_k(n_samples=1, requested_k=2)

    # Clamped when k < 2
    k, warnings = _adjust_k(n_samples=10, requested_k=1)
    assert k == 2
    assert len(warnings) == 1

    # Clamped when k >= n_samples
    k, warnings = _adjust_k(n_samples=5, requested_k=5)
    assert k == 4
    assert len(warnings) == 1


def test_tfidf_clustering_execution(sample_records):
    """Verify TF-IDF clustering executes, returns all records, and populates metadata."""
    k = 4
    result = run_tfidf_clustering(sample_records, n_clusters=k, random_state=42)

    assert isinstance(result, ModelResult)
    assert result.method == "TF-IDF + K-Means"
    assert len(result.feedback_ids) == len(sample_records)
    assert len(result.cluster_labels) == len(sample_records)
    assert set(result.cluster_labels).issubset(set(range(k)))
    assert result.runtime_seconds > 0

    # Ensure representatives and sizes exist for each cluster
    for cluster_id in range(k):
        assert cluster_id in result.cluster_sizes
        assert cluster_id in result.cluster_representatives
        assert cluster_id in result.cluster_keywords

    # Representative should be one of the feedback IDs
    for rep_id in result.cluster_representatives.values():
        if rep_id:
            assert rep_id in result.feedback_ids


def test_tfidf_evaluation_and_limitations(sample_records):
    """Verify evaluation generates metrics without crashes and documents limitations."""
    result = run_tfidf_clustering(sample_records, n_clusters=4, random_state=42)

    # Load ground truth labels
    labels_df = pd.read_csv("data/evaluation_labels.csv")
    gt_map = dict(zip(labels_df["feedback_id"], labels_df["ground_truth_theme"]))

    eval_report = evaluate_model_result(result, ground_truth_labels=gt_map)

    assert eval_report.n_samples == len(sample_records)
    assert eval_report.n_clusters == 4
    assert eval_report.silhouette is not None
    assert -1.0 <= eval_report.silhouette <= 1.0
    assert eval_report.adjusted_rand_index is not None
    assert len(eval_report.limitations) >= 3


def test_sentence_embedding_clustering(sample_records):
    """Verify Sentence Transformers clustering or verify descriptive exception if offline."""
    try:
        result = run_embedding_clustering(sample_records, n_clusters=4, random_state=42)
        assert result.method == "Sentence Transformers + K-Means"
        assert len(result.cluster_labels) == len(sample_records)
        assert result.vectors is not None
        assert result.vectors.shape[0] == len(sample_records)

        eval_report = evaluate_model_result(result)
        assert eval_report.silhouette is not None
    except RuntimeError as exc:
        # If internet/weights download is restricted in current environment, it must fail safely with clear error
        assert "Failed to load sentence-transformers model" in str(exc)


def test_compare_models_same_dataset_requirement(sample_records):
    """Verify compare_models enforces exact same feedback records."""
    res_a = run_tfidf_clustering(sample_records, n_clusters=3, random_state=42)

    # Modify IDs in copy to test safety check
    res_b = ModelResult(
        method="Mock",
        model_identifier="mock",
        parameters={},
        runtime_seconds=0.1,
        feedback_ids=res_a.feedback_ids[:-1],  # different length
        cluster_labels=res_a.cluster_labels[:-1],
        cluster_representatives={},
        cluster_keywords={},
        cluster_sizes={},
        vectors=None,
    )

    with pytest.raises(ValueError, match="different feedback IDs"):
        compare_models(res_a, res_b)


def test_embedding_caching_mechanism(sample_records):
    """Verify Sentence Transformers embeddings are cached to prevent redundant re-encoding."""
    from src.clustering import clear_embedding_cache, get_embedding_cache_size

    clear_embedding_cache()
    assert get_embedding_cache_size() == 0

    # First call encodes and populates cache
    res1 = run_embedding_clustering(sample_records, n_clusters=3, random_state=42)
    assert get_embedding_cache_size() == 1

    # Second call on same records reuses cached embeddings
    res2 = run_embedding_clustering(sample_records, n_clusters=4, random_state=42)
    assert get_embedding_cache_size() == 1
    # Verify vector equality
    import numpy as np
    assert np.allclose(res1.vectors, res2.vectors)


def test_metric_preconditions_safely_handled():
    """Verify that degenerate clusters return None with clear reasons rather than crashing."""
    records = [
        FeedbackRecord(id="FB1", original_text="One", cleaned_text="One"),
        FeedbackRecord(id="FB2", original_text="Two", cleaned_text="Two"),
    ]
    # k=2 on 2 samples -> clamped or evaluated
    res = run_tfidf_clustering(records, n_clusters=2, random_state=42)
    eval_report = evaluate_model_result(res)

    # 2 samples cannot compute silhouette (requires >= 3 samples and 2 <= k < n)
    assert eval_report.silhouette is None
    assert eval_report.metrics_validity["silhouette"] is False
    assert "preconditions not met" in eval_report.metric_reasons["silhouette"].lower()


def test_model_result_and_evaluation_representation_metadata(sample_records):
    """Verify representation and distance metric metadata are exposed in evaluation report."""
    res = run_tfidf_clustering(sample_records, n_clusters=3, random_state=42)
    assert res.representation == "sparse_tfidf"
    assert res.distance_metric == "cosine"

    eval_report = evaluate_model_result(res)
    assert eval_report.representation == "sparse_tfidf"
    assert eval_report.distance_metric == "cosine"
    assert any("sparse_tfidf" in lim for lim in eval_report.limitations)
