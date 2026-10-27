"""Evaluation module for clustering comparison and metric calculation.

Computes valid quantitative metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI/NMI),
enforces strict mathematical preconditions, and documents metric limitations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.clustering import ModelResult


@dataclass
class EvaluationReport:
    """Summary of cluster quality metrics, external validation, and known limitations."""

    method: str
    model_identifier: str
    n_samples: int
    n_clusters: int
    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    adjusted_rand_index: float | None = None
    normalized_mutual_info: float | None = None
    cluster_size_entropy: float | None = None
    metrics_validity: dict[str, bool] = field(default_factory=dict)
    metric_reasons: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    representation: str = ""
    distance_metric: str = ""


def _compute_cluster_entropy(labels: list[int], k: int) -> float:
    """Computes Shannon entropy of cluster size distribution normalized to [0, 1]."""
    if k <= 1 or not labels:
        return 0.0
    total = len(labels)
    counts = [labels.count(i) for i in range(k)]
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    max_entropy = math.log2(k)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def evaluate_model_result(
    result: ModelResult,
    ground_truth_labels: dict[str, str] | None = None,
) -> EvaluationReport:
    """Evaluates a single ModelResult, safely verifying preconditions for each metric."""
    warnings: list[str] = list(result.warnings)
    representation = getattr(result, "representation", "unspecified")
    distance_metric = getattr(result, "distance_metric", "unspecified")

    limitations: list[str] = [
        f"Model feature space is '{representation}' with distance metric '{distance_metric}'. "
        "Silhouette and spatial cluster scores evaluate geometric compactness within that specific representation, "
        "meaning raw numeric scores from different feature spaces (sparse TF-IDF vs dense embeddings) "
        "are not directly comparable as a single ground truth.",
        "K-Means optimizes for spherical, variance-equal clusters and may force outliers into unrelated themes.",
        "Small dataset sizes (n < 50) have high assignment variance under different random initializations.",
    ]
    validity: dict[str, bool] = {}
    metric_reasons: dict[str, str] = {}

    labels = np.array(result.cluster_labels)
    vectors = result.vectors
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_unique_clusters = len(unique_labels)

    sil_val: float | None = None
    db_val: float | None = None
    ch_val: float | None = None

    # Check mathematical preconditions for internal clustering metrics
    can_evaluate_internal = (
        vectors is not None
        and n_samples >= 3
        and 1 < n_unique_clusters < n_samples
    )

    if can_evaluate_internal:
        # Silhouette Score
        try:
            sil_val = float(silhouette_score(vectors, labels, metric="cosine" if result.method.startswith("TF-IDF") else "euclidean"))
            validity["silhouette"] = True
            metric_reasons["silhouette"] = "Calculated successfully."
        except Exception as exc:
            warnings.append(f"Silhouette score computation failed: {exc}")
            validity["silhouette"] = False
            metric_reasons["silhouette"] = f"Computation failed: {exc}"

        # Davies-Bouldin
        try:
            db_val = float(davies_bouldin_score(vectors, labels))
            validity["davies_bouldin"] = True
            metric_reasons["davies_bouldin"] = "Calculated successfully."
        except Exception as exc:
            warnings.append(f"Davies-Bouldin computation failed: {exc}")
            validity["davies_bouldin"] = False
            metric_reasons["davies_bouldin"] = f"Computation failed: {exc}"

        # Calinski-Harabasz
        try:
            ch_val = float(calinski_harabasz_score(vectors, labels))
            validity["calinski_harabasz"] = True
            metric_reasons["calinski_harabasz"] = "Calculated successfully."
        except Exception as exc:
            warnings.append(f"Calinski-Harabasz computation failed: {exc}")
            validity["calinski_harabasz"] = False
            metric_reasons["calinski_harabasz"] = f"Computation failed: {exc}"
    else:
        reason = (
            f"Preconditions not met for spatial clustering metrics (samples={n_samples}, "
            f"distinct clusters={n_unique_clusters}). Requires at least 3 samples and 2 <= k < n_samples."
        )
        warnings.append(reason)
        validity["silhouette"] = False
        validity["davies_bouldin"] = False
        validity["calinski_harabasz"] = False
        metric_reasons["silhouette"] = reason
        metric_reasons["davies_bouldin"] = reason
        metric_reasons["calinski_harabasz"] = reason

    # External evaluation against separate ground truth if provided
    ari_val: float | None = None
    nmi_val: float | None = None

    if ground_truth_labels:
        # Align ground truth labels with result feedback IDs
        aligned_true = []
        aligned_pred = []
        for fid, pred_label in zip(result.feedback_ids, result.cluster_labels):
            if fid in ground_truth_labels:
                aligned_true.append(ground_truth_labels[fid])
                aligned_pred.append(pred_label)

        if len(aligned_true) >= 2:
            ari_val = float(adjusted_rand_score(aligned_true, aligned_pred))
            nmi_val = float(normalized_mutual_info_score(aligned_true, aligned_pred))
            validity["adjusted_rand_index"] = True
            validity["normalized_mutual_info"] = True
            metric_reasons["adjusted_rand_index"] = f"Calculated against {len(aligned_true)} reference labels."
            metric_reasons["normalized_mutual_info"] = f"Calculated against {len(aligned_true)} reference labels."
            limitations.append(
                "Ground truth metrics (ARI/NMI) reflect agreement with manual annotator categories, "
                "which inherently incorporate subjective human grouping bias."
            )
        else:
            reason = "Insufficient matching ground truth labels to compute external benchmark metrics."
            warnings.append(reason)
            validity["adjusted_rand_index"] = False
            validity["normalized_mutual_info"] = False
            metric_reasons["adjusted_rand_index"] = reason
            metric_reasons["normalized_mutual_info"] = reason
    else:
        no_ref_reason = "No ground-truth reference labels provided for this dataset."
        validity["adjusted_rand_index"] = False
        validity["normalized_mutual_info"] = False
        metric_reasons["adjusted_rand_index"] = no_ref_reason
        metric_reasons["normalized_mutual_info"] = no_ref_reason

    # Cluster balance entropy
    k_param = result.parameters.get("n_clusters", n_unique_clusters)
    entropy_val = _compute_cluster_entropy(result.cluster_labels, k_param)

    return EvaluationReport(
        method=result.method,
        model_identifier=result.model_identifier,
        n_samples=n_samples,
        n_clusters=n_unique_clusters,
        silhouette=sil_val,
        davies_bouldin=db_val,
        calinski_harabasz=ch_val,
        adjusted_rand_index=ari_val,
        normalized_mutual_info=nmi_val,
        cluster_size_entropy=entropy_val,
        metrics_validity=validity,
        metric_reasons=metric_reasons,
        warnings=warnings,
        limitations=limitations,
        representation=representation,
        distance_metric=distance_metric,
    )


def compare_models(
    tfidf_result: ModelResult,
    embedding_result: ModelResult,
    ground_truth_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compares TF-IDF and Embedding clustering results on the exact same dataset."""
    if tfidf_result.feedback_ids != embedding_result.feedback_ids:
        raise ValueError("Cannot compare models: they were evaluated on different feedback IDs!")

    tfidf_eval = evaluate_model_result(tfidf_result, ground_truth_labels)
    embedding_eval = evaluate_model_result(embedding_result, ground_truth_labels)

    return {
        "dataset_size": len(tfidf_result.feedback_ids),
        "comparison_metrics": {
            "runtime_seconds": {
                "TF-IDF": round(tfidf_result.runtime_seconds, 4),
                "Embeddings": round(embedding_result.runtime_seconds, 4),
            },
            "silhouette": {
                "TF-IDF": round(tfidf_eval.silhouette, 4) if tfidf_eval.silhouette is not None else None,
                "Embeddings": round(embedding_eval.silhouette, 4) if embedding_eval.silhouette is not None else None,
            },
            "cluster_balance_entropy": {
                "TF-IDF": round(tfidf_eval.cluster_size_entropy, 4) if tfidf_eval.cluster_size_entropy is not None else None,
                "Embeddings": round(embedding_eval.cluster_size_entropy, 4) if embedding_eval.cluster_size_entropy is not None else None,
            },
            "adjusted_rand_index": {
                "TF-IDF": round(tfidf_eval.adjusted_rand_index, 4) if tfidf_eval.adjusted_rand_index is not None else None,
                "Embeddings": round(embedding_eval.adjusted_rand_index, 4) if embedding_eval.adjusted_rand_index is not None else None,
            },
        },
        "reports": {
            "TF-IDF": tfidf_eval,
            "Embeddings": embedding_eval,
        },
        "neutral_summary": (
            "Both models ran on the exact same feedback dataset. Metrics should be interpreted "
            "alongside manual cluster inspection. TF-IDF provides high keyword transparency and fast execution, "
            "while Sentence Transformers capture semantic paraphrasing but require more computation."
        ),
    }
