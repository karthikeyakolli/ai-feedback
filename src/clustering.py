"""Clustering module comparing TF-IDF baseline and Pretrained Sentence Transformers.

Implements K-Means clustering with cosine similarity alignment, representative feedback
discovery, cluster keyword extraction, and strict input validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.data_processing import FeedbackRecord


@dataclass
class ModelResult:
    """Encapsulates the output of a clustering run."""

    method: str
    model_identifier: str
    parameters: dict[str, Any]
    runtime_seconds: float
    feedback_ids: list[str]
    cluster_labels: list[int]
    cluster_representatives: dict[int, str]
    cluster_keywords: dict[int, list[str]]
    cluster_sizes: dict[int, int]
    warnings: list[str] = field(default_factory=list)
    vectors: np.ndarray | Any = None
    representation: str = "sparse_tfidf"
    distance_metric: str = "cosine"

    def get_assignment_map(self) -> dict[str, int]:
        """Maps each feedback ID to its assigned cluster label."""
        return dict(zip(self.feedback_ids, self.cluster_labels))


def _adjust_k(n_samples: int, requested_k: int) -> tuple[int, list[str]]:
    """Validates and safely clamps k given dataset size."""
    warnings: list[str] = []
    if n_samples < 2:
        raise ValueError(f"At least 2 feedback samples are required for clustering. Received: {n_samples}.")

    if requested_k < 2:
        warnings.append(f"Requested cluster count {requested_k} is less than 2. Clamped to k=2.")
        return 2, warnings

    if requested_k >= n_samples:
        adjusted_k = max(2, n_samples - 1)
        warnings.append(
            f"Requested cluster count k={requested_k} is greater than or equal to sample count {n_samples}. "
            f"Clamped k to {adjusted_k}."
        )
        return adjusted_k, warnings

    return requested_k, warnings


def _find_representatives_and_sizes(
    vectors: np.ndarray,
    labels: np.ndarray,
    feedback_ids: list[str],
    centroids: np.ndarray,
    k: int,
) -> tuple[dict[int, str], dict[int, int]]:
    """Finds the feedback record closest to each cluster centroid by cosine similarity."""
    representatives: dict[int, str] = {}
    sizes: dict[int, int] = {}

    for cluster_id in range(k):
        member_indices = np.where(labels == cluster_id)[0]
        sizes[cluster_id] = len(member_indices)

        if len(member_indices) == 0:
            representatives[cluster_id] = ""
            continue

        cluster_vecs = vectors[member_indices]
        centroid = centroids[cluster_id].reshape(1, -1)

        sims = cosine_similarity(cluster_vecs, centroid).flatten()
        best_member_idx = member_indices[np.argmax(sims)]
        representatives[cluster_id] = feedback_ids[best_member_idx]

    return representatives, sizes


def run_tfidf_clustering(
    records: Sequence[FeedbackRecord],
    n_clusters: int = 4,
    random_state: int = 42,
    max_features: int = 1000,
) -> ModelResult:
    """Runs TF-IDF vectorization followed by K-Means clustering on feedback records."""
    start_time = time.perf_counter()
    n_samples = len(records)
    k, warnings = _adjust_k(n_samples, n_clusters)

    feedback_ids = [rec.id for rec in records]
    texts = [rec.cleaned_text for rec in records]

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        lowercase=True,
        norm="l2",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            # Fallback without stop word filtering if texts contain only stopwords
            warnings.append("Input text contained only common stop words; disabled stop word filtering.")
            vectorizer = TfidfVectorizer(
                stop_words=None,
                max_features=max_features,
                lowercase=True,
                norm="l2",
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
        else:
            raise

    dense_vectors = tfidf_matrix.toarray()

    if tfidf_matrix.shape[1] == 0:
        raise ValueError("TF-IDF vocabulary is empty. Input texts may contain only whitespace or punctuation.")

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(dense_vectors)
    centroids = kmeans.cluster_centers_

    # Find representatives and sizes
    representatives, sizes = _find_representatives_and_sizes(
        dense_vectors, labels, feedback_ids, centroids, k
    )

    # Extract top keywords per cluster from centroid weights
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_keywords: dict[int, list[str]] = {}
    for cluster_id in range(k):
        centroid = centroids[cluster_id]
        top_indices = np.argsort(centroid)[::-1][:5]
        # Filter out features with 0 weight
        top_words = [feature_names[i] for i in top_indices if centroid[i] > 0]
        cluster_keywords[cluster_id] = top_words

    elapsed = time.perf_counter() - start_time

    return ModelResult(
        method="TF-IDF + K-Means",
        model_identifier="scikit-learn/TfidfVectorizer",
        parameters={
            "n_clusters": k,
            "random_state": random_state,
            "max_features": max_features,
            "norm": "l2",
        },
        runtime_seconds=elapsed,
        feedback_ids=feedback_ids,
        cluster_labels=labels.tolist(),
        cluster_representatives=representatives,
        cluster_keywords=cluster_keywords,
        cluster_sizes=sizes,
        warnings=warnings,
        vectors=dense_vectors,
        representation="sparse_tfidf",
        distance_metric="cosine",
    )


# Cache sentence-transformer model instance across calls
_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}
# Cache computed embeddings keyed by tuple of cleaned texts to avoid redundant encodes
_EMBEDDING_VECTORS_CACHE: dict[tuple[str, ...], np.ndarray] = {}


def clear_embedding_cache() -> None:
    """Clears the computed embedding vector cache."""
    _EMBEDDING_VECTORS_CACHE.clear()


def get_embedding_cache_size() -> int:
    """Returns the number of cached text-sequence embedding matrices."""
    return len(_EMBEDDING_VECTORS_CACHE)


def _get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Loads and caches the SentenceTransformer model."""
    if model_name not in _EMBEDDING_MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sentence-transformers model '{model_name}'. "
                f"Ensure sentence-transformers and PyTorch are installed and internet is accessible: {exc}"
            ) from exc
    return _EMBEDDING_MODEL_CACHE[model_name]


def run_embedding_clustering(
    records: Sequence[FeedbackRecord],
    n_clusters: int = 4,
    random_state: int = 42,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> ModelResult:
    """Runs Sentence Transformers embeddings followed by K-Means clustering.

    Reuses cached embeddings if identical feedback texts have already been encoded.
    """
    start_time = time.perf_counter()
    n_samples = len(records)
    k, warnings = _adjust_k(n_samples, n_clusters)

    feedback_ids = [rec.id for rec in records]
    texts = [rec.cleaned_text for rec in records]
    cache_key = tuple(texts)

    if cache_key in _EMBEDDING_VECTORS_CACHE:
        embeddings = _EMBEDDING_VECTORS_CACHE[cache_key].copy()
    else:
        model = _get_embedding_model(model_name)
        # Generate normalized sentence embeddings (cosine distance = Euclidean distance on normalized vectors)
        raw_embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = normalize(raw_embeddings, norm="l2")
        _EMBEDDING_VECTORS_CACHE[cache_key] = embeddings.copy()

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    centroids = normalize(kmeans.cluster_centers_, norm="l2")

    # Find representatives and sizes
    representatives, sizes = _find_representatives_and_sizes(
        embeddings, labels, feedback_ids, centroids, k
    )

    # Extract characteristic keywords per cluster using top distinct words from member texts
    cluster_keywords: dict[int, list[str]] = {}
    for cluster_id in range(k):
        member_indices = np.where(labels == cluster_id)[0]
        member_texts = [texts[idx] for idx in member_indices]
        if member_texts:
            # Use quick local count vectorizer for cluster-level vocabulary
            try:
                vec = TfidfVectorizer(stop_words="english", max_features=5)
                vec.fit(member_texts)
                cluster_keywords[cluster_id] = list(vec.get_feature_names_out())
            except Exception:
                cluster_keywords[cluster_id] = []
        else:
            cluster_keywords[cluster_id] = []

    elapsed = time.perf_counter() - start_time

    return ModelResult(
        method="Sentence Transformers + K-Means",
        model_identifier=model_name,
        parameters={
            "n_clusters": k,
            "random_state": random_state,
            "normalized_embeddings": True,
        },
        runtime_seconds=elapsed,
        feedback_ids=feedback_ids,
        cluster_labels=labels.tolist(),
        cluster_representatives=representatives,
        cluster_keywords=cluster_keywords,
        cluster_sizes=sizes,
        warnings=warnings,
        vectors=embeddings,
        representation="dense_sentence_embeddings",
        distance_metric="cosine_on_normalized_l2",
    )
