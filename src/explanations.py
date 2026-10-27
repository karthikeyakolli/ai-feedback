"""Explainability and education module for clustering algorithms and evaluation metrics.

Provides plain-language, transparent explanations of:
- TF-IDF vectorization and K-means mechanics
- Sentence Transformer neural embeddings and continuous semantic space
- Distance geometries and evaluation metrics (Silhouette, Davies-Bouldin, ARI, NMI)
- Fair comparison caveats preventing misleading 'winner' declarations across spaces
"""

from __future__ import annotations

from typing import Any


def get_tfidf_explanation() -> dict[str, str]:
    """Returns plain-language educational breakdown of the TF-IDF + K-Means pipeline."""
    return {
        "title": "TF-IDF Vectorization & K-Means Clustering",
        "how_text_is_converted": (
            "Each feedback submission is broken into individual words (tokenization). "
            "A dictionary of distinct words across the dataset forms a vocabulary. "
            "For each feedback record, the algorithm counts how many times each vocabulary word appears (Term Frequency)."
        ),
        "what_tfidf_represents": (
            "TF-IDF stands for Term Frequency-Inverse Document Frequency. "
            "It measures how distinctive a word is to a specific feedback record. "
            "Words that appear frequently in one comment but rarely in other comments receive high weights. "
            "Words that appear everywhere (e.g., 'the', 'app', 'system') are mathematically downweighted."
        ),
        "why_distinguishing_words_are_features": (
            "In customer feedback, the words that describe specific issues (e.g., 'export', 'billing', 'crash') "
            "distinguish one customer's problem from another. By assigning higher weights to these words, "
            "the algorithm creates numerical coordinates that emphasize specific friction points."
        ),
        "cosine_similarity_role": (
            "Because customer comments vary in length (some are one sentence, others are paragraphs), "
            "all TF-IDF vectors are normalized to unit length. In this normalized space, the angle between two vectors "
            "(cosine similarity) measures whether two comments use similar vocabulary proportions, regardless of comment length."
        ),
        "kmeans_assignment": (
            "K-Means chooses 'k' cluster centers (centroids) in the high-dimensional word space. "
            "It assigns each feedback record to the nearest centroid. "
            "When vectors are unit-normalized, minimizing Euclidean distance to centroids directly corresponds to "
            "maximizing cosine similarity."
        ),
        "wording_variance_limitation": (
            "TF-IDF operates on exact word tokens. If one customer writes 'the export crashes' and another writes "
            "'the download freezes', TF-IDF treats 'crashes' and 'freezes' as completely independent dimensions with 0 overlap. "
            "Synonyms and paraphrases can make related feedback harder to group together."
        ),
    }


def get_sentence_transformer_explanation() -> dict[str, str]:
    """Returns plain-language educational breakdown of the Sentence Transformer pipeline."""
    return {
        "title": "Sentence Transformers & Dense Semantic Embeddings",
        "how_text_is_converted": (
            "Each feedback record is passed into a pretrained MiniLM transformer neural network (all-MiniLM-L6-v2). "
            "The model processes the entire sentence structure, word order, and context to generate a dense "
            "384-dimensional numerical vector."
        ),
        "what_a_vector_represents": (
            "A vector is a point (or coordinate) in a continuous 384-dimensional mathematical space. "
            "Rather than representing isolated words, each coordinate captures abstract semantic relationships "
            "learned from vast language corpora."
        ),
        "why_related_sentences_are_close": (
            "Because the neural network was trained on conceptual associations and semantic relationships, sentences with similar meanings "
            "(e.g., 'app crashes on save' and 'program terminates when writing to disk') are mapped to nearby coordinates, "
            "even if they share few or no exact words."
        ),
        "kmeans_in_embedding_space": (
            "K-Means groups feedback records by partitioning the 384-dimensional continuous space into compact regions. "
            "Centroids represent the average semantic meaning of member records in that region."
        ),
        "embedding_limitations": (
            "While dense embeddings handle synonyms well, they do NOT understand everything. "
            "They can struggle with domain-specific jargon, proprietary acronyms, subtle sarcasm, "
            "double negatives ('not unhappy'), technical stack traces, or customer nuance that differs from "
            "general language training data."
        ),
    }


def get_metric_explanation(metric_name: str) -> dict[str, Any]:
    """Returns structured, cautious educational guidance for a specific evaluation metric."""
    metrics_info: dict[str, dict[str, Any]] = {
        "silhouette": {
            "name": "Silhouette Score",
            "measures": "Cluster cohesion (how close records are to their own cluster) vs separation (how far they are from the nearest neighboring cluster).",
            "range": "[-1.0 to +1.0]",
            "direction_of_preference": "Higher is mathematically preferred (closer to +1.0).",
            "cautious_interpretation": (
                "A higher score indicates that clusters are tighter and well-separated in vector space. "
                "However, a score near 0 indicates overlapping clusters, and negative scores indicate records assigned "
                "to the wrong cluster."
            ),
            "unavailability_reasons": "Cannot be computed when sample count is less than 2, when cluster count k < 2, or when k >= sample count.",
            "business_utility_caveat": (
                "A mathematically high Silhouette score does NOT mean the clusters are useful for product decisions. "
                "For example, a cluster cleanly grouping short boilerplate ('thanks', 'ok') achieves a high silhouette score "
                "while providing zero actionable product intelligence."
            ),
        },
        "davies_bouldin": {
            "name": "Davies–Bouldin Index",
            "measures": "Average similarity ratio between each cluster and its most similar neighboring cluster, comparing cluster size to distance between centers.",
            "range": "[0.0 to +infinity]",
            "direction_of_preference": "Lower is mathematically preferred (closer to 0.0).",
            "cautious_interpretation": (
                "Lower values indicate that clusters are compact internally and far apart from other clusters."
            ),
            "unavailability_reasons": "Cannot be computed when k < 2 or when sample count is less than 2.",
            "business_utility_caveat": (
                "Davies-Bouldin strictly penalizes overlapping cluster boundaries. In real user feedback, "
                "issues naturally overlap (e.g., performance issues occurring during export). A poor score may simply reflect "
                "real-world customer problem complexity, not a defective model."
            ),
        },
        "calinski_harabasz": {
            "name": "Calinski–Harabasz Index (Variance Ratio)",
            "measures": "Ratio of the sum of between-cluster dispersion to within-cluster dispersion.",
            "range": "[0.0 to +infinity]",
            "direction_of_preference": "Higher is mathematically preferred.",
            "cautious_interpretation": "Higher values indicate dense, well-separated clusters.",
            "unavailability_reasons": "Cannot be computed when k < 2 or when sample count is less than 2.",
            "business_utility_caveat": (
                "Tends to be higher for convex, spherical clusters and can unfairly favor algorithms that artificially split "
                "elongated or irregular natural themes."
            ),
        },
        "adjusted_rand_index": {
            "name": "Adjusted Rand Index (ARI)",
            "measures": "Similarity between two clusterings (or clustering vs ground-truth labels), corrected for chance.",
            "range": "[-1.0 to +1.0]",
            "direction_of_preference": "Higher is preferred (1.0 indicates perfect agreement, 0.0 indicates agreement expected by random chance).",
            "cautious_interpretation": (
                "Measures whether pairs of feedback items that belong together in one grouping also belong together in the other. "
                "Accounts for arbitrary cluster label permutation."
            ),
            "unavailability_reasons": (
                "Requires external reference labels (ground truth) or a second comparison run on the exact same dataset records."
            ),
            "business_utility_caveat": (
                "Ground-truth labels are created by human annotators with specific taxonomies. "
                "An unsupervised model that uncovers genuine emerging themes not captured in the taxonomy may receive a lower ARI "
                "despite being highly valuable to the product owner."
            ),
        },
        "normalized_mutual_info": {
            "name": "Normalized Mutual Information (NMI)",
            "measures": "Amount of shared information between two clusterings, normalized between 0 and 1.",
            "range": "[0.0 to 1.0]",
            "direction_of_preference": "Higher is preferred (1.0 = identical information, 0.0 = completely independent).",
            "cautious_interpretation": "Evaluates how much knowing the cluster assignment in one model reduces uncertainty about the other.",
            "unavailability_reasons": "Requires external reference labels or a paired clustering run.",
            "business_utility_caveat": (
                "Does not penalize cluster count differences in the same way ARI does, and can be sensitive to fragmented partitions."
            ),
        },
        "cluster_size_entropy": {
            "name": "Cluster Size Entropy",
            "measures": "Information entropy of cluster sizes, indicating balance vs skew in member counts.",
            "range": "[0.0 to log(k)]",
            "direction_of_preference": "Context-dependent. Maximum entropy indicates equal-sized clusters.",
            "cautious_interpretation": "High entropy means even distribution; very low entropy indicates a giant cluster with tiny outlier clusters.",
            "unavailability_reasons": "Cannot be computed if no clusters exist.",
            "business_utility_caveat": (
                "Real user issues are rarely equally distributed. One major outage may legitimately drive 70% of all feedback. "
                "A balanced cluster size distribution is not inherently superior."
            ),
        },
    }

    if metric_name.lower() in metrics_info:
        return metrics_info[metric_name.lower()]

    return {
        "name": metric_name,
        "measures": f"Evaluation measure for {metric_name}.",
        "range": "N/A",
        "direction_of_preference": "Refer to specific metric documentation.",
        "cautious_interpretation": "Interpret in context with supporting feedback evidence.",
        "unavailability_reasons": "Insufficient data or uncomputed metric.",
        "business_utility_caveat": "Internal mathematical metrics do not equal product utility.",
    }


def get_fair_comparison_caveat() -> dict[str, str]:
    """Provides essential caveats on why metrics cannot be directly compared across vector spaces."""
    return {
        "title": "Cross-Representation Metric Incomparability Caveat",
        "core_principle": (
            "Internal clustering metrics (such as Silhouette Score and Davies-Bouldin Index) calculated in "
            "TF-IDF space CANNOT be directly compared to those calculated in Sentence Transformer space."
        ),
        "why_spaces_differ": (
            "1. Dimensionality: TF-IDF operates in a sparse space with hundreds or thousands of orthogonal dimensions, "
            "where distances are bounded by cosine geometry on a unit sphere. "
            "Sentence Transformers operate in a compact, continuous 384-dimensional space with dense geometry.\n"
            "2. Distance Density: In sparse TF-IDF space, orthogonal vectors naturally produce different distance distributions "
            "than continuous dense neural manifolds.\n"
            "3. Mathematical incomparability: A Silhouette of 0.25 in TF-IDF does NOT mean the model is 'better' or 'worse' "
            "than a Silhouette of 0.20 in Sentence Transformer space."
        ),
        "four_pillars_to_distinguish": (
            "Four Evaluation Pillars to Distinguish:\n"
            "• Internal Cluster Geometry: Mathematical compactness in that specific representation.\n"
            "• Agreement with Reference Labels: Independent external benchmark (when legitimately available).\n"
            "• Evidence Quality: Traceability, quote coherence, and low noise in discovered themes.\n"
            "• Practical Usefulness: Whether the cluster themes help a human product owner make informed investigation decisions."
        ),
        "winner_declaration_warning": (
            "CRITICAL: Never declare one model the 'winner' or 'superior' solely because its internal metric is numerically higher. "
            "Both models provide complementary lenses on customer feedback."
        ),
    }
