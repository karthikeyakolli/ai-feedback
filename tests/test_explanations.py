"""Tests for Feature C (Explainable Clustering and Educational Guides).

Verifies algorithm mechanics descriptions, metric preference directions, unavailability reasons,
business utility caveats, and cross-representation metric incomparability.
"""

from __future__ import annotations

from src.explanations import (
    get_fair_comparison_caveat,
    get_metric_explanation,
    get_sentence_transformer_explanation,
    get_tfidf_explanation,
)


def test_tfidf_explanation_content():
    """Verify TF-IDF educational breakdown covers all required concepts."""
    tfidf = get_tfidf_explanation()

    assert "TF-IDF" in tfidf["title"]
    assert "vocabulary" in tfidf["how_text_is_converted"].lower()
    assert "term frequency-inverse document frequency" in tfidf["what_tfidf_represents"].lower()
    assert "distinguish" in tfidf["why_distinguishing_words_are_features"].lower()
    assert "cosine" in tfidf["cosine_similarity_role"].lower()
    assert "centroid" in tfidf["kmeans_assignment"].lower()
    # Limitations
    assert "synonyms" in tfidf["wording_variance_limitation"].lower()


def test_sentence_transformer_explanation_content():
    """Verify Sentence Transformer educational breakdown covers embeddings and limitations."""
    emb = get_sentence_transformer_explanation()

    assert "Sentence Transformers" in emb["title"]
    assert "384" in emb["how_text_is_converted"]
    assert "continuous" in emb["what_a_vector_represents"].lower()
    assert "semantic" in emb["why_related_sentences_are_close"].lower()
    assert "continuous" in emb["kmeans_in_embedding_space"].lower()
    # Limitations: domain jargon, sarcasm, etc.
    assert "sarcasm" in emb["embedding_limitations"].lower()
    assert "domain-specific" in emb["embedding_limitations"].lower()


def test_metric_explanations_direction_and_caveats():
    """Verify metric directions of preference and business utility warnings."""
    # 1. Silhouette
    sil = get_metric_explanation("silhouette")
    assert "higher" in sil["direction_of_preference"].lower()
    assert "[-1.0 to +1.0]" == sil["range"]
    assert "boilerplate" in sil["business_utility_caveat"].lower()

    # 2. Davies-Bouldin
    db = get_metric_explanation("davies_bouldin")
    assert "lower" in db["direction_of_preference"].lower()
    assert "overlap" in db["business_utility_caveat"].lower()

    # 3. Adjusted Rand Index
    ari = get_metric_explanation("adjusted_rand_index")
    assert "higher" in ari["direction_of_preference"].lower()
    assert "reference labels" in ari["unavailability_reasons"].lower()

    # 4. Normalized Mutual Info
    nmi = get_metric_explanation("normalized_mutual_info")
    assert "higher" in nmi["direction_of_preference"].lower()

    # 5. Cluster Size Entropy
    entropy = get_metric_explanation("cluster_size_entropy")
    assert "balance" in entropy["measures"].lower()


def test_fair_comparison_caveat():
    """Verify cross-representation incomparability principles and winner warning."""
    caveat = get_fair_comparison_caveat()

    assert "Incomparability" in caveat["title"]
    assert "cannot be directly compared" in caveat["core_principle"].lower()
    assert "sparse" in caveat["why_spaces_differ"].lower()
    assert "dense" in caveat["why_spaces_differ"].lower()
    assert "Four Evaluation Pillars" in caveat["four_pillars_to_distinguish"]
    assert "Never declare" in caveat["winner_declaration_warning"]
