"""Evidence-grounded insights generation module.

Synthesizes cluster results, supporting feedback records, and approved product context
into transparent, deterministic reports. Strictly avoids fabricated conclusions,
paid LLM hallucinatory claims, and unsupported business severity metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.applicability import (
    ApplicabilityAssessment,
    assess_cluster_applicability,
)
from src.clustering import ModelResult
from src.context_ingestion import ProductContext
from src.data_processing import FeedbackRecord, ValidationReport
from src.recommendations import (
    InvestigationRecommendation,
    build_investigation_queue,
    generate_cluster_recommendation,
)


@dataclass
class ClusterInsight:
    """Detailed evidence-grounded summary for a single discovered cluster."""

    cluster_id: int
    label: str
    member_count: int
    percentage_of_total: float
    unique_member_count: int
    duplicate_count: int
    keywords: list[str]
    keywords_derivation: str
    representative_quote: str
    representative_feedback_id: str
    member_feedback_ids: list[str]
    supporting_examples: list[dict[str, Any]]
    product_context_relevance: str
    evidence_strength: str  # 'High', 'Moderate', 'Preliminary / Low'
    notes: list[str] = field(default_factory=list)
    applicability: ApplicabilityAssessment | None = None
    recommendation: InvestigationRecommendation | None = None


@dataclass
class InsightsReport:
    """Comprehensive insights summary across all clusters for a given model."""

    model_name: str
    model_identifier: str
    total_feedback_count: int
    total_unique_count: int
    total_duplicate_count: int
    cluster_insights: list[ClusterInsight]
    prioritized_clusters: list[int]  # Cluster IDs sorted by unique feedback volume
    approved_product_context: ProductContext | None = None
    limitations: list[str] = field(default_factory=list)
    investigation_queue: list[InvestigationRecommendation] = field(default_factory=list)

    def get_cluster(self, cluster_id: int) -> ClusterInsight | None:
        """Finds a specific cluster insight by ID."""
        for c in self.cluster_insights:
            if c.cluster_id == cluster_id:
                return c
        return None


# English stopwords for token-overlap analysis
_COMMON_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
}


def _extract_content_tokens(text: str) -> set[str]:
    """Extracts non-stopword alphanumeric word tokens for transparent correlation checks."""
    if not text:
        return set()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in _COMMON_STOPWORDS}


def _determine_context_relevance(
    cluster_keywords: list[str],
    member_texts: list[str],
    context: ProductContext | None,
) -> tuple[str, list[str]]:
    """Cautiously evaluates whether a cluster theme correlates with documented product context.

    Employs deterministic token overlap. Never invents unsupported claims or causal links.
    """
    notes = [
        "Clustering reflects recurring mentions in user feedback; it does not prove code defects or business causality."
    ]

    if not context or not context.is_approved:
        return (
            "No approved product context provided. Contextual interpretation is withheld to avoid conjecture.",
            notes,
        )

    context_full_text = f"{context.name} {context.description} {context.goals_or_concerns} {context.target_users}"
    context_tokens = _extract_content_tokens(context_full_text)

    if not context_tokens:
        return ("Approved product context contains no identifiable feature terms for correlation.", notes)

    # Check keyword overlap
    cluster_terms = set(k.lower() for k in cluster_keywords)
    overlap = cluster_terms.intersection(context_tokens)

    # Check member text overlap if keyword overlap is empty
    if not overlap and member_texts:
        sample_member_tokens = set()
        for t in member_texts[:10]:
            sample_member_tokens.update(_extract_content_tokens(t))
        overlap = sample_member_tokens.intersection(context_tokens)

    if overlap:
        matched = ", ".join(sorted(list(overlap))[:3])
        return (
            f"Potential domain correlation with documented context terms: '{matched}'. "
            f"This feedback may relate to intended functionality or stated concerns.",
            notes,
        )

    return (
        "No direct terminology overlap detected with documented product description or goals. "
        "This theme may represent an unaddressed user concern, emerging issue, or unexpected workflow.",
        notes,
    )


def generate_insights_report(
    model_result: ModelResult,
    validation_report: ValidationReport,
    product_context: ProductContext | None = None,
) -> InsightsReport:
    """Generates an evidence-grounded insights report from actual clustering output.

    Preserves stable record IDs, links exact quotes, and explains keyword derivations.
    """
    feedback_by_id = {r.id: r for r in validation_report.records}
    total_samples = len(model_result.feedback_ids)

    unique_clusters = sorted(list(set(model_result.cluster_labels)))
    cluster_insights: list[ClusterInsight] = []

    is_tfidf = model_result.method.startswith("TF-IDF")
    derivation_explanation = (
        "Top TF-IDF feature weights derived from cluster centroid vector."
        if is_tfidf
        else "Most frequent informative vocabulary tokens across cluster member comments."
    )

    for cluster_id in unique_clusters:
        # Find all members
        member_indices = [
            idx for idx, lbl in enumerate(model_result.cluster_labels)
            if lbl == cluster_id
        ]
        member_ids = [model_result.feedback_ids[idx] for idx in member_indices]
        member_records = [feedback_by_id[fid] for fid in member_ids if fid in feedback_by_id]

        c_size = len(member_records)
        pct = (c_size / total_samples * 100) if total_samples > 0 else 0.0

        # Duplicate counts
        duplicate_count = sum(1 for r in member_records if r.is_duplicate)
        unique_count = c_size - duplicate_count

        # Representative quote
        rep_id = model_result.cluster_representatives.get(cluster_id, "")
        rep_record = feedback_by_id.get(rep_id)
        rep_quote = rep_record.original_text if rep_record else (
            member_records[0].original_text if member_records else "No feedback available."
        )

        keywords = model_result.cluster_keywords.get(cluster_id, [])

        # Evidence strength indicator based on volume
        if unique_count >= 5:
            strength = "High Recurrence"
        elif unique_count >= 3:
            strength = "Moderate Recurrence"
        else:
            strength = "Preliminary / Low Volume"

        # Cautious context relevance
        member_texts = [r.cleaned_text for r in member_records]
        relevance_text, notes = _determine_context_relevance(keywords, member_texts, product_context)

        # Supporting examples
        supporting_examples = []
        for r in member_records[:6]:  # Show top supporting quotes
            supporting_examples.append({
                "id": r.id,
                "text": r.original_text,
                "source": r.source,
                "is_duplicate": r.is_duplicate,
                "duplicate_of_id": r.duplicate_of_id,
            })

        actual_rep_id = rep_id if rep_record else (member_records[0].id if member_records else "")

        # Feature A: Applicability Assessment
        applicability = assess_cluster_applicability(
            cluster_id=cluster_id,
            cluster_keywords=keywords,
            representative_quote=rep_quote,
            representative_id=actual_rep_id,
            member_records=member_records,
            product_context=product_context,
        )

        # Feature B: Evidence-Backed Investigation Recommendation
        recommendation = generate_cluster_recommendation(
            cluster_id=cluster_id,
            model_name=model_result.method,
            keywords=keywords,
            representative_quote=rep_quote,
            representative_id=actual_rep_id,
            member_records=member_records,
            assessment=applicability,
            product_context=product_context,
        )

        insight = ClusterInsight(
            cluster_id=cluster_id,
            label=f"Cluster {cluster_id}",
            member_count=c_size,
            percentage_of_total=round(pct, 2),
            unique_member_count=unique_count,
            duplicate_count=duplicate_count,
            keywords=keywords,
            keywords_derivation=derivation_explanation,
            representative_quote=rep_quote,
            representative_feedback_id=actual_rep_id,
            member_feedback_ids=member_ids,
            supporting_examples=supporting_examples,
            product_context_relevance=relevance_text,
            evidence_strength=strength,
            notes=notes,
            applicability=applicability,
            recommendation=recommendation,
        )
        cluster_insights.append(insight)

    # Sort prioritization by unique feedback volume descending
    prioritized = sorted(
        cluster_insights,
        key=lambda c: (c.unique_member_count, c.member_count),
        reverse=True,
    )
    prioritized_cluster_ids = [c.cluster_id for c in prioritized]

    # Build Frequency-Based Investigation Queue across clusters
    investigation_queue = build_investigation_queue(
        [c.recommendation for c in cluster_insights if c.recommendation is not None]
    )

    limitations = [
        "Prioritization reflects reporting recurrence across users; it does not measure commercial severity or impact.",
        "Keywords are suggested descriptive markers and may omit subtle context or technical specifics.",
        "Unsupervised K-Means clustering does not verify whether an issue represents an actual defect.",
        "Investigation recommendations are structured exploration prompts; human review is required before taking product action.",
    ]

    total_dups = sum(c.duplicate_count for c in cluster_insights)
    total_uniques = total_samples - total_dups

    return InsightsReport(
        model_name=model_result.method,
        model_identifier=model_result.model_identifier,
        total_feedback_count=total_samples,
        total_unique_count=total_uniques,
        total_duplicate_count=total_dups,
        cluster_insights=cluster_insights,
        prioritized_clusters=prioritized_cluster_ids,
        approved_product_context=product_context,
        limitations=limitations,
        investigation_queue=investigation_queue,
    )
