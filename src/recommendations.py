"""Investigation Recommendations module for decision support and explainability.

Transforms cluster analysis and applicability assessments into actionable investigation
suggestions. Recommends WHAT TO INVESTIGATE, avoiding autonomous product decisions,
unsupported severity claims, or fabricated customer outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.applicability import (
    STATUS_INSUFFICIENT_INFORMATION,
    STATUS_LIKELY_RELEVANT,
    STATUS_NEEDS_VALIDATION,
    ApplicabilityAssessment,
)
from src.context_ingestion import ProductContext
from src.data_processing import FeedbackRecord


@dataclass
class InvestigationRecommendation:
    """Actionable investigation prompt grounded in cluster evidence."""

    title: str
    suggested_action: str
    rationale: str
    cluster_id: int
    model_name: str
    supporting_feedback_ids: list[str]
    supporting_quotes: list[str]
    contradictory_evidence: list[str]
    applicability_status: str
    limitations_and_missing_info: list[str]
    human_review_reminder: str = (
        "Recommendations are preliminary investigation prompts based on cluster recurrence; "
        "human validation is required before taking product action."
    )
    unique_mentions_count: int = 0
    duplicate_count: int = 0
    recurrence_score: int = 0


def generate_cluster_recommendation(
    cluster_id: int,
    model_name: str,
    keywords: list[str],
    representative_quote: str,
    representative_id: str,
    member_records: Sequence[FeedbackRecord],
    assessment: ApplicabilityAssessment,
    product_context: ProductContext | None = None,
) -> InvestigationRecommendation:
    """Generates an evidence-grounded investigation recommendation for a single cluster.

    Args:
        cluster_id: Numeric cluster identifier.
        model_name: Name of the clustering model (e.g., 'TF-IDF + K-Means').
        keywords: Descriptive keywords for the cluster.
        representative_quote: Verbatim representative feedback quote.
        representative_id: Feedback ID of the representative quote.
        member_records: Real feedback records belonging to the cluster.
        assessment: Computed ApplicabilityAssessment for this cluster.
        product_context: Approved product context, if available.

    Returns:
        A structured InvestigationRecommendation object.
    """
    total_count = len(member_records)
    duplicate_count = sum(1 for r in member_records if r.is_duplicate)
    unique_count = total_count - duplicate_count

    primary_keyword = keywords[0] if keywords else f"Cluster {cluster_id}"
    clean_keyword = primary_keyword.strip().capitalize()

    # Determine limitations based on duplicates and evidence volume
    limitations = list(assessment.missing_information)
    if duplicate_count > 0:
        limitations.append(
            f"{duplicate_count} duplicate record(s) detected. Duplicate submissions are treated as "
            "repeated reports, not independent corroborations by separate customers."
        )
    if unique_count < 3:
        limitations.append(
            f"Low unique evidence volume ({unique_count} distinct records). Conclusions should be treated with caution."
        )

    # Branch by applicability status
    if assessment.status == STATUS_LIKELY_RELEVANT:
        concept_str = (
            assessment.relevant_context_concepts[0]
            if assessment.relevant_context_concepts
            else "documented capabilities"
        )
        title = f"Inspect {clean_keyword} feedback against {concept_str}"
        suggested_action = (
            f"Inspect supporting feedback ({', '.join(assessment.supporting_feedback_ids[:3])}) to verify whether "
            f"reported user experiences in '{clean_keyword}' reflect a divergence from documented {concept_str} behavior."
        )
        rationale = (
            f"Recurring user mentions ({unique_count} unique records) correlate directly with approved product context. "
            "Evidence indicates potential friction or workflow issues within intended features."
        )

    elif assessment.status == STATUS_NEEDS_VALIDATION:
        if assessment.contradictory_evidence:
            title = f"Investigate conflicting feedback signals in {clean_keyword}"
            suggested_action = (
                f"Perform manual human review of member records ({', '.join(assessment.supporting_feedback_ids[:3])}) "
                "to separate contradictory user statements (e.g., positive testimonials vs defect reports) into distinct sub-themes."
            )
            rationale = (
                f"Cluster exhibits mixed or conflicting evidence ({len(assessment.contradictory_evidence)} contradiction signal(s)). "
                "Automatic clustering may have combined opposing user experiences."
            )
        else:
            title = f"Verify product scope relevance for {clean_keyword}"
            suggested_action = (
                f"Review member feedback ({', '.join(assessment.supporting_feedback_ids[:3])}) with the product owner "
                f"to determine whether '{clean_keyword}' falls within planned scope or represents an unaddressed feature request."
            )
            rationale = (
                "Terminology in this cluster lacks direct overlap with approved product context. "
                "Requires human validation to distinguish unexpected user workflows from out-of-scope submissions."
            )

    else:  # STATUS_INSUFFICIENT_INFORMATION
        title = f"Collect additional descriptive feedback on {clean_keyword}"
        suggested_action = (
            f"Gather more detailed, specific user feedback regarding '{clean_keyword}' and ensure approved "
            "product context documentation is up to date before making product decisions."
        )
        rationale = (
            "Current evidence is sparse, consists of uninformative fragments, or lacks approved product context. "
            "Insufficient grounding to support a concrete engineering or product investigation."
        )

    # Ensure supporting quotes and IDs strictly reflect actual feedback records
    supp_ids = list(assessment.supporting_feedback_ids)
    supp_quotes = list(assessment.supporting_quotes)

    # Fallback to representative or first member if empty
    if not supp_ids and member_records:
        supp_ids = [member_records[0].id]
        supp_quotes = [member_records[0].original_text]

    return InvestigationRecommendation(
        title=title,
        suggested_action=suggested_action,
        rationale=rationale,
        cluster_id=cluster_id,
        model_name=model_name,
        supporting_feedback_ids=supp_ids,
        supporting_quotes=supp_quotes,
        contradictory_evidence=list(assessment.contradictory_evidence),
        applicability_status=assessment.status,
        limitations_and_missing_info=limitations,
        human_review_reminder=(
            "Recommendations are preliminary investigation prompts based on cluster recurrence; "
            "human validation is required before taking product action."
        ),
        unique_mentions_count=unique_count,
        duplicate_count=duplicate_count,
        recurrence_score=unique_count,
    )


def build_investigation_queue(
    recommendations: Sequence[InvestigationRecommendation],
) -> list[InvestigationRecommendation]:
    """Orders recommendations into a deterministic frequency-based investigation queue.

    Sorted by:
    1. unique_mentions_count descending (highest unique user mentions first)
    2. duplicate_count descending
    3. cluster_id ascending (deterministic tie-breaker)

    Deduplicates equivalent recommendations across clusters if they suggest identical actions.
    """
    # Deterministic deduplication
    seen_actions: set[str] = set()
    deduplicated: list[InvestigationRecommendation] = []

    for rec in recommendations:
        # Normalize action signature for deduplication
        norm_key = (rec.suggested_action.lower().strip(), rec.model_name)
        if norm_key in seen_actions:
            continue
        seen_actions.add(norm_key)
        deduplicated.append(rec)

    # Sort into frequency-based queue
    queue = sorted(
        deduplicated,
        key=lambda r: (r.unique_mentions_count, r.duplicate_count, -r.cluster_id),
        reverse=True,
    )
    return queue
