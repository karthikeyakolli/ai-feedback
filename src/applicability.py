"""Applicability Assessment module for evaluating cluster relevance to product context.

Provides transparent, deterministic, evidence-grounded assessment of discovered clusters
against approved product context documents. Strictly adheres to three primary statuses:
- 'Likely relevant'
- 'Needs validation'
- 'Insufficient information'

Avoids fabricated claims, hallucinated business impacts, or black-box probability scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.context_ingestion import ProductContext
from src.data_processing import FeedbackRecord

# Three strictly required primary applicability statuses
STATUS_LIKELY_RELEVANT = "Likely relevant"
STATUS_NEEDS_VALIDATION = "Needs validation"
STATUS_INSUFFICIENT_INFORMATION = "Insufficient information"

VALID_STATUSES = {
    STATUS_LIKELY_RELEVANT,
    STATUS_NEEDS_VALIDATION,
    STATUS_INSUFFICIENT_INFORMATION,
}

# Standard English stopwords to prevent spurious lexical overlap
STOPWORDS = {
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


@dataclass
class ApplicabilityRulesConfig:
    """Configurable thresholds and rules for deterministic applicability assessment.

    Attributes:
        min_informative_tokens: Minimum distinct informative tokens in feedback for evaluation.
            Prevents evaluating empty, single-character, or pure stopword clusters.
        min_overlap_tokens: Minimum lexical overlap count with context to consider correlation.
        strong_overlap_tokens: Minimum overlap count with specific product goals or description
            to qualify for 'Likely relevant'.
        max_evidence_quotes: Maximum number of verbatim quotes to attach as supporting evidence.
        sample_member_limit: Number of actual member feedback records to inspect for signals.
    """

    min_informative_tokens: int = 2
    min_overlap_tokens: int = 1
    strong_overlap_tokens: int = 2
    max_evidence_quotes: int = 3
    sample_member_limit: int = 10


@dataclass
class ApplicabilityAssessment:
    """Structured, evidence-grounded applicability evaluation for a cluster."""

    status: str
    explanation: str
    relevant_context_concepts: list[str] = field(default_factory=list)
    supporting_feedback_ids: list[str] = field(default_factory=list)
    supporting_quotes: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    signals_used: dict[str, Any] = field(default_factory=dict)
    missing_information: list[str] = field(default_factory=list)
    next_validation_step: str = ""
    assessment_method: str = "deterministic_rule_v1"
    context_version: int | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid applicability status '{self.status}'. Must be one of {VALID_STATUSES}")


def extract_informative_tokens(text: str) -> set[str]:
    """Extracts lowercase alphabetic words of length >= 3 excluding English stopwords."""
    if not text:
        return set()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _detect_contradictions_and_noise(
    member_records: Sequence[FeedbackRecord],
) -> list[str]:
    """Identifies contradictory sentiment or conflicting problem reports within member feedback.

    Employs deterministic heuristic checks on polar opposing sentiment or conflicting statements.
    """
    if len(member_records) < 2:
        return []

    contradictions: list[str] = []
    has_positive = False
    has_negative = False
    pos_quotes = []
    neg_quotes = []

    pos_markers = {"great", "fast", "love", "smooth", "excellent", "working well", "perfect", "easy"}
    neg_markers = {"slow", "crash", "bug", "broken", "fail", "freeze", "terrible", "cannot", "error"}

    for r in member_records:
        text_lower = r.original_text.lower()
        if any(m in text_lower for m in pos_markers):
            has_positive = True
            pos_quotes.append(f"[{r.id}] \"{r.original_text[:80]}\"")
        if any(m in text_lower for m in neg_markers):
            has_negative = True
            neg_quotes.append(f"[{r.id}] \"{r.original_text[:80]}\"")

    if has_positive and has_negative and pos_quotes and neg_quotes:
        contradictions.append(
            f"Mixed feedback polarity detected within cluster: positive mention ({pos_quotes[0]}) "
            f"conflicts with defect report ({neg_quotes[0]}). Cluster may be overly broad or noisy."
        )

    return contradictions


def assess_cluster_applicability(
    cluster_id: int,
    cluster_keywords: list[str],
    representative_quote: str,
    representative_id: str,
    member_records: Sequence[FeedbackRecord],
    product_context: ProductContext | None,
    run_id: str | None = None,
    config: ApplicabilityRulesConfig | None = None,
) -> ApplicabilityAssessment:
    """Deterministically assesses cluster relevance against approved product context.

    Follows strict, transparent rules:
    1. If product context is missing or not approved -> 'Insufficient information'.
    2. If cluster is empty or feedback lacks informative tokens -> 'Insufficient information'.
    3. If evidence exhibits strong direct overlap with documented goals or description -> 'Likely relevant'.
    4. If evidence exhibits partial overlap or contradictory evidence -> 'Needs validation'.
    5. If no overlap is detected -> 'Needs validation' (emerging or undocumented issue).

    Args:
        cluster_id: Numeric cluster identifier.
        cluster_keywords: Top keywords representing the cluster.
        representative_quote: Verbatim representative feedback quote.
        representative_id: Feedback ID of the representative quote.
        member_records: Validated feedback records belonging to this cluster.
        product_context: Approved product context snapshot, if available.
        run_id: Optional ID of the analysis run.
        config: Thresholds configuration (defaults to ApplicabilityRulesConfig()).

    Returns:
        A structured, fully traceable ApplicabilityAssessment object.
    """
    cfg = config or ApplicabilityRulesConfig()
    ctx_version = product_context.version if (product_context and hasattr(product_context, "version")) else None

    # Rule 1: Check for missing or unapproved product context
    if product_context is None or not product_context.is_approved:
        return ApplicabilityAssessment(
            status=STATUS_INSUFFICIENT_INFORMATION,
            explanation=(
                "No approved product context is available. Applicability cannot be determined "
                "without a validated product specification baseline."
            ),
            relevant_context_concepts=[],
            supporting_feedback_ids=[representative_id] if representative_id else [],
            supporting_quotes=[representative_quote] if representative_quote else [],
            contradictory_evidence=[],
            signals_used={"has_approved_context": False, "cluster_id": cluster_id},
            missing_information=["Approved product context specification"],
            next_validation_step="Approve a product context definition before evaluating feature relevance.",
            assessment_method="deterministic_rule_v1",
            context_version=ctx_version,
            run_id=run_id,
        )

    # Rule 2: Check for empty cluster
    if not member_records:
        return ApplicabilityAssessment(
            status=STATUS_INSUFFICIENT_INFORMATION,
            explanation=f"Cluster {cluster_id} contains zero member feedback records.",
            relevant_context_concepts=[],
            supporting_feedback_ids=[],
            supporting_quotes=[],
            contradictory_evidence=[],
            signals_used={"member_count": 0, "cluster_id": cluster_id},
            missing_information=["Cluster member feedback records"],
            next_validation_step="Inspect data ingestion and clustering partition to verify why this cluster is empty.",
            assessment_method="deterministic_rule_v1",
            context_version=ctx_version,
            run_id=run_id,
        )

    # Collect informative tokens from cluster
    sampled_members = member_records[: cfg.sample_member_limit]
    cluster_member_tokens: set[str] = set()
    for rec in sampled_members:
        cluster_member_tokens.update(extract_informative_tokens(rec.cleaned_text))

    keyword_tokens = set()
    for kw in cluster_keywords:
        keyword_tokens.update(extract_informative_tokens(kw))

    combined_cluster_tokens = cluster_member_tokens.union(keyword_tokens)

    # Rule 3: Check for very short or uninformative feedback
    if len(combined_cluster_tokens) < cfg.min_informative_tokens:
        return ApplicabilityAssessment(
            status=STATUS_INSUFFICIENT_INFORMATION,
            explanation=(
                f"Cluster {cluster_id} feedback lacks sufficient distinct informative tokens "
                f"(found {len(combined_cluster_tokens)}, minimum required {cfg.min_informative_tokens}). "
                "The records consist primarily of stopwords, short fragments, or punctuation."
            ),
            relevant_context_concepts=[],
            supporting_feedback_ids=[r.id for r in member_records[: cfg.max_evidence_quotes]],
            supporting_quotes=[r.original_text for r in member_records[: cfg.max_evidence_quotes]],
            contradictory_evidence=[],
            signals_used={
                "informative_token_count": len(combined_cluster_tokens),
                "cluster_size": len(member_records),
            },
            missing_information=["Descriptive feedback content with domain-specific terminology"],
            next_validation_step="Collect more detailed feedback from users before re-evaluating this cluster.",
            assessment_method="deterministic_rule_v1",
            context_version=ctx_version,
            run_id=run_id,
        )

    # Context concept parsing
    context_sections = {
        "Product Name": extract_informative_tokens(product_context.name),
        "Description": extract_informative_tokens(product_context.description),
        "Goals / Concerns": extract_informative_tokens(product_context.goals_or_concerns),
        "Target Users": extract_informative_tokens(product_context.target_users),
    }

    matched_concepts: list[str] = []
    overlapping_tokens: set[str] = set()
    signals: dict[str, Any] = {
        "cluster_id": cluster_id,
        "cluster_size": len(member_records),
        "keyword_count": len(cluster_keywords),
    }

    for section_name, sec_tokens in context_sections.items():
        if not sec_tokens:
            continue
        kw_overlap = keyword_tokens.intersection(sec_tokens)
        mem_overlap = cluster_member_tokens.intersection(sec_tokens)
        sec_overlap = kw_overlap.union(mem_overlap)
        if sec_overlap:
            matched_concepts.append(f"{section_name} ({', '.join(sorted(sec_overlap))})")
            overlapping_tokens.update(sec_overlap)

    signals["overlapping_tokens"] = sorted(list(overlapping_tokens))
    signals["overlap_count"] = len(overlapping_tokens)

    # Detect contradictions or noisy cluster membership
    contradictions = _detect_contradictions_and_noise(member_records)

    # Build traceable supporting quotes (strictly from real records)
    supporting_ids: list[str] = []
    supporting_quotes: list[str] = []

    # Priority to representative quote
    if representative_id and representative_quote:
        supporting_ids.append(representative_id)
        supporting_quotes.append(representative_quote)

    for rec in member_records:
        if rec.id not in supporting_ids and len(supporting_ids) < cfg.max_evidence_quotes:
            supporting_ids.append(rec.id)
            supporting_quotes.append(rec.original_text)

    # Rule 4 & 5: Determine status based on overlap strength and internal consistency
    if len(overlapping_tokens) >= cfg.strong_overlap_tokens and not contradictions:
        status = STATUS_LIKELY_RELEVANT
        explanation = (
            f"Direct lexical overlap detected with documented product context: {', '.join(sorted(overlapping_tokens))}. "
            f"The cluster feedback correlates with explicitly stated concepts in {', '.join(matched_concepts)}. "
            "Plausible relevance to planned features or known product concerns."
        )
        missing_info: list[str] = []
        next_step = (
            f"Review supporting feedback ({', '.join(supporting_ids[:2])}) with the engineering/design team "
            "responsible for the matched product capabilities."
        )
    elif len(overlapping_tokens) >= cfg.min_overlap_tokens:
        status = STATUS_NEEDS_VALIDATION
        missing_info = [
            "Detailed workflow specification to distinguish intended behavior from defect",
            "Human confirmation on whether terminology matches intended product terminology",
        ]
        if contradictions:
            explanation = (
                f"Partial overlap detected ({', '.join(sorted(overlapping_tokens))}), but cluster contains "
                "contradictory or noisy feedback signals. Member records may group disparate user sentiments."
            )
            next_step = "Inspect individual feedback records to separate positive testimonials from bug reports."
        else:
            explanation = (
                f"Moderate or partial lexical overlap detected with {', '.join(matched_concepts)}. "
                "Feedback shares terminology with context, but the specific user issue is not explicitly detailed."
            )
            next_step = "Conduct a manual review of member records to determine whether this reflects a known feature gap."
    else:
        status = STATUS_NEEDS_VALIDATION
        missing_info = [
            "Documented product context covering this specific feature or user workflow",
            "Product roadmap reference to determine if this is an emerging or out-of-scope request",
        ]
        explanation = (
            "No direct terminology overlap detected between cluster feedback and approved product context. "
            "This theme may represent an emerging user concern, an undocumented workflow, or out-of-scope feedback."
        )
        next_step = (
            "Validate with the product owner whether this theme falls within current product scope "
            "or represents an unaddressed customer use case."
        )

    return ApplicabilityAssessment(
        status=status,
        explanation=explanation,
        relevant_context_concepts=matched_concepts,
        supporting_feedback_ids=supporting_ids,
        supporting_quotes=supporting_quotes,
        contradictory_evidence=contradictions,
        signals_used=signals,
        missing_information=missing_info,
        next_validation_step=next_step,
        assessment_method="deterministic_rule_v1",
        context_version=ctx_version,
        run_id=run_id,
    )
