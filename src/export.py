"""Export module for feedback analysis, cluster insights, and model metrics.

Generates downloadable CSV spreadsheets and structured Markdown reports
grounded in real computed results. Preserves original feedback text, record IDs,
and full decision support traceability for Phase 5.
"""

from __future__ import annotations

import io
from typing import Sequence

import pandas as pd

from src.clustering import ModelResult
from src.context_ingestion import ProductContext
from src.data_processing import ValidationReport
from src.evaluation import EvaluationReport
from src.explanations import (
    get_fair_comparison_caveat,
    get_sentence_transformer_explanation,
    get_tfidf_explanation,
)
from src.insights import InsightsReport
from src.recommendations import InvestigationRecommendation


def export_feedback_records_csv(
    validation_report: ValidationReport,
    tfidf_res: ModelResult | None = None,
    embedding_res: ModelResult | None = None,
) -> str:
    """Exports validated feedback records with cluster assignments and duplicate references."""
    tfidf_map = tfidf_res.get_assignment_map() if tfidf_res else {}
    emb_map = embedding_res.get_assignment_map() if embedding_res else {}

    rows = []
    for r in validation_report.records:
        rows.append({
            "record_id": r.id,
            "original_feedback": r.original_text,
            "cleaned_feedback": r.cleaned_text,
            "source": r.source,
            "rating": r.rating if r.rating is not None else "",
            "is_duplicate": r.is_duplicate,
            "duplicate_of_id": r.duplicate_of_id or "",
            "tfidf_cluster": tfidf_map.get(r.id, "N/A"),
            "embedding_cluster": emb_map.get(r.id, "N/A"),
        })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_cluster_summaries_csv(insights_report: InsightsReport) -> str:
    """Exports cluster summary statistics, applicability assessments, and investigation actions."""
    rows = []
    for c in insights_report.cluster_insights:
        app_status = c.applicability.status if c.applicability else "N/A"
        app_concepts = (
            "; ".join(c.applicability.relevant_context_concepts)
            if (c.applicability and c.applicability.relevant_context_concepts)
            else ""
        )
        rec_action = c.recommendation.suggested_action if c.recommendation else ""

        rows.append({
            "cluster_id": c.cluster_id,
            "label": c.label,
            "member_count": c.member_count,
            "unique_count": c.unique_member_count,
            "duplicate_count": c.duplicate_count,
            "percentage_of_total": c.percentage_of_total,
            "keywords": "; ".join(c.keywords),
            "keywords_derivation": c.keywords_derivation,
            "representative_id": c.representative_feedback_id,
            "representative_quote": c.representative_quote,
            "evidence_strength": c.evidence_strength,
            "context_relevance": c.product_context_relevance,
            "applicability_status": app_status,
            "relevant_concepts": app_concepts,
            "investigation_action": rec_action,
        })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_investigation_recommendations_csv(
    recommendations: Sequence[InvestigationRecommendation],
) -> str:
    """Exports structured, evidence-grounded investigation recommendations as CSV.

    Preserves exact verbatim quotes, traceable feedback IDs, duplicate counts,
    and human review reminders. Correctly escapes commas, quotes, and newlines.
    """
    rows = []
    for idx, r in enumerate(recommendations, 1):
        rows.append({
            "queue_rank": idx,
            "title": r.title,
            "cluster_id": r.cluster_id,
            "model_name": r.model_name,
            "applicability_status": r.applicability_status,
            "suggested_action": r.suggested_action,
            "rationale": r.rationale,
            "unique_mentions_count": r.unique_mentions_count,
            "duplicate_count": r.duplicate_count,
            "recurrence_score": r.recurrence_score,
            "supporting_feedback_ids": "; ".join(r.supporting_feedback_ids),
            "supporting_quotes": " | ".join(f'"{q}"' for q in r.supporting_quotes),
            "contradictory_evidence": " | ".join(r.contradictory_evidence) if r.contradictory_evidence else "None",
            "limitations": " | ".join(r.limitations_and_missing_info) if r.limitations_and_missing_info else "None",
            "human_review_reminder": r.human_review_reminder,
        })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_model_evaluation_csv(
    tfidf_eval: EvaluationReport | None,
    embedding_eval: EvaluationReport | None,
) -> str:
    """Exports model evaluation metrics and validity explanations as CSV."""
    rows = []
    reports = [r for r in [tfidf_eval, embedding_eval] if r is not None]

    for ev in reports:
        rows.append({
            "method": ev.method,
            "model_identifier": ev.model_identifier,
            "representation": ev.representation,
            "distance_metric": ev.distance_metric,
            "n_samples": ev.n_samples,
            "n_clusters": ev.n_clusters,
            "silhouette": ev.silhouette if ev.silhouette is not None else "Not available",
            "silhouette_reason": ev.metric_reasons.get("silhouette", ""),
            "davies_bouldin": ev.davies_bouldin if ev.davies_bouldin is not None else "Not available",
            "davies_bouldin_reason": ev.metric_reasons.get("davies_bouldin", ""),
            "calinski_harabasz": ev.calinski_harabasz if ev.calinski_harabasz is not None else "Not available",
            "calinski_harabasz_reason": ev.metric_reasons.get("calinski_harabasz", ""),
            "adjusted_rand_index": ev.adjusted_rand_index if ev.adjusted_rand_index is not None else "Not available",
            "ari_reason": ev.metric_reasons.get("adjusted_rand_index", ""),
            "normalized_mutual_info": ev.normalized_mutual_info if ev.normalized_mutual_info is not None else "Not available",
            "cluster_size_entropy": ev.cluster_size_entropy if ev.cluster_size_entropy is not None else "N/A",
        })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_full_markdown_report(
    insights_report: InsightsReport,
    evaluation_report: EvaluationReport | None = None,
    product_context: ProductContext | None = None,
) -> str:
    """Generates a complete, evidence-grounded Markdown report with Phase 5 decision support."""
    lines = [
        f"# AI Product Intelligence & Feedback Analysis Report",
        f"**Generated by:** AI Product Intelligence & Feedback Analyzer (Phase 5 Decision Support)",
        f"**Model Evaluated:** {insights_report.model_name} (`{insights_report.model_identifier}`)",
        f"",
        f"---",
        f"",
        f"## 1. Product Context",
    ]

    if product_context and product_context.is_approved:
        lines.extend([
            f"- **Product Name:** {product_context.name}",
            f"- **Description:** {product_context.description}",
            f"- **Target Users:** {product_context.target_users or 'Not specified'}",
            f"- **Goals/Known Concerns:** {product_context.goals_or_concerns or 'Not specified'}",
            f"- **Context Version:** {getattr(product_context, 'version', 1)}",
            f"- **Context Origin:** {product_context.source_type} ({product_context.source_url or 'Direct input'})",
            f"- **Approval Status:** Approved by reviewer",
        ])
    else:
        lines.append("*No approved product context provided for this analysis session.*")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Dataset Overview",
        f"- **Total Feedback Records:** {insights_report.total_feedback_count}",
        f"- **Unique Feedback Items:** {insights_report.total_unique_count}",
        f"- **Flagged Duplicate Comments:** {insights_report.total_duplicate_count}",
        f"- **Discovered Theme Groups:** {len(insights_report.cluster_insights)}",
        f"",
        f"---",
        f"",
        f"## 3. Discovered Themes & Evidence Findings",
    ])

    for c in insights_report.cluster_insights:
        kw_str = ", ".join(c.keywords) if c.keywords else "None"
        app_status = c.applicability.status if c.applicability else "Insufficient information"
        app_expl = c.applicability.explanation if c.applicability else c.product_context_relevance
        next_step = c.applicability.next_validation_step if c.applicability else "Review member feedback."

        lines.extend([
            f"### {c.label}: {kw_str}",
            f"- **Volume:** {c.member_count} records ({c.percentage_of_total}% of dataset) | **Unique:** {c.unique_member_count} | **Duplicates:** {c.duplicate_count}",
            f"- **Recurrence Strength:** {c.evidence_strength}",
            f"- **Applicability Status:** `{app_status}`",
            f"- **Applicability Explanation:** {app_expl}",
            f"- **Recommended Next Validation Step:** {next_step}",
            f"- **Keywords Derivation:** *{c.keywords_derivation}*",
            f"- **Representative Quote ({c.representative_feedback_id}):**",
            f"  > *\"{c.representative_quote}\"*",
            f"- **Supporting Feedback Samples:**",
        ])
        for ex in c.supporting_examples[:4]:
            dup_tag = f" [Duplicate of {ex['duplicate_of_id']}]" if ex["is_duplicate"] else ""
            lines.append(f"  - `{ex['id']}`{dup_tag}: {ex['text']}")

        if c.applicability and c.applicability.contradictory_evidence:
            lines.append(f"- **Contradictory / Noise Signals:**")
            for ce in c.applicability.contradictory_evidence:
                lines.append(f"  - ⚠️ {ce}")

        lines.append("")

    lines.extend([
        f"---",
        f"",
        f"## 4. Theme Prioritization & Frequency-Based Investigation Queue",
        f"| Rank | Recommendation Title | Cluster | Applicability | Unique Volume | Duplicates | Next Suggested Action |",
        f"|---|---|---|---|---|---|---|",
    ])

    if insights_report.investigation_queue:
        for idx, rec in enumerate(insights_report.investigation_queue, 1):
            lines.append(
                f"| {idx} | {rec.title} | Cluster {rec.cluster_id} | `{rec.applicability_status}` | "
                f"{rec.unique_mentions_count} | {rec.duplicate_count} | {rec.suggested_action} |"
            )
    else:
        for c_id in insights_report.prioritized_clusters:
            c = insights_report.get_cluster(c_id)
            if c:
                kw = ", ".join(c.keywords[:3])
                lines.append(f"| - | Explore Cluster {c.cluster_id} | Cluster {c.cluster_id} | N/A | {c.unique_member_count} | {c.duplicate_count} | Inspect feedback for '{kw}' |")

    lines.extend([
        f"",
        f"*Note on Prioritization:* The investigation queue is ordered strictly by unique feedback volume (frequency-based queue). "
        f"It does NOT establish defect severity, business priority, financial impact, or customer importance.",
        f"",
        f"---",
        f"",
        f"## 5. Methodological Limitations & Integrity Safeguards",
    ])
    for lim in insights_report.limitations:
        lines.append(f"- ⚠️ {lim}")

    if evaluation_report:
        lines.extend([
            f"",
            f"### Model Clustering Preconditions & Metrics",
            f"- **Representation Space:** `{evaluation_report.representation}` (Distance: `{evaluation_report.distance_metric}`)",
            f"- **Silhouette Score:** {evaluation_report.silhouette if evaluation_report.silhouette is not None else 'N/A'}",
            f"- **Davies-Bouldin Index:** {evaluation_report.davies_bouldin if evaluation_report.davies_bouldin is not None else 'N/A'}",
            f"- **Adjusted Rand Index:** {evaluation_report.adjusted_rand_index if evaluation_report.adjusted_rand_index is not None else 'N/A'}",
            f"- **Metric Notes:** {', '.join(f'{k}: {v}' for k, v in evaluation_report.metric_reasons.items())}",
        ])

    # Explainability & Caveats section
    fair_caveat = get_fair_comparison_caveat()
    lines.extend([
        f"",
        f"---",
        f"",
        f"## 6. Algorithm Explainability & Cross-Representation Caveat",
        f"### {fair_caveat['title']}",
        f"{fair_caveat['core_principle']}",
        f"",
        f"**Why Vector Spaces Differ:**",
        f"{fair_caveat['why_spaces_differ']}",
        f"",
        f"**Four Evaluation Pillars:**",
        f"{fair_caveat['four_pillars_to_distinguish']}",
        f"",
        f"> **Reminder:** {fair_caveat['winner_declaration_warning']}",
    ])

    return "\n".join(lines)
