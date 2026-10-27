"""Tests for Feature A (Applicability Assessment) and Feature B (Investigation Recommendations).

Verifies deterministic rule evaluation, three strict statuses, verbatim quote traceability,
duplicate accounting, deduplication, and frequency-based investigation queue ordering.
"""

from __future__ import annotations

import pytest

from src.applicability import (
    STATUS_INSUFFICIENT_INFORMATION,
    STATUS_LIKELY_RELEVANT,
    STATUS_NEEDS_VALIDATION,
    ApplicabilityAssessment,
    ApplicabilityRulesConfig,
    assess_cluster_applicability,
    extract_informative_tokens,
)
from src.clustering import run_tfidf_clustering
from src.context_ingestion import ProductContext
from src.data_processing import FeedbackRecord, ValidationReport, load_feedback_csv
from src.export import (
    export_cluster_summaries_csv,
    export_full_markdown_report,
    export_investigation_recommendations_csv,
)
from src.insights import generate_insights_report
from src.recommendations import (
    InvestigationRecommendation,
    build_investigation_queue,
    generate_cluster_recommendation,
)


@pytest.fixture
def sample_context() -> ProductContext:
    """Returns an approved test product context."""
    return ProductContext(
        name="DocuFlow PDF",
        description="Enterprise platform for document processing, OCR scanning, and PDF export workflows.",
        target_users="Accountants, Paralegals, Knowledge Workers",
        goals_or_concerns="Prevent application freezing on large PDF export and reduce timeout errors.",
        is_approved=True,
        version=2,
    )


def test_informative_tokens_extraction():
    """Verify stopword removal and token extraction."""
    text = "The application crashes repeatedly during PDF export!"
    tokens = extract_informative_tokens(text)
    assert "pdf" in tokens
    assert "export" in tokens
    assert "crashes" in tokens
    assert "the" not in tokens
    assert "during" not in tokens


def test_applicability_missing_or_unapproved_context():
    """Verify missing or unapproved context strictly returns 'Insufficient information'."""
    records = [
        FeedbackRecord(id="FB-01", original_text="PDF export crashes.", cleaned_text="pdf export crashes"),
    ]

    # Case 1: None context
    res_none = assess_cluster_applicability(
        cluster_id=0,
        cluster_keywords=["pdf", "export"],
        representative_quote="PDF export crashes.",
        representative_id="FB-01",
        member_records=records,
        product_context=None,
    )
    assert res_none.status == STATUS_INSUFFICIENT_INFORMATION
    assert "no approved product context" in res_none.explanation.lower()
    assert "Approved product context" in res_none.missing_information[0]

    # Case 2: Unapproved context
    unapproved_ctx = ProductContext(name="DocuFlow", description="PDF tool", is_approved=False)
    res_unapp = assess_cluster_applicability(
        cluster_id=0,
        cluster_keywords=["pdf", "export"],
        representative_quote="PDF export crashes.",
        representative_id="FB-01",
        member_records=records,
        product_context=unapproved_ctx,
    )
    assert res_unapp.status == STATUS_INSUFFICIENT_INFORMATION
    assert "no approved product context" in res_unapp.explanation.lower()


def test_applicability_empty_cluster(sample_context: ProductContext):
    """Verify empty cluster records return 'Insufficient information'."""
    res = assess_cluster_applicability(
        cluster_id=99,
        cluster_keywords=[],
        representative_quote="",
        representative_id="",
        member_records=[],
        product_context=sample_context,
    )
    assert res.status == STATUS_INSUFFICIENT_INFORMATION
    assert "zero member feedback" in res.explanation.lower()


def test_applicability_uninformative_or_stopword_feedback(sample_context: ProductContext):
    """Verify very short, punctuation, or stopword-only feedback returns 'Insufficient information'."""
    records = [
        FeedbackRecord(id="FB-01", original_text="it is what it is", cleaned_text="it is what it is"),
        FeedbackRecord(id="FB-02", original_text="the a an", cleaned_text="the a an"),
    ]
    res = assess_cluster_applicability(
        cluster_id=0,
        cluster_keywords=["is"],
        representative_quote="it is what it is",
        representative_id="FB-01",
        member_records=records,
        product_context=sample_context,
    )
    assert res.status == STATUS_INSUFFICIENT_INFORMATION
    assert "lacks sufficient distinct informative tokens" in res.explanation


def test_applicability_strongly_supported_overlap(sample_context: ProductContext):
    """Verify strong overlap with product description/goals returns 'Likely relevant'."""
    records = [
        FeedbackRecord(
            id="FB-10",
            original_text="The PDF export workflow constantly hangs on multi-page files.",
            cleaned_text="the pdf export workflow constantly hangs on multi page files",
        ),
        FeedbackRecord(
            id="FB-11",
            original_text="Large document export freezes the whole interface.",
            cleaned_text="large document export freezes the whole interface",
        ),
    ]
    res = assess_cluster_applicability(
        cluster_id=1,
        cluster_keywords=["pdf", "export", "freezes"],
        representative_quote="The PDF export workflow constantly hangs on multi-page files.",
        representative_id="FB-10",
        member_records=records,
        product_context=sample_context,
        run_id="run-test-01",
    )
    assert res.status == STATUS_LIKELY_RELEVANT
    assert "Direct lexical overlap detected" in res.explanation
    assert res.context_version == 2
    assert res.run_id == "run-test-01"
    assert "FB-10" in res.supporting_feedback_ids
    assert res.supporting_quotes[0] == "The PDF export workflow constantly hangs on multi-page files."
    assert len(res.relevant_context_concepts) >= 1


def test_applicability_partial_or_unrelated_returns_needs_validation(sample_context: ProductContext):
    """Verify feedback without overlap or with only partial overlap returns 'Needs validation'."""
    records = [
        FeedbackRecord(
            id="FB-20",
            original_text="Dark mode theme colors look too bright at night.",
            cleaned_text="dark mode theme colors look too bright at night",
        ),
        FeedbackRecord(
            id="FB-21",
            original_text="Night theme contrast needs improvement.",
            cleaned_text="night theme contrast needs improvement",
        ),
    ]
    res = assess_cluster_applicability(
        cluster_id=2,
        cluster_keywords=["dark", "theme", "night"],
        representative_quote="Dark mode theme colors look too bright at night.",
        representative_id="FB-20",
        member_records=records,
        product_context=sample_context,
    )
    assert res.status == STATUS_NEEDS_VALIDATION
    assert "No direct terminology overlap" in res.explanation
    assert "Validate with the product owner" in res.next_validation_step


def test_applicability_contradictory_evidence_detection(sample_context: ProductContext):
    """Verify mixed sentiment (great working well vs crash error) is detected as contradictory."""
    records = [
        FeedbackRecord(
            id="FB-30",
            original_text="The PDF export is great, love how fast and smooth it is!",
            cleaned_text="the pdf export is great love how fast and smooth it is",
        ),
        FeedbackRecord(
            id="FB-31",
            original_text="The PDF export is broken and slow, crashes with error code 500.",
            cleaned_text="the pdf export is broken and slow crashes with error code 500",
        ),
    ]
    res = assess_cluster_applicability(
        cluster_id=3,
        cluster_keywords=["pdf", "export"],
        representative_quote="The PDF export is broken and slow, crashes with error code 500.",
        representative_id="FB-31",
        member_records=records,
        product_context=sample_context,
    )
    assert len(res.contradictory_evidence) > 0
    assert "Mixed feedback polarity detected" in res.contradictory_evidence[0]
    assert res.status == STATUS_NEEDS_VALIDATION


def test_applicability_determinism(sample_context: ProductContext):
    """Verify identical inputs yield identical assessments."""
    records = [
        FeedbackRecord(id="FB-01", original_text="Export fails on big files.", cleaned_text="export fails on big files"),
        FeedbackRecord(id="FB-02", original_text="PDF exporter is broken.", cleaned_text="pdf exporter is broken"),
    ]
    res1 = assess_cluster_applicability(0, ["export", "pdf"], "Export fails on big files.", "FB-01", records, sample_context)
    res2 = assess_cluster_applicability(0, ["export", "pdf"], "Export fails on big files.", "FB-01", records, sample_context)

    assert res1.status == res2.status
    assert res1.explanation == res2.explanation
    assert res1.supporting_feedback_ids == res2.supporting_feedback_ids
    assert res1.supporting_quotes == res2.supporting_quotes


# =============================================================================
# RECOMMENDATIONS TESTS
# =============================================================================

def test_recommendation_evidence_grounding_and_traceability(sample_context: ProductContext):
    """Verify recommendations preserve real feedback IDs and verbatim quotes."""
    records = [
        FeedbackRecord(id="FB-100", original_text="Cannot export report to PDF format.", cleaned_text="cannot export report to pdf format"),
        FeedbackRecord(id="FB-101", original_text="PDF generation times out.", cleaned_text="pdf generation times out"),
    ]
    assessment = assess_cluster_applicability(
        cluster_id=0,
        cluster_keywords=["pdf", "export"],
        representative_quote="Cannot export report to PDF format.",
        representative_id="FB-100",
        member_records=records,
        product_context=sample_context,
    )

    rec = generate_cluster_recommendation(
        cluster_id=0,
        model_name="TF-IDF + K-Means",
        keywords=["pdf", "export"],
        representative_quote="Cannot export report to PDF format.",
        representative_id="FB-100",
        member_records=records,
        assessment=assessment,
        product_context=sample_context,
    )

    assert rec.cluster_id == 0
    assert "FB-100" in rec.supporting_feedback_ids
    assert "Cannot export report to PDF format." in rec.supporting_quotes
    assert rec.applicability_status == STATUS_LIKELY_RELEVANT
    assert "human validation is required" in rec.human_review_reminder.lower()
    assert rec.unique_mentions_count == 2
    assert rec.duplicate_count == 0


def test_recommendation_accounts_for_duplicates(sample_context: ProductContext):
    """Verify duplicate records are accounted for and not counted as independent corroboration."""
    records = [
        FeedbackRecord(id="FB-201", original_text="Login times out.", cleaned_text="login times out", is_duplicate=False),
        FeedbackRecord(id="FB-202", original_text="Login times out.", cleaned_text="login times out", is_duplicate=True, duplicate_of_id="FB-201"),
        FeedbackRecord(id="FB-203", original_text="Login times out.", cleaned_text="login times out", is_duplicate=True, duplicate_of_id="FB-201"),
    ]
    assessment = assess_cluster_applicability(
        cluster_id=1,
        cluster_keywords=["login", "timeout"],
        representative_quote="Login times out.",
        representative_id="FB-201",
        member_records=records,
        product_context=sample_context,
    )
    rec = generate_cluster_recommendation(
        cluster_id=1,
        model_name="TF-IDF + K-Means",
        keywords=["login", "timeout"],
        representative_quote="Login times out.",
        representative_id="FB-201",
        member_records=records,
        assessment=assessment,
        product_context=sample_context,
    )

    assert rec.unique_mentions_count == 1
    assert rec.duplicate_count == 2
    assert rec.recurrence_score == 1
    assert any("2 duplicate record(s) detected" in lim for lim in rec.limitations_and_missing_info)


def test_investigation_queue_ordering_and_deduplication():
    """Verify queue sorts by unique mentions count descending and deduplicates equivalent actions."""
    r1 = InvestigationRecommendation(
        title="Inspect Export",
        suggested_action="Inspect the PDF export workflow for memory spikes.",
        rationale="High volume.",
        cluster_id=0,
        model_name="TF-IDF + K-Means",
        supporting_feedback_ids=["FB-01"],
        supporting_quotes=["Export hangs"],
        contradictory_evidence=[],
        applicability_status=STATUS_LIKELY_RELEVANT,
        limitations_and_missing_info=[],
        unique_mentions_count=5,
        duplicate_count=0,
        recurrence_score=5,
    )
    r2 = InvestigationRecommendation(
        title="Inspect Login",
        suggested_action="Verify login session timeout.",
        rationale="Moderate volume.",
        cluster_id=1,
        model_name="TF-IDF + K-Means",
        supporting_feedback_ids=["FB-02"],
        supporting_quotes=["Login timeout"],
        contradictory_evidence=[],
        applicability_status=STATUS_NEEDS_VALIDATION,
        limitations_and_missing_info=[],
        unique_mentions_count=10,
        duplicate_count=1,
        recurrence_score=10,
    )
    r3_duplicate = InvestigationRecommendation(
        title="Inspect Export Dup",
        suggested_action="Inspect the PDF export workflow for memory spikes.",  # identical action
        rationale="High volume.",
        cluster_id=2,
        model_name="TF-IDF + K-Means",
        supporting_feedback_ids=["FB-03"],
        supporting_quotes=["Export hangs 2"],
        contradictory_evidence=[],
        applicability_status=STATUS_LIKELY_RELEVANT,
        limitations_and_missing_info=[],
        unique_mentions_count=3,
        duplicate_count=0,
        recurrence_score=3,
    )

    queue = build_investigation_queue([r1, r2, r3_duplicate])

    # Deduplication check: r3_duplicate should be deduplicated
    assert len(queue) == 2
    # Frequency ordering check: r2 (10 unique) comes before r1 (5 unique)
    assert queue[0].cluster_id == 1
    assert queue[1].cluster_id == 0


def test_export_investigation_recommendations_csv():
    """Verify CSV export formats quotes, commas, newlines, and headers correctly."""
    recs = [
        InvestigationRecommendation(
            title="Inspect Export",
            suggested_action="Inspect PDF export, retry count, and timeout limits.",
            rationale="Frequent mentions of 'freeze, crash'.",
            cluster_id=0,
            model_name="TF-IDF + K-Means",
            supporting_feedback_ids=["FB-01", "FB-02"],
            supporting_quotes=['Export fails, "unexpectedly" with error', 'Freeze on save\nnew line'],
            contradictory_evidence=[],
            applicability_status=STATUS_LIKELY_RELEVANT,
            limitations_and_missing_info=["Need server logs"],
            unique_mentions_count=4,
            duplicate_count=1,
            recurrence_score=4,
        )
    ]
    csv_str = export_investigation_recommendations_csv(recs)

    assert "queue_rank,title,cluster_id,model_name,applicability_status" in csv_str
    assert "Inspect Export" in csv_str
    assert "FB-01; FB-02" in csv_str
    assert "human validation is required" in csv_str.lower()
