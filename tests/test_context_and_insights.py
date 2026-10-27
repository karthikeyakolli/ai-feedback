"""Unit and integration tests for Product Context Ingestion, SSRF safety, Insights, and Exports."""

import socket
from unittest.mock import MagicMock, patch

import pytest

from src.clustering import run_tfidf_clustering
from src.context_ingestion import (
    ExtractionResult,
    ProductContext,
    extract_webpage_content,
    is_ip_allowed,
    validate_url_safety,
)
from src.data_processing import load_feedback_csv
from src.evaluation import evaluate_model_result
from src.export import (
    export_cluster_summaries_csv,
    export_feedback_records_csv,
    export_full_markdown_report,
    export_model_evaluation_csv,
)
from src.insights import generate_insights_report


# =========================================================================
# Part 1: Product Context Ingestion & SSRF Safety Tests
# =========================================================================

def test_ip_allowlist_blocking():
    """Verify loopback, private subnets, link-local, and broadcast IPs are blocked."""
    # Loopback
    allowed, _ = is_ip_allowed("127.0.0.1")
    assert allowed is False
    allowed, _ = is_ip_allowed("127.0.0.2")
    assert allowed is False

    # Private ranges
    assert is_ip_allowed("10.0.0.1")[0] is False
    assert is_ip_allowed("192.168.1.1")[0] is False
    assert is_ip_allowed("172.16.0.1")[0] is False

    # Link-local / metadata endpoint
    assert is_ip_allowed("169.254.169.254")[0] is False

    # IPv6 loopback
    assert is_ip_allowed("::1")[0] is False

    # Valid public IP (Cloudflare DNS 1.1.1.1)
    allowed, _ = is_ip_allowed("1.1.1.1")
    assert allowed is True


def test_url_safety_scheme_and_hostname():
    """Verify only http/https schemes and public hostnames pass validation."""
    # Invalid schemes
    assert validate_url_safety("ftp://example.com")[0] is False
    assert validate_url_safety("file:///etc/passwd")[0] is False
    assert validate_url_safety("gopher://example.com")[0] is False

    # Localhost aliases
    assert validate_url_safety("http://localhost")[0] is False
    assert validate_url_safety("http://127.0.0.1")[0] is False
    assert validate_url_safety("http://0.0.0.0")[0] is False

    # Malformed URL
    assert validate_url_safety("not a url")[0] is False


def test_url_extraction_blocks_private_dns():
    """Verify URL extraction rejects destinations resolving to private IP space."""
    with patch("socket.getaddrinfo") as mock_dns:
        # Simulate DNS returning a private IP
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.50", 80))
        ]
        result = extract_webpage_content("http://internal-corp-service.com")
        assert result.status == "failed"
        assert "restricted IP" in result.message


def test_url_extraction_timeout_and_error_handling():
    """Verify requests timeout and network failures produce clear failure results without crashing."""
    import requests

    with patch("src.context_ingestion.validate_url_safety", return_value=(True, "Safe", "93.184.216.34")):
        with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
            result = extract_webpage_content("https://example.com/slow-page")
            assert result.status == "failed"
            assert "timed out" in result.message.lower()


def test_url_extraction_unsupported_content_type():
    """Verify non-HTML/text responses (e.g. PDF or binary) are rejected safely."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.is_redirect = False
    mock_resp.headers = {"Content-Type": "application/pdf"}

    with patch("src.context_ingestion.validate_url_safety", return_value=(True, "Safe", "93.184.216.34")):
        with patch("requests.get", return_value=mock_resp):
            result = extract_webpage_content("https://example.com/manual.pdf")
            assert result.status == "failed"
            assert "unsupported content-type" in result.message.lower()


def test_url_extraction_success_mock():
    """Verify successful HTML extraction populates title and text correctly."""
    mock_html = (
        "<html><head><title>Test App Product Page</title></head>"
        "<body><main><p>Test App is a fast, modern platform for tracking customer feedback "
        "and resolving export crashes seamlessly.</p></main></body></html>"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.is_redirect = False
    mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_resp.encoding = "utf-8"
    mock_resp.iter_content = MagicMock(return_value=[mock_html.encode("utf-8")])

    with patch("src.context_ingestion.validate_url_safety", return_value=(True, "Safe", "93.184.216.34")):
        with patch("requests.get", return_value=mock_resp):
            result = extract_webpage_content("https://example.com/product")
            assert result.status in {"success", "partial"}
            assert result.title == "Test App Product Page"
            assert "tracking customer feedback" in result.extracted_text


def test_product_context_model_and_approval():
    """Verify ProductContext attributes, approval toggle, and summary generation."""
    ctx = ProductContext(
        name="FeedbackHub",
        description="A feedback tracking tool.",
        target_users="Engineers",
        goals_or_concerns="Reduce PDF export crash reports.",
        source_type="webpage_url",
        source_url="https://example.com",
        is_approved=False,
    )
    assert ctx.is_approved is False
    assert "FeedbackHub" in ctx.get_summary_text()
    assert "Reduce PDF export crash reports" in ctx.get_summary_text()

    # Simulate reviewer approval
    ctx.is_approved = True
    assert ctx.is_approved is True


# =========================================================================
# Part 2: Evidence-Grounded Insights Tests
# =========================================================================

@pytest.fixture
def sample_setup():
    """Loads sample feedback and runs TF-IDF baseline."""
    val_report = load_feedback_csv("data/sample_feedback.csv")
    model_res = run_tfidf_clustering(val_report.records, n_clusters=4, random_state=42)
    return val_report, model_res


def test_insights_generation_and_evidence_traceability(sample_setup):
    """Verify insights report matches cluster assignments and retains exact quotes."""
    val_report, model_res = sample_setup
    ctx = ProductContext(
        name="FeedbackHub",
        description="Tool for reporting software issues and managing document exports.",
        goals_or_concerns="Prevent application freezing on PDF export.",
        is_approved=True,
    )

    report = generate_insights_report(model_res, val_report, product_context=ctx)

    assert report.total_feedback_count == 20
    assert len(report.cluster_insights) == 4
    assert len(report.prioritized_clusters) == 4

    # Check each cluster insight
    total_assigned_members = 0
    feedback_dict = {r.id: r.original_text for r in val_report.records}

    for c in report.cluster_insights:
        total_assigned_members += c.member_count
        assert c.member_count == c.unique_member_count + c.duplicate_count
        assert c.percentage_of_total > 0

        # Representative quote must be real feedback
        assert c.representative_quote in feedback_dict.values()
        assert c.representative_feedback_id in feedback_dict

        # Supporting examples must have real IDs
        for ex in c.supporting_examples:
            assert ex["id"] in feedback_dict
            assert ex["text"] == feedback_dict[ex["id"]]

    assert total_assigned_members == 20


def test_context_relevance_cautious_correlation(sample_setup):
    """Verify context relevance identifies matching keywords without overclaiming."""
    val_report, model_res = sample_setup

    # Case 1: Approved context with overlapping words ("export")
    ctx_approved = ProductContext(
        name="DocExporter",
        description="An application that exports PDF files.",
        is_approved=True,
    )
    report = generate_insights_report(model_res, val_report, product_context=ctx_approved)
    relevance_texts = [c.product_context_relevance for c in report.cluster_insights]
    # At least one cluster mentioning export should detect the domain overlap
    assert any("export" in t.lower() or "correlation" in t.lower() for t in relevance_texts)

    # Case 2: Unapproved context - must withhold interpretation
    ctx_unapproved = ProductContext(
        name="DocExporter",
        description="An application that exports PDF files.",
        is_approved=False,
    )
    report_unapproved = generate_insights_report(model_res, val_report, product_context=ctx_unapproved)
    for c in report_unapproved.cluster_insights:
        assert "no approved product context" in c.product_context_relevance.lower()


# =========================================================================
# Part 3: Export Functionality Tests
# =========================================================================

def test_export_feedback_records_csv(sample_setup):
    """Verify feedback CSV export contains all records and headers."""
    val_report, model_res = sample_setup
    csv_str = export_feedback_records_csv(val_report, tfidf_res=model_res)

    assert "record_id,original_feedback,cleaned_feedback" in csv_str
    assert "FB01" in csv_str
    assert "FB20" in csv_str
    # 20 items + 1 header = 21 lines minimum
    assert len(csv_str.strip().splitlines()) >= 21


def test_export_cluster_summaries_csv(sample_setup):
    """Verify cluster summaries CSV export contains accurate metrics."""
    val_report, model_res = sample_setup
    report = generate_insights_report(model_res, val_report)
    csv_str = export_cluster_summaries_csv(report)

    assert "cluster_id,label,member_count,unique_count" in csv_str
    assert "Cluster 0" in csv_str
    assert len(csv_str.strip().splitlines()) == 5  # 1 header + 4 clusters


def test_export_model_evaluation_csv(sample_setup):
    """Verify model evaluation CSV export contains preconditions and reasons."""
    _, model_res = sample_setup
    eval_report = evaluate_model_result(model_res)
    csv_str = export_model_evaluation_csv(tfidf_eval=eval_report, embedding_eval=None)

    assert "method,model_identifier,representation" in csv_str
    assert "TF-IDF + K-Means" in csv_str
    assert "silhouette_reason" in csv_str


def test_export_full_markdown_report(sample_setup):
    """Verify markdown export generates structured report with real quotes."""
    val_report, model_res = sample_setup
    eval_report = evaluate_model_result(model_res)
    ctx = ProductContext(
        name="FeedbackHub",
        description="Product intelligence system.",
        is_approved=True,
    )
    report = generate_insights_report(model_res, val_report, product_context=ctx)
    md_str = export_full_markdown_report(report, eval_report, ctx)

    assert "# AI Product Intelligence & Feedback Analysis Report" in md_str
    assert "FeedbackHub" in md_str
    assert "## 3. Discovered Themes & Evidence Findings" in md_str
    assert "## 4. Theme Prioritization" in md_str
    assert "## 5. Methodological Limitations" in md_str
