"""Unit tests for feedback data processing, validation, and normalization."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import (
    FeedbackRecord,
    ValidationReport,
    load_feedback_csv,
    load_feedback_from_raw_text,
    normalize_whitespace,
    validate_and_prepare_feedback,
)


def test_normalize_whitespace():
    """Verify that multi-line spaces, tabs, and trailing spaces are cleanly collapsed."""
    raw = "  Hello \t\n world!   This   is   a test. \n"
    assert normalize_whitespace(raw) == "Hello world! This is a test."
    assert normalize_whitespace("") == ""
    assert normalize_whitespace(None) == ""


def test_empty_dataframe():
    """Verify empty dataframe returns zero records with clear warning and no crash."""
    df = pd.DataFrame()
    report = validate_and_prepare_feedback(df)

    assert report.total_input_rows == 0
    assert report.valid_records_count == 0
    assert report.is_usable_for_analysis is False
    assert any("empty" in w.lower() for w in report.warnings)


def test_missing_text_column():
    """Verify missing text column produces clear warning and records invalid rows."""
    df = pd.DataFrame({"number": [1, 2, 3], "rating": [5, 4, 3]})
    report = validate_and_prepare_feedback(df, text_column="feedback_text")

    assert report.valid_records_count == 0
    assert report.invalid_records_count == 3
    assert any("text column" in w.lower() for w in report.warnings)


def test_blank_and_null_texts_handled_safely():
    """Verify blank, whitespace-only, and null values do not crash and are marked invalid."""
    df = pd.DataFrame({
        "feedback_text": ["Legitimate feedback.", "   \n\t  ", None, "Another valid comment."]
    })
    report = validate_and_prepare_feedback(df)

    assert report.total_input_rows == 4
    assert report.valid_records_count == 2
    assert report.invalid_records_count == 2
    assert len(report.records) == 2
    assert report.records[0].cleaned_text == "Legitimate feedback."
    assert report.records[1].cleaned_text == "Another valid comment."


def test_original_text_strictly_preserved():
    """Verify original text is unmodified while cleaned text is normalized."""
    raw_text = "   App crashed with ERROR: #1234!!   \n\n Please fix.  "
    df = pd.DataFrame({"feedback_text": [raw_text]})
    report = validate_and_prepare_feedback(df)

    assert report.valid_records_count == 1
    rec = report.records[0]
    assert rec.original_text == raw_text
    assert rec.cleaned_text == "App crashed with ERROR: #1234!! Please fix."


def test_duplicate_flagging_without_silent_dropping():
    """Verify duplicate records are detected, linked to first occurrence, and NOT dropped."""
    df = pd.DataFrame({
        "id": ["FB1", "FB2", "FB3"],
        "feedback_text": [
            "Search takes too long to load.",
            "Different issue.",
            "  search takes too long to load.  ",  # duplicate of FB1
        ]
    })
    report = validate_and_prepare_feedback(df)

    assert report.total_input_rows == 3
    assert report.valid_records_count == 3
    assert report.duplicate_records_count == 1
    assert len(report.records) == 3

    # Check first record is not duplicate
    assert report.records[0].is_duplicate is False
    assert report.records[0].duplicate_of_id is None

    # Check third record is flagged as duplicate of FB1
    assert report.records[2].is_duplicate is True
    assert report.records[2].duplicate_of_id == "FB1"


def test_small_dataset_warning():
    """Verify very small datasets generate an informative warning."""
    df = pd.DataFrame({"feedback_text": ["Only one comment."]})
    report = validate_and_prepare_feedback(df)

    assert report.valid_records_count == 1
    assert report.is_usable_for_analysis is False
    assert any("small" in w.lower() for w in report.warnings)


def test_load_sample_feedback_csv():
    """Verify reading the included synthetic sample feedback CSV works end-to-end."""
    sample_path = Path("data/sample_feedback.csv")
    assert sample_path.exists(), "data/sample_feedback.csv must exist"

    report = load_feedback_csv(sample_path)
    assert report.total_input_rows == 20
    assert report.valid_records_count == 20
    assert report.invalid_records_count == 0
    assert report.duplicate_records_count >= 2
    assert report.is_usable_for_analysis is True


def test_pasted_multiline_feedback():
    """Verify multiline text is split line-by-line, blank lines skipped, stable IDs generated."""
    pasted = """
    First bug report about slow loading.

    Second complaint about dark mode missing.

    First bug report about slow loading.
    Third item with praise.
    """
    report = load_feedback_from_raw_text(pasted, source_name="user_paste")

    assert report.valid_records_count == 4
    assert report.duplicate_records_count == 1
    assert report.records[0].id == "FB0001"
    assert report.records[1].id == "FB0002"
    assert report.records[2].id == "FB0003"
    assert report.records[3].id == "FB0004"
    assert report.records[0].cleaned_text == "First bug report about slow loading."
    assert report.records[2].is_duplicate is True
    assert report.records[2].duplicate_of_id == "FB0001"


def test_custom_csv_column_selection():
    """Verify arbitrary custom column names can be mapped to text and ID."""
    df = pd.DataFrame({
        "ticket_num": ["T-101", "T-102"],
        "user_rant": ["The export button does nothing.", "Cannot reset password."],
        "internal_code": [99, 100],
    })
    report = validate_and_prepare_feedback(df, text_column="user_rant", id_column="ticket_num")

    assert report.valid_records_count == 2
    assert report.records[0].id == "T-101"
    assert report.records[0].cleaned_text == "The export button does nothing."
    assert report.records[1].id == "T-102"
    assert report.records[1].cleaned_text == "Cannot reset password."
    assert report.records[0].metadata.get("internal_code") == 99


def test_evaluation_label_column_isolation():
    """Verify that columns containing evaluation labels are never auto-selected as text column."""
    df = pd.DataFrame({
        "ground_truth_theme": ["bug_export", "auth_timeout"],
        "notes": ["Export fails on Win11", "Session dies in 2m"],
    })
    # When no text_column is passed, fallback should pick 'notes', NOT 'ground_truth_theme'
    report = validate_and_prepare_feedback(df)

    assert report.valid_records_count == 2
    assert report.records[0].cleaned_text == "Export fails on Win11"
    assert report.records[1].cleaned_text == "Session dies in 2m"


def test_empty_pasted_text():
    """Verify whitespace-only or empty pasted text produces clear warning and 0 records."""
    report = load_feedback_from_raw_text("   \n\t   \n  ")
    assert report.valid_records_count == 0
    assert report.total_input_rows == 0
    assert any("empty" in w.lower() for w in report.warnings)
