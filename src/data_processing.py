"""Data processing module for feedback validation, normalization, and preparation.

Preserves original text, identifies duplicates, handles empty/malformed inputs,
and outputs typed data records for downstream clustering without discarding raw evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FeedbackRecord:
    """Represents a validated feedback item."""

    id: str
    original_text: str
    cleaned_text: str
    source: str = "direct_input"
    rating: float | None = None
    is_duplicate: bool = False
    duplicate_of_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Summary of data loading and validation quality."""

    total_input_rows: int
    valid_records_count: int
    invalid_records_count: int
    duplicate_records_count: int
    warnings: list[str] = field(default_factory=list)
    records: list[FeedbackRecord] = field(default_factory=list)
    invalid_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_usable_for_analysis(self) -> bool:
        """Determines if there are enough valid records for clustering analysis."""
        return self.valid_records_count >= 2


def normalize_whitespace(text: str) -> str:
    """Collapses consecutive whitespace and trims outer bounds."""
    if not isinstance(text, str):
        return ""
    # Normalize unicode whitespace and collapse multiple spaces/newlines
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def validate_and_prepare_feedback(
    df: pd.DataFrame,
    text_column: str | None = None,
    id_column: str | None = None,
    source_column: str | None = None,
    rating_column: str | None = None,
) -> ValidationReport:
    """Validates and processes a DataFrame of feedback records.

    Original text is strictly preserved. Duplicates are flagged rather than dropped.
    Missing or empty texts are safely captured in invalid_rows.
    """
    warnings: list[str] = []
    invalid_rows: list[dict[str, Any]] = []
    records: list[FeedbackRecord] = []

    if df.empty:
        warnings.append("Input dataset is empty.")
        return ValidationReport(
            total_input_rows=0,
            valid_records_count=0,
            invalid_records_count=0,
            duplicate_records_count=0,
            warnings=warnings,
            records=[],
            invalid_rows=[],
        )

    # Resolve text column
    candidate_text_cols = [
        "feedback_text",
        "text",
        "comment",
        "review",
        "content",
        "feedback",
        "message",
    ]
    resolved_text_col = text_column

    if not resolved_text_col:
        for candidate in candidate_text_cols:
            if candidate in df.columns:
                resolved_text_col = candidate
                break

    if not resolved_text_col or resolved_text_col not in df.columns:
        # Fallback to the first object/string column if available
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        # Avoid picking ground truth columns automatically as fallback text
        filtered_str_cols = [
            c for c in str_cols if c.lower() not in {"ground_truth_theme", "theme", "label", "category", "target"}
        ]
        chosen_cols = filtered_str_cols if filtered_str_cols else str_cols

        if chosen_cols:
            resolved_text_col = chosen_cols[0]
            warnings.append(
                f"No explicit text column specified. Inferred '{resolved_text_col}' as the feedback text column."
            )
        else:
            warnings.append("Could not locate any valid text column in the dataset.")
            return ValidationReport(
                total_input_rows=len(df),
                valid_records_count=0,
                invalid_records_count=len(df),
                duplicate_records_count=0,
                warnings=warnings,
                records=[],
                invalid_rows=[{"row_index": idx, "reason": "No text column"} for idx in df.index],
            )

    # Track duplicates by canonical normalized lower text
    seen_texts: dict[str, str] = {}
    duplicate_count = 0

    for idx, row in df.iterrows():
        raw_val = row.get(resolved_text_col, None)

        if pd.isna(raw_val) or raw_val is None:
            invalid_rows.append(
                {"row_index": int(idx), "reason": "Missing or null text value", "raw_data": row.to_dict()}
            )
            continue

        raw_str = str(raw_val)
        cleaned = normalize_whitespace(raw_str)

        if not cleaned:
            invalid_rows.append(
                {"row_index": int(idx), "reason": "Blank or whitespace-only text", "raw_data": row.to_dict()}
            )
            continue

        # Resolve Record ID
        if id_column and id_column in row and pd.notna(row[id_column]):
            rec_id = str(row[id_column]).strip()
        elif "id" in row and pd.notna(row["id"]):
            rec_id = str(row["id"]).strip()
        else:
            rec_id = f"FB{len(records) + 1:04d}"

        # Resolve Source
        if source_column and source_column in row and pd.notna(row[source_column]):
            rec_source = str(row[source_column])
        elif "source" in row and pd.notna(row["source"]):
            rec_source = str(row["source"])
        else:
            rec_source = "direct_input"

        # Resolve Rating
        rec_rating: float | None = None
        if rating_column and rating_column in row and pd.notna(row[rating_column]):
            try:
                rec_rating = float(row[rating_column])
            except (ValueError, TypeError):
                rec_rating = None
        elif "rating" in row and pd.notna(row["rating"]):
            try:
                rec_rating = float(row["rating"])
            except (ValueError, TypeError):
                rec_rating = None

        # Check for duplicates using normalized lowercase
        canonical_key = cleaned.lower()
        is_dup = False
        dup_of: str | None = None

        if canonical_key in seen_texts:
            is_dup = True
            dup_of = seen_texts[canonical_key]
            duplicate_count += 1
        else:
            seen_texts[canonical_key] = rec_id

        # Retain other columns as metadata
        metadata = {}
        for col_name in row.index:
            if col_name not in {resolved_text_col, id_column, source_column, rating_column, "id", "source", "rating"}:
                val = row[col_name]
                if pd.notna(val):
                    metadata[str(col_name)] = val

        record = FeedbackRecord(
            id=rec_id,
            original_text=raw_str,
            cleaned_text=cleaned,
            source=rec_source,
            rating=rec_rating,
            is_duplicate=is_dup,
            duplicate_of_id=dup_of,
            metadata=metadata,
        )
        records.append(record)

    if duplicate_count > 0:
        warnings.append(
            f"Detected {duplicate_count} duplicate or repeated feedback items. They are flagged and retained."
        )

    if len(records) < 4:
        warnings.append(
            f"Dataset contains only {len(records)} valid records. Small datasets may not support meaningful clustering."
        )

    return ValidationReport(
        total_input_rows=len(df),
        valid_records_count=len(records),
        invalid_records_count=len(invalid_rows),
        duplicate_records_count=duplicate_count,
        warnings=warnings,
        records=records,
        invalid_rows=invalid_rows,
    )


def load_feedback_csv(
    filepath_or_buffer: str | Path | Any,
    text_column: str | None = None,
    id_column: str | None = None,
) -> ValidationReport:
    """Reads a CSV source and returns a ValidationReport."""
    try:
        df = pd.read_csv(filepath_or_buffer)
    except Exception as exc:
        return ValidationReport(
            total_input_rows=0,
            valid_records_count=0,
            invalid_records_count=0,
            duplicate_records_count=0,
            warnings=[f"Failed to read CSV input: {str(exc)}"],
            records=[],
            invalid_rows=[],
        )

    return validate_and_prepare_feedback(df, text_column=text_column, id_column=id_column)


def load_feedback_from_raw_text(raw_text: str, source_name: str = "pasted_text") -> ValidationReport:
    """Parses pasted multiline text where each non-empty line is a feedback item.

    Trims empty lines safely, preserves original text, and generates stable sequential IDs.
    """
    if not raw_text or not raw_text.strip():
        return ValidationReport(
            total_input_rows=0,
            valid_records_count=0,
            invalid_records_count=0,
            duplicate_records_count=0,
            warnings=["Pasted feedback text is empty."],
            records=[],
            invalid_rows=[],
        )

    lines = raw_text.splitlines()
    rows = []
    for line in lines:
        rows.append({"feedback_text": line, "source": source_name})

    df = pd.DataFrame(rows)
    return validate_and_prepare_feedback(df, text_column="feedback_text", source_column="source")
