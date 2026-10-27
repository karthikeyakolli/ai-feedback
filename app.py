"""AI Product Intelligence & Feedback Analyzer - Streamlit Interactive Application.

Phase 4:
- SQLite Persistence & Project Management (Projects, Context Versions, Source Metadata, Feedback Items)
- Safe Feedback Imports with Conflict Strategies (skip, reassign_id, replace_all with confirmation)
- Atomic Analysis Run Recording (running -> completed / failed with rollback)
- Full Offline Historical Run Reopening & Inspection (Zero ML Model Recomputation)
- Historical Run Comparison with Partition Agreement (ARI) on Identical Datasets
- Preserved Separation of Product Context from Feedback Clustering Inputs
- Core AI Experiment: TF-IDF vs Sentence Transformers (all-MiniLM-L6-v2) on Identical Records
- Evidence-Grounded Insights Reports & Download Center
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.clustering import run_embedding_clustering, run_tfidf_clustering
from src.context_ingestion import (
    ExtractionResult,
    ProductContext,
    extract_webpage_content,
    validate_url_safety,
)
from src.data_processing import (
    FeedbackRecord,
    ValidationReport,
    load_feedback_csv,
    load_feedback_from_raw_text,
    validate_and_prepare_feedback,
)
from src.applicability import (
    STATUS_INSUFFICIENT_INFORMATION,
    STATUS_LIKELY_RELEVANT,
    STATUS_NEEDS_VALIDATION,
    assess_cluster_applicability,
)
from src.database import DatabaseManager, ProjectRecord
from src.evaluation import compare_models, evaluate_model_result
from src.explanations import (
    get_fair_comparison_caveat,
    get_metric_explanation,
    get_sentence_transformer_explanation,
    get_tfidf_explanation,
)
from src.export import (
    export_cluster_summaries_csv,
    export_feedback_records_csv,
    export_full_markdown_report,
    export_investigation_recommendations_csv,
    export_model_evaluation_csv,
)
from src.insights import InsightsReport, generate_insights_report
from src.recommendations import (
    build_investigation_queue,
    generate_cluster_recommendation,
)


# =============================================================================
# SESSION STATE INITIALIZATION & PROJECT SWITCHING
# =============================================================================

def init_session_state() -> None:
    """Initializes persistent Streamlit session state and SQLite database connection."""
    if "db" not in st.session_state:
        db_path = os.getenv("DATABASE_PATH", "data/product_intelligence.db")
        st.session_state.db = DatabaseManager(db_path=db_path)

    db: DatabaseManager = st.session_state.db

    # Ensure at least one project exists
    projects = db.list_projects()
    if not projects:
        default_proj = db.create_project(
            name="Default Product Intelligence Workspace",
            description="Default project workspace for customer feedback analysis and clustering.",
        )
        active_pid = default_proj.project_id
    else:
        active_pid = projects[0].project_id

    if "active_project_id" not in st.session_state or not db.get_project(st.session_state.active_project_id):
        st.session_state.active_project_id = active_pid

    active_pid = st.session_state.active_project_id

    # Load or initialize project product context
    if "product_context" not in st.session_state:
        latest_res = db.get_latest_product_context(active_pid)
        if latest_res:
            st.session_state.product_context = latest_res[0]
        else:
            default_ctx = ProductContext(
                name="FeedbackHub Workspace",
                description="Decision-support prototype for analyzing product feedback and customer support reports.",
                target_users="Product Managers, Engineers, UX Researchers",
                goals_or_concerns="Reduce application freezing on PDF export and decrease login timeout friction.",
                source_type="manual",
                is_approved=True,
            )
            ver = db.save_product_context(active_pid, default_ctx)
            default_ctx.version = ver
            st.session_state.product_context = default_ctx

    # Load or initialize project feedback records
    if "validation_report" not in st.session_state:
        saved_records = db.get_feedback_records(active_pid)
        if saved_records:
            st.session_state.validation_report = ValidationReport(
                records=saved_records,
                valid_records_count=len(saved_records),
                invalid_records_count=0,
                duplicate_records_count=sum(1 for r in saved_records if r.is_duplicate),
                total_input_rows=len(saved_records),
                warnings=[],
            )
            st.session_state.dataset_source_desc = f"Project SQLite Database ({len(saved_records)} records)"
        else:
            sample_path = Path("data/sample_feedback.csv")
            if sample_path.exists():
                sample_report = load_feedback_csv(sample_path)
                db.import_feedback_records(active_pid, sample_report.records, on_conflict="skip")
                saved_records = db.get_feedback_records(active_pid)
                st.session_state.validation_report = ValidationReport(
                    records=saved_records,
                    valid_records_count=len(saved_records),
                    invalid_records_count=0,
                    duplicate_records_count=sum(1 for r in saved_records if r.is_duplicate),
                    total_input_rows=len(saved_records),
                    warnings=[],
                )
                st.session_state.dataset_source_desc = f"Built-in Synthetic Sample Dataset ({len(saved_records)} items saved to SQLite)"
            else:
                st.session_state.validation_report = None
                st.session_state.dataset_source_desc = "None"

    # Transient analysis run states
    if "tfidf_result" not in st.session_state:
        st.session_state.tfidf_result = None

    if "embedding_result" not in st.session_state:
        st.session_state.embedding_result = None

    if "embedding_error" not in st.session_state:
        st.session_state.embedding_error = None

    if "active_run_id" not in st.session_state:
        st.session_state.active_run_id = None

    if "is_viewing_historical" not in st.session_state:
        st.session_state.is_viewing_historical = False

    if "historical_run_info" not in st.session_state:
        st.session_state.historical_run_info = None

    if "active_tfidf_eval" not in st.session_state:
        st.session_state.active_tfidf_eval = None

    if "active_embedding_eval" not in st.session_state:
        st.session_state.active_embedding_eval = None

    # Benchmark ground truth labels
    if "ground_truth_map" not in st.session_state:
        eval_path = Path("data/evaluation_labels.csv")
        if eval_path.exists():
            try:
                gt_df = pd.read_csv(eval_path)
                st.session_state.ground_truth_map = dict(
                    zip(gt_df["feedback_id"].astype(str), gt_df["ground_truth_theme"].astype(str))
                )
            except Exception:
                st.session_state.ground_truth_map = None
        else:
            st.session_state.ground_truth_map = None

    if "url_extraction_result" not in st.session_state:
        st.session_state.url_extraction_result = None


def switch_project(new_project_id: str) -> None:
    """Switches the active project, clearing transient run state and reloading project data."""
    db: DatabaseManager = st.session_state.db
    st.session_state.active_project_id = new_project_id

    # Reset transient model results & historical viewing state
    st.session_state.tfidf_result = None
    st.session_state.embedding_result = None
    st.session_state.embedding_error = None
    st.session_state.active_run_id = None
    st.session_state.is_viewing_historical = False
    st.session_state.historical_run_info = None
    st.session_state.active_tfidf_eval = None
    st.session_state.active_embedding_eval = None
    st.session_state.url_extraction_result = None

    # Load context for selected project
    latest_res = db.get_latest_product_context(new_project_id)
    if latest_res:
        st.session_state.product_context = latest_res[0]
    else:
        new_ctx = ProductContext(
            name="New Project Workspace",
            description="",
            target_users="",
            goals_or_concerns="",
            source_type="manual",
            is_approved=True,
        )
        ver = db.save_product_context(new_project_id, new_ctx)
        new_ctx.version = ver
        st.session_state.product_context = new_ctx

    # Load feedback for selected project
    records = db.get_feedback_records(new_project_id)
    if records:
        st.session_state.validation_report = ValidationReport(
            records=records,
            valid_records_count=len(records),
            invalid_records_count=0,
            duplicate_records_count=sum(1 for r in records if r.is_duplicate),
            total_input_rows=len(records),
            warnings=[],
        )
        st.session_state.dataset_source_desc = f"Project SQLite Database ({len(records)} records)"
    else:
        st.session_state.validation_report = None
        st.session_state.dataset_source_desc = "None"


# =============================================================================
# SIDEBAR: PROJECT MANAGEMENT & HISTORICAL RUN BROWSER
# =============================================================================

def render_sidebar_project_manager(db: DatabaseManager) -> None:
    """Renders project selection, metadata, creation, editing, deletion, and historical run loader."""
    st.sidebar.title("📁 Project Management")
    st.sidebar.caption(f"SQLite Storage: `{db.db_path}`")

    projects = db.list_projects()
    if not projects:
        st.sidebar.warning("No projects found.")
        return

    proj_map = {p.project_id: p for p in projects}
    current_pid = st.session_state.active_project_id
    if current_pid not in proj_map:
        current_pid = projects[0].project_id
        st.session_state.active_project_id = current_pid

    current_idx = list(proj_map.keys()).index(current_pid)

    selected_pid = st.sidebar.selectbox(
        "Active Project",
        options=list(proj_map.keys()),
        index=current_idx,
        format_func=lambda pid: f"{proj_map[pid].name} ({pid[:8]})",
    )

    if selected_pid != current_pid:
        switch_project(selected_pid)
        st.rerun()

    active_proj = proj_map[selected_pid]

    # Project Metadata Card
    fb_count = db.get_feedback_count(selected_pid)
    runs_list = db.list_analysis_runs(selected_pid, status="completed")
    latest_ctx_res = db.get_latest_product_context(selected_pid)
    latest_ver = latest_ctx_res[1] if latest_ctx_res else 1

    with st.sidebar.container(border=True):
        st.markdown(f"**Name:** {active_proj.name}")
        if active_proj.description:
            st.caption(active_proj.description)
        st.markdown(f"**ID:** `{active_proj.project_id[:8]}...`")
        st.markdown(f"**Saved Feedback:** `{fb_count}` records")
        st.markdown(f"**Saved Runs:** `{len(runs_list)}` completed")
        st.markdown(f"**Context Version:** `v{latest_ver}`")
        st.caption(f"Updated: {active_proj.updated_at[:19]} UTC")

    # Project Actions: Create, Edit, Delete
    with st.sidebar.expander("➕ Create New Project", expanded=False):
        with st.form("create_project_form", clear_on_submit=True):
            new_p_name = st.text_input("Project Name", placeholder="e.g. Mobile App Redesign")
            new_p_desc = st.text_area("Description (Optional)", placeholder="Project scope, feedback sources...", height=68)
            create_submitted = st.form_submit_button("Create Project", type="primary")

            if create_submitted:
                if not new_p_name.strip():
                    st.error("Project name cannot be empty.")
                else:
                    try:
                        new_proj = db.create_project(name=new_p_name.strip(), description=new_p_desc.strip())
                        st.success(f"Project '{new_proj.name}' created.")
                        switch_project(new_proj.project_id)
                        st.rerun()
                    except ValueError as err:
                        st.error(str(err))

    with st.sidebar.expander("⚙️ Rename / Edit Project", expanded=False):
        with st.form("rename_project_form"):
            edit_p_name = st.text_input("Project Name", value=active_proj.name)
            edit_p_desc = st.text_area("Description", value=active_proj.description or "", height=68)
            rename_submitted = st.form_submit_button("Update Project")

            if rename_submitted:
                if not edit_p_name.strip():
                    st.error("Project name cannot be empty.")
                else:
                    try:
                        db.rename_project(active_proj.project_id, edit_p_name.strip(), edit_p_desc.strip())
                        st.success("Project updated.")
                        st.rerun()
                    except ValueError as err:
                        st.error(str(err))

    with st.sidebar.expander("🗑️ Delete Project", expanded=False):
        st.caption("Permanently deletes this project, its product contexts, feedback records, and analysis runs.")
        confirm_delete = st.checkbox("Confirm permanent deletion", key="confirm_delete_project_chk")
        delete_btn = st.button("Delete Active Project", type="primary", disabled=not confirm_delete)

        if delete_btn and confirm_delete:
            deleted_name = active_proj.name
            db.delete_project(active_proj.project_id)
            st.success(f"Project '{deleted_name}' permanently deleted.")
            # Switch to remaining or create new default
            remaining = db.list_projects()
            if remaining:
                switch_project(remaining[0].project_id)
            else:
                new_def = db.create_project(name="Default Product Intelligence Workspace")
                switch_project(new_def.project_id)
            st.rerun()

    # Historical Analysis Runs Quick-Loader
    st.sidebar.divider()
    st.sidebar.subheader("📜 Historical Runs")
    if not runs_list:
        st.sidebar.caption("No completed analysis runs saved for this project.")
    else:
        st.sidebar.caption("Reopen and inspect past analysis runs completely offline.")
        run_options = {
            r["run_id"]: f"{r['created_at'][:19]} (k={r['cluster_count_k']}, {r['record_count']} items) - {r['run_id'][:8]}"
            for r in runs_list
        }
        selected_run_id = st.sidebar.selectbox(
            "Select Saved Run",
            options=list(run_options.keys()),
            format_func=lambda rid: run_options[rid],
        )

        load_run_btn = st.sidebar.button("📂 Open Historical Run", type="secondary", use_container_width=True)
        if load_run_btn:
            try:
                run_info, model_results, eval_reports = db.reconstruct_run_results(selected_run_id)
                # Map results
                t_res = next((m for m in model_results if "TF-IDF" in m.method), None)
                e_res = next((m for m in model_results if "MiniLM" in m.method or "Embedding" in m.method), None)
                t_eval = next((e for e in eval_reports if "TF-IDF" in e.method), None)
                e_eval = next((e for e in eval_reports if "MiniLM" in e.method or "Embedding" in e.method), None)

                # Reconstruct evaluated feedback records
                run_records = db.get_run_feedback_records(selected_run_id)
                if not run_records:
                    run_records = db.get_feedback_records(active_proj.project_id)

                st.session_state.validation_report = ValidationReport(
                    records=run_records,
                    valid_records_count=len(run_records),
                    invalid_records_count=0,
                    duplicate_records_count=sum(1 for r in run_records if r.is_duplicate),
                    total_input_rows=len(run_records),
                    warnings=[],
                )
                st.session_state.dataset_source_desc = f"Historical Run {selected_run_id[:8]} ({len(run_records)} records)"

                # Load associated product context version if available
                if run_info.get("product_context_version"):
                    v_ctx = db.get_product_context_version(active_proj.project_id, run_info["product_context_version"])
                    if v_ctx:
                        st.session_state.product_context = v_ctx

                st.session_state.tfidf_result = t_res
                st.session_state.embedding_result = e_res
                st.session_state.active_tfidf_eval = t_eval
                st.session_state.active_embedding_eval = e_eval
                st.session_state.active_run_id = selected_run_id
                st.session_state.is_viewing_historical = True
                st.session_state.historical_run_info = run_info
                st.session_state.embedding_error = None

                st.sidebar.success(f"Historical run {selected_run_id[:8]} loaded!")
                st.rerun()
            except Exception as err:
                st.sidebar.error(f"Failed to load run: {err}")

        if st.session_state.is_viewing_historical:
            exit_hist_btn = st.sidebar.button("🔄 Return to Live Workspace", type="primary", use_container_width=True)
            if exit_hist_btn:
                st.session_state.is_viewing_historical = False
                st.session_state.historical_run_info = None
                st.session_state.tfidf_result = None
                st.session_state.embedding_result = None
                st.session_state.active_tfidf_eval = None
                st.session_state.active_embedding_eval = None
                # Restore current project records
                switch_project(active_proj.project_id)
                st.rerun()


# =============================================================================
# HEADER & BANNERS
# =============================================================================

def render_header() -> None:
    """Renders page header, research safeguards, and historical mode banner."""
    st.title("🔍 AI Product Intelligence & Feedback Analyzer")
    st.caption("Phase 4: SQLite Persistence, Project Management & Historical Analysis Explorer")

    if st.session_state.get("is_viewing_historical") and st.session_state.get("historical_run_info"):
        r_info = st.session_state.historical_run_info
        st.info(
            f"📜 **Historical Mode Active:** Viewing saved run `{r_info['run_id'][:8]}` "
            f"(Created: {r_info['created_at'][:19]} UTC | k={r_info['cluster_count_k']} | {r_info['record_count']} items | Seed: {r_info['random_seed']}). "
            "All cluster assignments and evaluation metrics were loaded directly from SQLite without re-running ML models."
        )

    with st.expander("ℹ️ Research Objective & Methodological Safeguards", expanded=False):
        st.markdown(
            r"""
            **Core Research Experiment:**
            - **Method A (Baseline):** Scikit-learn TF-IDF vectorization (sparse n-grams, L2 normalized) + K-Means ($k$ clusters).
            - **Method B (Semantic):** Pretrained Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`, dense 384-d L2 normalized) + K-Means ($k$ clusters).

            **Methodological & Persistence Safeguards:**
            1. **Local SQLite Persistence:** Reliable transactional storage for projects, versioned product contexts, feedback records, and analysis runs (`data/product_intelligence.db`).
            2. **Same-Dataset Comparison:** Both methods are evaluated on the exact same feedback record IDs in identical order.
            3. **Evidence Preservation:** Original feedback text is never mutated or discarded. Duplicates are flagged with pointers (`duplicate_of_id`) rather than dropped.
            4. **Isolated Product Context:** Product descriptions, documentation, and webpage text are kept strictly separate from the feedback clustering inputs.
            5. **Offline Historical Reopening:** Saved runs can be reopened, inspected, and exported without re-running vectorizers or embedding encoders.
            6. **Precondition-Safe Metrics:** Silhouette, Davies-Bouldin, and Calinski-Harabasz are calculated only when mathematical preconditions ($2 \le k < n$, $>1$ cluster) are met.
            7. **Zero Paid APIs Required:** Entire core analysis runs locally on CPU with open-source models and libraries.
            """
        )


# =============================================================================
# TAB 1: PRODUCT CONTEXT & FEEDBACK INGESTION
# =============================================================================

def render_product_context_tab() -> None:
    """Renders Product Definition and Feedback Input tab with SQLite persistence and conflict handling."""
    db: DatabaseManager = st.session_state.db
    active_pid = st.session_state.active_project_id

    st.subheader("1. Product Context Ingestion & Domain Grounding")
    st.caption(
        "Product context provides domain grounding for interpreting findings. "
        "It is strictly isolated from the clustering models and is never fed as feature input. "
        "Context updates are automatically saved as versioned snapshots in SQLite."
    )

    ctx: ProductContext = st.session_state.product_context

    # Context Input Mode Selector
    context_mode = st.radio(
        "Product Context Source",
        options=["Manual Entry", "Paste Product Documentation", "Safe Public Webpage Extraction"],
        horizontal=True,
    )

    if context_mode == "Manual Entry":
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                p_name = st.text_input("Product Name", value=ctx.name)
                p_users = st.text_input("Target Users / Audience (Optional)", value=ctx.target_users)
            with col2:
                p_desc = st.text_area("Product Description", value=ctx.description, height=68)
                p_goals = st.text_input("Product Goals / Known Concerns (Optional)", value=ctx.goals_or_concerns)

            c_btn1, c_btn2 = st.columns([1, 4])
            with c_btn1:
                if st.button("Save & Approve Context", type="primary"):
                    new_ctx = ProductContext(
                        name=p_name.strip(),
                        description=p_desc.strip(),
                        target_users=p_users.strip(),
                        goals_or_concerns=p_goals.strip(),
                        source_type="manual",
                        is_approved=True,
                    )
                    ver = db.save_product_context(active_pid, new_ctx)
                    new_ctx.version = ver
                    st.session_state.product_context = new_ctx
                    st.success(f"Product context saved as Version {ver} in SQLite.")
                    st.rerun()

    elif context_mode == "Paste Product Documentation":
        with st.container(border=True):
            doc_p_name = st.text_input("Product Name", value=ctx.name)
            pasted_doc = st.text_area(
                "Paste Product Documentation, Release Notes, or Specifications",
                height=140,
                placeholder="Paste product documentation, feature lists, or release notes here...",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                doc_goals = st.text_input("Goals or Known Concerns (Optional)", value=ctx.goals_or_concerns)
            with col_b:
                doc_users = st.text_input("Target Users (Optional)", value=ctx.target_users)

            if st.button("Review & Approve Pasted Documentation", type="primary"):
                if not pasted_doc.strip():
                    st.warning("Please paste documentation text before approving.")
                else:
                    new_ctx = ProductContext(
                        name=doc_p_name.strip() or "Unnamed Product",
                        description=pasted_doc.strip(),
                        target_users=doc_users.strip(),
                        goals_or_concerns=doc_goals.strip(),
                        source_type="pasted_docs",
                        raw_extracted_text=pasted_doc.strip(),
                        is_approved=True,
                        extraction_status="success",
                    )
                    ver = db.save_product_context(active_pid, new_ctx)
                    new_ctx.version = ver
                    db.save_source_metadata(
                        active_pid,
                        source_type="pasted_docs",
                        extracted_title=doc_p_name.strip() or "Pasted Documentation",
                        extraction_status="success",
                    )
                    st.session_state.product_context = new_ctx
                    st.success(f"Pasted documentation approved and saved as Version {ver} in SQLite.")
                    st.rerun()

    elif context_mode == "Safe Public Webpage Extraction":
        with st.container(border=True):
            st.markdown(
                "**SSRF Security Notice:** Only public `http` and `https` URLs are allowed. "
                "Localhost, private subnets (e.g. 192.168.x.x, 10.x.x.x), loopback addresses, "
                "and link-local endpoints are strictly rejected. Text is extracted safely using Trafilatura."
            )
            web_p_name = st.text_input("Product Name", value=ctx.name)
            web_url = st.text_input(
                "Public Product Webpage URL",
                placeholder="https://example.com/product",
                help="Must be a public routable webpage.",
            )

            if st.button("Extract Webpage Context", type="primary"):
                if not web_url.strip():
                    st.warning("Please enter a valid URL.")
                else:
                    with st.spinner("Validating URL safety and extracting readable text..."):
                        ext_res = extract_webpage_content(web_url.strip())
                        st.session_state.url_extraction_result = ext_res
                        if ext_res.status in {"success", "partial"}:
                            st.session_state.product_context = ProductContext(
                                name=web_p_name.strip() or (ext_res.title or "Extracted Product"),
                                description=ext_res.extracted_text[:2000],
                                source_type="webpage_url",
                                source_url=ext_res.url,
                                extracted_title=ext_res.title,
                                raw_extracted_text=ext_res.extracted_text,
                                is_approved=False,  # Requires explicit reviewer approval below!
                                extraction_status=ext_res.status,
                                extraction_message=ext_res.message,
                                retrieved_at=datetime.now(timezone.utc).isoformat(),
                            )
                            # Save source provenance
                            db.save_source_metadata(
                                active_pid,
                                source_type="public_url",
                                source_url=ext_res.url,
                                extracted_title=ext_res.title,
                                extraction_status=ext_res.status,
                                warning_metadata={"message": ext_res.message},
                            )
                        else:
                            st.error(f"Extraction failed: {ext_res.message}")

            # Display Extraction Review Card if available
            ext_res: ExtractionResult | None = st.session_state.url_extraction_result
            if ext_res and ext_res.status in {"success", "partial"}:
                st.markdown("#### Review Extracted Content")
                st.caption("Review and edit the extracted content before approving it for contextual grounding.")

                r_col1, r_col2 = st.columns([1, 3])
                with r_col1:
                    st.markdown(f"**Source URL:** `{ext_res.url}`")
                    st.markdown(f"**Extracted Title:** `{ext_res.title or 'N/A'}`")
                    st.markdown(f"**Status:** `{ext_res.status.upper()}`")
                with r_col2:
                    edited_desc = st.text_area(
                        "Extracted Text (Editable before approval)",
                        value=st.session_state.product_context.description,
                        height=120,
                    )

                app_col1, app_col2 = st.columns([1, 4])
                with app_col1:
                    if st.button("Approve Extracted Context", type="primary"):
                        st.session_state.product_context.description = edited_desc.strip()
                        st.session_state.product_context.is_approved = True
                        ver = db.save_product_context(active_pid, st.session_state.product_context)
                        st.session_state.product_context.version = ver
                        st.success(f"Extracted context approved and saved as Version {ver} in SQLite.")
                        st.rerun()

    # Active Context Status Card
    active_ctx = st.session_state.product_context
    with st.container(border=True):
        badge = f"✅ Approved (Version {active_ctx.version})" if active_ctx.is_approved else f"⚠️ Draft (Version {active_ctx.version} - Requires Approval)"
        st.markdown(f"**Active Product Context Status:** {badge} | **Source:** `{active_ctx.source_type}`")
        if active_ctx.description:
            preview_snippet = active_ctx.description[:300] + ("..." if len(active_ctx.description) > 300 else "")
            st.markdown(f"> *\"{preview_snippet}\"*")
        if active_ctx.source_url:
            st.caption(f"Traceable Source URL: `{active_ctx.source_url}` | Retrieved at: `{active_ctx.retrieved_at}`")

    st.divider()

    # Feedback Dataset Input
    st.subheader("2. Feedback Dataset Input & SQLite Import")
    existing_fb_count = db.get_feedback_count(active_pid)
    st.caption(f"Active project currently holds **{existing_fb_count}** feedback records in SQLite.")

    # Conflict Strategy Selector when importing
    conflict_strategy = "skip"
    confirm_replace = False
    if existing_fb_count > 0:
        with st.expander("⚙️ Import Strategy & ID Conflict Handling", expanded=False):
            st.markdown(
                "Select how to handle feedback imports when records already exist in this project:"
            )
            strategy_choice = st.radio(
                "Import Conflict Resolution Mode",
                options=[
                    "Add new records (Skip conflicting IDs)",
                    "Add new records (Auto-reassign conflicting IDs to unique FB-xxxx)",
                    "Replace active project dataset (Destructive overwrite)",
                ],
                index=0,
            )
            if "Skip" in strategy_choice:
                conflict_strategy = "skip"
            elif "Auto-reassign" in strategy_choice:
                conflict_strategy = "reassign_id"
            else:
                conflict_strategy = "replace_all"
                confirm_replace = st.checkbox(
                    "⚠️ Confirm replacement: I understand this will remove all existing feedback records in this project.",
                    value=False,
                )

    input_source = st.radio(
        "Select Feedback Input Method",
        options=[
            "Built-in synthetic sample dataset",
            "Upload custom CSV file",
            "Paste raw multiline text",
        ],
        horizontal=True,
    )

    def _execute_import(imported_records: list[FeedbackRecord], source_desc: str) -> None:
        if conflict_strategy == "replace_all" and not confirm_replace:
            st.error("Destructive dataset replacement requires confirmation. Please check the confirmation box above.")
            return

        try:
            inserted, skipped, conflicts = db.import_feedback_records(
                active_pid, imported_records, on_conflict=conflict_strategy
            )
            all_records = db.get_feedback_records(active_pid)
            st.session_state.validation_report = ValidationReport(
                records=all_records,
                valid_records_count=len(all_records),
                invalid_records_count=0,
                duplicate_records_count=sum(1 for r in all_records if r.is_duplicate),
                total_input_rows=len(all_records),
                warnings=[],
            )
            st.session_state.dataset_source_desc = f"{source_desc} ({len(all_records)} total records saved to SQLite)"
            # Reset transient model results since dataset changed
            st.session_state.tfidf_result = None
            st.session_state.embedding_result = None
            st.session_state.active_run_id = None
            st.session_state.is_viewing_historical = False
            st.session_state.historical_run_info = None

            msg = f"Successfully imported {inserted} records into SQLite!"
            if skipped > 0:
                msg += f" (Skipped {skipped} duplicate IDs)"
            if conflicts > 0 and conflict_strategy == "reassign_id":
                msg += f" (Reassigned {conflicts} conflicting IDs)"
            st.success(msg)
            st.rerun()
        except Exception as exc:
            st.error(f"Persistence error during import: {exc}")

    if input_source == "Built-in synthetic sample dataset":
        st.info("Using checked-in synthetic sample dataset (`data/sample_feedback.csv`) with 20 varied items.")
        if st.button("Import Built-in Dataset into SQLite", type="secondary"):
            sample_path = Path("data/sample_feedback.csv")
            if sample_path.exists():
                rep = load_feedback_csv(sample_path)
                _execute_import(rep.records, "Built-in Synthetic Sample")
            else:
                st.error("Sample dataset file not found at data/sample_feedback.csv")

    elif input_source == "Upload custom CSV file":
        st.write("Upload a custom CSV file containing user reviews, feedback, or support tickets.")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success(f"CSV uploaded successfully: {len(uploaded_df)} total rows detected.")
                all_cols = list(uploaded_df.columns)

                c1, c2 = st.columns(2)
                with c1:
                    default_text_idx = 0
                    for candidate in ["feedback_text", "text", "comment", "review", "content", "feedback", "message"]:
                        if candidate in all_cols:
                            default_text_idx = all_cols.index(candidate)
                            break
                    selected_text_col = st.selectbox(
                        "Select Feedback Text Column (Required)",
                        options=all_cols,
                        index=default_text_idx,
                    )
                with c2:
                    selected_id_col = st.selectbox(
                        "Select Record ID Column (Optional)",
                        options=["[None - Auto-generate stable sequential IDs]"] + all_cols,
                        index=0,
                    )

                if st.button("Validate & Save Uploaded CSV into SQLite", type="primary"):
                    resolved_id = None if selected_id_col.startswith("[None") else selected_id_col
                    rep = validate_and_prepare_feedback(
                        uploaded_df,
                        text_column=selected_text_col,
                        id_column=resolved_id,
                    )
                    _execute_import(rep.records, f"Uploaded CSV ({uploaded_file.name})")
            except Exception as exc:
                st.error(f"Failed to parse uploaded CSV: {exc}")

    elif input_source == "Paste raw multiline text":
        st.write("Paste feedback comments below. Each non-empty line will be treated as one individual feedback record.")
        pasted_text = st.text_area(
            "Raw feedback (one comment per line)",
            height=140,
            placeholder="App crashed when saving settings\nSearching items takes too long\nLove the dark mode!\nApp crashed when saving settings",
        )
        if st.button("Validate & Save Pasted Feedback into SQLite", type="primary"):
            rep = load_feedback_from_raw_text(pasted_text, source_name="user_paste")
            _execute_import(rep.records, "User Pasted Text")

    # Display Validation Report
    report: ValidationReport | None = st.session_state.validation_report
    if report is not None:
        st.divider()
        st.subheader("3. Data Quality & Active Dataset Report")
        st.caption(f"Active Source: **{st.session_state.dataset_source_desc}**")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Records in SQLite", report.valid_records_count)
        kpi2.metric("Unique Feedback Items", sum(1 for r in report.records if not r.is_duplicate))
        kpi3.metric("Flagged Duplicate Items", report.duplicate_records_count)
        kpi4.metric("Input Rows Processed", report.total_input_rows)

        if report.warnings:
            for w in report.warnings:
                st.warning(f"⚠️ {w}")

        if report.valid_records_count > 0:
            preview_rows = []
            for r in report.records:
                preview_rows.append({
                    "Record ID": r.id,
                    "Original Feedback": r.original_text,
                    "Source": r.source,
                    "Rating": r.rating if r.rating is not None else "N/A",
                    "Duplicate Status": "⚠️ Duplicate" if r.is_duplicate else "✅ Unique",
                    "Duplicate Of": r.duplicate_of_id if r.duplicate_of_id else "-",
                })
            preview_df = pd.DataFrame(preview_rows)

            show_dups_only = st.checkbox("Filter preview to show only duplicate records", value=False)
            display_df = preview_df[preview_df["Duplicate Status"] == "⚠️ Duplicate"] if show_dups_only else preview_df
            st.dataframe(display_df, use_container_width=True, height=220)
        else:
            st.error("No valid feedback records in active project. Please import feedback above.")


# =============================================================================
# TAB 2: ANALYSIS WORKSPACE & HISTORICAL RUN COMPARISON
# =============================================================================

def render_analysis_tab() -> None:
    """Renders Analysis Workspace, Model Execution controls, and Run Comparison."""
    db: DatabaseManager = st.session_state.db
    active_pid = st.session_state.active_project_id

    st.subheader("Analysis Workspace & Controls")

    # Historical Mode Banner inside Tab 2
    if st.session_state.get("is_viewing_historical") and st.session_state.get("historical_run_info"):
        r_info = st.session_state.historical_run_info
        with st.container(border=True):
            st.markdown(f"### 📜 Viewing Historical Run: `{r_info['run_id'][:8]}`")
            st.caption(f"Run recorded on {r_info['created_at'][:19]} UTC for Project `{active_pid[:8]}`")

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Evaluated Records", f"{r_info['record_count']} items")
            h2.metric("Cluster Count (k)", str(r_info['cluster_count_k']))
            h3.metric("Random Seed", str(r_info['random_seed']))
            h4.metric("Recorded Runtime", f"{r_info['runtime_seconds']:.4f}s")

            st.markdown(f"**Dataset Fingerprint:** `{r_info['dataset_fingerprint']}`")
            if r_info.get("product_context_version"):
                st.markdown(f"**Associated Product Context Version:** `v{r_info['product_context_version']}`")

            if st.button("Return to Live Analysis Workspace", type="primary"):
                st.session_state.is_viewing_historical = False
                st.session_state.historical_run_info = None
                switch_project(active_pid)
                st.rerun()

        st.divider()

    # Active Live Workspace
    report: ValidationReport | None = st.session_state.validation_report
    if report is None or report.valid_records_count < 2:
        st.warning("Please load at least 2 valid feedback records in the 'Product & Input Data' tab before analyzing.")
        return

    n_samples = report.valid_records_count
    max_k_slider = max(2, min(10, n_samples - 1))

    with st.container(border=True):
        st.markdown(f"**Target Product:** `{st.session_state.product_context.name}`")
        st.markdown(f"**Validated Dataset Size:** `{n_samples}` feedback records available in SQLite.")

        c1, c2 = st.columns(2)
        with c1:
            k = st.slider(
                "Number of Clusters (k)",
                min_value=2,
                max_value=max_k_slider,
                value=min(4, max_k_slider),
                help="Defines the number of theme groups to discover. Capped at n_samples - 1.",
            )
        with c2:
            seed = st.number_input(
                "Random Seed",
                value=42,
                step=1,
                help="Ensures reproducible K-Means centroid initialization.",
            )

        run_btn = st.button("🚀 Run Both AI Methods (Save to SQLite)", type="primary", use_container_width=True)

    if run_btn:
        records = report.records
        latest_res = db.get_latest_product_context(active_pid)
        ctx_ver = latest_res[1] if latest_res else None

        # 1. Create analysis run in SQLite with status 'running'
        run_record = db.create_analysis_run(
            project_id=active_pid,
            cluster_count_k=k,
            random_state=int(seed),
            dataset_records=records,
            product_context_version=ctx_ver,
        )

        try:
            with st.spinner("Executing Method A: TF-IDF vectorization + K-Means..."):
                tfidf_res = run_tfidf_clustering(records, n_clusters=k, random_state=int(seed))

            emb_res = None
            emb_error = None
            with st.spinner("Executing Method B: Pretrained Sentence Transformers + K-Means..."):
                try:
                    emb_res = run_embedding_clustering(records, n_clusters=k, random_state=int(seed))
                except Exception as exc:
                    emb_error = str(exc)

            # Evaluate both models
            gt_map = st.session_state.ground_truth_map
            tfidf_eval = evaluate_model_result(tfidf_res, ground_truth_labels=gt_map)
            embedding_eval = evaluate_model_result(emb_res, ground_truth_labels=gt_map) if emb_res else None

            # 2. Atomically persist results to SQLite and mark run completed
            db.save_run_results(
                run_id=run_record.run_id,
                tfidf_result=tfidf_res,
                embedding_result=emb_res,
                tfidf_eval=tfidf_eval,
                embedding_eval=embedding_eval,
            )

            # 3. Update session state only after successful DB persistence
            st.session_state.tfidf_result = tfidf_res
            st.session_state.embedding_result = emb_res
            st.session_state.embedding_error = emb_error
            st.session_state.active_run_id = run_record.run_id
            st.session_state.active_tfidf_eval = tfidf_eval
            st.session_state.active_embedding_eval = embedding_eval
            st.session_state.is_viewing_historical = False
            st.session_state.historical_run_info = None

            st.success(f"✅ Analysis completed and saved to SQLite! (Run ID: `{run_record.run_id[:8]}`)")
        except Exception as exc:
            db.record_run_failure(run_record.run_id, str(exc))
            st.error(f"❌ Analysis run failed: {exc}. Run failure was recorded safely in SQLite.")

    # Execution Summary Display
    if st.session_state.tfidf_result is not None:
        t_res = st.session_state.tfidf_result
        e_res = st.session_state.embedding_result

        st.markdown("#### Active Run Execution Summary")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Dataset Evaluated", f"{len(t_res.feedback_ids)} items")
        mcol2.metric("TF-IDF Runtime", f"{t_res.runtime_seconds:.4f}s")
        if e_res:
            mcol3.metric("Sentence Transformers Runtime", f"{e_res.runtime_seconds:.4f}s")
            mcol4.metric("Clusters Formed", f"{len(set(t_res.cluster_labels))} clusters")
        else:
            mcol3.metric("Sentence Transformers", "Unavailable")
            mcol4.metric("Clusters Formed", f"{len(set(t_res.cluster_labels))} clusters")

        if st.session_state.embedding_error:
            st.error(f"Sentence Transformers Error: {st.session_state.embedding_error}")

        st.info("👉 Switch to **Cluster Explorer**, **Model Comparison**, or **Insights Report** to review results.")

    # Run Comparison Expander
    st.divider()
    with st.expander("⚖️ Compare Two Historical Runs", expanded=False):
        completed_runs = db.list_analysis_runs(active_pid, status="completed")
        if len(completed_runs) < 2:
            st.info("At least 2 completed analysis runs are required in this project to perform a comparison.")
        else:
            run_choices = {
                r["run_id"]: f"{r['created_at'][:19]} (k={r['cluster_count_k']}, {r['record_count']} items) - {r['run_id'][:8]}"
                for r in completed_runs
            }
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                run_a_id = st.selectbox(
                    "Select Baseline Run (Run A)",
                    options=list(run_choices.keys()),
                    format_func=lambda rid: run_choices[rid],
                    index=0,
                )
            with comp_col2:
                run_b_id = st.selectbox(
                    "Select Comparison Run (Run B)",
                    options=list(run_choices.keys()),
                    format_func=lambda rid: run_choices[rid],
                    index=min(1, len(completed_runs) - 1),
                )

            if st.button("Compare Selected Runs", type="primary"):
                if run_a_id == run_b_id:
                    st.warning("Please choose two different runs to compare.")
                else:
                    try:
                        comp = db.compare_historical_runs(run_a_id, run_b_id)
                        ra = comp["run_a"]
                        rb = comp["run_b"]

                        st.markdown("#### 1. Configuration & Dataset Comparison")
                        meta_table = [
                            {"Property": "Run ID", "Run A": ra["run_id"][:8], "Run B": rb["run_id"][:8]},
                            {"Property": "Execution Timestamp", "Run A": ra["created_at"][:19], "Run B": rb["created_at"][:19]},
                            {"Property": "Record Count", "Run A": str(ra["record_count"]), "Run B": str(rb["record_count"])},
                            {"Property": "Cluster Count (k)", "Run A": str(ra["cluster_count_k"]), "Run B": str(rb["cluster_count_k"])},
                            {"Property": "Random Seed", "Run A": str(ra["random_seed"]), "Run B": str(rb["random_seed"])},
                            {"Property": "Dataset Fingerprint", "Run A": ra["dataset_fingerprint"][:16] + "...", "Run B": rb["dataset_fingerprint"][:16] + "..."},
                            {"Property": "Total Runtime", "Run A": f"{ra['runtime_seconds']:.4f}s", "Run B": f"{rb['runtime_seconds']:.4f}s"},
                        ]
                        st.table(pd.DataFrame(meta_table))

                        st.markdown("#### 2. Partition Agreement & Metric Comparison")
                        for mname, mdata in comp["model_comparisons"].items():
                            st.markdown(f"**Algorithm:** `{mname}`")
                            st.info(mdata["partition_ari_note"])

                            mets_a = mdata["metrics_a"]
                            mets_b = mdata["metrics_b"]
                            metric_table = [
                                {
                                    "Metric": "Silhouette Score",
                                    "Run A": f"{mets_a['silhouette']:.4f}" if mets_a['silhouette'] is not None else "N/A",
                                    "Run B": f"{mets_b['silhouette']:.4f}" if mets_b['silhouette'] is not None else "N/A",
                                },
                                {
                                    "Metric": "Davies-Bouldin Index (Lower is better)",
                                    "Run A": f"{mets_a['davies_bouldin']:.4f}" if mets_a['davies_bouldin'] is not None else "N/A",
                                    "Run B": f"{mets_b['davies_bouldin']:.4f}" if mets_b['davies_bouldin'] is not None else "N/A",
                                },
                                {
                                    "Metric": "Calinski-Harabasz Index",
                                    "Run A": f"{mets_a['calinski_harabasz']:.4f}" if mets_a['calinski_harabasz'] is not None else "N/A",
                                    "Run B": f"{mets_b['calinski_harabasz']:.4f}" if mets_b['calinski_harabasz'] is not None else "N/A",
                                },
                                {
                                    "Metric": "Adjusted Rand Index vs Reference Labels",
                                    "Run A": f"{mets_a['ari_reference']:.4f}" if mets_a['ari_reference'] is not None else "N/A",
                                    "Run B": f"{mets_b['ari_reference']:.4f}" if mets_b['ari_reference'] is not None else "N/A",
                                },
                            ]
                            st.table(pd.DataFrame(metric_table))

                            # Cluster Size Distribution Chart
                            st.markdown(f"**Cluster Size Distribution (`{mname}`):**")
                            dist_a = mdata["cluster_size_distribution_a"]
                            dist_b = mdata["cluster_size_distribution_b"]
                            all_cids = sorted(set(dist_a.keys()).union(set(dist_b.keys())))
                            chart_df = pd.DataFrame({
                                "Cluster": [f"Cluster #{c}" for c in all_cids],
                                "Run A": [dist_a.get(c, 0) for c in all_cids],
                                "Run B": [dist_b.get(c, 0) for c in all_cids],
                            }).set_index("Cluster")
                            st.bar_chart(chart_df)

                        st.markdown("#### 3. Methodological Caveats on Run Comparison")
                        with st.container(border=True):
                            for cav in comp["caveats"]:
                                st.markdown(f"- ⚠️ {cav}")
                    except Exception as err:
                        st.error(f"Error during run comparison: {err}")


# =============================================================================
# TAB 3: CLUSTER EXPLORER
# =============================================================================

def render_cluster_explorer_tab() -> None:
    """Renders interactive cluster evidence exploration."""
    st.subheader("Cluster Explorer & Evidence Inspection")

    t_res = st.session_state.tfidf_result
    e_res = st.session_state.embedding_result
    report: ValidationReport | None = st.session_state.validation_report

    if t_res is None or report is None:
        st.info("No active clustering results. Run the models in the **Analysis Workspace** tab or open a saved run from the sidebar.")
        return

    model_choices = ["Method A: TF-IDF + K-Means"]
    if e_res is not None:
        model_choices.append("Method B: Sentence Transformers + K-Means")

    selected_model_name = st.radio("Select Model to Inspect", options=model_choices, horizontal=True)
    active_res = t_res if "TF-IDF" in selected_model_name else e_res

    unique_clusters = sorted(list(set(active_res.cluster_labels)))
    cluster_labels_display = []
    for c_id in unique_clusters:
        kw = ", ".join(active_res.cluster_keywords.get(c_id, [])[:3])
        size = active_res.cluster_sizes.get(c_id, 0)
        cluster_labels_display.append(f"Cluster #{c_id} ({size} items) - [{kw}]")

    selected_cluster_idx = st.selectbox(
        "Select Cluster to Explore",
        options=range(len(unique_clusters)),
        format_func=lambda i: cluster_labels_display[i],
    )
    cluster_id = unique_clusters[selected_cluster_idx]

    c_size = active_res.cluster_sizes.get(cluster_id, 0)
    pct = (c_size / len(active_res.feedback_ids)) * 100 if active_res.feedback_ids else 0
    keywords = active_res.cluster_keywords.get(cluster_id, [])
    rep_id = active_res.cluster_representatives.get(cluster_id, "")
    rep_record = next((r for r in report.records if r.id == rep_id), None)

    all_cluster_records = [
        r for r, lbl in zip(report.records, active_res.cluster_labels)
        if lbl == cluster_id
    ]
    duplicate_count = sum(1 for r in all_cluster_records if r.is_duplicate)
    unique_count = c_size - duplicate_count

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Cluster Size", f"{c_size} items ({pct:.1f}%)")
        st.caption(f"Unique Mentions: **{unique_count}** | Duplicates: **{duplicate_count}**")
        st.markdown(f"**Top Extracted Terms:** `{', '.join(keywords) if keywords else 'None'}`")
        st.caption(
            "*Terms derivation:* "
            + ("Highest TF-IDF feature weights in cluster centroid." if "TF-IDF" in selected_model_name
               else "Top distinct vocabulary terms across cluster member texts.")
        )
    with col2:
        with st.container(border=True):
            st.markdown("🎯 **Centroid Representative Feedback:**")
            if rep_record:
                st.markdown(f"> *\"{rep_record.original_text}\"*")
                st.caption(f"Feedback ID: `{rep_record.id}` | Source: `{rep_record.source}` | Duplicate: `{rep_record.is_duplicate}`")
            else:
                st.write("No representative identified.")
            st.caption("*Selected as the feedback item with the highest cosine similarity to the cluster centroid.*")

    # Feature A & B: Phase 5 Decision Support & Explainability for Selected Cluster
    ctx = st.session_state.product_context
    rep_quote = rep_record.original_text if rep_record else (all_cluster_records[0].original_text if all_cluster_records else "")
    actual_rep_id = rep_record.id if rep_record else (all_cluster_records[0].id if all_cluster_records else "")

    assessment = assess_cluster_applicability(
        cluster_id=cluster_id,
        cluster_keywords=keywords,
        representative_quote=rep_quote,
        representative_id=actual_rep_id,
        member_records=all_cluster_records,
        product_context=ctx,
        run_id=st.session_state.get("active_run_id"),
    )

    recommendation = generate_cluster_recommendation(
        cluster_id=cluster_id,
        model_name=selected_model_name,
        keywords=keywords,
        representative_quote=rep_quote,
        representative_id=actual_rep_id,
        member_records=all_cluster_records,
        assessment=assessment,
        product_context=ctx,
    )

    app_col, rec_col = st.columns(2)
    with app_col:
        with st.container(border=True):
            st.markdown("#### 🧭 Applicability Assessment")
            if assessment.status == STATUS_LIKELY_RELEVANT:
                st.success(f"Status: **{assessment.status}**")
            elif assessment.status == STATUS_NEEDS_VALIDATION:
                st.warning(f"Status: **{assessment.status}**")
            else:
                st.info(f"Status: **{assessment.status}**")

            st.markdown(f"**Explanation:** {assessment.explanation}")

            if assessment.relevant_context_concepts:
                st.markdown(f"**Aligned Context Concepts:** `{', '.join(assessment.relevant_context_concepts)}`")

            if assessment.contradictory_evidence:
                st.markdown("**Contradictory / Noisy Signals:**")
                for ce in assessment.contradictory_evidence:
                    st.caption(f"⚠️ {ce}")
                st.caption("*Explicit Warning: This cluster may be overly broad or noisy; member feedback reflects mixed sentiments.*")

            if assessment.missing_information:
                st.markdown(f"**Missing Information:** {'; '.join(assessment.missing_information)}")

            st.info(f"👉 **Next Validation Step:** {assessment.next_validation_step}")
            st.caption(f"Method: `{assessment.assessment_method}` | Context Version: `{assessment.context_version or 'N/A'}`")

    with rec_col:
        with st.container(border=True):
            st.markdown("#### 💡 Investigation Recommendation")
            st.markdown(f"**Title:** {recommendation.title}")
            st.markdown(f"**Suggested Action:** {recommendation.suggested_action}")
            st.markdown(f"**Rationale:** {recommendation.rationale}")

            st.caption(
                f"Recurrence Score: **{recommendation.unique_mentions_count} unique user mentions** "
                f"({recommendation.duplicate_count} duplicate repeats)"
            )

            with st.expander("🔍 Supporting Evidence Quotes", expanded=False):
                for sid, squote in zip(recommendation.supporting_feedback_ids, recommendation.supporting_quotes):
                    st.markdown(f"- **`{sid}`**: *\"{squote}\"*")

            st.warning(f"⚠️ **Human Review Reminder:** {recommendation.human_review_reminder}")

    st.markdown("#### Member Feedback Evidence")
    search_query = st.text_input("Search within this cluster's comments", placeholder="Filter by keyword...")

    member_records = list(all_cluster_records)
    if search_query:
        member_records = [
            r for r in member_records
            if search_query.lower() in r.original_text.lower() or search_query.lower() in r.id.lower()
        ]

    st.caption(f"Showing {len(member_records)} of {c_size} feedback items in Cluster #{cluster_id}.")

    table_data = []
    for r in member_records:
        table_data.append({
            "Record ID": r.id,
            "Original Feedback": r.original_text,
            "Source": r.source,
            "Rating": r.rating if r.rating is not None else "N/A",
            "Duplicate Status": "⚠️ Duplicate" if r.is_duplicate else "✅ Unique",
            "Duplicate Of": r.duplicate_of_id if r.duplicate_of_id else "-",
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=260)


# =============================================================================
# TAB 4: MODEL COMPARISON & EVALUATION
# =============================================================================

def render_model_comparison_tab() -> None:
    """Renders side-by-side model comparison, distributions, and evaluation limitations."""
    st.subheader("Model Comparison & Evaluation Findings")

    t_res = st.session_state.tfidf_result
    e_res = st.session_state.embedding_result
    gt_map = st.session_state.ground_truth_map

    if t_res is None:
        st.info("No clustering runs available for comparison. Run the models in the **Analysis Workspace** tab or open a saved run.")
        return

    # Use saved evaluation report in historical mode, or recompute for live runs
    if st.session_state.get("active_tfidf_eval"):
        t_eval = st.session_state.active_tfidf_eval
    else:
        t_eval = evaluate_model_result(t_res, ground_truth_labels=gt_map)

    if st.session_state.get("active_embedding_eval"):
        e_eval = st.session_state.active_embedding_eval
    elif e_res:
        e_eval = evaluate_model_result(e_res, ground_truth_labels=gt_map)
    else:
        e_eval = None

    st.markdown("#### 1. Side-by-Side Metric Comparison")
    st.caption("Both algorithms were executed on the exact same records.")

    comparison_rows = [
        {
            "Property / Metric": "Method Architecture",
            "TF-IDF Baseline": t_res.method,
            "Sentence Transformers": e_res.method if e_res else "Failed / Offline",
        },
        {
            "Property / Metric": "Model Identifier",
            "TF-IDF Baseline": t_res.model_identifier,
            "Sentence Transformers": e_res.model_identifier if e_res else "N/A",
        },
        {
            "Property / Metric": "Feature Representation",
            "TF-IDF Baseline": t_res.representation,
            "Sentence Transformers": e_res.representation if e_res else "N/A",
        },
        {
            "Property / Metric": "Distance Metric",
            "TF-IDF Baseline": t_res.distance_metric,
            "Sentence Transformers": e_res.distance_metric if e_res else "N/A",
        },
        {
            "Property / Metric": "Execution Runtime",
            "TF-IDF Baseline": f"{t_res.runtime_seconds:.4f}s",
            "Sentence Transformers": f"{e_res.runtime_seconds:.4f}s" if e_res else "N/A",
        },
        {
            "Property / Metric": "Clusters Formed (k)",
            "TF-IDF Baseline": str(len(set(t_res.cluster_labels))),
            "Sentence Transformers": str(len(set(e_res.cluster_labels))) if e_res else "N/A",
        },
        {
            "Property / Metric": "Silhouette Score",
            "TF-IDF Baseline": (
                f"{t_eval.silhouette:.4f}" if t_eval.silhouette is not None
                else f"Not available ({t_eval.metric_reasons.get('silhouette', 'Preconditions not met')})"
            ),
            "Sentence Transformers": (
                f"{e_eval.silhouette:.4f}" if e_eval and e_eval.silhouette is not None
                else f"Not available ({e_eval.metric_reasons.get('silhouette', 'N/A') if e_eval else 'N/A'})"
            ),
        },
        {
            "Property / Metric": "Davies-Bouldin Index (Lower is better)",
            "TF-IDF Baseline": (
                f"{t_eval.davies_bouldin:.4f}" if t_eval.davies_bouldin is not None
                else f"Not available ({t_eval.metric_reasons.get('davies_bouldin', 'N/A')})"
            ),
            "Sentence Transformers": (
                f"{e_eval.davies_bouldin:.4f}" if e_eval and e_eval.davies_bouldin is not None
                else f"Not available ({e_eval.metric_reasons.get('davies_bouldin', 'N/A') if e_eval else 'N/A'})"
            ),
        },
        {
            "Property / Metric": "Adjusted Rand Index (ARI)",
            "TF-IDF Baseline": (
                f"{t_eval.adjusted_rand_index:.4f}" if t_eval.adjusted_rand_index is not None
                else f"Not available ({t_eval.metric_reasons.get('adjusted_rand_index', 'No reference labels')})"
            ),
            "Sentence Transformers": (
                f"{e_eval.adjusted_rand_index:.4f}" if e_eval and e_eval.adjusted_rand_index is not None
                else f"Not available ({e_eval.metric_reasons.get('adjusted_rand_index', 'No reference labels') if e_eval else 'N/A'})"
            ),
        },
        {
            "Property / Metric": "Cluster Balance Entropy",
            "TF-IDF Baseline": f"{t_eval.cluster_size_entropy:.4f}" if t_eval.cluster_size_entropy is not None else "N/A",
            "Sentence Transformers": f"{e_eval.cluster_size_entropy:.4f}" if e_eval and e_eval.cluster_size_entropy is not None else "N/A",
        },
    ]

    st.table(pd.DataFrame(comparison_rows))

    st.markdown("#### 2. Cluster Size Distributions")
    all_clusters = sorted(list(set(t_res.cluster_labels)))
    dist_data: dict[str, list[int]] = {"Cluster ID": [f"Cluster #{c}" for c in all_clusters]}
    dist_data["TF-IDF"] = [t_res.cluster_sizes.get(c, 0) for c in all_clusters]
    if e_res:
        dist_data["Sentence Transformers"] = [e_res.cluster_sizes.get(c, 0) for c in all_clusters]

    dist_df = pd.DataFrame(dist_data).set_index("Cluster ID")
    st.bar_chart(dist_df)

    st.markdown("#### 3. Methodological Limitations & Explanation")
    with st.container(border=True):
        st.markdown(
            """
            - **Cross-Representation Metric Incomparability:** Silhouette score measures geometric compactness in a model's *specific* feature space. Sparse TF-IDF (1000 n-gram dimensions) and dense Sentence Transformers (384 continuous dimensions) have fundamentally different geometries; a higher raw score in one space does not prove it discovered more meaningful product themes.
            - **K-Means Geometric Constraints:** K-Means assumes spherical, isotropic clusters with roughly equal variance. It will force outliers or mixed feedback into the nearest centroid.
            - **Sample Size Sensitivity:** Feedback datasets with fewer than 50 samples exhibit high assignment variance across random seeds.
            - **Ground Truth Subjectivity:** Adjusted Rand Index (ARI) evaluates alignment with manual annotator categories (`data/evaluation_labels.csv`), which inherently reflect subjective human categorization choices.
            """
        )

    st.markdown("#### 4. 📚 Algorithm & Metric Explainability Guide (For Product Owners)")
    st.caption("Plain-language educational reference explaining machine learning mechanics, distance geometries, and metrics.")

    with st.expander("🔬 How TF-IDF Vectorization & K-Means Work", expanded=False):
        tfidf_exp = get_tfidf_explanation()
        st.markdown(f"**How Text Becomes Numbers:** {tfidf_exp['how_text_is_converted']}")
        st.markdown(f"**What TF-IDF Represents:** {tfidf_exp['what_tfidf_represents']}")
        st.markdown(f"**Why Distinctive Words Are Features:** {tfidf_exp['why_distinguishing_words_are_features']}")
        st.markdown(f"**Cosine Similarity Role:** {tfidf_exp['cosine_similarity_role']}")
        st.markdown(f"**K-Means Assignment:** {tfidf_exp['kmeans_assignment']}")
        st.warning(f"⚠️ **Wording Limitation:** {tfidf_exp['wording_variance_limitation']}")

    with st.expander("🧠 How Sentence Transformers & Dense Embeddings Work", expanded=False):
        emb_exp = get_sentence_transformer_explanation()
        st.markdown(f"**How Sentences Are Converted:** {emb_exp['how_text_is_converted']}")
        st.markdown(f"**What a Dense Vector Represents:** {emb_exp['what_a_vector_represents']}")
        st.markdown(f"**Why Semantically Related Sentences Cluster:** {emb_exp['why_related_sentences_are_close']}")
        st.markdown(f"**K-Means in Continuous Space:** {emb_exp['kmeans_in_embedding_space']}")
        st.warning(f"⚠️ **Embedding Nuance Limitations:** {emb_exp['embedding_limitations']}")

    with st.expander("📐 Metric Interpretation & Preconditions Guide", expanded=False):
        for m_key in ["silhouette", "davies_bouldin", "calinski_harabasz", "adjusted_rand_index", "normalized_mutual_info", "cluster_size_entropy"]:
            m_info = get_metric_explanation(m_key)
            st.markdown(f"##### {m_info['name']}")
            st.markdown(f"- **What It Measures:** {m_info['measures']}")
            st.markdown(f"- **Preferred Direction:** `{m_info['direction_of_preference']}` (Range: `{m_info['range']}`)")
            st.markdown(f"- **Cautious Interpretation:** {m_info['cautious_interpretation']}")
            st.markdown(f"- **When Unavailable:** {m_info['unavailability_reasons']}")
            st.markdown(f"- ⚠️ **Business Utility Caveat:** {m_info['business_utility_caveat']}")
            st.divider()

    with st.expander("⚖️ Fair Comparison Caveat (Why Comparing Spaces Directly Is Misleading)", expanded=False):
        fair_info = get_fair_comparison_caveat()
        st.markdown(f"**{fair_info['title']}**")
        st.info(fair_info['core_principle'])
        st.markdown(f"**Why Spaces Differ Mathematically:**\n{fair_info['why_spaces_differ']}")
        st.markdown(f"**Four Evaluation Pillars to Distinguish:**\n{fair_info['four_pillars_to_distinguish']}")
        st.error(fair_info['winner_declaration_warning'])


# =============================================================================
# TAB 5: INSIGHTS REPORT & EXPORTS
# =============================================================================

def render_insights_tab() -> None:
    """Renders the Evidence-Grounded Insights Report and Download Center."""
    st.subheader("💡 Evidence-Grounded Insights Report")

    t_res = st.session_state.tfidf_result
    e_res = st.session_state.embedding_result
    report: ValidationReport | None = st.session_state.validation_report
    ctx: ProductContext = st.session_state.product_context
    gt_map = st.session_state.ground_truth_map

    if t_res is None or report is None:
        st.info("No clustering analysis available. Run the models in the **Analysis Workspace** tab or open a saved run from the sidebar.")
        return

    # Choose model to base insights upon
    model_opts = ["Method A: TF-IDF + K-Means"]
    if e_res is not None:
        model_opts.append("Method B: Sentence Transformers + K-Means")

    sel_model_name = st.radio("Generate Insights Report For:", options=model_opts, horizontal=True)
    active_res = t_res if "TF-IDF" in sel_model_name else e_res

    insights: InsightsReport = generate_insights_report(active_res, report, product_context=ctx)

    if st.session_state.get("active_tfidf_eval") and "TF-IDF" in sel_model_name:
        eval_rep = st.session_state.active_tfidf_eval
    elif st.session_state.get("active_embedding_eval") and "TF-IDF" not in sel_model_name:
        eval_rep = st.session_state.active_embedding_eval
    else:
        eval_rep = evaluate_model_result(active_res, ground_truth_labels=gt_map)

    # Overview KPI Cards
    top_cluster_id = insights.prioritized_clusters[0] if insights.prioritized_clusters else 0
    top_cluster = insights.get_cluster(top_cluster_id)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Analyzed Records", f"{insights.total_feedback_count} items")
    k2.metric("Unique Records", f"{insights.total_unique_count} items")
    k3.metric("Duplicate Items", f"{insights.total_duplicate_count} items")
    k4.metric("Most Frequent Theme", f"Cluster #{top_cluster_id}" if top_cluster else "N/A")

    st.markdown("#### 1. Theme Prioritization & Recurrence Ranking")
    st.caption(
        "Ranked by unique feedback volume (excluding duplicate repetitions). "
        "Recurrence reflects user reporting frequency, not verified defect severity or urgency."
    )

    prioritization_rows = []
    for c_id in insights.prioritized_clusters:
        c = insights.get_cluster(c_id)
        if c:
            prioritization_rows.append({
                "Cluster Label": c.label,
                "Unique Mentions": c.unique_member_count,
                "Duplicate Repeats": c.duplicate_count,
                "Total Volume": c.member_count,
                "Dataset Share": f"{c.percentage_of_total}%",
                "Evidence Strength": c.evidence_strength,
                "Top Keywords": ", ".join(c.keywords[:4]),
            })

    st.table(pd.DataFrame(prioritization_rows))

    st.markdown("#### 2. Discovered Themes & Grounded Evidence Panels")
    for c in insights.cluster_insights:
        app_status_tag = f"[{c.applicability.status}]" if c.applicability else ""
        with st.expander(
            f"📌 {c.label} ({c.member_count} items, {c.unique_member_count} unique) — {app_status_tag} [{', '.join(c.keywords[:3])}]",
            expanded=False,
        ):
            c_col1, c_col2 = st.columns([1, 2])
            with c_col1:
                st.markdown(f"**Cluster Share:** `{c.percentage_of_total}%`")
                st.markdown(f"**Evidence Recurrence:** `{c.evidence_strength}`")
                st.markdown(f"**Keywords Derivation:** *{c.keywords_derivation}*")
                if c.applicability:
                    st.markdown(f"**Applicability:** `{c.applicability.status}`")
                    if c.applicability.relevant_context_concepts:
                        st.caption(f"Concepts: {', '.join(c.applicability.relevant_context_concepts)}")
            with c_col2:
                st.markdown(f"**Representative Feedback ({c.representative_feedback_id}):**")
                st.markdown(f"> *\"{c.representative_quote}\"*")

            st.markdown(f"**Product Context Relevance:** {c.product_context_relevance}")

            if c.recommendation:
                with st.container(border=True):
                    st.markdown(f"💡 **Suggested Investigation:** {c.recommendation.title}")
                    st.markdown(f"- **Action:** {c.recommendation.suggested_action}")
                    st.caption(f"Rationale: {c.recommendation.rationale}")

            st.markdown("**Supporting Feedback Quotes:**")
            for ex in c.supporting_examples:
                dup_tag = f" `[Duplicate of {ex['duplicate_of_id']}]`" if ex["is_duplicate"] else ""
                st.write(f"- `{ex['id']}`{dup_tag}: {ex['text']}")

    # Feature B: Frequency-Based Investigation Queue
    st.markdown("#### 3. 🧭 Frequency-Based Investigation Queue")
    st.caption(
        "Ordered strictly by unique feedback volume (frequency-based queue). "
        "Does NOT represent defect severity, business priority, financial impact, or customer importance. "
        "Recommendations suggest WHAT TO INVESTIGATE, not autonomous product decisions."
    )

    if insights.investigation_queue:
        queue_rows = []
        for idx, rec in enumerate(insights.investigation_queue, 1):
            queue_rows.append({
                "Rank": idx,
                "Recommendation Title": rec.title,
                "Cluster": f"Cluster #{rec.cluster_id}",
                "Applicability": rec.applicability_status,
                "Unique Mentions": rec.unique_mentions_count,
                "Duplicate Repeats": rec.duplicate_count,
                "Suggested Action": rec.suggested_action,
                "Supporting IDs": ", ".join(rec.supporting_feedback_ids),
            })
        st.dataframe(pd.DataFrame(queue_rows), use_container_width=True, height=220)

        with st.expander("🔍 Inspect Full Recommendation Evidence & Quotes", expanded=False):
            for idx, rec in enumerate(insights.investigation_queue, 1):
                st.markdown(f"**#{idx} — {rec.title}** (`Cluster #{rec.cluster_id}`) [{rec.applicability_status}]")
                st.markdown(f"- **Action:** {rec.suggested_action}")
                st.markdown(f"- **Rationale:** {rec.rationale}")
                st.markdown(f"- **Evidence Quotes:**")
                for q_id, q_text in zip(rec.supporting_feedback_ids, rec.supporting_quotes):
                    st.markdown(f"  - `{q_id}`: *\"{q_text}\"*")
                if rec.contradictory_evidence:
                    st.markdown(f"- ⚠️ **Contradictory Signals:** {'; '.join(rec.contradictory_evidence)}")
                st.warning(f"⚠️ **Human Review:** {rec.human_review_reminder}")
                st.divider()

    # Export Center
    st.divider()
    st.subheader("📥 Export Center")
    st.caption("Download real computed analysis results, decision support queues, and reproducible reports.")

    # Generate Export Payloads
    t_eval_rep = st.session_state.get("active_tfidf_eval") or evaluate_model_result(t_res, gt_map)
    e_eval_rep = st.session_state.get("active_embedding_eval") or (evaluate_model_result(e_res, gt_map) if e_res else None)

    feedback_csv = export_feedback_records_csv(report, tfidf_res=t_res, embedding_res=e_res)
    summaries_csv = export_cluster_summaries_csv(insights)
    recs_csv = export_investigation_recommendations_csv(insights.investigation_queue)
    eval_csv = export_model_evaluation_csv(
        tfidf_eval=t_eval_rep,
        embedding_eval=e_eval_rep,
    )
    full_markdown = export_full_markdown_report(insights, evaluation_report=eval_rep, product_context=ctx)

    dcol1, dcol2, dcol3, dcol4, dcol5 = st.columns(5)
    with dcol1:
        st.download_button(
            label="📄 Feedback Records CSV",
            data=feedback_csv,
            file_name="feedback_records_with_clusters.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dcol2:
        st.download_button(
            label="📊 Cluster Summaries CSV",
            data=summaries_csv,
            file_name="cluster_summaries.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dcol3:
        st.download_button(
            label="🧭 Investigation Queue CSV",
            data=recs_csv,
            file_name="investigation_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dcol4:
        st.download_button(
            label="📈 Model Metrics CSV",
            data=eval_csv,
            file_name="model_evaluation_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dcol5:
        st.download_button(
            label="📝 Full Markdown Report",
            data=full_markdown,
            file_name="product_intelligence_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =============================================================================
# MAIN APP ENTRY POINT
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="AI Product Intelligence Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_sidebar_project_manager(st.session_state.db)
    render_header()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Product & Input Data",
        "⚙️ Analysis Workspace",
        "🔍 Cluster Explorer",
        "📊 Model Comparison",
        "💡 Insights Report & Export",
    ])

    with tab1:
        render_product_context_tab()

    with tab2:
        render_analysis_tab()

    with tab3:
        render_cluster_explorer_tab()

    with tab4:
        render_model_comparison_tab()

    with tab5:
        render_insights_tab()


if __name__ == "__main__":
    main()
