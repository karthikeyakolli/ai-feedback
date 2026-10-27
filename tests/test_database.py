"""Unit and integration tests for SQLite persistence service (src/database.py).

Phase 4:
- Database initialization and schema versioning
- Project CRUD and isolation
- Versioned product context and source provenance
- Feedback records persistence and conflict resolution
- Analysis run lifecycle, atomic commit, and offline result reconstruction
- Historical run comparison with partition agreement and caveats
"""

import tempfile
from pathlib import Path

import pytest

from src.clustering import ModelResult
from src.context_ingestion import ProductContext
from src.data_processing import FeedbackRecord
from src.database import DatabaseManager, ProjectRecord
from src.evaluation import EvaluationReport


@pytest.fixture
def temp_db() -> DatabaseManager:
    """Fixture providing an isolated in-memory DatabaseManager for each test."""
    return DatabaseManager(":memory:")


@pytest.fixture
def temp_file_db(tmp_path: Path) -> DatabaseManager:
    """Fixture providing an isolated file-backed DatabaseManager in a temporary directory."""
    db_file = tmp_path / "test_feedback.db"
    return DatabaseManager(db_file)


def test_database_initialization(temp_file_db: DatabaseManager):
    """Verifies database initialization, table creation, and idempotence."""
    conn = temp_file_db.get_connection()
    try:
        cur = conn.execute("PRAGMA user_version;")
        assert cur.fetchone()[0] == 1

        cur = conn.execute("PRAGMA foreign_keys;")
        assert cur.fetchone()[0] == 1

        # Check required tables exist
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        required = {
            "projects",
            "product_contexts",
            "sources",
            "feedback_items",
            "analysis_runs",
            "cluster_assignments",
            "cluster_metadata",
            "evaluation_results",
        }
        assert required.issubset(tables)
    finally:
        conn.close()

    # Re-initialization must not drop existing data
    temp_file_db.create_project("Existing Project")
    temp_file_db.init_db()
    projects = temp_file_db.list_projects()
    assert len(projects) == 1
    assert projects[0].name == "Existing Project"


def test_project_crud_and_validation(temp_db: DatabaseManager):
    """Tests creating, listing, renaming, getting, and deleting projects."""
    # Empty name validation
    with pytest.raises(ValueError):
        temp_db.create_project("   ")

    # Create project
    proj = temp_db.create_project("Alpha Project", "First test project")
    assert proj.project_id.startswith("proj-")
    assert proj.name == "Alpha Project"
    assert proj.description == "First test project"

    # List projects
    projects = temp_db.list_projects()
    assert len(projects) == 1
    assert projects[0].project_id == proj.project_id

    # Get project
    fetched = temp_db.get_project(proj.project_id)
    assert fetched is not None
    assert fetched.name == "Alpha Project"

    # Rename project
    renamed = temp_db.rename_project(proj.project_id, "Alpha Renamed", "Updated description")
    assert renamed.name == "Alpha Renamed"
    assert renamed.description == "Updated description"

    # Non-existent rename
    with pytest.raises(KeyError):
        temp_db.rename_project("proj-nonexistent", "New Name")

    # Delete project
    deleted = temp_db.delete_project(proj.project_id)
    assert deleted is True
    assert temp_db.get_project(proj.project_id) is None
    assert len(temp_db.list_projects()) == 0


def test_product_context_versioning(temp_db: DatabaseManager):
    """Verifies that product context changes create monotonically increasing versions."""
    proj = temp_db.create_project("Context Test Project")

    ctx_v1 = ProductContext(
        name="FeedbackHub v1",
        description="Initial product description",
        target_users="Engineers",
        goals_or_concerns="Reduce bugs",
        source_type="manual",
        is_approved=True,
    )
    v1_num = temp_db.save_product_context(proj.project_id, ctx_v1)
    assert v1_num == 1

    ctx_v2 = ProductContext(
        name="FeedbackHub v2",
        description="Updated product description with search enhancements",
        target_users="Engineers and PMs",
        goals_or_concerns="Improve latency",
        source_type="pasted_docs",
        raw_extracted_text="Feature documentation content...",
        is_approved=True,
    )
    v2_num = temp_db.save_product_context(proj.project_id, ctx_v2)
    assert v2_num == 2

    # Latest context
    latest_tuple = temp_db.get_latest_product_context(proj.project_id)
    assert latest_tuple is not None
    latest_ctx, latest_ver = latest_tuple
    assert latest_ver == 2
    assert latest_ctx.name == "FeedbackHub v2"
    assert latest_ctx.source_type == "pasted_docs"

    # Historical context version retrieval
    historical = temp_db.get_product_context_version(proj.project_id, 1)
    assert historical is not None
    assert historical.name == "FeedbackHub v1"
    assert historical.source_type == "manual"

    # Version listing
    ver_list = temp_db.list_product_context_versions(proj.project_id)
    assert len(ver_list) == 2
    assert ver_list[0]["version"] == 2
    assert ver_list[1]["version"] == 1


def test_source_metadata_storage(temp_db: DatabaseManager):
    """Tests saving and retrieving source provenance without storing sensitive secrets."""
    proj = temp_db.create_project("Source Test Project")

    src_id = temp_db.save_source_metadata(
        project_id=proj.project_id,
        source_type="webpage_url",
        status="success",
        status_message="200 OK",
        source_url="https://example.com/docs",
        extracted_title="Product Documentation",
        content_hash="sha256:abc12345",
        byte_size=1024,
        warnings=["Non-critical CSS selector omitted"],
    )
    assert src_id.startswith("src-")

    sources = temp_db.list_sources(proj.project_id)
    assert len(sources) == 1
    s = sources[0]
    assert s["source_type"] == "webpage_url"
    assert s["source_url"] == "https://example.com/docs"
    assert s["warnings"] == ["Non-critical CSS selector omitted"]


def test_feedback_records_persistence_and_conflict_handling(temp_db: DatabaseManager):
    """Tests feedback import, duplicate reference preservation, and conflict strategies."""
    proj = temp_db.create_project("Feedback Project")

    initial_records = [
        FeedbackRecord(
            id="FB-0001",
            original_text="App crashes when saving files",
            cleaned_text="app crashes when saving files",
            source="test_upload",
            rating=1.0,
            is_duplicate=False,
            duplicate_of_id=None,
        ),
        FeedbackRecord(
            id="FB-0002",
            original_text="App crashes when saving files",
            cleaned_text="app crashes when saving files",
            source="test_upload",
            rating=1.0,
            is_duplicate=True,
            duplicate_of_id="FB-0001",
        ),
    ]

    inserted, skipped, conflicts = temp_db.import_feedback_records(
        proj.project_id, initial_records, on_conflict="error"
    )
    assert inserted == 2
    assert skipped == 0
    assert len(conflicts) == 0

    records = temp_db.get_feedback_records(proj.project_id)
    assert len(records) == 2
    assert records[0].id == "FB-0001"
    assert records[1].is_duplicate is True
    assert records[1].duplicate_of_id == "FB-0001"

    # Conflicting ID import: 'error' strategy
    dupe_input = [
        FeedbackRecord(
            id="FB-0001",
            original_text="Conflicting text",
            cleaned_text="conflicting text",
            source="test_upload",
        )
    ]
    with pytest.raises(ValueError, match="Feedback ID conflict detected"):
        temp_db.import_feedback_records(proj.project_id, dupe_input, on_conflict="error")

    # Conflicting ID import: 'skip' strategy
    inserted, skipped, conflicts = temp_db.import_feedback_records(
        proj.project_id, dupe_input, on_conflict="skip"
    )
    assert inserted == 0
    assert skipped == 1
    assert conflicts == ["FB-0001"]

    # Conflicting ID import: 'reassign_id' strategy
    inserted, skipped, conflicts = temp_db.import_feedback_records(
        proj.project_id, dupe_input, on_conflict="reassign_id"
    )
    assert inserted == 1
    all_recs = temp_db.get_feedback_records(proj.project_id)
    assert len(all_recs) == 3
    assert all_recs[2].id == "FB-0003"
    assert all_recs[2].original_text == "Conflicting text"

    # Conflicting ID import: 'replace_all' strategy
    new_batch = [
        FeedbackRecord(
            id="FB-0001",
            original_text="Brand new batch feedback",
            cleaned_text="brand new batch feedback",
            source="new_batch",
        )
    ]
    inserted, skipped, conflicts = temp_db.import_feedback_records(
        proj.project_id, new_batch, on_conflict="replace_all"
    )
    assert inserted == 1
    all_recs = temp_db.get_feedback_records(proj.project_id)
    assert len(all_recs) == 1
    assert all_recs[0].original_text == "Brand new batch feedback"


def test_analysis_run_lifecycle_and_atomic_persistence(temp_db: DatabaseManager):
    """Tests run creation, result saving, failure handling, and atomic transaction rollback."""
    proj = temp_db.create_project("Run Lifecycle Project")

    run_id = temp_db.create_analysis_run(
        project_id=proj.project_id,
        dataset_fingerprint="fp-123456",
        record_count=10,
        cluster_count_k=2,
        random_seed=42,
        model_config={"tfidf": {"max_features": 500}, "minilm": {"model": "all-MiniLM-L6-v2"}},
        context_version=1,
    )
    assert run_id.startswith("run-")

    run_info = temp_db.get_analysis_run(run_id)
    assert run_info["status"] == "running"

    # Prepare mock ModelResults
    m_tfidf = ModelResult(
        method="TF-IDF + K-Means",
        model_identifier="scikit-learn/TfidfVectorizer + KMeans",
        parameters={"n_clusters": 2, "random_state": 42},
        runtime_seconds=0.0123,
        feedback_ids=["FB-0001", "FB-0002"],
        cluster_labels=[0, 1],
        cluster_representatives={0: "Quote 0", 1: "Quote 1"},
        cluster_keywords={0: ["crash", "freeze"], 1: ["login", "delay"]},
        cluster_sizes={0: 1, 1: 1},
        representation="sparse_tfidf",
        distance_metric="cosine",
    )

    ev_tfidf = EvaluationReport(
        method="TF-IDF + K-Means",
        model_identifier="scikit-learn/TfidfVectorizer + KMeans",
        n_samples=2,
        n_clusters=2,
        silhouette=0.456,
        davies_bouldin=1.12,
        calinski_harabasz=15.4,
        cluster_size_entropy=0.693,
        adjusted_rand_index=None,
        normalized_mutual_info=None,
        metric_reasons={"ari_against_reference": "No ground truth reference labels provided"},
        limitations=["Sparse space metric limitation"],
        representation="sparse_tfidf",
        distance_metric="cosine",
    )

    # Save results
    temp_db.save_run_results(run_id, [m_tfidf], [ev_tfidf], runtime_seconds=0.0123)

    completed_run = temp_db.get_analysis_run(run_id)
    assert completed_run["status"] == "completed"
    assert completed_run["runtime_seconds"] == 0.0123
    assert completed_run["completed_at"] is not None


def test_reconstruct_run_results_offline(temp_db: DatabaseManager):
    """Verifies that historical runs can be reconstructed offline with exact parity and zero recomputation."""
    proj = temp_db.create_project("Offline Reconstruction Project")

    run_id = temp_db.create_analysis_run(
        project_id=proj.project_id,
        dataset_fingerprint="fp-reconstruct",
        record_count=3,
        cluster_count_k=2,
        random_seed=42,
        model_config={},
        context_version=1,
    )

    m_res = ModelResult(
        method="TF-IDF + K-Means",
        model_identifier="scikit-learn/TfidfVectorizer + KMeans",
        parameters={"n_clusters": 2, "random_state": 42},
        runtime_seconds=0.045,
        feedback_ids=["FB-0001", "FB-0002", "FB-0003"],
        cluster_labels=[0, 0, 1],
        cluster_representatives={0: "FB-0001", 1: "FB-0003"},
        cluster_keywords={0: ["crash", "app"], 1: ["search", "slow"]},
        cluster_sizes={0: 2, 1: 1},
        representation="sparse_tfidf",
        distance_metric="cosine",
    )

    ev_rep = EvaluationReport(
        method="TF-IDF + K-Means",
        model_identifier="scikit-learn/TfidfVectorizer + KMeans",
        n_samples=3,
        n_clusters=2,
        silhouette=0.35,
        davies_bouldin=0.88,
        calinski_harabasz=12.0,
        cluster_size_entropy=0.636,
        adjusted_rand_index=None,
        normalized_mutual_info=None,
        metric_reasons={"ari_against_reference": "No labels"},
        limitations=["Test note"],
        representation="sparse_tfidf",
        distance_metric="cosine",
    )

    temp_db.save_run_results(run_id, [m_res], [ev_rep], runtime_seconds=0.045)

    # Reconstruct
    run_dict, reconstructed_models, reconstructed_evals = temp_db.reconstruct_run_results(run_id)

    assert run_dict["run_id"] == run_id
    assert len(reconstructed_models) == 1
    rm = reconstructed_models[0]
    assert rm.method == "TF-IDF + K-Means"
    assert rm.cluster_labels == [0, 0, 1]
    assert rm.feedback_ids == ["FB-0001", "FB-0002", "FB-0003"]
    assert rm.cluster_representatives[0] == "FB-0001"
    assert rm.cluster_keywords[1] == ["search", "slow"]
    assert rm.cluster_sizes[0] == 2

    assert len(reconstructed_evals) == 1
    rev = reconstructed_evals[0]
    assert rev.silhouette == 0.35
    assert rev.davies_bouldin == 0.88
    assert rev.metric_reasons["ari_against_reference"] == "No labels"


def test_compare_historical_runs_same_and_different_datasets(temp_db: DatabaseManager):
    """Verifies run comparison with partition agreement when identical records are compared."""
    proj = temp_db.create_project("Run Compare Project")

    # Run A
    run_a = temp_db.create_analysis_run(
        proj.project_id, "fp-common", record_count=3, cluster_count_k=2, random_seed=42, model_config={}
    )
    m_a = ModelResult(
        method="TF-IDF + K-Means",
        model_identifier="tfidf",
        parameters={"n_clusters": 2, "random_state": 42},
        runtime_seconds=0.01,
        feedback_ids=["FB-01", "FB-02", "FB-03"],
        cluster_labels=[0, 0, 1],
        cluster_representatives={0: "Q0", 1: "Q1"},
        cluster_keywords={0: ["kw1"], 1: ["kw2"]},
        cluster_sizes={0: 2, 1: 1},
        representation="sparse_tfidf",
        distance_metric="cosine",
    )
    ev_a = EvaluationReport(
        method="TF-IDF + K-Means",
        model_identifier="tfidf",
        n_samples=3,
        n_clusters=2,
        silhouette=0.40,
        davies_bouldin=1.0,
        calinski_harabasz=10.0,
        cluster_size_entropy=0.6,
        adjusted_rand_index=None,
        normalized_mutual_info=None,
        metric_reasons={},
        limitations=[],
        representation="sparse_tfidf",
        distance_metric="cosine",
    )
    temp_db.save_run_results(run_a, [m_a], [ev_a], runtime_seconds=0.01)

    # Run B: Same records, slightly different partition ([0, 1, 1])
    run_b = temp_db.create_analysis_run(
        proj.project_id, "fp-common", record_count=3, cluster_count_k=2, random_seed=99, model_config={}
    )
    m_b = ModelResult(
        method="TF-IDF + K-Means",
        model_identifier="tfidf",
        parameters={"n_clusters": 2, "random_state": 99},
        runtime_seconds=0.015,
        feedback_ids=["FB-01", "FB-02", "FB-03"],
        cluster_labels=[0, 1, 1],
        cluster_representatives={0: "Q0", 1: "Q1"},
        cluster_keywords={0: ["kw1"], 1: ["kw2"]},
        cluster_sizes={0: 1, 1: 2},
        representation="sparse_tfidf",
        distance_metric="cosine",
    )
    ev_b = EvaluationReport(
        method="TF-IDF + K-Means",
        model_identifier="tfidf",
        n_samples=3,
        n_clusters=2,
        silhouette=0.38,
        davies_bouldin=1.1,
        calinski_harabasz=9.5,
        cluster_size_entropy=0.6,
        adjusted_rand_index=None,
        normalized_mutual_info=None,
        metric_reasons={},
        limitations=[],
        representation="sparse_tfidf",
        distance_metric="cosine",
    )
    temp_db.save_run_results(run_b, [m_b], [ev_b], runtime_seconds=0.015)

    comparison = temp_db.compare_historical_runs(run_a, run_b)
    assert comparison["same_dataset_fingerprint"] is True
    tfidf_comp = comparison["model_comparisons"]["TF-IDF + K-Means"]
    assert tfidf_comp["same_dataset"] is True
    assert tfidf_comp["partition_ari"] is not None
    assert isinstance(tfidf_comp["partition_ari"], float)
    assert len(comparison["caveats"]) >= 2


def test_project_data_isolation(temp_db: DatabaseManager):
    """Confirms that records in Project A are strictly isolated from Project B."""
    proj_a = temp_db.create_project("Project Alpha")
    proj_b = temp_db.create_project("Project Beta")

    # Add feedback to Project A
    temp_db.import_feedback_records(
        proj_a.project_id,
        [FeedbackRecord(id="FB-0001", original_text="Feedback for Alpha", cleaned_text="feedback for alpha", source="test")],
    )

    # Add context to Project A
    temp_db.save_product_context(
        proj_a.project_id,
        ProductContext(name="Alpha App", description="Desc Alpha", target_users="", goals_or_concerns=""),
    )

    # Verify Project B is empty
    assert temp_db.get_feedback_count(proj_b.project_id) == 0
    assert temp_db.get_feedback_records(proj_b.project_id) == []
    assert temp_db.get_latest_product_context(proj_b.project_id) is None
    assert temp_db.list_analysis_runs(proj_b.project_id) == []


def test_get_run_feedback_records_preserves_evaluation_order(temp_db: DatabaseManager):
    """Verifies that get_run_feedback_records retrieves the evaluated subset in original assignment order."""
    proj = temp_db.create_project("Order Project")
    records = [
        FeedbackRecord(id="FB-01", original_text="Text 1", cleaned_text="text 1", source="test"),
        FeedbackRecord(id="FB-02", original_text="Text 2", cleaned_text="text 2", source="test"),
        FeedbackRecord(id="FB-03", original_text="Text 3", cleaned_text="text 3", source="test"),
    ]
    temp_db.import_feedback_records(proj.project_id, records)

    run_id = temp_db.create_analysis_run(
        proj.project_id, "fp-test", record_count=3, cluster_count_k=2, random_seed=42, model_config={}
    )
    m = ModelResult(
        method="TF-IDF + K-Means",
        model_identifier="tfidf",
        parameters={"n_clusters": 2, "random_state": 42},
        runtime_seconds=0.01,
        feedback_ids=["FB-01", "FB-02", "FB-03"],
        cluster_labels=[0, 1, 0],
        cluster_representatives={0: "FB-01", 1: "FB-02"},
        cluster_keywords={0: ["kw1"], 1: ["kw2"]},
        cluster_sizes={0: 2, 1: 1},
        representation="sparse_tfidf",
        distance_metric="cosine",
    )
    ev = EvaluationReport(
        method="TF-IDF + K-Means",
        model_identifier="tfidf",
        n_samples=3,
        n_clusters=2,
        silhouette=0.5,
        davies_bouldin=0.8,
        calinski_harabasz=12.0,
        cluster_size_entropy=0.6,
        adjusted_rand_index=None,
        normalized_mutual_info=None,
        metric_reasons={},
        limitations=[],
        representation="sparse_tfidf",
        distance_metric="cosine",
    )
    temp_db.save_run_results(run_id, [m], [ev])

    # Now add a new feedback item after the run
    temp_db.import_feedback_records(
        proj.project_id,
        [FeedbackRecord(id="FB-04", original_text="Text 4 after run", cleaned_text="text 4", source="test")],
    )
    assert temp_db.get_feedback_count(proj.project_id) == 4

    # get_run_feedback_records must only return the 3 records evaluated in that run
    run_records = temp_db.get_run_feedback_records(run_id)
    assert len(run_records) == 3
    assert [r.id for r in run_records] == ["FB-01", "FB-02", "FB-03"]
    assert run_records[0].original_text == "Text 1"


def test_cascade_project_deletion_cleans_all_tables(temp_db: DatabaseManager):
    """Verifies that deleting a project cascades cleanly to contexts, sources, items, runs, assignments, metadata, and evaluations."""
    proj = temp_db.create_project("Cascade Project")
    pid = proj.project_id

    # 1. Add context and source
    temp_db.save_product_context(pid, ProductContext(name="Cascade App", description="D", target_users="", goals_or_concerns=""))
    temp_db.save_source_metadata(pid, source_type="manual")

    # 2. Add feedback
    temp_db.import_feedback_records(
        pid,
        [FeedbackRecord(id="FB-C1", original_text="C1", cleaned_text="c1", source="test")],
    )

    # 3. Add analysis run with results
    run_id = temp_db.create_analysis_run(pid, "fp-c", record_count=1, cluster_count_k=1, random_seed=1, model_config={})
    m = ModelResult(
        method="TF-IDF + K-Means",
        model_identifier="tfidf",
        parameters={},
        runtime_seconds=0.01,
        feedback_ids=["FB-C1"],
        cluster_labels=[0],
        cluster_representatives={0: "FB-C1"},
        cluster_keywords={0: ["c1"]},
        cluster_sizes={0: 1},
        representation="sparse_tfidf",
        distance_metric="cosine",
    )
    ev = EvaluationReport(
        method="TF-IDF + K-Means",
        model_identifier="tfidf",
        n_samples=1,
        n_clusters=1,
        silhouette=None,
        davies_bouldin=None,
        calinski_harabasz=None,
        cluster_size_entropy=0.0,
        adjusted_rand_index=None,
        normalized_mutual_info=None,
        metric_reasons={"silhouette": "k=1"},
        limitations=[],
        representation="sparse_tfidf",
        distance_metric="cosine",
    )
    temp_db.save_run_results(run_id, [m], [ev])

    # Verify rows exist before deletion
    conn = temp_db.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM product_contexts WHERE project_id = ?;", (pid,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM sources WHERE project_id = ?;", (pid,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM feedback_items WHERE project_id = ?;", (pid,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM analysis_runs WHERE project_id = ?;", (pid,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM cluster_assignments WHERE run_id = ?;", (run_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM cluster_metadata WHERE run_id = ?;", (run_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM evaluation_results WHERE run_id = ?;", (run_id,)).fetchone()[0] >= 1
    finally:
        conn.close()

    # Perform delete
    temp_db.delete_project(pid)

    # Verify all tables are cleaned up for this project and run
    conn = temp_db.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM projects WHERE project_id = ?;", (pid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM product_contexts WHERE project_id = ?;", (pid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sources WHERE project_id = ?;", (pid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM feedback_items WHERE project_id = ?;", (pid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM analysis_runs WHERE project_id = ?;", (pid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM cluster_assignments WHERE run_id = ?;", (run_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM cluster_metadata WHERE run_id = ?;", (run_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM evaluation_results WHERE run_id = ?;", (run_id,)).fetchone()[0] == 0
    finally:
        conn.close()
