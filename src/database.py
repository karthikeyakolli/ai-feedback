"""SQLite persistence service layer for AI Product Intelligence & Feedback Analyzer.

Phase 4:
- Local SQLite database management with foreign keys enabled
- Projects CRUD and project-scoped isolation
- Versioned product context and source provenance storage
- Feedback records persistence with conflict resolution (error, replace_all, skip, reassign_id)
- Atomic analysis run recording (running -> completed/failed)
- Historical run reconstruction without model recomputation
- Historical run comparison with partition agreement (ARI) and caveats
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from sklearn.metrics import adjusted_rand_score

from src.clustering import ModelResult
from src.context_ingestion import ProductContext
from src.data_processing import FeedbackRecord, ValidationReport
from src.evaluation import EvaluationReport


def get_default_db_path() -> Path:
    """Returns the default database path, configurable via DATABASE_PATH environment variable."""
    env_path = os.environ.get("DATABASE_PATH")
    if env_path:
        return Path(env_path)
    return Path("data/product_intelligence.db")


@dataclass
class ProjectRecord:
    """Data representation of a project."""

    project_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    schema_version: int = 1


class DatabaseManager:
    """Manages SQLite database connections, schema migrations, and queries."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._is_memory = False
        self._keepalive_conn: sqlite3.Connection | None = None

        if db_path is None:
            self.db_path = get_default_db_path()
        elif str(db_path) == ":memory:":
            self._is_memory = True
            self.db_path = Path(":memory:")
            self._uri = f"file:mem_{uuid.uuid4().hex}?mode=memory&cache=shared"
            # Keep a persistent connection open so the shared in-memory DB is preserved
            self._keepalive_conn = sqlite3.connect(
                self._uri, uri=True, timeout=30.0, check_same_thread=False
            )
            self._keepalive_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            self.db_path = Path(db_path)

        if not self._is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Creates a context-managed sqlite3 connection with foreign keys and row factory."""
        if self._is_memory:
            conn = sqlite3.connect(
                self._uri,
                uri=True,
                timeout=30.0,
                check_same_thread=False,
            )
        else:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
            )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if not self._is_memory:
            conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides an atomic transaction scope with automatic commit/rollback."""
        conn = self.get_connection()
        try:
            conn.execute("BEGIN TRANSACTION;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initializes database schema and ensures required tables and indexes exist."""
        with self.transaction() as conn:
            # Check user_version
            cur = conn.execute("PRAGMA user_version;")
            row = cur.fetchone()
            current_version = row[0] if row else 0

            # 1. projects table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1
                );
                """
            )

            # 2. product_contexts table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS product_contexts (
                    context_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    product_name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    target_users TEXT DEFAULT '',
                    goals_or_concerns TEXT DEFAULT '',
                    approved_text TEXT DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT 'manual',
                    source_url TEXT,
                    extracted_title TEXT,
                    is_approved INTEGER NOT NULL DEFAULT 0,
                    warnings_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, version)
                );
                """
            )

            # 3. sources table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                    context_id TEXT REFERENCES product_contexts(context_id) ON DELETE SET NULL,
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    extracted_title TEXT,
                    status TEXT NOT NULL DEFAULT 'success',
                    status_message TEXT DEFAULT '',
                    content_hash TEXT DEFAULT '',
                    byte_size INTEGER DEFAULT 0,
                    warnings_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                """
            )

            # 4. feedback_items table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_items (
                    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                    feedback_id TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    rating REAL,
                    submitted_at TEXT,
                    is_duplicate INTEGER NOT NULL DEFAULT 0,
                    duplicate_of_id TEXT,
                    source_type TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, feedback_id)
                );
                """
            )

            # 5. analysis_runs table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                    context_version INTEGER,
                    status TEXT NOT NULL DEFAULT 'running',
                    dataset_fingerprint TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    cluster_count_k INTEGER NOT NULL,
                    random_seed INTEGER NOT NULL,
                    runtime_seconds REAL DEFAULT 0.0,
                    model_config_json TEXT NOT NULL DEFAULT '{}',
                    error_message TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                """
            )

            # 6. cluster_assignments table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                    feedback_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    is_representative INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(run_id, model_name, feedback_id)
                );
                """
            )

            # 7. cluster_metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_metadata (
                    meta_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                    model_name TEXT NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    keywords_json TEXT NOT NULL DEFAULT '[]',
                    representative_quote TEXT NOT NULL DEFAULT '',
                    representative_id TEXT NOT NULL DEFAULT '',
                    member_ids_json TEXT NOT NULL DEFAULT '[]',
                    UNIQUE(run_id, model_name, cluster_id)
                );
                """
            )

            # 8. evaluation_results table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    evaluation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    unavailability_reason TEXT,
                    details_json TEXT DEFAULT '{}',
                    UNIQUE(run_id, model_name, metric_name)
                );
                """
            )

            # Indexes for fast lookup
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contexts_proj ON product_contexts(project_id, version);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_proj ON sources(project_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_proj ON feedback_items(project_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_proj ON analysis_runs(project_id, created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assignments_run ON cluster_assignments(run_id, model_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_clustermeta_run ON cluster_metadata(run_id, model_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_results(run_id, model_name);")

            if current_version == 0:
                conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION};")

    # =========================================================================
    # PROJECT OPERATIONS
    # =========================================================================

    def create_project(self, name: str, description: str = "") -> ProjectRecord:
        """Creates a new project record."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Project name cannot be empty or whitespace only.")

        project_id = f"proj-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO projects (project_id, name, description, created_at, updated_at, schema_version)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (project_id, clean_name, description.strip(), now, now, self.SCHEMA_VERSION),
            )

        return ProjectRecord(
            project_id=project_id,
            name=clean_name,
            description=description.strip(),
            created_at=now,
            updated_at=now,
            schema_version=self.SCHEMA_VERSION,
        )

    def list_projects(self) -> list[ProjectRecord]:
        """Lists all projects ordered by updated_at descending."""
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT project_id, name, description, created_at, updated_at, schema_version
                FROM projects
                ORDER BY updated_at DESC;
                """
            )
            rows = cur.fetchall()
            return [
                ProjectRecord(
                    project_id=r["project_id"],
                    name=r["name"],
                    description=r["description"] or "",
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    schema_version=r["schema_version"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_project(self, project_id: str) -> ProjectRecord | None:
        """Retrieves a single project record by project_id."""
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT project_id, name, description, created_at, updated_at, schema_version
                FROM projects
                WHERE project_id = ?;
                """,
                (project_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            return ProjectRecord(
                project_id=r["project_id"],
                name=r["name"],
                description=r["description"] or "",
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                schema_version=r["schema_version"],
            )
        finally:
            conn.close()

    def rename_project(self, project_id: str, new_name: str, new_description: str | None = None) -> ProjectRecord:
        """Renames a project and updates its timestamp."""
        clean_name = new_name.strip()
        if not clean_name:
            raise ValueError("Project name cannot be empty.")

        existing = self.get_project(project_id)
        if not existing:
            raise KeyError(f"Project '{project_id}' not found.")

        now = datetime.now(timezone.utc).isoformat()
        desc = existing.description if new_description is None else new_description.strip()

        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE projects
                SET name = ?, description = ?, updated_at = ?
                WHERE project_id = ?;
                """,
                (clean_name, desc, now, project_id),
            )

        return ProjectRecord(
            project_id=project_id,
            name=clean_name,
            description=desc,
            created_at=existing.created_at,
            updated_at=now,
            schema_version=existing.schema_version,
        )

    def delete_project(self, project_id: str) -> bool:
        """Deletes a project and cascades deletion to all related records."""
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM projects WHERE project_id = ?;", (project_id,))
            return cur.rowcount > 0

    # =========================================================================
    # PRODUCT CONTEXT OPERATIONS
    # =========================================================================

    def save_product_context(self, project_id: str, context: ProductContext) -> int:
        """Saves a new version of product context for the specified project.

        Returns the newly created version number.
        """
        existing_proj = self.get_project(project_id)
        if not existing_proj:
            raise KeyError(f"Project '{project_id}' does not exist.")

        context_id = f"ctx-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        with self.transaction() as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM product_contexts WHERE project_id = ?;",
                (project_id,),
            )
            last_version = cur.fetchone()[0]
            new_version = last_version + 1

            approved_text = context.raw_extracted_text or context.description or ""
            status_msg = context.extraction_message or ""

            conn.execute(
                """
                INSERT INTO product_contexts (
                    context_id, project_id, version, product_name, description,
                    target_users, goals_or_concerns, approved_text, source_type,
                    source_url, extracted_title, is_approved, warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    context_id,
                    project_id,
                    new_version,
                    context.name,
                    context.description,
                    context.target_users,
                    context.goals_or_concerns,
                    approved_text,
                    context.source_type,
                    context.source_url,
                    context.extracted_title,
                    1 if context.is_approved else 0,
                    json.dumps([]),
                    now,
                ),
            )

            # Touch project updated_at
            conn.execute("UPDATE projects SET updated_at = ? WHERE project_id = ?;", (now, project_id))

        return new_version

    def get_latest_product_context(self, project_id: str) -> tuple[ProductContext, int] | None:
        """Retrieves the latest version of product context for a project."""
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT context_id, version, product_name, description, target_users,
                       goals_or_concerns, approved_text, source_type, source_url,
                       extracted_title, is_approved, created_at
                FROM product_contexts
                WHERE project_id = ?
                ORDER BY version DESC
                LIMIT 1;
                """,
                (project_id,),
            )
            r = cur.fetchone()
            if not r:
                return None

            ctx = ProductContext(
                name=r["product_name"],
                description=r["description"] or "",
                target_users=r["target_users"] or "",
                goals_or_concerns=r["goals_or_concerns"] or "",
                source_type=r["source_type"] or "manual",
                source_url=r["source_url"],
                extracted_title=r["extracted_title"],
                raw_extracted_text=r["approved_text"],
                is_approved=bool(r["is_approved"]),
                version=r["version"],
            )
            return ctx, r["version"]
        finally:
            conn.close()

    def get_product_context_version(self, project_id: str, version: int) -> ProductContext | None:
        """Retrieves a specific historical version of product context."""
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT context_id, version, product_name, description, target_users,
                       goals_or_concerns, approved_text, source_type, source_url,
                       extracted_title, is_approved, created_at
                FROM product_contexts
                WHERE project_id = ? AND version = ?;
                """,
                (project_id, version),
            )
            r = cur.fetchone()
            if not r:
                return None

            return ProductContext(
                name=r["product_name"],
                description=r["description"] or "",
                target_users=r["target_users"] or "",
                goals_or_concerns=r["goals_or_concerns"] or "",
                source_type=r["source_type"] or "manual",
                source_url=r["source_url"],
                extracted_title=r["extracted_title"],
                raw_extracted_text=r["approved_text"],
                is_approved=bool(r["is_approved"]),
                version=r["version"],
            )
        finally:
            conn.close()

    def list_product_context_versions(self, project_id: str) -> list[dict[str, Any]]:
        """Lists metadata for all stored context versions of a project."""
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT version, product_name, source_type, is_approved, created_at
                FROM product_contexts
                WHERE project_id = ?
                ORDER BY version DESC;
                """,
                (project_id,),
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    # =========================================================================
    # SOURCE METADATA OPERATIONS
    # =========================================================================

    def save_source_metadata(
        self,
        project_id: str,
        source_type: str,
        status: str = "success",
        status_message: str = "",
        source_url: str | None = None,
        extracted_title: str | None = None,
        content_hash: str | None = None,
        byte_size: int = 0,
        warnings: list[str] | None = None,
        context_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Stores source provenance metadata without refetching or credentials."""
        if "extraction_status" in kwargs and kwargs["extraction_status"]:
            status = kwargs["extraction_status"]
        if "warning_metadata" in kwargs and isinstance(kwargs["warning_metadata"], dict):
            status_message = kwargs["warning_metadata"].get("message", status_message)

        source_id = f"src-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        warnings_json = json.dumps(warnings or [])

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sources (
                    source_id, project_id, context_id, source_type, source_url,
                    extracted_title, status, status_message, content_hash,
                    byte_size, warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    source_id,
                    project_id,
                    context_id,
                    source_type,
                    source_url,
                    extracted_title,
                    status,
                    status_message,
                    content_hash or "",
                    byte_size,
                    warnings_json,
                    now,
                ),
            )
        return source_id

    def list_sources(self, project_id: str) -> list[dict[str, Any]]:
        """Lists sources recorded for a project."""
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT source_id, source_type, source_url, extracted_title, status,
                       status_message, byte_size, warnings_json, created_at
                FROM sources
                WHERE project_id = ?
                ORDER BY created_at DESC;
                """,
                (project_id,),
            )
            rows = []
            for r in cur.fetchall():
                d = dict(r)
                d["warnings"] = json.loads(d.pop("warnings_json", "[]"))
                rows.append(d)
            return rows
        finally:
            conn.close()

    # =========================================================================
    # FEEDBACK IMPORT AND RETRIEVAL
    # =========================================================================

    def import_feedback_records(
        self,
        project_id: str,
        records: list[FeedbackRecord],
        on_conflict: str = "error",
    ) -> tuple[int, int, list[str]]:
        """Imports feedback records into a project with conflict resolution.

        Parameters:
        - project_id: Target project ID.
        - records: List of FeedbackRecord objects.
        - on_conflict: Strategy when incoming feedback_id exists in project:
            - 'error': Raise ValueError, transaction rolls back.
            - 'replace_all': Deletes all existing feedback in project and inserts new records.
            - 'skip': Silently skips incoming records with existing IDs.
            - 'reassign_id': Assigns a new sequential ID (e.g. FB-XXXX) to conflicting records.

        Returns:
        - tuple: (inserted_count, skipped_count, list_of_conflicts)
        """
        if not self.get_project(project_id):
            raise KeyError(f"Project '{project_id}' does not exist.")

        if not records:
            return 0, 0, []

        now = datetime.now(timezone.utc).isoformat()

        with self.transaction() as conn:
            if on_conflict == "replace_all":
                conn.execute("DELETE FROM feedback_items WHERE project_id = ?;", (project_id,))
                existing_ids = set()
                max_seq = 0
            else:
                cur = conn.execute("SELECT feedback_id FROM feedback_items WHERE project_id = ?;", (project_id,))
                existing_ids = {row[0] for row in cur.fetchall()}
                max_seq = 0
                for fid in existing_ids:
                    if fid.startswith("FB-") and fid[3:].isdigit():
                        max_seq = max(max_seq, int(fid[3:]))

            conflicts = []
            to_insert = []
            skipped = 0

            for rec in records:
                target_id = rec.id
                if target_id in existing_ids:
                    conflicts.append(target_id)
                    if on_conflict == "error":
                        raise ValueError(
                            f"Feedback ID conflict detected: ID '{target_id}' already exists in project '{project_id}'."
                        )
                    elif on_conflict == "skip":
                        skipped += 1
                        continue
                    elif on_conflict == "reassign_id":
                        max_seq += 1
                        target_id = f"FB-{max_seq:04d}"
                        existing_ids.add(target_id)

                existing_ids.add(target_id)
                sub_at = rec.metadata.get("submitted_at") if isinstance(rec.metadata, dict) else None
                to_insert.append((
                    project_id,
                    target_id,
                    rec.original_text,
                    rec.cleaned_text,
                    rec.rating,
                    sub_at,
                    1 if rec.is_duplicate else 0,
                    rec.duplicate_of_id,
                    rec.source,
                    now,
                    now,
                ))

            if to_insert:
                conn.executemany(
                    """
                    INSERT INTO feedback_items (
                        project_id, feedback_id, original_text, cleaned_text,
                        rating, submitted_at, is_duplicate, duplicate_of_id,
                        source_type, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    to_insert,
                )

            # Update project timestamp
            conn.execute("UPDATE projects SET updated_at = ? WHERE project_id = ?;", (now, project_id))

        return len(to_insert), skipped, conflicts

    def get_feedback_records(self, project_id: str) -> list[FeedbackRecord]:
        """Retrieves all feedback records for a project ordered by feedback_id."""
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT feedback_id, original_text, cleaned_text, rating,
                       submitted_at, is_duplicate, duplicate_of_id, source_type
                FROM feedback_items
                WHERE project_id = ?
                ORDER BY feedback_id ASC;
                """,
                (project_id,),
            )
            records = []
            for r in cur.fetchall():
                meta = {"submitted_at": r["submitted_at"]} if r["submitted_at"] else {}
                records.append(
                    FeedbackRecord(
                        id=r["feedback_id"],
                        original_text=r["original_text"],
                        cleaned_text=r["cleaned_text"],
                        source=r["source_type"],
                        rating=r["rating"],
                        is_duplicate=bool(r["is_duplicate"]),
                        duplicate_of_id=r["duplicate_of_id"],
                        metadata=meta,
                    )
                )
            return records
        finally:
            conn.close()

    def get_run_feedback_records(self, run_id: str) -> list[FeedbackRecord]:
        """Retrieves the feedback records that were evaluated in a specific run, in assignment order."""
        run = self.get_analysis_run(run_id)
        if not run:
            return []
        conn = self.get_connection()
        try:
            # Query feedback items joining on cluster assignments for the run
            cur = conn.execute(
                """
                SELECT fi.feedback_id, fi.original_text, fi.cleaned_text, fi.rating,
                       fi.submitted_at, fi.is_duplicate, fi.duplicate_of_id, fi.source_type
                FROM cluster_assignments ca
                JOIN feedback_items fi ON ca.feedback_id = fi.feedback_id AND fi.project_id = ?
                WHERE ca.run_id = ? AND ca.model_name = (
                    SELECT model_name FROM cluster_assignments WHERE run_id = ? LIMIT 1
                )
                ORDER BY ca.assignment_id ASC;
                """,
                (run["project_id"], run_id, run_id),
            )
            records = []
            for r in cur.fetchall():
                meta = {"submitted_at": r["submitted_at"]} if r["submitted_at"] else {}
                records.append(
                    FeedbackRecord(
                        id=r["feedback_id"],
                        original_text=r["original_text"],
                        cleaned_text=r["cleaned_text"],
                        source=r["source_type"],
                        rating=r["rating"],
                        is_duplicate=bool(r["is_duplicate"]),
                        duplicate_of_id=r["duplicate_of_id"],
                        metadata=meta,
                    )
                )
            return records
        finally:
            conn.close()

    def get_feedback_count(self, project_id: str) -> int:
        """Returns total count of feedback items stored for a project."""
        conn = self.get_connection()
        try:
            cur = conn.execute("SELECT COUNT(*) FROM feedback_items WHERE project_id = ?;", (project_id,))
            return cur.fetchone()[0]
        finally:
            conn.close()

    def delete_project_feedback(self, project_id: str) -> int:
        """Deletes all feedback items for a project."""
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM feedback_items WHERE project_id = ?;", (project_id,))
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("UPDATE projects SET updated_at = ? WHERE project_id = ?;", (now, project_id))
            return cur.rowcount

    # =========================================================================
    # ANALYSIS RUNS & HISTORICAL RESULTS
    # =========================================================================

    def create_analysis_run(
        self,
        project_id: str,
        dataset_fingerprint: str,
        record_count: int,
        cluster_count_k: int,
        random_seed: int,
        model_config: dict[str, Any],
        context_version: int | None = None,
    ) -> str:
        """Initializes a new analysis run with 'running' status.

        Returns run_id.
        """
        if not self.get_project(project_id):
            raise KeyError(f"Project '{project_id}' does not exist.")

        run_id = f"run-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        config_json = json.dumps(model_config)

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO analysis_runs (
                    run_id, project_id, context_version, status, dataset_fingerprint,
                    record_count, cluster_count_k, random_seed, runtime_seconds,
                    model_config_json, error_message, created_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, 0.0, ?, '', ?);
                """,
                (
                    run_id,
                    project_id,
                    context_version,
                    dataset_fingerprint,
                    record_count,
                    cluster_count_k,
                    random_seed,
                    config_json,
                    now,
                ),
            )
        return run_id

    def save_run_results(
        self,
        run_id: str,
        model_results: list[ModelResult],
        eval_reports: list[EvaluationReport],
        runtime_seconds: float = 0.0,
    ) -> None:
        """Atomically saves cluster assignments, metadata, and evaluation results.

        Sets run status to 'completed'.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self.transaction() as conn:
            # 1. Verify run exists
            cur = conn.execute("SELECT project_id, status FROM analysis_runs WHERE run_id = ?;", (run_id,))
            run_row = cur.fetchone()
            if not run_row:
                raise KeyError(f"Analysis run '{run_id}' not found.")

            # 2. Insert assignments and cluster metadata for each model
            for m_res in model_results:
                mname = m_res.method
                assignments = []
                for idx, (fid, clabel) in enumerate(zip(m_res.feedback_ids, m_res.cluster_labels)):
                    assignment_id = f"asgn-{idx:06d}-{uuid.uuid4().hex[:6]}"
                    rep_id = m_res.cluster_representatives.get(clabel, "")
                    is_rep = 1 if (rep_id and fid == rep_id) else 0
                    assignments.append((assignment_id, run_id, fid, mname, clabel, is_rep))

                conn.executemany(
                    """
                    INSERT INTO cluster_assignments (
                        assignment_id, run_id, feedback_id, model_name, cluster_id, is_representative
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    assignments,
                )

                # Insert cluster metadata
                cluster_metas = []
                unique_clusters = sorted(set(m_res.cluster_labels))
                for cid in unique_clusters:
                    meta_id = f"cmeta-{uuid.uuid4().hex[:10]}"
                    keywords = m_res.cluster_keywords.get(cid, [])
                    rep_id = m_res.cluster_representatives.get(cid, "")
                    members = [fid for fid, lab in zip(m_res.feedback_ids, m_res.cluster_labels) if lab == cid]

                    cluster_metas.append((
                        meta_id,
                        run_id,
                        mname,
                        cid,
                        json.dumps(keywords),
                        "",
                        rep_id,
                        json.dumps(members),
                    ))

                conn.executemany(
                    """
                    INSERT INTO cluster_metadata (
                        meta_id, run_id, model_name, cluster_id, keywords_json,
                        representative_quote, representative_id, member_ids_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    cluster_metas,
                )

            # 3. Insert evaluation results
            for ev in eval_reports:
                eval_entries = [
                    ("silhouette_score", ev.silhouette),
                    ("davies_bouldin_index", ev.davies_bouldin),
                    ("calinski_harabasz_index", ev.calinski_harabasz),
                    ("cluster_entropy", ev.cluster_size_entropy),
                    ("ari_against_reference", ev.adjusted_rand_index),
                    ("nmi_against_reference", ev.normalized_mutual_info),
                ]

                details = {
                    "representation": ev.representation,
                    "distance_metric": ev.distance_metric,
                    "limitations": ev.limitations,
                    "metrics_validity": ev.metrics_validity,
                }
                details_json = json.dumps(details)

                eval_rows = []
                for m_name, m_val in eval_entries:
                    eval_id = f"eval-{uuid.uuid4().hex[:10]}"
                    unavail_reason = ev.metric_reasons.get(m_name, "") if m_val is None else ""
                    eval_rows.append((eval_id, run_id, ev.method, m_name, m_val, unavail_reason, details_json))

                conn.executemany(
                    """
                    INSERT INTO evaluation_results (
                        evaluation_id, run_id, model_name, metric_name, metric_value,
                        unavailability_reason, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    eval_rows,
                )

            # 4. Mark run completed
            conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'completed', runtime_seconds = ?, completed_at = ?
                WHERE run_id = ?;
                """,
                (runtime_seconds, now, run_id),
            )

    def record_run_failure(self, run_id: str, error_message: str) -> None:
        """Safely records run failure without throwing an exception."""
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'failed', error_message = ?, completed_at = ?
                WHERE run_id = ?;
                """,
                (error_message, now, run_id),
            )

    def list_analysis_runs(self, project_id: str, status: str | None = None) -> list[dict[str, Any]]:
        """Lists all analysis runs for a project ordered by created_at descending."""
        conn = self.get_connection()
        try:
            if status:
                cur = conn.execute(
                    """
                    SELECT run_id, project_id, context_version, status, dataset_fingerprint,
                           record_count, cluster_count_k, random_seed, runtime_seconds,
                           error_message, created_at, completed_at
                    FROM analysis_runs
                    WHERE project_id = ? AND status = ?
                    ORDER BY created_at DESC;
                    """,
                    (project_id, status),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT run_id, project_id, context_version, status, dataset_fingerprint,
                           record_count, cluster_count_k, random_seed, runtime_seconds,
                           error_message, created_at, completed_at
                    FROM analysis_runs
                    WHERE project_id = ?
                    ORDER BY created_at DESC;
                    """,
                    (project_id,),
                )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def get_analysis_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieves raw run record by run_id."""
        conn = self.get_connection()
        try:
            cur = conn.execute(
                """
                SELECT run_id, project_id, context_version, status, dataset_fingerprint,
                       record_count, cluster_count_k, random_seed, runtime_seconds,
                       model_config_json, error_message, created_at, completed_at
                FROM analysis_runs
                WHERE run_id = ?;
                """,
                (run_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            res = dict(r)
            res["model_config"] = json.loads(res.pop("model_config_json", "{}"))
            return res
        finally:
            conn.close()

    def reconstruct_run_results(
        self, run_id: str
    ) -> tuple[dict[str, Any], list[ModelResult], list[EvaluationReport]]:
        """Reconstructs complete ModelResult and EvaluationReport objects from saved records.

        Executes 100% locally from SQLite without running ML vectorizers or encoders.
        """
        run_info = self.get_analysis_run(run_id)
        if not run_info:
            raise KeyError(f"Run '{run_id}' not found.")

        conn = self.get_connection()
        try:
            # 1. Fetch assignments
            cur = conn.execute(
                """
                SELECT model_name, feedback_id, cluster_id, is_representative
                FROM cluster_assignments
                WHERE run_id = ?
                ORDER BY cluster_id ASC, feedback_id ASC;
                """,
                (run_id,),
            )
            assignment_rows = cur.fetchall()

            # Group assignments by model
            models_assignments: dict[str, list[dict]] = {}
            for row in assignment_rows:
                mname = row["model_name"]
                if mname not in models_assignments:
                    models_assignments[mname] = []
                models_assignments[mname].append(dict(row))

            # 2. Fetch cluster metadata
            cur = conn.execute(
                """
                SELECT model_name, cluster_id, keywords_json, representative_quote,
                       representative_id, member_ids_json
                FROM cluster_metadata
                WHERE run_id = ?
                ORDER BY cluster_id ASC;
                """,
                (run_id,),
            )
            meta_rows = cur.fetchall()
            models_meta: dict[str, dict[int, dict]] = {}
            for row in meta_rows:
                mname = row["model_name"]
                if mname not in models_meta:
                    models_meta[mname] = {}
                models_meta[mname][row["cluster_id"]] = {
                    "keywords": json.loads(row["keywords_json"]),
                    "representative_quote": row["representative_quote"],
                    "representative_id": row["representative_id"],
                    "member_ids": json.loads(row["member_ids_json"]),
                }

            # 3. Fetch evaluation results
            cur = conn.execute(
                """
                SELECT model_name, metric_name, metric_value, unavailability_reason, details_json
                FROM evaluation_results
                WHERE run_id = ?;
                """,
                (run_id,),
            )
            eval_rows = cur.fetchall()
            models_eval: dict[str, dict[str, Any]] = {}
            for row in eval_rows:
                mname = row["model_name"]
                if mname not in models_eval:
                    models_eval[mname] = {"metrics": {}, "reasons": {}, "details": {}}
                models_eval[mname]["metrics"][row["metric_name"]] = row["metric_value"]
                if row["unavailability_reason"]:
                    models_eval[mname]["reasons"][row["metric_name"]] = row["unavailability_reason"]
                if row["details_json"]:
                    models_eval[mname]["details"] = json.loads(row["details_json"])

            # 4. Rebuild ModelResult objects
            model_results = []
            for mname, asgns in models_assignments.items():
                f_ids = [a["feedback_id"] for a in asgns]
                c_labels = [a["cluster_id"] for a in asgns]

                meta_dict = models_meta.get(mname, {})
                rep_ids = {cid: d["representative_id"] for cid, d in meta_dict.items()}
                keywords = {cid: d["keywords"] for cid, d in meta_dict.items()}
                cluster_sizes = {cid: len(d["member_ids"]) for cid, d in meta_dict.items()}

                eval_details = models_eval.get(mname, {}).get("details", {})

                is_tfidf = "TF-IDF" in mname
                model_ident = "scikit-learn/TfidfVectorizer + KMeans" if is_tfidf else "sentence-transformers/all-MiniLM-L6-v2 + KMeans"

                m_res = ModelResult(
                    method=mname,
                    model_identifier=model_ident,
                    parameters={"n_clusters": run_info["cluster_count_k"], "random_state": run_info["random_seed"]},
                    runtime_seconds=run_info["runtime_seconds"],
                    feedback_ids=f_ids,
                    cluster_labels=c_labels,
                    cluster_representatives=rep_ids,
                    cluster_keywords=keywords,
                    cluster_sizes=cluster_sizes,
                    representation=eval_details.get("representation", "sparse_tfidf" if is_tfidf else "dense_embeddings"),
                    distance_metric=eval_details.get("distance_metric", "cosine"),
                )
                model_results.append(m_res)

            # 5. Rebuild EvaluationReport objects
            eval_reports = []
            for mname, ev_data in models_eval.items():
                mets = ev_data["metrics"]
                reasons = ev_data["reasons"]
                dets = ev_data["details"]

                ev_rep = EvaluationReport(
                    method=mname,
                    model_identifier="scikit-learn/TfidfVectorizer + KMeans" if "TF-IDF" in mname else "sentence-transformers/all-MiniLM-L6-v2 + KMeans",
                    n_samples=run_info["record_count"],
                    n_clusters=run_info["cluster_count_k"],
                    silhouette=mets.get("silhouette_score"),
                    davies_bouldin=mets.get("davies_bouldin_index"),
                    calinski_harabasz=mets.get("calinski_harabasz_index"),
                    cluster_size_entropy=mets.get("cluster_entropy"),
                    adjusted_rand_index=mets.get("ari_against_reference"),
                    normalized_mutual_info=mets.get("nmi_against_reference"),
                    metric_reasons=reasons,
                    metrics_validity=dets.get("metrics_validity", {}),
                    limitations=dets.get("limitations", []),
                    representation=dets.get("representation", ""),
                    distance_metric=dets.get("distance_metric", ""),
                )
                eval_reports.append(ev_rep)

            return run_info, model_results, eval_reports
        finally:
            conn.close()

    # =========================================================================
    # HISTORICAL RUN COMPARISON
    # =========================================================================

    def compare_historical_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compares two historical runs from the same project.

        Calculates partition agreement (Adjusted Rand Index) between the two runs
        if and only if both runs evaluated the identical set of feedback record IDs.
        Explains limitations regarding arbitrary cluster numbering.
        """
        run_a = self.get_analysis_run(run_id_a)
        run_b = self.get_analysis_run(run_id_b)

        if not run_a:
            raise KeyError(f"Run '{run_id_a}' not found.")
        if not run_b:
            raise KeyError(f"Run '{run_id_b}' not found.")
        if run_a["project_id"] != run_b["project_id"]:
            raise ValueError("Cannot compare runs from two different projects.")

        _, models_a, evals_a = self.reconstruct_run_results(run_id_a)
        _, models_b, evals_b = self.reconstruct_run_results(run_id_b)

        # Map by model method
        m_a_dict = {m.method: m for m in models_a}
        m_b_dict = {m.method: m for m in models_b}

        ev_a_dict = {e.method: e for e in evals_a}
        ev_b_dict = {e.method: e for e in evals_b}

        shared_models = sorted(set(m_a_dict.keys()).intersection(set(m_b_dict.keys())))

        model_comparisons = {}
        for mname in shared_models:
            ma = m_a_dict[mname]
            mb = m_b_dict[mname]
            ea = ev_a_dict.get(mname)
            eb = ev_b_dict.get(mname)

            # Compare feedback sets
            fids_a = ma.feedback_ids
            fids_b = mb.feedback_ids
            same_dataset = (fids_a == fids_b)

            partition_ari = None
            partition_ari_note = ""
            if same_dataset and len(fids_a) > 0:
                partition_ari = float(adjusted_rand_score(ma.cluster_labels, mb.cluster_labels))
                partition_ari_note = (
                    f"Partition Adjusted Rand Index: {partition_ari:.4f}. Measures cluster assignment agreement "
                    "across runs independently of cluster permutation."
                )
            else:
                partition_ari_note = (
                    "Partition agreement (ARI) cannot be computed because Run A and Run B evaluated different feedback records "
                    f"({len(fids_a)} vs {len(fids_b)} records)."
                )

            # Cluster size distributions
            dist_a = ma.cluster_sizes
            dist_b = mb.cluster_sizes

            model_comparisons[mname] = {
                "same_dataset": same_dataset,
                "partition_ari": partition_ari,
                "partition_ari_note": partition_ari_note,
                "cluster_size_distribution_a": dist_a,
                "cluster_size_distribution_b": dist_b,
                "metrics_a": {
                    "silhouette": ea.silhouette if ea else None,
                    "davies_bouldin": ea.davies_bouldin if ea else None,
                    "calinski_harabasz": ea.calinski_harabasz if ea else None,
                    "ari_reference": ea.adjusted_rand_index if ea else None,
                },
                "metrics_b": {
                    "silhouette": eb.silhouette if eb else None,
                    "davies_bouldin": eb.davies_bouldin if eb else None,
                    "calinski_harabasz": eb.calinski_harabasz if eb else None,
                    "ari_reference": eb.adjusted_rand_index if eb else None,
                },
            }

        caveats = [
            "Cluster numbers (e.g. 'Cluster 1') are arbitrary internal integers assigned by K-Means. "
            "Cluster 1 in Run A does not automatically correspond to Cluster 1 in Run B.",
            "Higher metric values in one run do not prove superior business utility.",
            "Changes in cluster count k, random seed, or input dataset will alter cluster boundaries.",
        ]

        return {
            "project_id": run_a["project_id"],
            "run_a": {
                "run_id": run_a["run_id"],
                "created_at": run_a["created_at"],
                "cluster_count_k": run_a["cluster_count_k"],
                "random_seed": run_a["random_seed"],
                "record_count": run_a["record_count"],
                "dataset_fingerprint": run_a["dataset_fingerprint"],
                "runtime_seconds": run_a["runtime_seconds"],
            },
            "run_b": {
                "run_id": run_b["run_id"],
                "created_at": run_b["created_at"],
                "cluster_count_k": run_b["cluster_count_k"],
                "random_seed": run_b["random_seed"],
                "record_count": run_b["record_count"],
                "dataset_fingerprint": run_b["dataset_fingerprint"],
                "runtime_seconds": run_b["runtime_seconds"],
            },
            "same_dataset_fingerprint": (run_a["dataset_fingerprint"] == run_b["dataset_fingerprint"]),
            "model_comparisons": model_comparisons,
            "caveats": caveats,
        }
