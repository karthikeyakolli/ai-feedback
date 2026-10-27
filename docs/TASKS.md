# Active Task Scratchpad

Update this file after each work session. Mark a task completed only after verifying it.

Legend:
- `[ ]` Todo (Not implemented)
- `[/]` In progress (Implemented but not verified)
- `[x]` Completed (Implemented and verified)
- `[!]` Blocked

## Current Phase
**Phase 6 — Experimental Evaluation, Reproducibility & Capstone Delivery (COMPLETED & VERIFIED)**

## Decisions Already Made
- [x] Python + Streamlit prototype rather than the full social platform.
- [x] Core experiment compares TF-IDF + K-means with sentence embeddings + K-means.
- [x] Product context and actual feedback must remain separate.
- [x] Public URLs are best-effort inputs; extraction status and provenance must be visible.
- [x] Core analysis must not require a paid AI API.
- [x] AI results must be evidence-backed, reproducible where practical, and not fabricated.
- [x] Python 3.12.10 selected as compatible target (default system 3.15 preview lacks torch/sklearn wheels; Windows AppLocker requires running via `python -m <module>`).
- [x] SSRF-safe URL extraction: DNS resolution checking, private/loopback/link-local IP blocking, 10s timeout, 2MB size cap, max 3 redirects, no JS execution.
- [x] Deterministic insights: cluster evidence quotes directly traceable to original feedback IDs; neutral cluster labels; cautious context correlation; prioritization by volume rather than fabricated severity.
- [x] Three strict applicability statuses: `Likely relevant`, `Needs validation`, `Insufficient information` evaluated via deterministic lexical overlap and contradiction checks.
- [x] Investigation recommendations recommend WHAT TO INVESTIGATE, avoiding autonomous product decisions or fabricated business claims.
- [x] Duplicates explicitly tracked as repetitions, not independent customer confirmations.
- [x] Frequency-Based Investigation Queue ordered strictly by unique feedback volume, disclaiming commercial severity or urgency.
- [x] Educational Explainability Layer disclaiming cross-space metric comparisons (sparse TF-IDF vs dense embeddings).
- [x] Phase 6 Controlled Benchmark: Multi-seed (42, 100, 2024) and multi-k (3, 4, 5, 6) evaluation of TF-IDF vs. `all-MiniLM-L6-v2` with strict label isolation.
- [x] Clean Installation Verification: Automated 10-point health check script (`scripts/verify_install.py`).
- [x] Complete Academic Capstone Suite: 33-section report, 15-slide PowerPoint deck, transcript/notes, and A5 poster.

## Phase 0 — Scope and Environment (COMPLETED & VERIFIED)
- [x] Inspect current project folder/repository before creating or overwriting files (Verified: analyzed 8 planning documents).
- [x] Create virtual environment using Python 3.11+ (Verified: created `.venv` with Python 3.12.10).
- [x] Create clean directory structure (`src/`, `data/`, `tests/`, `docs/`, `app.py`, `requirements.txt`, `.gitignore`).
- [x] Create and validate `requirements.txt` (Verified: pip installed all packages without errors).
- [x] Create sample feedback dataset with varied categories, duplicates, and synthetic marker (Verified: `data/sample_feedback.csv`).
- [x] Create manually labelled examples/pairs for evaluation held strictly separate from model inputs (Verified: `data/evaluation_labels.csv`).
- [x] Create minimal Streamlit shell (Verified: `app.py`).
- [x] Run app smoke test and record exact command/output (Verified: `python -m streamlit run app.py`, HTTP 200 OK).

## Phase 1 — Core AI Experiment (COMPLETED & VERIFIED)
- [x] Implement feedback loader and validator preserving original text (`src/data_processing.py`, verified with 7 unit tests).
- [x] Implement preprocessing and duplicate flagging without silent record deletion (`src/data_processing.py`, verified).
- [x] Implement TF-IDF baseline + K-Means with representative item discovery and keyword extraction (`src/clustering.py`, verified).
- [x] Implement sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) + K-Means (`src/clustering.py`, weights downloaded & verified).
- [x] Ensure both methods execute on the identical feedback record IDs (`src/evaluation.py:compare_models`, verified).
- [x] Implement evaluation metrics with precondition validation (Silhouette, Davies-Bouldin, ARI, NMI, entropy) and documented limitations (`src/evaluation.py`, verified).
- [x] Add automated test suite and execute (Verified: 13/13 pytest tests passing).

## Phase 2 — Streamlit Interactive MVP (COMPLETED & VERIFIED)
- [x] Implement Product Definition section isolated from model inputs (`app.py`, verified).
- [x] Implement 3 feedback input sources: Built-in sample, Custom CSV with column mapping, and Pasted multiline text (`src/data_processing.py`, `app.py`, verified).
- [x] Implement session-level in-memory embedding caching to avoid redundant model encodings on unchanged feedback (`src/clustering.py:run_embedding_clustering`, verified).
- [x] Implement descriptive metric unavailability reasons (`metric_reasons`) and feature space metadata in `EvaluationReport` (`src/evaluation.py`, verified).
- [x] Build interactive Cluster Explorer with model switcher, keyword inspection, centroid representative feedback, member feedback table, duplicate badges, and text filter (`app.py`, verified).
- [x] Build Model Comparison workspace with side-by-side metric tables, cluster size distribution bar chart, and explicit limitations callout (`app.py`, verified).
- [x] Add automated test coverage for Phase 2 additions (20/20 pytest tests passing in 28.4s).
- [x] Verify Streamlit startup smoke test and HTTP 200 responsiveness on port 8501 (`Invoke-WebRequest`, verified).

## Phase 3 — Product Context Ingestion, Insights & Export (COMPLETED & VERIFIED)
- [x] Add Product Context models with provenance and approval tracking (`src/context_ingestion.py`, verified).
- [x] Implement bounded, SSRF-safe public webpage text extractor (`src/context_ingestion.py`, verified with unit tests for IP blocking, redirect validation, and timeout handling).
- [x] Display extraction review card with approval, editing, and origin attribution separated from feedback inputs (`app.py`, verified).
- [x] Build Evidence-Grounded Insights report generating neutral theme summaries, centroid quotes, and traceable record IDs (`src/insights.py`, verified).
- [x] Implement cautious, non-speculative product-context correlation based on verifiable token overlap (`src/insights.py`, verified).
- [x] Add measurable prioritization view (record count, unique records, dataset share) without fabricated severity scores (`src/insights.py`, `app.py`, verified).
- [x] Implement multi-format export: Feedback CSV, Cluster Summaries CSV, Evaluation Metrics CSV, and Markdown Summary Report (`src/export.py`, `app.py`, verified).
- [x] Expand automated tests to 33 passing tests (Verified: 33/33 pytest tests passing in 19.9s).
- [x] Verify Streamlit server responsiveness on port 8501 (Verified: HTTP 200 OK).

## Phase 4 — SQLite Persistence, Project Management & Analysis History (COMPLETED & VERIFIED)
- [x] Implement local SQLite persistence service module (`src/database.py`, verified with 8 relational tables, WAL mode, foreign keys, schema versioning, and parameterized queries).
- [x] Implement Project Management CRUD (Create, List, Rename, Open, Delete with confirmation) and multi-project isolation (`src/database.py`, `app.py`, verified).
- [x] Implement Product Context versioning preserving domain history across updates (`src/database.py`, `app.py`, verified).
- [x] Implement Source metadata and provenance storage without storing secrets or refetching URLs upon reopening (`src/database.py`, `app.py`, verified).
- [x] Implement Feedback record persistence with conflict strategies (`error`, `replace_all` with confirmation, `skip`, `reassign_id`) preserving original verbatim text and duplicate links (`src/database.py`, `app.py`, verified).
- [x] Implement atomic analysis run persistence (`running` -> `completed` / `failed`) with full transaction rollback on failure (`src/database.py`, `app.py`, verified).
- [x] Implement 100% offline historical run reopening and inspection without re-running vectorizers or embedding encoders (`src/database.py`, `app.py`, verified).
- [x] Implement historical run comparison view with Adjusted Rand Index (ARI) for identical datasets, cluster distribution diffs, and arbitrary cluster ID caveats (`src/database.py`, `app.py`, verified).
- [x] Add comprehensive SQLite and integration test suite (`tests/test_database.py`, 11 tests; total suite: 44/44 tests passing).
- [x] Verify Streamlit startup and full AppTest execution with 5 tabs and zero exceptions (`AppTest.from_file('app.py')`, HTTP 200 OK).

## Phase 5 — Decision Support & Explainability (COMPLETED & VERIFIED)
- [x] Implement Feature A: Applicability assessment (`Likely relevant`, `Needs validation`, `Insufficient information`) with deterministic rules, centralized config, token overlap, and contradiction detection (`src/applicability.py`, verified with 8 tests).
- [x] Implement Feature B: Evidence-backed investigation recommendations (`src/recommendations.py`, verified with 4 tests), frequency-based investigation queue, duplicate accounting, and deterministic deduplication.
- [x] Implement Feature C: Educational explainable clustering guide (`src/explanations.py`, verified with 4 tests) covering TF-IDF, Sentence Transformers, evaluation metrics, and the cross-space fair comparison caveat.
- [x] Implement Feature D: Cluster Explorer extension with neutral labels, applicability assessment cards, investigation recommendations, and member feedback inspection (`app.py`, verified).
- [x] Implement Feature E: SQLite persistence and historical isolation, pure deterministic derivation from immutable run data + context snapshot without DB schema breaking changes (`app.py`, `src/database.py`, verified).
- [x] Implement Feature F: Export Center integration with `investigation_recommendations.csv`, enhanced summaries CSV, and full markdown report with explanations and caveats (`src/export.py`, `app.py`, verified).
- [x] Expand automated tests to 60 passing tests (Verified: 60/60 pytest tests passing in 21.02s).
- [x] Verify Streamlit startup and full AppTest execution with 5 tabs and zero exceptions (`AppTest.from_file('app.py')`, HTTP 200 OK).

## Phase 6 — Delivery & Capstone Documentation (COMPLETED & VERIFIED)
- [x] Run full test suite across all modules (Verified: 60/60 tests passing in 20.69s).
- [x] Test clean installation and verification procedure (`scripts/verify_install.py`, 10/10 checks passing).
- [x] Execute controlled multi-seed, multi-k comparative benchmark (`scripts/run_experiments.py`, 12 configurations evaluated).
- [x] Conduct dataset audit and provenance report (`results/dataset_audit.md`).
- [x] Conduct qualitative error analysis on verbatim records (`results/error_analysis.md`).
- [x] Produce reproduction instructions with exact seeds and configs (`results/reproduction_instructions.md`).
- [x] Author comprehensive 33-section academic capstone report (`docs/FINAL_CAPSTONE_REPORT.md` and `product_intelligence_project_docs/FINAL_CAPSTONE_REPORT.md`).
- [x] Generate editable 15-slide PowerPoint capstone defense deck (`docs/capstone_presentation.pptx`) and speaker notes (`docs/slides_content.md`).
- [x] Generate A5 academic poster in vector PDF (`docs/capstone_poster.pdf`) and 300 DPI preview PNG (`docs/capstone_poster.png`).
- [x] Update documentation, diagrams, tasks, and repository hygiene.

## Blockers / Decisions Needed
- None. All Phase 6 deliverables and acceptance criteria are completed, verified, and operational. Product is fully delivered.
