# Phase-Wise Build Plan / Roadmap

## Guiding Strategy
Build a complete, small vertical slice before adding optional features. The first goal is an end-to-end run with sample feedback and two genuine AI methods. Then add sources, persistence, explanations, and polish.

## Phase 0 — Scope and Environment
### Tasks
- [x] Review PRD and architecture.
- [x] Create Python 3.11+ virtual environment.
- [x] Create project structure and `.gitignore`.
- [x] Add verified dependencies.
- [x] Add a small sample feedback dataset and manual evaluation labels.
- [x] Confirm Streamlit launches.
### Verification
- [x] Fresh environment installs successfully.
- [x] `streamlit run app.py` starts.
- [x] No secrets or user data are committed.
### Exit condition
A runnable shell app and a reproducible environment exist.

## Phase 1 — Core AI Experiment (highest priority)
### Tasks
- [x] Implement data loading and validation.
- [x] Implement preprocessing while preserving original text.
- [x] Implement TF-IDF + cosine similarity + K-means.
- [x] Implement sentence embeddings + cosine similarity + K-means.
- [x] Ensure both methods use the same records.
- [x] Record model configuration, random seed, runtime, and dataset fingerprint.
- [x] Implement valid metrics and manual error inspection.
- [x] Add tests for small datasets, invalid cluster counts, blank records, and model failures.
### Verification
- [x] Both methods run on the sample dataset.
- [x] Outputs contain a cluster assignment for every included record.
- [x] Actual metrics are generated or clearly marked unavailable with a reason.
- [x] Runtime and configuration are recorded.
- [x] No metric or result is hardcoded.
### Exit condition
The experiment can be run from a Python command without Streamlit.

## Phase 2 — Streamlit MVP
### Tasks
- [x] Build overview and source/data input pages.
- [x] Add CSV upload and pasted-feedback input.
- [x] Add sample-data option.
- [x] Show data preview and quality warnings.
- [x] Display both methods' clusters and representative feedback.
- [x] Display model comparison and error examples.
### Verification
- [x] User can complete the sample workflow from UI.
- [x] Invalid input gives an understandable message.
- [x] UI never displays fake metrics.
### Exit condition
The core experiment is usable through the browser.

## Phase 3 — Product Context and Public Sources
### Tasks
- [x] Add project/product metadata.
- [x] Add bounded public website extraction.
- [x] Add source preview and provenance.
- [x] Store product context separately from feedback.
- [x] Add source status, partial extraction, and failure handling.
- [ ] Add GitHub public README/issues only if core milestones are stable.
### Verification
- [x] Accessible site text can be previewed and attributed to its URL.
- [x] Failed extraction is visible and does not crash the app.
- [x] Website text is not silently treated as user feedback.
- [x] URL-fetch safeguards are tested.
### Exit condition
Product context and feedback can be combined without mixing their roles.

## Phase 4 — Persistence and Traceability
### Tasks
- [x] Add SQLite storage for projects, sources, feedback, and analysis runs (`src/database.py`).
- [x] Save cluster assignments, metrics, and settings atomically with rollback on error.
- [x] Reopen a saved project and inspect prior runs offline without re-running ML models.
- [x] Link each finding back to original feedback and source with provenance.
- [x] Compare historical analysis runs with partition agreement (Adjusted Rand Index) on identical records.
### Verification
- [x] Data remains after restarting the app or refreshing browser.
- [x] Each cluster member resolves to its original record in SQLite.
- [x] A new run does not silently overwrite previous run results.
- [x] Deleting a project cascades cleanly across all associated tables with confirmation.
### Exit condition
The prototype retains projects, versioned context, and analysis run history across sessions.

## Phase 5 — Explanation and Decision Support
### Tasks
- [x] Show cluster keywords and representative examples with neutral cluster labels.
- [x] Explain similarity and clustering in simple language (TF-IDF mechanics, Sentence Transformer continuous geometry).
- [x] Add applicability statuses (`Likely relevant`, `Needs validation`, `Insufficient information`) and missing-evidence prompts.
- [x] Add transparent, rule-based investigation recommendations and deduplicated frequency-based queue.
- [x] Add cross-representation metric incomparability caveats (sparse TF-IDF vs dense embeddings).
### Verification
- [x] Each recommendation links to verifiable evidence and real feedback IDs.
- [x] Missing information, contradictions, and noise are explicitly shown.
- [x] No unsupported causal or business-impact claims appear; ordering is frequency-based only.
- [x] All 60 automated unit/integration tests pass.
### Exit condition
Users can understand and inspect the results, explainability mechanics, and actionable next validation steps.

## Phase 6 — Export, Quality, and Capstone Delivery
### Tasks
- [x] Export cluster results and metrics as CSV (Implemented in Phase 3 & 5).
- [x] Export a report as Markdown or HTML (Implemented in Phase 3 & 5).
- [x] Complete unit/integration tests (60/60 pytest passing).
- [x] Run a clean setup test from README instructions (`scripts/verify_install.py`, 10/10 passed).
- [x] Record actual experiment results and failure cases (`results/model_comparison.csv`, `results/error_analysis.md`).
- [x] Complete methodology, limitations, references, presentation, and A5 poster (`docs/FINAL_CAPSTONE_REPORT.md`, `docs/capstone_presentation.pptx`, `docs/capstone_poster.pdf`).
### Verification
- [x] All documented commands are tested.
- [x] Results are reproducible with documented settings as far as practical (`scripts/run_experiments.py`).
- [x] Documentation matches actual implementation.
- [x] Team members can explain the full system.
### Exit condition
The prototype is demonstrable and its results are defensible. (COMPLETED & VERIFIED)

## Scope Gate
If time becomes short, complete Phases 0–2 first. Do not sacrifice the real experiment for extra UI polish. Phases 3–6 should be prioritised by their acceptance criteria and deadline; optional integrations and sentiment must not block the core comparison. (ALL PHASES 0-6 COMPLETED).
