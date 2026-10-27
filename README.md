# AI Product Intelligence & Feedback Analyzer

A local-first Python + Streamlit prototype that helps product teams analyze user feedback, discover recurring themes, inspect supporting evidence, and compare two natural language processing approaches:
- **Baseline:** TF-IDF vectorization + K-Means clustering
- **Semantic:** Pretrained Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`) + K-Means clustering

---

## Core Research Question
> *Does sentence-embedding-based clustering group differently worded product feedback more accurately and meaningfully than a traditional TF-IDF baseline when evaluated on the exact same dataset?*

---

## Key Architectural Principles
1. **Raw Feedback Preservation:** Original feedback text is never overwritten or mutated. Cleaned text is stored separately.
2. **Flag, Don't Drop:** Exact and near-duplicates are flagged and linked to their original instance rather than silently dropped.
3. **Independent Evaluation Labels:** Ground truth evaluation labels are stored in a dedicated file (`data/evaluation_labels.csv`) and are strictly isolated from model inputs.
4. **Isolated Product Context:** Product domain descriptions and goals are kept separate from feedback records and are not passed as clustering features.
5. **Local SQLite Persistence:** Reliable transactional storage for projects, versioned product contexts, feedback records, and analysis runs (`data/product_intelligence.db`, configurable via `DATABASE_PATH`).
6. **Offline Historical Run Reopening:** Saved runs can be reopened, inspected, and exported completely offline without re-running vectorizers or embedding encoders.
7. **Session-Level Embedding Caching:** Unchanged feedback texts reuse in-memory precomputed embeddings to prevent redundant encoding.
8. **Mathematical Precondition Safety:** Clustering metrics (Silhouette score, Davies-Bouldin, Calinski-Harabasz, ARI/NMI) are computed only when preconditions are met ($2 \le k < n$, $>1$ cluster).
9. **Zero Paid APIs Required:** Entire core analysis runs locally on CPU with open-source models and libraries.

---

## Project Structure
```text
ai-feedback-project/
├── app.py                     # Streamlit interactive interface (5 tabs + project manager sidebar)
├── requirements.txt           # Core Python dependencies
├── README.md                  # Project overview & usage instructions
├── .gitignore                 # Standard Python, Streamlit, and SQLite cache ignore rules
├── data/
│   ├── sample_feedback.csv    # Varied synthetic feedback dataset (20 items)
│   ├── evaluation_labels.csv  # Separate ground-truth evaluation labels
│   └── product_intelligence.db# Local SQLite database (auto-created on first run)
├── src/
│   ├── __init__.py
│   ├── database.py            # SQLite schema, transactional migrations, project CRUD, runs & offline reconstruction
│   ├── data_processing.py     # Data loading, validation, column mapping, and duplicate detection
│   ├── clustering.py          # TF-IDF baseline, Sentence Transformers, and embedding cache
│   ├── evaluation.py          # Metrics calculation, precondition guards, and limitations
│   ├── context_ingestion.py   # Product context data model, SSRF-safe URL extraction & review
│   ├── applicability.py      # Feature A: Deterministic applicability assessment against approved product context
│   ├── recommendations.py    # Feature B: Evidence-backed investigation recommendations & deduplicated queue
│   ├── explanations.py       # Feature C: Educational explainable clustering guide and cross-representation caveat
│   ├── insights.py            # Evidence-grounded insights, deterministic context matching & prioritization
│   └── export.py              # Feature F: Multi-format CSV exports (including recommendations CSV) and Markdown reports
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py# Unit tests for input validation, custom CSV, and pasted text (12 tests)
│   ├── test_clustering.py     # Tests for K-Means, embedding caching, and precondition safety (8 tests)
│   ├── test_context_and_insights.py # SSRF protection, context approval, insights & exports (13 tests)
│   ├── test_database.py       # SQLite persistence, project isolation, runs, and comparison (11 tests)
│   ├── test_applicability_and_recommendations.py # Feature A & B applicability, recommendations, and queue tests (12 tests)
│   └── test_explanations.py   # Feature C explainability, metric directions, and fair comparison tests (4 tests)
└── docs/                      # Specification, architecture, and testing guides
    ├── PRD.md
    ├── ARCHITECTURE.md
    ├── CONTEXT.md
    ├── PLAN.md
    ├── TASKS.md
    ├── TESTING.md
    └── README.md
```

---

## Setup & Quickstart

### 1. Environment Setup
Recommended Python version: **Python 3.11+** (Verified on Python 3.12.10).

On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
*(All 60 unit and integration tests passing in ~21s).*

### 3. Verify Clean Installation (10-Point Health Check)
```powershell
.\.venv\Scripts\python.exe scripts/verify_install.py
```

### 4. Run Controlled Empirical Evaluation Benchmark
```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py
```
*(Executes TF-IDF vs. Sentence Transformers across 12 configurations, generating CSVs, JSON, and charts in `results/`).*

### 5. Generate Academic Capstone Presentation & Poster
```powershell
.\.venv\Scripts\python.exe scripts/generate_slides.py
.\.venv\Scripts\python.exe scripts/generate_poster.py
```
*(Produces `docs/capstone_presentation.pptx`, `docs/capstone_poster.pdf`, and `docs/capstone_poster.png`).*

### 6. Launch Streamlit Application
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*(Note: On Windows systems with Application Control policies, invoking via `python -m streamlit` avoids executable shim restrictions).*

---

## Interactive Workflow (Phase 5 & 6)
The application consists of a project management sidebar and 5 focused workspaces:

0. **📁 Project Management (Sidebar)**:
   - Select active project, inspect creation and updated timestamps, feedback counts, and completed runs.
   - Create new projects, rename active projects, or permanently delete projects with mandatory confirmation.
   - **Historical Runs Loader:** Reopen saved historical analysis runs completely offline without re-running ML models.

1. **📋 Product & Input Data**:
   - Define product metadata (Name, Description, Target Users, Goals); auto-saved as versioned snapshots in SQLite.
   - Ingest product documentation via pasted text or SSRF-safe public webpage extraction with human review.
   - Import feedback from synthetic sample, custom CSV, or pasted text with explicit conflict handling (`skip`, `reassign_id`, `replace_all` with confirmation).

2. **⚙️ Analysis Workspace**:
   - Configure cluster count $k$ (dynamically bounded by dataset size) and random seed.
   - Execute both TF-IDF and Sentence Transformers on identical records; atomically saves results to SQLite.
   - **⚖️ Compare Historical Runs:** Select two completed runs from the project to compare runtime, configurations, metric scores, cluster distributions, and partition agreement (Adjusted Rand Index) across identical record sets.

3. **🔍 Cluster Explorer (Feature D: Explain Cluster Membership)**:
   - Inspect individual clusters for either model with neutral labels (`Cluster #X`).
   - View top keywords and centroid-closest representative feedback quotes.
   - **Applicability Assessment Card:** Status badge (`Likely relevant`, `Needs validation`, `Insufficient information`), plain-language explanation, matching concepts, contradictory signals, and next validation step.
   - **Investigation Recommendation:** Suggested action, rationale, unique recurrence score, duplicate counts, and verbatim supporting quotes.
   - Search within member feedback comments.

4. **📊 Model Comparison (Feature C: Explainable Clustering)**:
   - Side-by-side metric comparison (Runtime, Silhouette, Davies-Bouldin, ARI against reference labels, Entropy).
   - Cluster size distribution bar chart.
   - **📚 Algorithm & Metric Explainability Guide:** Educational breakdowns for beginners on how TF-IDF and Sentence Transformers work, what metrics measure, directions of preference, preconditions, and business utility caveats.
   - **⚖️ Fair Comparison Caveat:** Explicit warning explaining why internal metrics calculated in sparse TF-IDF vs dense neural spaces cannot be directly compared numerically.

5. **💡 Insights Report & Export (Feature B & Feature F)**:
   - **Theme Overview & Evidence:** Centroid quotes, traceable feedback IDs, and non-speculative product-context correlation.
   - **🧭 Frequency-Based Investigation Queue:** Structured queue of actionable investigation prompts ordered strictly by unique feedback recurrence (with explicit disclaimer: does not indicate severity or urgency).
   - **Multi-Format Exports:** Download Feedback Records CSV (with cluster labels), Cluster Summaries CSV (with applicability and actions), Investigation Recommendations CSV (with quotes and rationale), Model Evaluation CSV, and full Markdown Analytical Summary Report with run provenance.

---

## Academic Capstone Deliverables (Phase 6)
- **Final Capstone Report:** [`docs/FINAL_CAPSTONE_REPORT.md`](file:///docs/FINAL_CAPSTONE_REPORT.md) (33 complete academic sections).
- **Presentation Slide Deck:** [`docs/capstone_presentation.pptx`](file:///docs/capstone_presentation.pptx) (15-slide editable PowerPoint).
- **Presentation Transcript & Notes:** [`docs/slides_content.md`](file:///docs/slides_content.md).
- **A5 Academic Poster:** [`docs/capstone_poster.pdf`](file:///docs/capstone_poster.pdf) and [`docs/capstone_poster.png`](file:///docs/capstone_poster.png).
- **Empirical Evaluation Results:**
  - Config: [`results/experiment_config.json`](file:///results/experiment_config.json)
  - Comparison Table: [`results/model_comparison.csv`](file:///results/model_comparison.csv)
  - Cluster Quality Analysis: [`results/cluster_quality_analysis.csv`](file:///results/cluster_quality_analysis.csv)
  - Dataset Audit: [`results/dataset_audit.md`](file:///results/dataset_audit.md)
  - Error Analysis: [`results/error_analysis.md`](file:///results/error_analysis.md)
  - Summary: [`results/experiment_summary.md`](file:///results/experiment_summary.md)
  - Reproduction Instructions: [`results/reproduction_instructions.md`](file:///results/reproduction_instructions.md)

---

## Status & Roadmap
- [x] **Phase 0: Scope and Environment** (Completed & Verified)
- [x] **Phase 1: Minimal Project Foundation & Core AI Pipeline** (Completed & Verified)
- [x] **Phase 2: Streamlit Interactive MVP** (Completed & Verified)
- [x] **Phase 3: Product Context Ingestion, Evidence-Grounded Insights & Export** (Completed & Verified)
- [x] **Phase 4: SQLite Persistence, Project Management & Analysis History** (Completed & Verified)
- [x] **Phase 5: Decision Support & Explainability** (Completed & Verified)
- [x] **Phase 6: Experimental Evaluation, Reproducibility & Capstone Delivery** (Completed & Verified)
