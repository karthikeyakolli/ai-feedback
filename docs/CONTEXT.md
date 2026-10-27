# AI Coding Context and Project Rules

## 1. Project Identity
Name: **AI Product Intelligence & Feedback Analyzer**
Purpose: Python mini-capstone prototype for explainable product feedback analysis and an AI-method comparison experiment.

## 2. Non-Negotiable Scope Rules
- Build the Python prototype, not the full Reddit-style social platform.
- Keep product context separate from actual feedback.
- Main research experiment: TF-IDF + K-means versus sentence embeddings + K-means.
- Both methods must use the same feedback dataset.
- Core analysis must work without a paid AI API.
- Do not invent metrics, source content, review counts, confidence values, or business impact.
- Every finding must trace to original feedback and source provenance when available.
- Never silently delete or merge user feedback.
- Label recommendations as suggestions, not established facts.
- Implement one phase at a time; do not jump ahead without passing its verification checklist.

## 3. Recommended Directory Layout
```text
product-intelligence/
├── app.py
├── pages/
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── ai/
│   ├── storage/
│   └── reporting/
├── data/
│   ├── sample_feedback.csv
│   └── evaluation_labels.csv
├── experiments/
├── tests/
├── docs/
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── PRD.md
├── ARCHITECTURE.md
├── CONTEXT.md
├── PLAN.md
├── TASKS.md
└── TESTING.md
```

## 4. Code Style
- Use Python 3.11+ and type hints for public functions.
- Prefer small, single-purpose functions.
- Keep Streamlit rendering in `app.py` and `pages/`; do not import Streamlit into core AI functions.
- Separate extraction, validation, preprocessing, AI, persistence, evaluation, and reporting.
- Use descriptive `snake_case` for functions/variables and `PascalCase` for classes.
- Use dataclasses or typed structures for non-trivial result objects.
- Use `pathlib.Path` for filesystem paths.
- Use `logging` for diagnostic messages; show safe, actionable errors in the UI.
- Do not swallow exceptions with bare `except:`.
- Do not put API keys in source code or commit `.env`.
- Avoid unnecessary frameworks, services, databases, and abstractions.
- Preserve original feedback text. Store processed text separately.
- Set deterministic random seeds where applicable.
- Add tests for core logic and error paths alongside each feature.

## 5. Naming Conventions
- Modules and files: `snake_case.py`.
- Functions and variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Test files: `test_*.py`.
- Experiment output: include a run ID or timestamp; never overwrite the only copy of previous results silently.

## 6. Environment and Configuration
- Use a virtual environment.
- Keep dependency versions pinned or bounded in `requirements.txt` after verifying installation on the target machine.
- `.env.example` may contain names of optional environment variables, never real secrets.
- Core configuration should use a validated config object or `config.py`, not scattered constants.
- Suggested settings:
  - `DATABASE_PATH=./data/product_intelligence.db`
  - `ARTIFACTS_DIR=./artifacts`
  - `MAX_UPLOAD_MB=10`
  - `HTTP_TIMEOUT_SECONDS=10`
  - `MAX_RESPONSE_MB=5`
  - `DEFAULT_RANDOM_STATE=42`
  - `DEFAULT_CLUSTER_COUNT=4`
- Values are defaults for local development and should be validated; they are not universal guarantees.
- Optional API keys must be optional, ignored by Git, and unnecessary for P0.

## 7. UI and Explanation Rules
- Use plain language before technical details.
- Distinguish observed evidence, model output, assumptions, and recommendations.
- Show the original feedback that supports every cluster/issue.
- Explain what each metric means and its limitations.
- Show loading, empty, partial-success, and failure states.
- Never show placeholder metric values as real results.
- Do not present similarity as a calibrated probability.
- Let users inspect source records and correct suggested theme labels where feasible.

## 8. Working With AI Coding Agents
Before coding:
1. Read `PRD.md`, `ARCHITECTURE.md`, `CONTEXT.md`, `PLAN.md`, `TASKS.md`, and `TESTING.md`.
2. Identify the active phase and task.
3. Inspect the current repository before modifying files.
4. State the files you intend to change.
5. Implement only the requested phase/task.
6. Run relevant tests and report the exact commands/results.
7. Update `TASKS.md` only for work actually completed.
8. Record blockers and decisions rather than guessing.
9. Do not rewrite unrelated working files.
10. Do not claim success unless the app/tests were run and the result is known.

## 9. Git Rules
- Commit after each verified milestone.
- Never commit databases with private user data, uploaded sensitive feedback, `.env`, model caches, or large generated artifacts.
- Include a `.gitignore` before running the app.

## 10. Phase 4 SQLite Persistence & Project Rules
- **Parameterized SQL:** All queries must use parameterized placeholders (`?`). String interpolation of user input is strictly prohibited.
- **Foreign Keys:** `PRAGMA foreign_keys = ON;` must be enabled on every connection.
- **Atomic Transactions:** Wrap multi-table persistence in `with db.transaction() as conn:`. Roll back completely on errors.
- **Project Isolation:** Strictly isolate project datasets, contexts, and runs (`WHERE project_id = ?`).
- **Context Versioning:** Editing product context creates a new monotonic version (`UNIQUE(project_id, version)`). Never overwrite historical context referenced by earlier runs.
- **Feedback Preservation:** Verbatim feedback text is never mutated. Duplicate entries are linked via `duplicate_of_id`, never deleted. Conflicting IDs support explicit resolution strategies (`skip`, `reassign_id`, `replace_all` with confirmation).
- **Offline Run Reconstruction:** Historical runs must be reconstructed directly from SQLite without re-running Sentence Transformers or TF-IDF models.
- **Partition Comparison:** Cluster IDs are arbitrary integer labels. When comparing historical runs, calculate Adjusted Rand Index (ARI) only across identical record sets and display explicit caveats.
- **Storage Limitations:** Document that local SQLite storage provides single-user file-level atomicity, not encryption at rest or multi-user cloud synchronization.
