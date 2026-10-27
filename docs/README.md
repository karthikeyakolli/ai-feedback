# AI Product Intelligence & Feedback Analyzer

A Python + Streamlit prototype that collects accessible product context and user feedback, groups related feedback with NLP, explains its evidence, and compares two clustering approaches for an AI mini-capstone.

## Core Research Question
Does sentence-embedding-based clustering group differently worded product feedback more usefully than a traditional TF-IDF-based approach?

## Status
**Planning/starter specification.** The documentation defines the intended architecture and phased build. It does not mean the application has already been implemented.

## Features in the planned prototype
- Product context and public website references.
- CSV and pasted feedback input.
- Source preview and extraction status.
- TF-IDF + cosine similarity + K-means baseline.
- Sentence embeddings + cosine similarity + K-means comparison.
- Evaluation metrics, manual cluster inspection, runtime, and error analysis.
- Evidence-linked findings and cautious recommendations.
- Local project persistence and export, introduced in later phases.

## Requirements
- Python 3.11 or newer recommended.
- Internet access may be needed to install packages and download the pretrained embedding model the first time.
- A machine with enough memory to run the selected sentence-embedding model.
- No paid AI API is required for the core analysis.

## Setup
Create and activate a virtual environment.

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Tests
```bash
pytest -q
```

These commands become fully usable after the initial implementation phase creates `requirements.txt`, `app.py`, and the test suite. Follow `PLAN.md` and `TASKS.md` for current status.

## Documentation
- `PRD.md` — product problem, users, scope and acceptance criteria.
- `ARCHITECTURE.md` — architecture, Mermaid diagram, data model, data flow and safety rules.
- `CONTEXT.md` — rules for AI coding agents, code style and project layout.
- `PLAN.md` — phases, milestones and verification criteria.
- `TASKS.md` — active scratchpad and current tasks.
- `TESTING.md` — test strategy and local commands.

## Important limitations
- A URL does not guarantee access to all product screens or reviews.
- Product documentation and actual user feedback are different evidence types and must remain separate.
- Similarity and clustering are suggestions, not proof that two comments describe the same defect.
- Sentiment is not equivalent to severity or business impact.
- Results must be computed from actual data; do not use invented demonstration metrics.
- The current documents are a starter specification. Confirm implementation and commands by running the application and tests.
