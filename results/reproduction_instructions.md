# Experimental Reproduction & Protocol Instructions

## 1. Overview & Reproducibility Guarantee

This document provides exact, step-by-step instructions for reproducing the Phase 6 controlled comparative evaluation of **TF-IDF + K-Means** versus **Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`) + K-Means** on the **AI Product Intelligence & Feedback Analyzer**.

All data, scripts, and evaluation harnesses are deterministic when run with the specified seeds.

---

## 2. Environment Specifications

- **Operating System:** Windows 10/11 or Linux / macOS
- **Python Version:** Python 3.12.x (Verified on 3.12.10)
- **Virtual Environment Directory:** `.venv`
- **Core Dependencies:**
  - `streamlit>=1.41.0`
  - `pandas>=2.2.0`
  - `scikit-learn>=1.6.0`
  - `sentence-transformers>=3.4.0`
  - `pytest>=8.3.0`
  - `matplotlib>=3.8.0`
  - `python-pptx>=1.0.0`

---

## 3. Step-by-Step Reproduction Procedure

### Step 1: Activate Virtual Environment
From the project root:
```powershell
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Or on Linux/macOS:
source .venv/bin/activate
```

### Step 2: Verify Python Environment & Dependencies
```powershell
python --version
pip list
```
Confirm that `scikit-learn`, `sentence-transformers`, `pandas`, `matplotlib`, and `python-pptx` are installed.

### Step 3: Run the Verification Script
```powershell
python scripts/verify_install.py
```
This runs a 10-point health check, executes the complete automated unit and integration test suite (60+ tests), and executes a smoke clustering run.

### Step 4: Run the Full Controlled Experiment Benchmark
Execute the automated benchmark runner:
```powershell
python scripts/run_experiments.py
```

### Expected Console Output:
```text
=== Starting Controlled Experiment (12 configurations) ===
=== Experiment Completed! Outputs saved to 'results' ===
```

---

## 4. Benchmark Execution Parameters

The runner iterates over the Cartesian product of:
- **Cluster Counts ($k$):** `[3, 4, 5, 6]`
- **Random Seeds:** `[42, 100, 2024]`
- **Total Executions:** 12 configurations $\times$ 2 models = 24 clustering fits.

### Exact Hyperparameters:
- **TF-IDF Pipeline:**
  - `ngram_range`: `(1, 2)` (unigrams and bigrams)
  - `max_features`: `1000`
  - `sublinear_tf`: `True`
  - `stop_words`: `english`
  - Distance: Cosine distance (Euclidean metric on L2-normalized unit vectors)
- **Sentence Transformer Pipeline:**
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Embedding Dimensions: 384 continuous dense values
  - Normalization: L2 unit normalization (`normalize_embeddings=True`)
  - Distance: Euclidean distance on normalized unit hypersphere
- **Clustering Algorithm:**
  - `sklearn.cluster.KMeans`
  - `init`: `'k-means++'`
  - `n_init`: 10
  - `max_iter`: 300

---

## 5. Artifact Verification & Output Locations

After execution completes, verify that the following files are populated in `results/`:

| Output Artifact | Purpose | Expected Size / Format |
|---|---|---|
| `results/experiment_config.json` | Complete metadata, system info, and hyperparameter logs | JSON format |
| `results/model_comparison.csv` | Granular table containing all 12 evaluated runs and metrics | 12 rows, 25 columns CSV |
| `results/cluster_quality_analysis.csv` | Per-cluster keyword, size, and representative quotes at $k=4$ | 8 rows CSV |
| `results/metrics_comparison.png` | Matplotlib plot comparing ARI and Pairwise F1 across $k$ | 300 DPI PNG |
| `results/runtime_comparison.png` | Bar chart comparing execution latencies (ms) | 300 DPI PNG |
| `results/pairwise_comparison.png` | Grouped bar chart of cross-model partition agreement | 300 DPI PNG |
| `results/experiment_summary.md` | Executive markdown summary of all numerical results | Markdown |

---

## 6. Offline Execution & Model Caching

- The sentence-transformer model (`all-MiniLM-L6-v2`) is cached locally in `~/.cache/huggingface/hub/`.
- Once downloaded during initial setup, all subsequent benchmark executions run **100% offline without network connectivity**.
- **Zero Paid APIs:** The entire pipeline requires zero external API keys, OpenAI tokens, or cloud services.
