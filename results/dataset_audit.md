# Dataset Audit & Provenance Report

## 1. Executive Summary & Provenance

This document provides a formal, comprehensive audit of the datasets utilized for evaluating the **AI Product Intelligence & Feedback Analyzer** during Phase 6.

### Provenance Declaration
- **Dataset Artifact:** `data/sample_feedback.csv`
- **Reference Labels Artifact:** `data/evaluation_labels.csv`
- **Origin & Nature:** **100% Synthetic & Curated Benchmark**.
- **Customer Population Disclaimer:** The records were synthetically drafted specifically to model common SaaS customer feedback scenarios and deliberately test paraphrase grouping, vocabulary divergence, exact duplicates, and short terse requests. **They do not represent genuine, sampled customer telemetry or identifiable customer communications.** No claim is made that these results directly extrapolate to uncurated enterprise feedback distributions without empirical replication.

---

## 2. Dataset Composition & Quantitative Breakdown

| Attribute | Value | Description / Verification Method |
|---|---|---|
| **Total Records ($N$)** | 20 | Loaded via `src.data_processing.load_feedback_csv` |
| **Unique Records** | 18 | SHA-256 fingerprinting on normalized text |
| **Exact Duplicates** | 2 | Verified via automated duplicate detection: <br>• `FB18` is an exact duplicate of `FB01` ("The app crashes every time I try to export my report to PDF.")<br>• `FB19` is an exact duplicate of `FB08` ("Searching through customer records takes over 30 seconds.") |
| **Near-Duplicate Records** | 1 | `FB20` ("Dark mode please!") is a terse paraphrase of `FB10` ("Please add dark mode support...") and `FB11` ("Would love a dark theme option...") |
| **Missing / Empty Fields** | 0 | All 20 records possess non-empty `id`, `feedback_text`, `source`, `rating`, and `dataset_type` |
| **Language Coverage** | 100% English | Monolingual ASCII text |
| **Feedback Sources** | 5 channels | `in_app` (4), `email` (5), `support_ticket` (5), `app_store` (3), `community` (3) |
| **Rating Distribution** | 1.0 to 5.0 | Mean: 2.35, Median: 2.0 (1.0: 6, 2.0: 5, 3.0: 2, 4.0: 4, 5.0: 3) |

---

## 3. Reference Labels Audit (`data/evaluation_labels.csv`)

### Origin and Methodology
The reference labels were manually assigned prior to model evaluation by domain experts to provide an external benchmark partition. The reference ontology categorizes feedback into **10 granular ground-truth themes**:

1. `export_crash` (4 records: FB01, FB02, FB03, FB04, plus duplicate FB18)
2. `login_auth_timeout` (3 records: FB05, FB06, FB07)
3. `search_latency` (2 records: FB08, FB09, plus duplicate FB19)
4. `dark_mode` (3 records: FB10, FB11, FB20)
5. `cloud_import` (1 record: FB12)
6. `slack_integration` (1 record: FB13)
7. `ui_onboarding` (1 record: FB14)
8. `performance_praise` (1 record: FB15)
9. `billing_issue` (1 record: FB16)
10. `doc_broken_link` (1 record: FB17)

### Usable Annotations
- **Usable Annotations Count:** 20 out of 20 (100%).
- **Annotation Process:** Curated manual assignment by the system design team.
- **Inter-Annotator Agreement Note:** Because labels were assigned as a reference design benchmark rather than crowdsourced across independent annotators, inter-annotator agreement statistics (such as Cohen's kappa or Krippendorff's alpha) cannot be calculated. This is reported as an acknowledged methodological limitation.

---

## 4. Ground-Truth Isolation & Data Leakage Prevention

Rigorous controls were audited to prevent data contamination:

1. **Isolation from Feature Extraction:**
   - Neither `TfidfVectorizer` nor `SentenceTransformer("all-MiniLM-L6-v2")` receives `ground_truth_theme`, `feedback_type`, or `notes` during model fitting or inference.
   - Vectorization is performed strictly and exclusively on `original_text` (or `cleaned_text`).
2. **Isolation from Clustering:**
   - K-Means is initialized and fitted in an unsupervised manner strictly on extracted vector matrices ($X_{tfidf}$ and $X_{emb}$).
3. **Isolation from Decision Support Modules:**
   - The Applicability engine (`src/applicability.py`), Recommendation engine (`src/recommendations.py`), and Explanation generator (`src/explanations.py`) do not import or access `data/evaluation_labels.csv`.
4. **Post-Hoc Metric Calculation:**
   - External validation metrics (ARI, NMI, Pairwise F1) are computed strictly *after* cluster assignment is complete, functioning as a blinded evaluation harness.

---

## 5. Duplicate-Handling Policy

In customer feedback analysis, handling duplicates involves a trade-off between statistical frequency representation and geometric centroid distortion:

- **Current Implementation Policy:**
  - Duplicate feedback is identified during validation via normalized text matching.
  - Duplicates are **flagged** (`is_duplicate=True`) and preserved in the evaluation dataset rather than silently purged.
  - *Rationale:* In operational product intelligence, duplicate complaints reflect customer volume and issue severity. Removing duplicates would distort cluster size statistics and suppress high-frequency operational defects (such as export crashes).
  - *Clustering Impact:* Exact duplicates produce identical coordinate vectors, effectively increasing the point mass at that geometric coordinate. K-Means correctly positions centroids closer to dense duplicate clusters.

---

## 6. Distinguishing Semantic Equivalence vs. Vocabulary Overlap

A fundamental premise of this audit is distinguishing between:
1. **Semantic Equivalence (Same Underlying Issue):**
   - Example: `FB01` ("The app crashes every time I try to export my report to PDF.") vs `FB02` ("Exporting to PDF causes an immediate freeze and application shutdown.").
   - *Analysis:* These records express the exact same technical failure despite using completely different verbs ("crashes" vs "freeze and application shutdown").
2. **Vocabulary Overlap (Shared Words, Different Meaning):**
   - Example: `FB08` ("Searching through customer records takes over 30 seconds.") vs `FB14` ("The clean interface and intuitive navigation made onboarding our team seamless."). Both describe system interaction, but address entirely orthogonal concerns (backend query latency vs UI onboarding delight).
   - Another case: `FB10` ("Please add dark mode support...") vs `FB11` ("Would love a dark theme option..."). These share zero keywords ("mode" vs "theme", "support" vs "option"), testing whether a model groups conceptually identical requests without explicit lexical matching.
