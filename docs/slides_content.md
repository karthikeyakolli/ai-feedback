# Academic Capstone Presentation Transcript & Speaker Notes

**Project:** AI Product Intelligence & Feedback Analyzer
**Presentation Time:** 8–12 Minutes
**Format:** 15 Slides Widescreen 16:9 (`docs/capstone_presentation.pptx`)
**Target Audience:** Academic Examination Panel, Applied ML Faculty, Software Engineering Mentors

---

## Slide 1: Title & Candidacy Details

### Slide Content
- **Title:** AI Product Intelligence & Feedback Analyzer
- **Subtitle:** A Rigorous Empirical Comparison of Sparse N-Gram (TF-IDF) vs. Dense Semantic (Sentence Transformers) Clustering for Product Feedback Triage
- **Candidate:** Project Candidate [Placeholder]
- **Faculty / Institution:** Faculty of Computer Science & Software Engineering [Placeholder]
- **Mentor:** Academic Mentor / Advisor [Placeholder]
- **Date:** October 2026

### Speaker Notes (0:00 - 0:45)
> "Good morning, members of the evaluation committee and distinguished faculty. Today, I am presenting our capstone engineering and research project: the *AI Product Intelligence & Feedback Analyzer*.
>
> In modern software organizations, product managers are inundated with thousands of incoming customer feedback items every week. Our project investigates whether modern dense neural sentence embeddings objectively outperform classical TF-IDF vectorization when grouping differently worded product feedback, and embeds this comparative intelligence into a fully local, privacy-preserving decision support application."

---

## Slide 2: Problem Statement & Industry Motivation

### Slide Content
- **The Voice-of-Customer Dilemma:**
  - *Volume Overload:* SaaS platforms receive thousands of unstructured messages across app stores, emails, and support tickets.
  - *Vocabulary Mismatch:* Customers describe identical defects using different words (*"freezes on export"* vs *"application shutdown during document download"*).
  - *Manual Triage Bottleneck:* Tagging tickets manually causes delayed bug fixes and engineering churn.
- **Core Engineering Objectives:**
  - 100% Local & Deterministic: Zero dependency on commercial cloud LLMs or recurring paid API keys.
  - Rigorous Comparative Evaluation: Fair, matched empirical testing across multiple cluster counts and random seeds.
  - Decision Support: Automatic translation of cluster statistics into actionable, quote-grounded engineering recommendations.
  - Explainability: Demystified centroid distributions, keywords, and representative exemplars.

### Speaker Notes (0:45 - 1:45)
> "The central operational bottleneck in feedback management is the *vocabulary mismatch problem*. When an outage occurs, ten users might submit tickets using ten completely different phrases without sharing a single technical keyword.
>
> While modern teams often look to cloud LLM APIs, enterprise privacy policies, GDPR compliance, and subscription costs make sending proprietary customer telemetry to external servers unfeasible for many teams. We set out to build an end-to-end local system that provides deep product intelligence with zero paid APIs."

---

## Slide 3: Formal Research Question & Hypotheses

### Slide Content
- **Primary Research Question:**
  *“Does sentence-embedding-based clustering group differently worded product feedback more usefully than a traditional TF-IDF-based approach?”*
- **Hypothesis H1 (Semantic Capture):**
  Dense continuous representations (`sentence-transformers/all-MiniLM-L6-v2`) will achieve statistically higher external reference agreement (ARI) and pairwise grouping F1 on paraphrased complaints lacking exact lexical overlap.
- **Hypothesis H2 (Computational Trade-Off):**
  Classical TF-IDF will achieve orders-of-magnitude lower execution latency (<100ms) and demonstrate superior precision in isolating distinct vocabulary outliers (e.g., billing bugs or rare error codes).

### Speaker Notes (1:45 - 2:45)
> "Our research question is explicitly two-sided: we do not presuppose that deep neural networks are universally superior.
>
> We formalized two operational hypotheses. Hypothesis 1 tests whether pretrained semantic geometry prevents issue fragmentation when paraphrases occur. Hypothesis 2 tests whether TF-IDF's sparse lexical specificity provides faster, cleaner outlier detection for unique technical terms. We established these evaluation criteria before computing our final benchmark tables."

---

## Slide 4: System Architecture

### Slide Content
- **1. Ingestion Layer:** Multi-channel feedback parsing (CSV, paste), URL context extraction with SSRF security, text cleaning, duplicate detection.
- **2. Dual ML Engine:** Parallel feature extraction via `TfidfVectorizer` (1–2 n-grams) and `SentenceTransformer` (384-dim dense vectors), followed by $K$-Means++ clustering.
- **3. Decision Support Layer:** Domain applicability scoring, Negative Impact Score calculation, rule-based recommendation mapping.
- **4. Explainability & UI:** Centroid distance distributions, top TF-IDF keywords, interactive Streamlit frontend, and SQLite persistence.

### Speaker Notes (2:45 - 3:30)
> "Here is our four-layer modular architecture. Notice that the system separates unsupervised machine learning from deterministic decision support. The feedback is ingested, normalized, and simultaneously fed through both vector pipelines.
>
> Once clusters are formed, our decision support engine calculates a Negative Impact Score and maps actionable recommendations directly to verbatim customer quotes. All runs are persisted locally in SQLite."

---

## Slide 5: Methodological Rigor & Ground-Truth Isolation

### Slide Content
- **Strict Label Isolation:**
  - `data/evaluation_labels.csv` is isolated from feature extraction and clustering pipelines.
  - Zero data leakage: vectorizers and $K$-Means operate solely on unsupervised text.
  - Post-hoc metric calculation: ARI and NMI are computed strictly as a blinded benchmark harness.
- **Multi-Faceted Metric Suite:**
  - *External Agreement:* Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) against 10 ground truth themes.
  - *Pairwise Semantic Grouping:* Precision, Recall, and F1 across all 190 record pairs.
  - *Internal Geometric Scores:* Silhouette and Davies-Bouldin (reported within native spaces).
  - *System Runtime:* Sub-millisecond timing across cold-start and warmed states.

### Speaker Notes (3:30 - 4:30)
> "Methodological validity is paramount. We implemented strict physical and algorithmic label isolation. Neither the TF-IDF vectorizer nor the Sentence Transformer has access to ground-truth labels during fitting or inference.
>
> Furthermore, we evaluate both partition-level metrics like ARI, and pairwise metrics across all 190 document pairs to measure exact issue grouping fidelity."

---

## Slide 6: Controlled Benchmark Results

### Slide Content
- **Embedded Visualization:** `results/metrics_comparison.png` (ARI vs Reference & Pairwise F1 across $k \in [3, 4, 5, 6]$).
- **Key Quantitative Summary:**
  - *External Agreement (ARI):* Sentence Transformers: **0.6149 ± 0.1534** vs. TF-IDF: **0.2647 ± 0.1764** (+132% higher alignment).
  - *Pairwise F1:* Sentence Transformers: **0.6680 ± 0.1275** vs. TF-IDF: **0.3744 ± 0.1402**.
  - *Pairwise Recall:* Sentence Transformers achieved **1.0000** recall across all tested seeds (19/19 known pairs united).
  - *Cross-Model Partition ARI:* **0.3339 ± 0.1861** (NMI: **0.5867 ± 0.2119**).

### Speaker Notes (4:30 - 5:45)
> "Turning to our empirical results, across 12 controlled configurations testing $k$ from 3 to 6 across three random seeds:
>
> Sentence Transformers achieved a Mean ARI of 0.6149 compared to 0.2647 for TF-IDF. Even more striking is the Pairwise F1 score: Sentence Transformers achieved 0.6680, maintaining perfect 1.0 recall across all seeds. This confirms Hypothesis 1: dense embeddings effectively prevent issue fragmentation across paraphrased complaints."

---

## Slide 7: Execution Latency & Cross-Model Divergence

### Slide Content
- **Embedded Visualizations:**
  - `results/runtime_comparison.png` (Warmed execution latency in ms vs. cold-start overhead in seconds).
  - `results/pairwise_comparison.png` (Cross-model Partition Agreement ARI & NMI).
- **Latency Analysis:**
  - TF-IDF runtime: **64.41 ± 30.80 ms** (sub-100ms across all 12 fits).
  - Sentence Transformers Warmed: **62.78 ± 5.00 ms** (fully comparable to TF-IDF once cached).
  - Sentence Transformers Cold-Start: **19.57 s** (initial weight loading from disk to CPU memory).

### Speaker Notes (5:45 - 6:30)
> "On Slide 7, we examine computational latency with an essential methodological distinction between cold-start initialization and warmed inference.
>
> When loading neural weights on the very first execution, Sentence Transformers requires approximately 19 seconds. However, once warmed in memory, subsequent inferences take only 62 milliseconds—virtually indistinguishable from TF-IDF at 64 milliseconds! For serverless cold starts, TF-IDF is superior; for long-running services, dense embeddings introduce negligible runtime penalty."

---

## Slide 8: Qualitative Case Study: Paraphrase Grouping

### Slide Content
- **Records Evaluated:**
  - `FB08`: *"Searching through customer records takes over 30 seconds."*
  - `FB09`: *"Search functionality is painfully sluggish on larger lists."*
- **Model Behavior:**
  - *TF-IDF:* Split into Cluster 3 (size 2) and Cluster 2 (size 11). Lexical sparsity diverged cosine similarity below centroid threshold.
  - *Sentence Transformers:* Successfully united both into Cluster 0 (size 5). Mapped *"takes over 30 seconds"* and *"painfully sluggish"* into proximate geometric coordinates.

### Speaker Notes (6:30 - 7:30)
> "Let us inspect the qualitative error analysis. Here we have two real feedback items from our benchmark describing search latency.
>
> TF-IDF completely fragmented them: FB08 was isolated in a small cluster, while FB09 was dumped into a massive 11-record catch-all cluster. Because one used '30 seconds' and the other used 'painfully sluggish', TF-IDF found zero n-gram intersection. Sentence Transformers, however, projected both into the exact same performance cluster."

---

## Slide 9: Qualitative Case Study: The TF-IDF Catch-All Failure Mode

### Slide Content
- **TF-IDF Cluster 2 ($N=11$ records, 55% of dataset):**
  - Absorbed: 2FA delays, login failures, search sluggishness, dark mode requests, Google Drive import, Slack webhooks, onboarding praise, and API 404 documentation errors.
- **Mathematical Cause:**
  - In small datasets, distinct vocabularies are orthogonal in sparse space ($d = \sqrt{2}$). $K$-Means assigns orthogonal items to a diffuse residual centroid.
- **Product Hazard:** Critical authentication vulnerabilities (`FB06`, `FB07`) are submerged beneath cosmetic UI suggestions.

### Speaker Notes (7:30 - 8:30)
> "Slide 9 highlights the primary failure mode of sparse representations on short texts: the 'catch-all' cluster. Under TF-IDF at $k=4$, Cluster 2 absorbed over half the dataset.
>
> This is mathematically caused by vector orthogonality. If an item does not share frequent words with the dominant crash cluster, it collapses into a diffuse residual bucket. For an engineering team, this is dangerous: severe login and 2FA failures were mixed with dark mode feature requests."

---

## Slide 10: Decision Support & Evidence Grounding

### Slide Content
- **Negative Impact Score:** Formulated as $\text{Size} \times (5 - \overline{\text{Rating}}) \times \text{Bug Ratio}$. Prioritizes severe bugs over cosmetic items.
- **Applicability Engine:** Scores relevance against target audience and product domain, filtering out out-of-scope noise.
- **Rule-Based Recommendations:** Maps identified defect patterns into concrete engineering, UX, or documentation actions.
- **Verbatim Grounding:** Every action links directly to customer quotes, ensuring total traceability.

### Speaker Notes (8:30 - 9:30)
> "Clustering alone does not solve the product owner's problem. Our application translates clusters into prioritized decision support.
>
> We formulate an automated Negative Impact Score combining issue volume, average star rating, and bug proportion. This elevates critical defects like PDF export crashes to the top of the backlog, accompanied by concrete action plans grounded in verbatim customer quotes."

---

## Slide 11: Interactive Streamlit MVP

### Slide Content
- **Tab 1: Product Definition & Context Ingestion** (pasted context, secure URL scraping).
- **Tab 2: Feedback Ingestion & Validation** (CSV upload, deduplication report).
- **Tab 3: Dual Clustering & Model Comparison** (side-by-side metric inspection).
- **Tab 4: Evidence-Grounded Recommendations** (ranked action cards).
- **Tab 5: Analysis History & Persistence** (local SQLite database).

### Speaker Notes (9:30 - 10:15)
> "The system is packaged as an interactive, five-tab Streamlit web application. A product manager can upload a CSV, see deduplication statistics, execute both clustering pipelines side-by-side, view explainability histograms, and review prioritized recommendation cards.
>
> All data and run histories persist locally in SQLite without cloud dependencies."

---

## Slide 12: Threats to Validity & Limitations

### Slide Content
- **Scientific Limitations:**
  - Synthetic benchmark dataset ($N=20$) curated to test paraphrase detection; empirical scaling to real customer telemetry is required.
  - Single annotator ground truth; no multi-rater Cohen's Kappa.
  - $K$-Means spherical geometry forces outliers into nearest centroid.
  - Topic attraction vs. sentiment polarity: Sentence Transformers groups positive praise with negative complaints if they discuss the same feature.
- **Ethical Safeguards:**
  - 100% local processing protects customer PII.
  - Rule-based decision support prevents generative hallucination.

### Speaker Notes (10:15 - 11:00)
> "In academic research, acknowledging limitations is essential. Our benchmark dataset is synthetic, designed specifically to evaluate paraphrase handling.
>
> Furthermore, Sentence Transformers groups by semantic topic regardless of sentiment tone. For instance, praise for recent speed improvements was grouped with complaints about query latency. In our application, we address this by coupling clustering with independent sentiment and rating filters."

---

## Slide 13: Conclusion & Key Findings

### Slide Content
- **1. Clear Answer to Research Question:** Sentence Transformers objectively outperforms TF-IDF in grouping differently worded feedback (+132% higher ARI, 0.6680 Pairwise F1 vs 0.3744).
- **2. Complementary Baselines:** TF-IDF remains valuable for sub-100ms real-time triage and isolating unique vocabulary outliers (such as billing bugs).
- **3. Hybrid Architecture Recommended:** The optimal industry system pairs dense embeddings for theme discovery with TF-IDF indexing for keyword search.
- **4. Capstone Delivery:** 60 passing automated tests, clean install verification script, SQLite database, and complete documentation.

### Speaker Notes (11:00 - 11:45)
> "In conclusion, our research demonstrates that sentence embeddings definitively solve the vocabulary mismatch problem in product feedback triage.
>
> However, neither approach is universally superior. A hybrid architecture—leveraging dense representations for semantic discovery and sparse indexing for exact keyword triage—provides the optimal balance of accuracy and latency."

---

## Slide 14: Future Scope

### Slide Content
- **HDBSCAN & Density-Based Clustering:** Eliminate fixed $k$ requirement and support variable-density outlier detection.
- **Multi-Annotator Industry Corpus:** Benchmark on 10,000+ open-source telemetry items with multi-rater Fleiss' Kappa.
- **Contrastive Domain Fine-Tuning:** Fine-tune MiniLM on SaaS error logs using triplet loss.
- **Automated Issue Tracker Sync:** Two-way integration with Jira, GitHub Issues, and Linear.

### Speaker Notes (11:45 - 12:15)
> "Future iterations will explore density-based algorithms like HDBSCAN to eliminate the fixed $k$ parameter and support automated syncing directly into engineering issue trackers like Jira and GitHub."

---

## Slide 15: References & Defense Closure

### Slide Content
- Complete formal citations (Reimers & Gurevych 2019, Salton & Buckley 1988, MacQueen 1967, Hubert & Arabie 1985, Strehl & Ghosh 2002, Rousseeuw 1987).
- Formal acknowledgments.
- Defense floor open for questions.

### Speaker Notes (12:15 - 12:30)
> "Thank you for your time and attention. I look forward to your questions."
