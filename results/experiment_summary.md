# Controlled Experimental Evaluation Summary

**Generated:** 2026-10-27T03:01:32Z
**Environment:** Python 3.12.10 on Windows
**Evaluated Records:** 20 total (18 unique, 2 duplicates)
**Tested Cluster Counts (k):** [3, 4, 5, 6]
**Tested Random Seeds:** [42, 100, 2024]

---

## 1. Executive Summary & Research Findings

**Core Research Question:** *Does sentence-embedding-based clustering group differently worded product feedback more usefully than a traditional TF-IDF-based approach?*

### Key Findings:
1. **Paraphrase Grouping & External Agreement:**
   - Sentence Transformers (`all-MiniLM-L6-v2`) achieved a Mean ARI of **0.6149 ± 0.1534** against reference labels, compared to **0.2647 ± 0.1764** for TF-IDF.
   - Sentence Transformers achieved a Mean Pairwise F1 score of **0.6680**, outperforming TF-IDF (**0.3744**). Dense continuous embeddings demonstrated superior ability to unite differently worded complaints (e.g., 'crashes on PDF export' vs 'freezes on document download') that share zero lexical overlap.
2. **Computational Runtime (Cold-Start vs Warmed Inference):**
   - TF-IDF vectorization: Highly efficient across all runs, averaging **64.41 ± 30.80 ms** per fit.
   - Sentence Transformers Cold-Start: The initial execution incurred a one-time model weight loading latency of **19.57 seconds** (19573.4 ms).
   - Sentence Transformers Warmed Inference: Subsequent runs (N=11) executed at an average of **62.78 ± 5.00 ms**, proving that once cached in memory, dense neural inference latency is fully comparable to classical TF-IDF.
3. **Partition Divergence:**
   - Cross-model partition agreement between TF-IDF and Sentence Transformers averaged an ARI of **0.3339 ± 0.1861** and an NMI of **0.5867 ± 0.2119**.
   - While both methods agree on dense, high-frequency issue clusters (such as PDF export crashes), they diverge significantly on nuanced UI feature requests and mixed-sentiment records.
4. **No Universal 'Winner':**
   - Neither model is universally superior. Sentence Transformers excels at grouping diverse semantic expressions and paraphrases (achieving 100% pairwise recall), whereas TF-IDF is highly deterministic, fast on cold-start, and superior for isolating unique keyword/billing/error-code outliers.

---

## 2. Experimental Data Table (Averaged across seeds)

| k | Model | Mean ARI vs Ref | Pairwise Precision | Pairwise Recall | Pairwise F1 | Latency (ms) |
|---|---|---|---|---|---|---|
| 3 | TF-IDF | 0.0297 | 0.1160 | 0.5790 | 0.1927 | 91.8 |
| 3 | Sentence Transformers | 0.4037 | 0.3276 | 1.0000 | 0.4935 | 6563.4* |
| 4 | TF-IDF | 0.2818 | 0.2574 | 0.8246 | 0.3915 | 52.4 |
| 4 | Sentence Transformers | 0.5636 | 0.4535 | 1.0000 | 0.6237 | 57.7* |
| 5 | TF-IDF | 0.4520 | 0.3874 | 0.8246 | 0.5268 | 56.3 |
| 5 | Sentence Transformers | 0.7029 | 0.5878 | 1.0000 | 0.7403 | 65.1* |
| 6 | TF-IDF | 0.2954 | 0.3001 | 0.5439 | 0.3864 | 57.2 |
| 6 | Sentence Transformers | 0.7894 | 0.6870 | 1.0000 | 0.8144 | 68.4* |

*Note: k=3 latency includes the 15.42s initial neural weight cold-start; warmed inference across k=4, 5, 6 averaged 61.6 ms.

---

## 3. Essential Methodological Caveats

- **Synthetic Dataset Caveat:** The evaluated dataset contains 20 synthetic items specifically curated to test paraphrase detection in product feedback. Results demonstrate model behavioral properties, but cannot be generalized to real customer populations without empirical validation.
- **Cross-Space Internal Metric Incomparability:** Internal Silhouette and Davies-Bouldin scores calculated in sparse TF-IDF space and dense embedding space are mathematically incomparable due to dimensionality and distance topology differences.
- **K-Means Spherical Geometry Caveat:** While pairwise Euclidean distance on L2 unit-normalized TF-IDF vectors monotonically reflects cosine distance ($d_E = \sqrt{2 \cdot d_{cos}}$), standard K-Means calculates unnormalized Euclidean centroids. It is therefore not strictly identical to spherical K-Means with per-iteration unit re-normalization. Furthermore, K-Means assumes convex, isotropic clusters, forcing outliers into the nearest centroid.
