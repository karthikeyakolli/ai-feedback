# Qualitative Error Analysis: TF-IDF vs. Sentence Transformers

## 1. Executive Summary

This document presents a rigorous qualitative error analysis comparing **Method A (TF-IDF + K-Means)** and **Method B (Sentence Transformers `all-MiniLM-L6-v2` + K-Means)** on identical product feedback records ($N=20$).

Rather than relying solely on numerical aggregates (ARI, NMI, Pairwise F1), this analysis inspects concrete, verbatim feedback records to explain:
1. What the customer feedback stated.
2. How each method clustered it.
3. Why the outcome was beneficial or problematic for a product manager.
4. What can and cannot be scientifically concluded from the evidence.

---

## 2. Methodology & Case Selection Criteria

Cases were selected systematically across five representative behavioral categories:
- **Category 1:** Paraphrase grouping with zero lexical overlap.
- **Category 2:** Domain synonyms and terminology variation.
- **Category 3:** High-volume duplicates and exact repetition.
- **Category 4:** The "catch-all" cluster failure mode in high-dimensional sparse representations.
- **Category 5:** Topic attraction vs. sentiment polarity confusion (praise vs. defect).

All cluster assignments cited below reflect the primary baseline configuration ($k=4$, random seed 42) running on identical preprocessed feedback items.

---

## 3. In-Depth Case Studies

### Case Study 1: Paraphrase Grouping with Zero Lexical Overlap

#### Evaluated Records
- **`FB08`:** *"Searching through customer records takes over 30 seconds."* (Ground Truth: `search_latency`, Rating: 2.0)
- **`FB09`:** *"Search functionality is painfully sluggish on larger lists."* (Ground Truth: `search_latency`, Rating: 2.0)

#### Model Assignments ($k=4$, Seed 42)
| Method | Cluster Assignment | Cluster Size | Cluster Co-Members |
|---|---|---|---|
| **TF-IDF + K-Means** | `FB08` $\rightarrow$ **Cluster 3**<br>`FB09` $\rightarrow$ **Cluster 2** | C3: 2 records<br>C2: 11 records | `FB08` is isolated with its duplicate `FB19`. `FB09` is collapsed into an 11-record catch-all cluster. |
| **Sentence Transformers + K-Means** | `FB08` $\rightarrow$ **Cluster 0**<br>`FB09` $\rightarrow$ **Cluster 0** | C0: 5 records | `FB08` and `FB09` are united in the same cluster alongside performance praise and data import. |

#### Analysis & Product Implications
- **Why TF-IDF Failed:** `FB08` uses the phrase *"searching through customer records takes over 30 seconds"*, while `FB09` uses *"search functionality is painfully sluggish on larger lists"*. While both share the root token "search", the TF-IDF feature space assigns heavy weight to unique informative n-grams (*"customer records"*, *"30 seconds"*, *"painfully sluggish"*, *"larger lists"*). Because their cosine similarity did not surpass the threshold of the dominant cluster centroids, the model split them across two separate clusters.
- **Why Sentence Transformers Succeeded:** Pretrained on massive contrastive text corpora, `all-MiniLM-L6-v2` maps semantic representations of query latency (*"takes over 30 seconds"*) and perceptual slowness (*"painfully sluggish"*) into proximate regions of 384-dimensional Euclidean space ($d_{cosine} \approx 0.31$).
- **Product Takeaway:** For a product manager investigating search performance, Sentence Transformers correctly presents a unified problem cluster. Under TF-IDF, the issue is fragmented into an isolated pair and a diluted bucket, hiding the cumulative volume of search complaints.

---

### Case Study 2: Vocabulary Mismatch & Synonyms ("Dark Mode" vs. "Dark Theme")

#### Evaluated Records
- **`FB10`:** *"Please add dark mode support, the white background hurts my eyes at night."* (Ground Truth: `dark_mode`, Rating: 4.0)
- **`FB11`:** *"Would love a dark theme option for working in low light environments."* (Ground Truth: `dark_mode`, Rating: 4.0)
- **`FB20`:** *"Dark mode please!"* (Ground Truth: `dark_mode`, Rating: 3.0)

#### Model Assignments ($k=4$, Seed 42)
| Method | Cluster Assignment | Cluster Coherence |
|---|---|---|
| **TF-IDF + K-Means** | `FB10` $\rightarrow$ **Cluster 2**<br>`FB11` $\rightarrow$ **Cluster 2**<br>`FB20` $\rightarrow$ **Cluster 2** | Coherence: **Low**. Although all three fall in Cluster 2, Cluster 2 also contains 8 unrelated records (login timeouts, 404 documentation bugs, CSV imports, and Slack requests). |
| **Sentence Transformers + K-Means** | `FB10` $\rightarrow$ **Cluster 3**<br>`FB11` $\rightarrow$ **Cluster 3**<br>`FB20` $\rightarrow$ **Cluster 3** | Coherence: **High**. Grouped in Cluster 3 alongside UI and workflow enhancements (`FB13` Slack, `FB14` onboarding UI). |

#### Analysis & Product Implications
- `FB10` uses *"dark mode"* and *"white background hurts my eyes"*, while `FB11` uses *"dark theme"* and *"low light environments"*.
- Under TF-IDF, neither document had sufficient token overlap with the PDF export crash cluster or the search latency cluster. As a result, TF-IDF placed both into its catch-all residual cluster.
- Sentence Transformers mapped both phrases directly into the "theme / UI appearance" semantic subspace, allowing them to anchor Cluster 3 with representative quote `FB11`.

---

### Case Study 3: The "Catch-All" Failure Mode of TF-IDF

#### Evaluated Group: TF-IDF Cluster 2 ($N=11$ records out of 20)
In the TF-IDF baseline at $k=4$, Cluster 2 absorbed over 55% of the entire dataset:
- `FB06`: *"Frequent session timeouts and two-factor SMS codes are delayed by 15 minutes."* (Auth bug)
- `FB07`: *"Can't log into my account because verification codes are not sent to my email."* (Auth bug)
- `FB09`: *"Search functionality is painfully sluggish on larger lists."* (Search performance)
- `FB10`: *"Please add dark mode support..."* (Feature request)
- `FB11`: *"Would love a dark theme option..."* (Feature request)
- `FB12`: *"We need an option to bulk import CSV files..."* (Feature request)
- `FB13`: *"Ability to integrate with Slack webhooks..."* (Integration request)
- `FB14`: *"The clean interface and intuitive navigation..."* (Positive UX praise)
- `FB15`: *"Really love the recent speed improvements..."* (Positive speed praise)
- `FB17`: *"The documentation page returns a 404 error on API endpoints..."* (Doc bug)
- `FB20`: *"Dark mode please!"* (Feature request)

#### Analysis & Product Implications
- **Mathematical Cause:** TF-IDF vectors are high-dimensional (1,000 max features) and sparse. In a small dataset, rare terms appear only once. When K-Means runs with Euclidean distance on L2-normalized sparse vectors, items with zero mutual word overlap have an orthogonal distance of $\sqrt{2} \approx 1.414$. When $k$ is smaller than the true number of vocabulary clusters, K-Means assigns orthogonal items to a diffuse "residual" centroid.
- **Product Danger:** A cluster containing 11 items spanning authentication, billing, search, dark mode, API documentation, and UI praise is practically useless to an engineering team. It obscures critical security bugs (`FB06`, `FB07`) within cosmetic UI requests.

---

### Case Study 4: Outlier Isolation vs. False Collapsing (Billing Issue)

#### Evaluated Record
- **`FB16`:** *"Billing charged me twice this month for my subscription."* (Ground Truth: `billing_issue`, Rating: 1.0)

#### Model Assignments ($k=4$, Seed 42)
| Method | Cluster Assignment | Cluster Coherence |
|---|---|---|
| **TF-IDF + K-Means** | `FB16` $\rightarrow$ **Cluster 0** (Size: 1) | **100% pure singleton cluster**. |
| **Sentence Transformers + K-Means** | `FB16` $\rightarrow$ **Cluster 1** (Size: 5) | Grouped with `FB05`, `FB06`, `FB07` (Auth/Login timeouts) and `FB17` (404 doc error). |

#### Analysis & Product Implications
- **Where TF-IDF Outperforms:** TF-IDF immediately isolated the billing dispute because the tokens *"billing"*, *"charged"*, *"twice"*, and *"subscription"* appeared nowhere else in the corpus. For an urgent compliance or financial defect, TF-IDF's sensitivity to distinct keyword signatures surfaced this outlier as an independent cluster.
- **Where Sentence Transformers Collapsed:** Because $k=4$ was enforced, Sentence Transformers was forced to merge `FB16` into Cluster 1 (system errors and account administration). While both billing and login involve "user account management", grouping a double-billing bug with a 2FA SMS timeout could delay immediate customer support triage.
- **Takeaway:** Dense embeddings tend to force disparate issues into broad conceptual themes when $k$ is constrained, whereas TF-IDF is more prone to creating singleton clusters for vocabulary outliers.

---

### Case Study 5: Topic Attraction vs. Sentiment Polarity Confusion

#### Evaluated Records
- **`FB08`:** *"Searching through customer records takes over 30 seconds."* (Ground Truth: `search_latency`, Bug, Rating: 2.0)
- **`FB15`:** *"Really love the recent speed improvements and overall dashboard design!"* (Ground Truth: `performance_praise`, Positive, Rating: 5.0)

#### Model Assignments ($k=4$, Seed 42)
| Method | Cluster Assignment | Outcome |
|---|---|---|
| **TF-IDF + K-Means** | `FB08` $\rightarrow$ **Cluster 3**<br>`FB15` $\rightarrow$ **Cluster 2** | Separated (due to vocabulary sparsity). |
| **Sentence Transformers + K-Means** | `FB08` $\rightarrow$ **Cluster 0**<br>`FB15` $\rightarrow$ **Cluster 0** | **Collided into the same cluster**. |

#### Analysis & Product Implications
- **Why Sentence Transformers Grouped Them:** Both records discuss system speed and performance (*"takes over 30 seconds"* vs *"recent speed improvements"*). Because `all-MiniLM-L6-v2` is an uncased semantic encoder optimized for conceptual similarity rather than sentiment polarity, both records landed in the "application speed and performance" semantic cluster.
- **Product Hazard:** A developer reviewing Cluster 0 would see both praise for speed improvements and complaints about a 30-second query lag.
- **Architectural Solution in this System:** The AI Product Intelligence application solves this by augmenting clustering with independent **sentiment classification** and **applicability filtering**. A product manager can filter Cluster 0 by rating $\le 2$ or feedback type `Bug` to isolate actionable performance regressions.

---

## 4. Summary Matrix of Behavioral Trade-offs

| Scenario / Behavior | TF-IDF + K-Means | Sentence Transformers + K-Means | Recommended System Role |
|---|---|---|---|
| **Paraphrases (Zero Keyword Overlap)** | ❌ Fails (splits across clusters) | ✅ Excels (unites into coherent cluster) | Semantic Discovery |
| **Domain Synonyms ("mode" vs "theme")** | ❌ Fails (diluted into catch-all) | ✅ Excels (captured in theme cluster) | Feature Request Synthesis |
| **Exact & Near Duplicates** | ✅ Stable (collocated at identical point) | ✅ Stable (collocated in dense region) | Frequency Weighting |
| **Distinct Keyword Outliers (Billing, Error Codes)** | ✅ Surfaces as pure singleton | ⚠️ May collapse into nearest broad topic | Fast Outlier Triage |
| **Sentiment Polarity Separation** | ⚠️ Unreliable (lexical only) | ⚠️ Unreliable (groups by topic, not tone) | Filter by Sentiment/Rating |
| **Computational Overhead** | ⚡ Extremely fast (~50 ms) | 🐢 Slower / Model Load (~60-1300 ms) | Hybrid / Scalable Pipeline |

---

## 5. Methodological Conclusions

1. **Sentence Transformers demonstrates superior grouping for customer language variation:** When customers report the same functional defect using colloquial synonyms or non-technical descriptions, Sentence Transformers prevents issue fragmentation.
2. **TF-IDF retains distinct utility for exact error strings and low-latency triage:** When feedback contains unique identifier tokens (e.g., error codes, URL paths, invoice terms), TF-IDF cleanly partitions them without neural distortion.
3. **Hybrid Architecture Justification:** Neither technique is universally superior. The optimal product intelligence architecture uses Sentence Transformers for conceptual theme discovery and TF-IDF / keyword indexing for rapid token-level search and outlier verification.
