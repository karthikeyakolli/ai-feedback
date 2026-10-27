# Testing and Verification Guide

## 1. Test Strategy
Use pytest for unit and integration tests. Test the data pipeline, clustering algorithms, and evaluation functions independently of Streamlit. Use the included synthetic dataset (`data/sample_feedback.csv`) and separate evaluation labels (`data/evaluation_labels.csv`) for repeatable verification.

## 2. Environment Setup
Recommended Python version: 3.11+ (Verified on Python 3.12.10).

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Verified Execution Commands
```powershell
# Run the complete automated test suite (60 tests)
.\.venv\Scripts\python.exe -m pytest -v

# Start the interactive Streamlit application
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*(Note: On Windows machines with AppLocker or Application Control policies, invoking via `python.exe -m <module>` guarantees execution without binary shim blocking).*

## 4. Test Suite Coverage (60 Tests Passing)

### Input Validation & Data Preparation (`tests/test_data_processing.py` - 12 Tests)
1. `test_normalize_whitespace`: Collapses consecutive spaces, newlines, and tabs.
2. `test_empty_dataframe`: Confirms empty inputs safely yield 0 records with an informative warning.
3. `test_missing_text_column`: Detects missing text columns and flags them in invalid row reports.
4. `test_blank_and_null_texts_handled_safely`: Filters null, empty, and whitespace-only records without crashing.
5. `test_original_text_strictly_preserved`: Guarantees `original_text` remains unmodified while `cleaned_text` is normalized.
6. `test_duplicate_flagging_without_silent_dropping`: Detects exact duplicates, assigns `duplicate_of_id`, and retains them.
7. `test_small_dataset_warning`: Warns when fewer than 4 valid records exist.
8. `test_load_sample_feedback_csv`: Validates end-to-end loading of the 20-item synthetic sample CSV.
9. `test_pasted_multiline_feedback`: Parses multiline text one line per item, skips empty lines, and assigns stable IDs.
10. `test_custom_csv_column_selection`: Allows custom text and ID column mapping on arbitrary CSV formats.
11. `test_evaluation_label_column_isolation`: Ensures label/target columns are never automatically picked as text features.
12. `test_empty_pasted_text`: Returns clean zero-record warning on whitespace-only pasted text.

### AI Clustering & Evaluation (`tests/test_clustering.py` - 8 Tests)
13. `test_adjust_k_safety`: Clamps $k$ when $k \ge n$ or $k < 2$; raises ValueError when $n < 2$.
14. `test_tfidf_clustering_execution`: Runs TF-IDF + K-Means, discovers centroid representatives, and extracts keywords.
15. `test_tfidf_evaluation_and_limitations`: Computes Silhouette and ARI against ground truth, documenting limitations.
16. `test_sentence_embedding_clustering`: Runs Sentence Transformers (`all-MiniLM-L6-v2`) and verifies normalized vectors.
17. `test_compare_models_same_dataset_requirement`: Enforces strict record ID parity between compared models.
18. `test_embedding_caching_mechanism`: Confirms unchanged feedback inputs reuse precomputed in-memory embeddings.
19. `test_metric_preconditions_safely_handled`: Confirms spatial metrics are marked unavailable with clear reasons when preconditions fail.
20. `test_model_result_and_evaluation_representation_metadata`: Confirms representation and distance metric metadata are exposed.

### Context Ingestion, SSRF Protection, Insights & Export (`tests/test_context_and_insights.py` - 13 Tests)
21. `test_ip_allowlist_blocking`: Validates blocking of private (10.x, 192.168.x), loopback (127.0.0.1), link-local, and reserved IP ranges.
22. `test_url_safety_scheme_and_hostname`: Enforces HTTP/HTTPS scheme restrictions and rejects localhost or non-FQDN hostnames.
23. `test_url_extraction_blocks_private_dns`: Confirms URLs resolving to internal/private IPs are rejected prior to network transmission.
24. `test_url_extraction_timeout_and_error_handling`: Mocks network timeout (10s) and connection failures, returning graceful error statuses without app crashes.
25. `test_url_extraction_unsupported_content_type`: Rejects binary or non-HTML/text payloads (e.g., application/pdf, image/png).
26. `test_url_extraction_success_mock`: Confirms successful readable text extraction using Trafilatura parsing on mocked HTML responses.
27. `test_product_context_model_and_approval`: Tests manual description and pasted documentation modeling, provenance metadata, and explicit user approval flag.
28. `test_insights_generation_and_evidence_traceability`: Confirms cluster theme generation with centroid quotes, exact member IDs, and duplicate counts.
29. `test_context_relevance_cautious_correlation`: Verifies cautious, non-speculative keyword matching against approved product context without unsupported causal claims.
30. `test_export_feedback_records_csv`: Verifies CSV export containing stable record IDs, original feedback, duplicate flags, and assigned cluster labels.
31. `test_export_cluster_summaries_csv`: Verifies CSV export of cluster metrics (size, percentage share, keywords, duplicate count).
32. `test_export_model_evaluation_csv`: Verifies CSV export of model evaluation scores and unavailable metric reasons.
33. `test_export_full_markdown_report`: Verifies generation of a complete, human-readable Markdown summary report with evidence provenance.

### SQLite Persistence, Project Management & Run History (`tests/test_database.py` - 11 Tests)
34. `test_database_initialization`: Verifies database file creation, schema initialization (`PRAGMA user_version = 1`), foreign key enforcement, WAL mode, and idempotence.
35. `test_project_crud_and_validation`: Tests project creation, validation, retrieval, listing, renaming, and safe deletion.
36. `test_product_context_versioning`: Verifies monotonically increasing versioning of product context across updates without overwriting historical context snapshots.
37. `test_source_metadata_storage`: Validates recording of source provenance (manual, pasted docs, public URL) without storing credentials or refetching URLs.
38. `test_feedback_records_persistence_and_conflict_handling`: Validates feedback storage and conflict strategies (`error`, `replace_all`, `skip`, `reassign_id`).
39. `test_analysis_run_lifecycle_and_atomic_persistence`: Tests atomic transition (`running` -> `completed`/`failed`) and transaction rollback on error.
40. `test_reconstruct_run_results_offline`: Reconstructs complete `ModelResult` and `EvaluationReport` objects from SQLite with 100% offline fidelity without ML model re-computation.
41. `test_compare_historical_runs_same_and_different_datasets`: Calculates partition agreement (Adjusted Rand Index) across identical record partitions, handles differing datasets, and verifies caveats.
42. `test_project_data_isolation`: Confirms that records, contexts, and runs belonging to Project Alpha never leak into Project Beta.
43. `test_get_run_feedback_records_preserves_evaluation_order`: Confirms that reopening a run retrieves the evaluated subset of feedback records in their exact evaluated order.
44. `test_cascade_project_deletion_cleans_all_tables`: Verifies that permanently deleting a project cascades cleanly across all 7 dependent tables.

### Applicability Assessment & Investigation Recommendations (`tests/test_applicability_and_recommendations.py` - 12 Tests)
45. `test_informative_tokens_extraction`: Verifies alphanumeric extraction, length filter, and stopword exclusion.
46. `test_applicability_missing_or_unapproved_context`: Confirms missing or unapproved context strictly returns `Insufficient information`.
47. `test_applicability_empty_cluster`: Confirms zero member records return `Insufficient information`.
48. `test_applicability_uninformative_or_stopword_feedback`: Confirms short fragments or stopword-only records return `Insufficient information`.
49. `test_applicability_strongly_supported_overlap`: Verifies strong lexical overlap returns `Likely relevant` with traceable quotes and context concept linkage.
50. `test_applicability_partial_or_unrelated_returns_needs_validation`: Confirms partial or unaligned feedback returns `Needs validation`.
51. `test_applicability_contradictory_evidence_detection`: Flags mixed polarity/contradictions (working well vs crashing) in `contradictory_evidence`.
52. `test_applicability_determinism`: Proves identical inputs yield identical outputs.
53. `test_recommendation_evidence_grounding_and_traceability`: Confirms recommendations retain real feedback IDs and verbatim quotes.
54. `test_recommendation_accounts_for_duplicates`: Proves duplicates are accounted as repetitions rather than independent corroborations.
55. `test_investigation_queue_ordering_and_deduplication`: Verifies deduplication and frequency-based ordering by unique mentions.
56. `test_export_investigation_recommendations_csv`: Validates CSV export formatting quotes, commas, newlines, and human review reminders.

### Explainable Clustering Guides & Caveats (`tests/test_explanations.py` - 4 Tests)
57. `test_tfidf_explanation_content`: Verifies educational coverage of text conversion, TF-IDF weights, cosine distance, K-Means centroids, and wording limitations.
58. `test_sentence_transformer_explanation_content`: Verifies explanation of MiniLM dense 384-d vectors, continuous semantic geometry, and nuance limitations.
59. `test_metric_explanations_direction_and_caveats`: Confirms directions of preference and utility caveats for Silhouette, Davies-Bouldin, ARI, NMI, and Entropy.
60. `test_fair_comparison_caveat`: Verifies cross-representation incomparability rules, four evaluation pillars, and winner declaration warning.

## 5. Definition of Done (Phase 5 Verified)
A feature is complete only when:
- The implementation exists, is fully typed, and adheres to local-first rules.
- Automated pytest tests pass with full output evidence (60/60 passed in 21.0s).
- Applicability assessments strictly adhere to `Likely relevant`, `Needs validation`, and `Insufficient information`.
- Recommendations suggest what to investigate, avoiding autonomous product decisions or fabricated business claims.
- Every supporting quote and feedback ID is traceable to real feedback records verbatim.
- Cross-representation internal metric incomparability is prominently explained with educational caveats.
- SQLite persistence and historical run reconstruction remain strictly isolated and non-destructive.
- Multi-format exports include `investigation_recommendations.csv` and enhanced summary reports.
- Streamlit application executes end-to-end with zero exceptions (`AppTest` verified, HTTP 200 OK).
- Documentation and `TASKS.md` accurately reflect reality.
