# Product Requirements Document (PRD)
## AI Product Intelligence & Feedback Analyzer

**Status:** Baseline scope frozen for prototype
**Target:** Working Python prototype for an AI mini-capstone
**Primary implementation:** Python + Streamlit
**Research focus:** Compare traditional TF-IDF clustering with sentence-embedding clustering for product feedback.

## 1. Product Vision
Help a product owner understand accessible product information and user feedback, identify recurring themes, inspect supporting evidence, and decide what to investigate next. The application must explain its findings rather than presenting unexplained AI labels.

The prototype is an analysis and decision-support tool. It is not a complete public social network or an autonomous product manager.

## 2. Problem Statement
Product feedback can arrive in different formats and sources. Similar issues may be described with different wording, making manual organisation slow. A product owner also needs to distinguish what the product documentation says from what users actually report, understand the evidence behind AI-generated groups, and know where the available data is insufficient.

## 3. Target Audience
### Primary persona: Indie developer / small product team
- Has a website, app, or software product.
- Has feedback from users, public sources, pasted text, or CSV files.
- Needs to discover repeated issues and feature requests.
- Wants transparent findings and practical next steps, not unsupported claims.

### Secondary persona: Student / AI evaluator
- Needs to demonstrate a real AI pipeline.
- Needs to compare algorithms on the same dataset.
- Needs reproducible measurements, error analysis, and clear documentation.

## 4. User Stories
1. As a developer, I can create a named analysis project for a product.
2. I can provide a product URL and optional documentation/reference URLs.
3. I can see which source content was extracted, what failed, and where it came from.
4. I can add feedback through pasted text or a CSV file.
5. I can keep product context separate from feedback records.
6. I can inspect data quality before running analysis.
7. I can run a TF-IDF baseline and an embedding-based clustering approach on the same feedback dataset.
8. I can inspect themes, cluster members, similar feedback, representative examples, and source references.
9. I can see the measured metrics, runtime, parameters, and known failure cases.
10. I can see possible recommendations with evidence, missing information, and uncertainty clearly labelled.
11. I can export a report and return to saved project results later.
12. I can use the application when an optional AI API is unavailable; the core experiment must not require a paid API.

## 5. Product Scope
### P0 — Required MVP
- Streamlit multipage application.
- Create/select a local analysis project.
- Product context fields and public website URL input.
- Safe, bounded extraction of accessible public website text.
- CSV upload and pasted feedback input.
- Source preview, source status, and extraction errors.
- Separate product-context and feedback records.
- Input validation and data-quality report.
- Original feedback preserved; exact and near-duplicate records are flagged, not silently deleted.
- TF-IDF vectorisation, cosine similarity, and K-means.
- Small pretrained sentence-embedding model, cosine similarity, and K-means.
- Same-dataset comparison and reproducible experiment settings.
- Evaluation metrics where valid, manual cluster-quality checks, runtime, and error examples.
- Cluster detail view with original evidence and source provenance.
- Clear explanation of what is known, inferred, suggested, and unknown.
- Basic export of results and experiment details.
- Tests, README, limitations, and capstone documentation.

### P1 — Add only after P0 works
- GitHub public README / issue ingestion using supported public API endpoints.
- PDF and TXT readers; DOCX only after it is tested.
- Sentiment estimation with documented limitations.
- Editable theme names and issue categories.
- Transparent heuristic priority indicator based on available evidence.
- Applicability assessment: `Likely relevant`, `Needs validation`, or `Insufficient information`.
- Recommendations using deterministic templates/rules.
- Report export in Markdown/HTML and CSV.

### P2 — Future scope
- Direct integrations with multiple app stores/review platforms.
- Broad crawling, authenticated sources, or scraping that bypasses access controls.
- Social network features, accounts, public discussions, voting, live chat, notifications.
- Hosted multi-user workspaces and role-based permissions.
- Scheduled source monitoring and change tracking.
- Advanced causal impact or churn prediction.
- Autonomous code changes or claims that an issue has been fixed.
- Mandatory generative AI API use.

## 6. Non-Goals
- Do not promise full analysis of every website/app from a URL.
- Do not infer that product documentation is user feedback.
- Do not fabricate metrics, review counts, confidence values, or business impact.
- Do not automatically delete or merge similar user comments.
- Do not claim that a cluster proves a real product defect.
- Do not treat cosine similarity as a probability of correctness.
- Do not require a paid API for the core pipeline.
- Do not build the complete social product in this capstone phase.

## 7. Core User Workflow
1. Create a project and enter product name/context.
2. Add a website/reference URL and/or feedback sources.
3. Extract accessible source text and preview it.
4. Add or import actual feedback.
5. Review data quality and confirm the feedback set.
6. Run both AI approaches on the same prepared feedback set.
7. Inspect clusters, related feedback, metrics, errors, and evidence.
8. Review applicability and suggested investigation steps.
9. Export the analysis and experiment results.

## 8. Explainability Contract
Every important finding should show:
- Theme or cluster label (clearly marked as a suggested label if generated).
- Original feedback items supporting it.
- Source URL or input source where available.
- Number of records and distinct reviewers only when reliably available.
- Algorithm, model identifier, settings, and relevant metrics.
- What the result may indicate.
- Missing evidence and uncertainty.
- Potential next step, clearly labelled as a suggestion.
- Limitations and possible failure modes.

## 9. Success Criteria
- A user can complete the core workflow with the included sample data.
- Both clustering methods run successfully on the same dataset.
- The app records actual results and settings rather than using placeholder numbers.
- Original feedback can be traced from a cluster to its source record.
- Invalid or inaccessible inputs produce understandable errors rather than crashing the app.
- The experiment includes metrics where valid, manual inspection, runtime, and failure examples.
- A report can be exported.
- Unit tests cover core data and AI functions.
- A team member can explain the data flow, algorithms, results, and limitations.

## 10. Assumptions and Risks
- External websites may block requests, require JavaScript, or change structure.
- Public URL input is untrusted; SSRF protections and bounded requests are required.
- Embedding-model download and inference may be slow on some machines.
- Small or poorly varied datasets may not support meaningful clustering.
- Sentiment and applicability estimates can be wrong.
- Free API quotas and terms can change; the core pipeline must work without them.

## 11. Acceptance Criteria
The P0 scope is accepted only when the app runs locally, the core sample workflow works end-to-end, both methods produce real results, results are reproducible enough to compare, source evidence is traceable, errors are handled, and documentation/test commands are verified.
