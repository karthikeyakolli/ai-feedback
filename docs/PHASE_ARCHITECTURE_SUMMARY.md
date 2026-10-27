# Phase-Wise Architecture Summary

| Phase | Main outcome | Main modules | Exit gate |
|---|---|---|---|
| 0. Environment | Runnable shell and pinned dependencies | `app.py`, config, requirements | Clean install and app launch |
| 1. AI experiment | Two real clustering methods and evaluation | `src/ai/*`, `experiments/*` | Same dataset, real metrics, tests |
| 2. UI MVP | Upload/paste/sample workflow in Streamlit | `pages/*` | Complete browser-based demo |
| 3. Source ingestion | Product URL and provenance | `src/ingestion/*` | Preview and graceful source failures |
| 4. Persistence | Projects and analysis history | `src/storage/*`, SQLite | Data survives restart |
| 5. Decision support | Evidence, applicability and recommendations | `src/ai/applicability.py`, UI pages | Every suggestion is evidence-linked |
| 6. Delivery | Export, tests, report and poster | `src/reporting/*`, `docs/*` | Clean run, actual results and documentation |

**Order rule:** Do not start advanced connectors or optional sentiment before Phase 1 is working. If time is limited, prioritise the genuine experiment over UI polish.
