"""A5 Academic Poster Generator for Phase 6 Capstone Delivery.

Renders a publication-ready single-page A5 poster (148 mm x 210 mm = 5.827 x 8.268 inches)
using Matplotlib and Pillow. Generates:
- docs/capstone_poster.pdf (Vector / printable PDF)
- docs/capstone_poster.png (High-resolution 300 DPI raster preview)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def generate_a5_poster() -> tuple[Path, Path]:
    # Exact A5 dimensions: 148 mm x 210 mm = 5.827 x 8.268 inches
    width_in = 5.827
    height_in = 8.268
    dpi = 300

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_bg = "#F8FAFC"        # Slate 50
    c_navy = "#0F172A"      # Slate 900
    c_blue = "#1E40AF"      # Blue 800
    c_accent = "#2563EB"    # Blue 600
    c_card_bg = "#FFFFFF"
    c_card_border = "#CBD5E1"
    c_text_dark = "#1E293B"
    c_text_muted = "#475569"
    c_green = "#166534"
    c_orange = "#C2410C"

    # Background
    bg_rect = patches.Rectangle((0, 0), 100, 100, facecolor=c_bg, edgecolor="none")
    ax.add_patch(bg_rect)

    # 1. Header Banner
    header_rect = patches.Rectangle((0, 88.5), 100, 11.5, facecolor=c_navy, edgecolor="none")
    ax.add_patch(header_rect)

    ax.text(
        50, 96.8, "AI PRODUCT INTELLIGENCE & FEEDBACK ANALYZER",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#93C5FD", family="sans-serif"
    )
    ax.text(
        50, 93.4, "Empirical Evaluation: Sparse N-Gram (TF-IDF) vs. Dense Semantic (MiniLM) Clustering",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#FFFFFF", family="sans-serif"
    )
    ax.text(
        50, 90.3, "Candidate: Project Candidate [Placeholder]  |  Faculty of Computer Science & Software Engineering  |  Advisor: Faculty Mentor [Placeholder]",
        ha="center", va="center", fontsize=6.2, color="#CBD5E1", family="sans-serif"
    )

    # Helper function for section cards
    def draw_card(x, y, w, h, title="", header_color=c_accent):
        card = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=c_card_bg, edgecolor=c_card_border, linewidth=0.8
        )
        ax.add_patch(card)
        if title:
            ax.text(
                x + 1.2, y + h - 2.2, title.upper(),
                fontsize=7.2, fontweight="bold", color=header_color, family="sans-serif"
            )
            # Accent underline
            ax.plot([x + 1.2, x + w - 1.2], [y + h - 3.0, y + h - 3.0], color=c_card_border, linewidth=0.5)

    # ----------------------------------------------------
    # LEFT COLUMN (x: 2.5 to 48.5)
    # ----------------------------------------------------

    # Card 1: Problem & Motivation (y: 69 to 87)
    draw_card(2.5, 69.0, 46.5, 18.2, "1. Problem & Research Question")
    ax.text(
        3.5, 82.5,
        "• Unstructured Voice-of-Customer: SaaS platforms receive\n"
        "  thousands of tickets, reviews, and bug reports weekly.\n"
        "• Vocabulary Mismatch: Customers use colloquial terms\n"
        "  ('freezes on export' vs 'application shutdown during download').\n"
        "• Research Question: Does sentence-embedding-based clustering\n"
        "  group differently worded product feedback more usefully than\n"
        "  a traditional TF-IDF-based approach?",
        fontsize=5.9, color=c_text_dark, linespacing=1.35, family="sans-serif"
    )
    # Highlight box for RQ
    rq_box = patches.FancyBboxPatch(
        (3.5, 69.8), 44.5, 3.8, boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=0.6
    )
    ax.add_patch(rq_box)
    ax.text(
        25.7, 71.7, "Hypothesis: Dense continuous geometry prevents ticket fragmentation\nwithout requiring expensive commercial cloud LLM APIs.",
        ha="center", va="center", fontsize=5.3, fontweight="bold", color=c_blue, family="sans-serif"
    )

    # Card 2: System Architecture & Workflow (y: 44 to 67.5)
    draw_card(2.5, 44.0, 46.5, 23.8, "2. System Architecture & Workflow")
    ax.text(
        3.5, 63.8,
        "A 4-tier modular pipeline with strict evaluation label isolation:\n"
        "1. Ingestion: Multi-channel input, deduplication, URL context extraction.\n"
        "2. Parallel Vectorization: TF-IDF (1-2 n-grams) vs. MiniLM (384-dim).\n"
        "3. K-Means++ Clustering: Partitioning across k in [3, 4, 5, 6].\n"
        "4. Decision Support: Negative Impact Score, rules & quote grounding.",
        fontsize=5.8, color=c_text_dark, linespacing=1.3, family="sans-serif"
    )
    # Workflow mini-diagram boxes
    steps = [("Ingest & Clean", 3.5), ("Dual Vectors", 14.8), ("K-Means++", 26.1), ("Action Cards", 37.4)]
    for name, sx in steps:
        sbox = patches.FancyBboxPatch((sx, 45.2), 9.8, 4.2, boxstyle="round,pad=0.2,rounding_size=0.5",
                                      facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=0.6)
        ax.add_patch(sbox)
        ax.text(sx + 4.9, 47.3, name, ha="center", va="center", fontsize=5.2, fontweight="bold", color=c_navy)

    # Card 3: Methodology & Experimental Controls (y: 20 to 42.8)
    draw_card(2.5, 20.0, 46.5, 22.8, "3. Rigorous Experimental Setup")
    ax.text(
        3.5, 39.0,
        "• Controlled Benchmark: Identical feedback records (N=20)\n"
        "  evaluated across 12 configurations (k in [3..6], seeds: 42, 100, 2024).\n"
        "• Strict Label Isolation: 'evaluation_labels.csv' kept physically\n"
        "  separate; zero label leakage into vectorizers or clustering.\n"
        "• Multi-Metric Harness:\n"
        "  - External Agreement: Adjusted Rand Index (ARI) & NMI vs. 10 themes.\n"
        "  - Pairwise Grouping: Precision, Recall, F1 over 190 record pairs.\n"
        "  - Computational Latency: Sub-millisecond timing on CPU.\n"
        "• Clean Install Verification: 10 automated health checks passed.",
        fontsize=5.8, color=c_text_dark, linespacing=1.3, family="sans-serif"
    )

    # Card 4: Decision Support & Explainability (y: 1.5 to 18.8)
    draw_card(2.5, 1.5, 46.5, 17.3, "4. Evidence-Grounded Decision Support")
    ax.text(
        3.5, 15.0,
        "• Negative Impact Score = Size * (5 - Avg Rating) * Bug Ratio.\n"
        "  Prioritizes widespread technical regressions over cosmetic feedback.\n"
        "• Traceable Explainability: Centroid distance distributions and\n"
        "  verbatim representative quotes anchor every recommended action.\n"
        "• Persistence: Fully local SQLite database for history & auditing.",
        fontsize=5.7, color=c_text_dark, linespacing=1.3, family="sans-serif"
    )

    # ----------------------------------------------------
    # RIGHT COLUMN (x: 51.0 to 97.5)
    # ----------------------------------------------------

    # Card 5: Empirical Benchmark Results (y: 53 to 87)
    draw_card(51.0, 53.0, 46.5, 34.2, "5. Controlled Experimental Results")
    ax.text(
        52.0, 83.5,
        "Averaged across 12 runs (Seeds: 42, 100, 2024 | k in [3, 4, 5, 6]):",
        fontsize=5.8, fontweight="bold", color=c_text_dark, family="sans-serif"
    )

    # Table of Results
    headers = ["Model", "Mean ARI", "Pairwise F1", "Latency"]
    rows = [
        ["TF-IDF + K-Means", "0.2647", "0.3744", "64.4 ms"],
        ["Sentence Transformers", "0.6149", "0.6680", "62.8 ms*"]
    ]
    ty = 78.5
    for c_idx, h in enumerate(headers):
        ax.text(52.5 + c_idx * 11.0, ty, h, fontsize=5.6, fontweight="bold", color=c_navy)
    ax.plot([52.0, 96.0], [ty - 1.2, ty - 1.2], color="#94A3B8", linewidth=0.6)

    for r_idx, row in enumerate(rows):
        r_y = ty - 4.0 - (r_idx * 3.4)
        bg_col = "#EFF6FF" if r_idx == 1 else "#FFFFFF"
        tbox = patches.Rectangle((51.8, r_y - 1.0), 44.8, 3.2, facecolor=bg_col, edgecolor="none")
        ax.add_patch(tbox)
        for c_idx, val in enumerate(row):
            weight = "bold" if r_idx == 1 else "normal"
            col = c_blue if r_idx == 1 else c_text_dark
            ax.text(52.5 + c_idx * 11.0, r_y, val, fontsize=5.5, fontweight=weight, color=col)

    ax.text(
        52.0, 67.5,
        "*Note: MiniLM warmed inference is 62.8 ms (initial cold-start: 19.6s).\n"
        "Key Finding: Sentence Transformers achieved +132% higher ARI and\n"
        "perfect 1.0 Pairwise Recall (19/19 pairs), uniting differently worded items.",
        fontsize=5.3, color=c_text_muted, linespacing=1.2, family="sans-serif"
    )

    # Mini Chart Inset (from results/metrics_comparison.png)
    chart_path = RESULTS_DIR / "metrics_comparison.png"
    if chart_path.is_file():
        img = Image.open(chart_path)
        # Add image to axis
        ax_inset = fig.add_axes([0.53, 0.54, 0.43, 0.12])
        ax_inset.imshow(img)
        ax_inset.axis("off")

    # Card 6: Qualitative Case Study & Error Analysis (y: 28 to 51.8)
    draw_card(51.0, 28.0, 46.5, 23.8, "6. Qualitative Error Analysis")
    ax.text(
        52.0, 48.0,
        "Paraphrase Grouping Case Study (Search Latency):\n"
        "• FB08: 'Searching customer records takes over 30 seconds.'\n"
        "• FB09: 'Search functionality is painfully sluggish on larger lists.'\n"
        "  - TF-IDF Result: Split into Cluster 3 & Cluster 2 (catch-all).\n"
        "  - MiniLM Result: Correctly united into Cluster 0 (purity 1.0).\n\n"
        "The TF-IDF Catch-All Problem:\n"
        "• At k=4, TF-IDF Cluster 2 absorbed 11/20 records (55%), mixing\n"
        "  auth timeouts, 404 bugs, and dark mode due to vector orthogonality.",
        fontsize=5.6, color=c_text_dark, linespacing=1.25, family="sans-serif"
    )

    # Card 7: Threats to Validity & Conclusion (y: 1.5 to 26.8)
    draw_card(51.0, 1.5, 46.5, 25.3, "7. Threats to Validity & Conclusion")
    ax.text(
        52.0, 23.0,
        "• Threats to Validity:\n"
        "  - Synthetic dataset (N=20) designed for paraphrase evaluation;\n"
        "    empirical replication on large customer telemetry is needed.\n"
        "  - MiniLM groups by topic, not sentiment (speed praise + bug).\n"
        "• Conclusion:\n"
        "  - Dense embeddings definitively resolve vocabulary mismatch.\n"
        "  - TF-IDF remains an essential baseline for sub-100ms triage\n"
        "    and isolating rare keyword outliers (billing issues).\n"
        "  - Optimal enterprise architecture is hybrid.",
        fontsize=5.5, color=c_text_dark, linespacing=1.25, family="sans-serif"
    )
    # References
    ax.text(
        52.0, 3.8,
        "Refs: Reimers & Gurevych (EMNLP 2019); Salton & Buckley (1988);\n"
        "MacQueen (1967); Hubert & Arabie (1985). Zero paid cloud APIs.",
        fontsize=4.9, color=c_text_muted, family="sans-serif"
    )

    # Save PDF and PNG
    pdf_out = DOCS_DIR / "capstone_poster.pdf"
    png_out = DOCS_DIR / "capstone_poster.png"

    plt.savefig(pdf_out, format="pdf", dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.savefig(png_out, format="png", dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close()

    print(f"Poster successfully generated:\n- PDF: {pdf_out}\n- PNG: {png_out}")
    return pdf_out, png_out


if __name__ == "__main__":
    generate_a5_poster()
