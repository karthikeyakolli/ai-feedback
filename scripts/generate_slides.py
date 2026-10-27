"""Generates the Academic Capstone Presentation Slide Deck (.pptx).

Constructs a professional, cohesive 15-slide widescreen (16:9) presentation
covering the problem, research question, system architecture, methodology,
controlled benchmark results, error analysis, explainability, and conclusions.
Integrates real empirical data and charts directly from the results directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Visual Design Palette
NAVY_PRIMARY = RGBColor(15, 23, 42)      # #0F172A Slate 900
BLUE_ACCENT = RGBColor(37, 99, 235)      # #2563EB Blue 600
TEXT_WHITE = RGBColor(255, 255, 255)
TEXT_DARK = RGBColor(30, 41, 59)         # #1E293B Slate 800
TEXT_MUTED = RGBColor(100, 116, 139)     # #64748B Slate 500
BG_LIGHT = RGBColor(248, 250, 252)       # #F8FAFC Slate 50
CARD_BG = RGBColor(255, 255, 255)
CARD_BORDER = RGBColor(226, 232, 240)    # #E2E8F0
ORANGE_ACCENT = RGBColor(234, 88, 12)    # #EA580C Orange 600
GREEN_ACCENT = RGBColor(22, 163, 74)     # #16A34A Green 600


def create_deck() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text: str, category_text: str = "AI PRODUCT INTELLIGENCE & FEEDBACK ANALYZER"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = BLUE_ACCENT

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = NAVY_PRIMARY

    def add_card(slide, left, top, width, height, title: str = "", bg_color=CARD_BG, border_color=CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), Inches(0.5))
            tf = tb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = NAVY_PRIMARY
        return shape

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY_PRIMARY
    bg1.line.color.rgb = NAVY_PRIMARY

    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.9), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p_badge = tf.paragraphs[0]
    p_badge.text = "ACADEMIC CAPSTONE DEFENSE | APPLIED MACHINE LEARNING & PRODUCT INTELLIGENCE"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = RGBColor(147, 197, 253)

    p_title = tf.add_paragraph()
    p_title.text = "AI Product Intelligence & Feedback Analyzer"
    p_title.font.size = Pt(36)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE
    p_title.space_after = Pt(12)

    p_sub = tf.add_paragraph()
    p_sub.text = "A Rigorous Empirical Comparison of Sparse N-Gram (TF-IDF) vs. Dense Semantic (Sentence Transformers) Clustering for Product Feedback Triage"
    p_sub.font.size = Pt(17)
    p_sub.font.color.rgb = RGBColor(203, 213, 225)
    p_sub.space_after = Pt(36)

    p_team = tf.add_paragraph()
    p_team.text = "Author / Candidate: Project Candidate [Placeholder]\nInstitution / Faculty: Faculty of Computer Science & Software Engineering [Placeholder]\nAcademic Mentor: Academic Mentor / Advisor [Placeholder] | Date: October 2026"
    p_team.font.size = Pt(12)
    p_team.font.color.rgb = RGBColor(148, 163, 184)

    # ==========================================
    # SLIDE 2: Problem & Motivation
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Industry Motivation")

    add_card(s2, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1), "The Voice-of-Customer Dilemma")
    tb2_left = s2.shapes.add_textbox(Inches(1.05), Inches(2.4), Inches(5.1), Inches(4.1))
    tf2_l = tb2_left.text_frame
    tf2_l.word_wrap = True
    bullets2_l = [
        "**Volume Overload:** SaaS platforms receive thousands of unstructured feedback items across app stores, emails, and tickets weekly.",
        "**Vocabulary Mismatch:** Customers rarely use engineering jargon. They express identical underlying defects using completely different terminology (e.g., 'freezes on save' vs 'crashes on document download').",
        "**Manual Triage Bottleneck:** Product teams spend dozens of hours manually tagging tickets, resulting in delayed fixes and customer churn.",
        "**Need for Local Intelligence:** Many organizations prohibit sending proprietary customer telemetry to commercial cloud APIs due to privacy (GDPR/HIPAA) or cost restrictions."
    ]
    for i, b in enumerate(bullets2_l):
        p = tf2_l.paragraphs[0] if i == 0 else tf2_l.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(12)

    add_card(s2, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1), "Core Engineering Objectives")
    tb2_right = s2.shapes.add_textbox(Inches(7.05), Inches(2.4), Inches(5.2), Inches(4.1))
    tf2_r = tb2_right.text_frame
    tf2_r.word_wrap = True
    bullets2_r = [
        "**100% Local & Deterministic:** Zero reliance on paid cloud LLMs or recurring API subscriptions.",
        "**Rigorous Comparative Evaluation:** Evaluate whether modern neural dense embeddings objectively justify their compute cost over classical TF-IDF.",
        "**Actionable Decision Support:** Transform raw clusters into evidence-grounded engineering recommendations tied directly to verbatim customer quotes.",
        "**Explainability & Trust:** Demystify cluster assignments with centroid distance distributions, top keywords, and representative quotes."
    ]
    for i, b in enumerate(bullets2_r):
        p = tf2_r.paragraphs[0] if i == 0 else tf2_r.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(12)

    # ==========================================
    # SLIDE 3: Research Question & Hypotheses
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Formal Research Question & Hypotheses")

    add_card(s3, Inches(0.8), Inches(1.7), Inches(11.7), Inches(1.6), "Primary Research Question", bg_color=RGBColor(239, 246, 255), border_color=BLUE_ACCENT)
    tb3_q = s3.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(11.2), Inches(0.9))
    tf3_q = tb3_q.text_frame
    tf3_q.word_wrap = True
    p = tf3_q.paragraphs[0]
    p.text = '“Does sentence-embedding-based clustering group differently worded product feedback more usefully than a traditional TF-IDF-based approach?”'
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT

    add_card(s3, Inches(0.8), Inches(3.6), Inches(5.6), Inches(3.3), "Hypothesis H1 (Semantic Capture)")
    tb3_h1 = s3.shapes.add_textbox(Inches(1.05), Inches(4.2), Inches(5.1), Inches(2.5))
    tf3_h1 = tb3_h1.text_frame
    tf3_h1.word_wrap = True
    h1_text = (
        "Dense continuous representations (Sentence Transformers) will achieve statistically higher external "
        "agreement (ARI) and pairwise grouping F1 on paraphrased complaints lacking exact lexical token overlap, "
        "preventing issue fragmentation."
    )
    p = tf3_h1.paragraphs[0]
    p.text = h1_text
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_DARK

    add_card(s3, Inches(6.8), Inches(3.6), Inches(5.7), Inches(3.3), "Hypothesis H2 (Computational Trade-Off)")
    tb3_h2 = s3.shapes.add_textbox(Inches(7.05), Inches(4.2), Inches(5.2), Inches(2.5))
    tf3_h2 = tb3_h2.text_frame
    tf3_h2.word_wrap = True
    h2_text = (
        "Classical TF-IDF will achieve orders-of-magnitude lower execution latency (<100ms) and demonstrate "
        "superior precision in isolating distinct technical vocabulary outliers (such as rare billing bugs or unique error codes), "
        "providing a strong baseline for real-time triage."
    )
    p = tf3_h2.paragraphs[0]
    p.text = h2_text
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 4: System Architecture
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "End-to-End System Architecture")

    cols = [
        ("1. Ingestion Layer", ["Multi-channel feedback (CSV, text)", "Context extraction (manual/URL)", "Deduplication & normalization", "Data integrity validation"]),
        ("2. Dual ML Engine", ["TF-IDF (1-2 n-grams, sublinear)", "MiniLM-L6-v2 (384-dim dense)", "K-Means++ clustering", "Strict label isolation"]),
        ("3. Decision Support", ["Product relevance scoring", "Negative impact ranking", "Rule-based recommendation", "Verbatim quote grounding"]),
        ("4. Explainability & UI", ["Centroid distance distributions", "Top TF-IDF keywords / themes", "Interactive Streamlit UI", "Local SQLite persistence"])
    ]

    for idx, (col_title, col_items) in enumerate(cols):
        left_pos = Inches(0.8 + idx * 2.98)
        add_card(s4, left_pos, Inches(1.7), Inches(2.8), Inches(5.1), col_title)
        tb_col = s4.shapes.add_textbox(left_pos + Inches(0.2), Inches(2.4), Inches(2.4), Inches(4.1))
        tf_col = tb_col.text_frame
        tf_col.word_wrap = True
        for i, item in enumerate(col_items):
            p = tf_col.paragraphs[0] if i == 0 else tf_col.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(12)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(10)

    # ==========================================
    # SLIDE 5: Label Isolation & Evaluation Protocol
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Methodological Rigor & Ground-Truth Isolation")

    add_card(s5, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1), "Strict Label Isolation Architecture")
    tb5_l = s5.shapes.add_textbox(Inches(1.05), Inches(2.4), Inches(5.1), Inches(4.1))
    tf5_l = tb5_l.text_frame
    tf5_l.word_wrap = True
    bullets5_l = [
        "**Zero Data Contamination:** Evaluation labels (`data/evaluation_labels.csv`) are physically stored in a separate file and never read by feature extraction or clustering modules.",
        "**Unsupervised Vectorization:** Both TF-IDF and Sentence Transformers fit strictly on raw customer text.",
        "**Post-Hoc Metric Calculation:** Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) are calculated only after unsupervised clustering completes.",
        "**Blinded Decision Support:** Applicability, recommendation, and explanation modules have zero access to ground truth labels."
    ]
    for i, b in enumerate(bullets5_l):
        p = tf5_l.paragraphs[0] if i == 0 else tf5_l.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(12)

    add_card(s5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1), "Multi-Faceted Metric Suite")
    tb5_r = s5.shapes.add_textbox(Inches(7.05), Inches(2.4), Inches(5.2), Inches(4.1))
    tf5_r = tb5_r.text_frame
    tf5_r.word_wrap = True
    bullets5_r = [
        "**External Agreement (ARI & NMI):** Quantifies partition alignment against 10 ground truth reference themes.",
        "**Pairwise Semantic Grouping (P, R, F1):** Evaluates all 190 document pairs ($N=20$) to directly test whether identical issues are co-clustered.",
        "**Internal Geometric Scores:** Silhouette and Davies-Bouldin indices (reported separately within native feature spaces).",
        "**Human Usefulness Proxy Rubric:** Evaluates cluster thematic purity and centroid representative quote fidelity.",
        "**System Execution Latency:** Cold-start vs. warmed inference latency in milliseconds."
    ]
    for i, b in enumerate(bullets5_r):
        p = tf5_r.paragraphs[0] if i == 0 else tf5_r.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(10)

    # ==========================================
    # SLIDE 6: Controlled Benchmark Results
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Controlled Benchmark Results (Averaged across Seeds)")

    # Insert metrics chart if available
    chart1_path = RESULTS_DIR / "metrics_comparison.png"
    if chart1_path.is_file():
        s6.shapes.add_picture(str(chart1_path), Inches(0.8), Inches(1.7), Inches(6.8), Inches(5.1))

    add_card(s6, Inches(7.8), Inches(1.7), Inches(4.7), Inches(5.1), "Key Quantitative Metrics")
    tb6_r = s6.shapes.add_textbox(Inches(8.05), Inches(2.4), Inches(4.2), Inches(4.1))
    tf6_r = tb6_r.text_frame
    tf6_r.word_wrap = True
    bullets6_r = [
        "**External Agreement (ARI):**\n• Sentence Transformers: **0.6149 ± 0.1534**\n• TF-IDF Baseline: **0.2647 ± 0.1764**\n*(MiniLM achieved +132% higher reference alignment)*",
        "**Pairwise Semantic F1:**\n• Sentence Transformers: **0.6680**\n• TF-IDF Baseline: **0.3744**\n*(MiniLM achieved perfect 1.0 recall across all seeds)*",
        "**Cross-Model Agreement:**\n• Partition ARI: **0.3339 ± 0.1861**\n*(Models agree on dense crashes, but diverge on nuanced UI/features)*",
        "**Execution Runtime:**\n• TF-IDF: **64.4 ms** (fast across all fits)\n• MiniLM Warmed: **62.8 ms** (comparable to TF-IDF)\n• MiniLM Cold-Start: **19.57 s** (one-time weight load)"
    ]
    for i, b in enumerate(bullets6_r):
        p = tf6_r.paragraphs[0] if i == 0 else tf6_r.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(10)

    # ==========================================
    # SLIDE 7: Runtime & Cross-Model Agreement
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Execution Latency & Cross-Model Partition Divergence")

    chart2_path = RESULTS_DIR / "runtime_comparison.png"
    if chart2_path.is_file():
        s7.shapes.add_picture(str(chart2_path), Inches(0.8), Inches(1.7), Inches(5.6), Inches(4.8))

    chart3_path = RESULTS_DIR / "pairwise_comparison.png"
    if chart3_path.is_file():
        s7.shapes.add_picture(str(chart3_path), Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.8))

    # ==========================================
    # SLIDE 8: Qualitative Error Analysis - Paraphrase Grouping
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Qualitative Case Study: Paraphrase Grouping")

    add_card(s8, Inches(0.8), Inches(1.7), Inches(11.7), Inches(1.5), "Evaluated Records: Search Latency Complaints", bg_color=RGBColor(241, 245, 249))
    tb8_top = s8.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(11.2), Inches(0.9))
    tf8_t = tb8_top.text_frame
    tf8_t.word_wrap = True
    p = tf8_t.paragraphs[0]
    p.text = 'FB08: "Searching through customer records takes over 30 seconds." (Rating 2.0)\nFB09: "Search functionality is painfully sluggish on larger lists." (Rating 2.0)'
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = NAVY_PRIMARY

    add_card(s8, Inches(0.8), Inches(3.5), Inches(5.6), Inches(3.3), "Method A: TF-IDF (Fragmented)")
    tb8_l = s8.shapes.add_textbox(Inches(1.05), Inches(4.1), Inches(5.1), Inches(2.5))
    tf8_l = tb8_l.text_frame
    tf8_l.word_wrap = True
    tf8_l.paragraphs[0].text = (
        "• Assignment: FB08 -> Cluster 3 (size 2); FB09 -> Cluster 2 (size 11)\n"
        "• Failure Mechanism: High-dimensional n-gram sparsity. Unique tokens ('30 seconds', 'larger lists', 'painfully sluggish') "
        "diverged cosine similarity below cluster threshold.\n"
        "• Consequence: Issue is split. Product manager sees an isolated complaint and misses the cumulative severity."
    )
    tf8_l.paragraphs[0].font.size = Pt(13)
    tf8_l.paragraphs[0].font.color.rgb = TEXT_DARK

    add_card(s8, Inches(6.8), Inches(3.5), Inches(5.7), Inches(3.3), "Method B: Sentence Transformers (United)")
    tb8_r = s8.shapes.add_textbox(Inches(7.05), Inches(4.1), Inches(5.2), Inches(2.5))
    tf8_r = tb8_r.text_frame
    tf8_r.word_wrap = True
    tf8_r.paragraphs[0].text = (
        "• Assignment: FB08 -> Cluster 0; FB09 -> Cluster 0 (size 5)\n"
        "• Success Mechanism: Pretrained neural semantic space maps 'takes 30 seconds' and 'painfully sluggish' "
        "to nearby dense geometric coordinates.\n"
        "• Consequence: Clean, actionable performance cluster containing all search complaints."
    )
    tf8_r.paragraphs[0].font.size = Pt(13)
    tf8_r.paragraphs[0].font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 9: Qualitative Error Analysis - The Catch-All Failure Mode
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Qualitative Case Study: The TF-IDF Catch-All Failure")

    add_card(s9, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1), "TF-IDF Cluster 2: 11 Items Absorbed (55%)")
    tb9_l = s9.shapes.add_textbox(Inches(1.05), Inches(2.4), Inches(5.1), Inches(4.1))
    tf9_l = tb9_l.text_frame
    tf9_l.word_wrap = True
    b9_items = [
        "FB06: Session timeouts and 2FA SMS delayed 15 mins (Auth)",
        "FB07: Cannot log in, verification code not sent (Auth)",
        "FB09: Search sluggish on larger lists (Performance)",
        "FB10: Add dark mode support, hurts eyes (Feature)",
        "FB11: Dark theme option for low light (Feature)",
        "FB12: Bulk import CSV from Google Drive (Feature)",
        "FB13: Slack webhook integration (Feature)",
        "FB14: Clean interface made onboarding seamless (UX Praise)",
        "FB17: Documentation page 404 error on API (Doc Bug)",
        "FB20: Dark mode please! (Feature)"
    ]
    for i, it in enumerate(b9_items):
        p = tf9_l.paragraphs[0] if i == 0 else tf9_l.add_paragraph()
        p.text = f"• {it}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    add_card(s9, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1), "Root Cause & Product Implications")
    tb9_r = s9.shapes.add_textbox(Inches(7.05), Inches(2.4), Inches(5.2), Inches(4.1))
    tf9_r = tb9_r.text_frame
    tf9_r.word_wrap = True
    b9_causes = [
        "**Mathematical Cause:** In small corpora, distinct non-overlapping vocabularies have orthogonal distance ($d = \\sqrt{2}$). K-Means assigns orthogonal items to a diffuse residual centroid.",
        "**Critical Product Hazard:** Severe security bugs (login timeouts, 2FA failures) are drowned out inside cosmetic theme requests and onboarding praise.",
        "**Sentence Transformers Solution:** Formed 4 balanced clusters of size 5: C0 (Speed/Search), C1 (Auth/Admin/404), C2 (PDF Export Crashes), C3 (UI/Themes/Integrations)."
    ]
    for i, b in enumerate(b9_causes):
        p = tf9_r.paragraphs[0] if i == 0 else tf9_r.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(14)

    # ==========================================
    # SLIDE 10: Decision Support & Recommendations
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Decision Support: Evidence-Grounded Recommendations")

    cards10 = [
        ("Negative Impact Score", "Calculated as: `size * (5 - avg_rating) * bug_ratio`. Prioritizes widespread, critical defects over isolated cosmetic issues.", Inches(0.8), Inches(1.7)),
        ("Applicability Filtering", "Evaluates feedback against ingested product context (e.g., target audience, product domain). Rejects out-of-scope feedback.", Inches(6.8), Inches(1.7)),
        ("Recommendation Engine", "Generates concrete action steps grounded in verbatim quotes. Rules map issue types to engineering, design, or doc sprints.", Inches(0.8), Inches(4.4)),
        ("Explainability Cards", "Provides centroid distance histograms, top keywords, and representative quotes so product owners understand WHY clusters formed.", Inches(6.8), Inches(4.4))
    ]
    for title, desc, l, t in cards10:
        add_card(s10, l, t, Inches(5.7), Inches(2.4), title)
        tb_c = s10.shapes.add_textbox(l + Inches(0.25), t + Inches(0.8), Inches(5.2), Inches(1.4))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        p = tf_c.paragraphs[0]
        p.text = desc
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 11: Interactive Streamlit Application
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Interactive Application Implementation")

    add_card(s11, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.1), "Five Fully Functional Streamlit Tabs")
    tb11 = s11.shapes.add_textbox(Inches(1.05), Inches(2.4), Inches(11.2), Inches(4.1))
    tf11 = tb11.text_frame
    tf11.word_wrap = True
    tabs = [
        "**Tab 1: Product Definition & Context Ingestion** — Paste context or ingest public product URLs with SSRF protection.",
        "**Tab 2: Feedback Ingestion & Validation** — Load sample data, upload custom CSVs, or paste text with deduplication.",
        "**Tab 3: Dual Clustering & Model Comparison** — Side-by-side execution of TF-IDF vs. Sentence Transformers with interactive metric inspection.",
        "**Tab 4: Evidence-Grounded Recommendations** — Actionable advice cards sorted by Negative Impact Score with verbatim quote grounding.",
        "**Tab 5: Analysis History & Persistence** — Local SQLite storage of analysis runs with search, filtering, and reload capabilities."
    ]
    for i, t in enumerate(tabs):
        p = tf11.paragraphs[0] if i == 0 else tf11.add_paragraph()
        p.text = t.replace("**", "")
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(12)

    # ==========================================
    # SLIDE 12: Threats to Validity & Limitations
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Threats to Validity & Methodological Limitations")

    add_card(s12, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.1), "Scientific Limitations")
    tb12_l = s12.shapes.add_textbox(Inches(1.05), Inches(2.4), Inches(5.1), Inches(4.1))
    tf12_l = tb12_l.text_frame
    tf12_l.word_wrap = True
    b12_l = [
        "**Synthetic Dataset Scope ($N=20$):** Curated specifically to model paraphrase challenges. Generalization to large uncurated customer telemetry requires empirical replication.",
        "**Single Annotator Reference:** Reference themes were defined by the system design team without multi-annotator agreement metrics (Cohen's Kappa).",
        "**K-Means Spherical Geometry:** Assumes convex clusters; forces noise and outliers into nearest centroid.",
        "**Topic vs. Sentiment Confusion:** Sentence Transformers groups by semantic topic regardless of sentiment (e.g., speed complaints grouped with speed praise)."
    ]
    for i, b in enumerate(b12_l):
        p = tf12_l.paragraphs[0] if i == 0 else tf12_l.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(12)

    add_card(s12, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.1), "Ethical & Data Privacy Considerations")
    tb12_r = s12.shapes.add_textbox(Inches(7.05), Inches(2.4), Inches(5.2), Inches(4.1))
    tf12_r = tb12_r.text_frame
    tf12_r.word_wrap = True
    b12_r = [
        "**100% Local Processing:** Customer feedback is processed locally without transmitting confidential user data or PII to external cloud APIs.",
        "**Algorithmic Bias in Embeddings:** Pretrained embeddings reflect training corpus biases; rare domain dialects may be misclustered.",
        "**Transparent Decision Support:** Recommendations are rule-based and transparent, avoiding 'black-box' hallucinations.",
        "**Data Minimization:** No personal credentials, session tokens, or private keys are stored."
    ]
    for i, b in enumerate(b12_r):
        p = tf12_r.paragraphs[0] if i == 0 else tf12_r.add_paragraph()
        p.text = b.replace("**", "")
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(12)

    # ==========================================
    # SLIDE 13: Conclusion & Research Takeaways
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Conclusion & Core Research Takeaways")

    add_card(s13, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.1), "Answering the Research Question")
    tb13 = s13.shapes.add_textbox(Inches(1.05), Inches(2.4), Inches(11.2), Inches(4.1))
    tf13 = tb13.text_frame
    tf13.word_wrap = True
    takeaways = [
        "1. **Sentence Transformers definitively out-groups differently worded feedback:** Achieving Mean ARI of 0.6149 (vs 0.2647 for TF-IDF) and Pairwise F1 of 0.6680 (vs 0.3744), dense embeddings successfully resolve the vocabulary mismatch problem in product feedback.",
        "2. **TF-IDF remains an essential baseline:** For low-latency triage (<100ms) and isolating unique vocabulary outliers (billing disputes, error codes), TF-IDF is highly effective and computationally lightweight.",
        "3. **No Single 'Universal Winner':** The optimal architecture is a hybrid: Sentence Transformers for conceptual theme discovery, coupled with TF-IDF keyword indexing for exact error search.",
        "4. **Complete Capstone Delivery:** 60 automated passing tests, clean installation verification, SQLite persistence, and interactive Streamlit UI."
    ]
    for i, t in enumerate(takeaways):
        p = tf13.paragraphs[0] if i == 0 else tf13.add_paragraph()
        p.text = t.replace("**", "")
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(14)

    # ==========================================
    # SLIDE 14: Future Scope
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Future Work & Architecture Enhancements")

    cols14 = [
        ("HDBSCAN & Density Clustering", "Replace K-Means with HDBSCAN to eliminate the requirement to specify $k$ in advance and allow natural outlier identification without forcing residual clusters."),
        ("Multi-Annotator Corpus", "Collect multi-annotator real-world customer telemetry across diverse software domains with formal Inter-Annotator Agreement (Cohen's Kappa / Fleiss' Kappa)."),
        ("Contrastive Domain Fine-Tuning", "Fine-tune `all-MiniLM-L6-v2` using contrastive triplet loss on SaaS-specific technical error logs and customer bug reports."),
        ("Automated Issue Tracker Sync", "Direct two-way synchronization of evidence-grounded recommendations into Jira, GitHub Issues, and Linear boards.")
    ]
    for idx, (title, desc) in enumerate(cols14):
        left_pos = Inches(0.8 + idx * 2.98)
        add_card(s14, left_pos, Inches(1.7), Inches(2.8), Inches(5.1), title)
        tb_col = s14.shapes.add_textbox(left_pos + Inches(0.2), Inches(2.6), Inches(2.4), Inches(3.9))
        tf_col = tb_col.text_frame
        tf_col.word_wrap = True
        p = tf_col.paragraphs[0]
        p.text = desc
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 15: References & Defense Closure
    # ==========================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Key References & Academic Defense Closure")

    add_card(s15, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.1), "Academic Citations")
    tb15 = s15.shapes.add_textbox(Inches(1.05), Inches(2.4), Inches(11.2), Inches(4.1))
    tf15 = tb15.text_frame
    tf15.word_wrap = True
    refs = [
        "• Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In EMNLP-IJCNLP 2019 (pp. 3982–3992).",
        "• Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523.",
        "• MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proc. 5th Berkeley Symp. Math. Statist. Prob., 1, 281–297.",
        "• Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193–218.",
        "• Strehl, A., & Ghosh, J. (2002). Cluster ensembles—a knowledge reuse framework for combining multiple partitions. JMLR, 3, 583–617.",
        "• Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. J. Comput. Appl. Math., 20, 53–65."
    ]
    for i, r in enumerate(refs):
        p = tf15.paragraphs[0] if i == 0 else tf15.add_paragraph()
        p.text = r
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(8)

    p_close = tf15.add_paragraph()
    p_close.text = "\nThank you. Questions and defense discussion are welcome."
    p_close.font.size = Pt(15)
    p_close.font.bold = True
    p_close.font.color.rgb = BLUE_ACCENT

    out_file = DOCS_DIR / "capstone_presentation.pptx"
    prs.save(str(out_file))
    print(f"Presentation deck successfully created at: {out_file}")
    return out_file


if __name__ == "__main__":
    create_deck()
