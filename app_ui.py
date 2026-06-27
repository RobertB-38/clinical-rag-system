"""Gradio demo UI — the deployment surface for Hugging Face Spaces.

Glassmorphism styling in light medical/clinical tones. Calls the RAG pipeline
in-process (no separate API server needed on Spaces).
Run locally:  python app_ui.py   ->  http://127.0.0.1:7860
On Spaces, this file is the entry point.
"""
from __future__ import annotations

import gradio as gr

from app.models import DISCLAIMER
from app.rag.pipeline import RagPipeline

_pipeline: RagPipeline | None = None


def pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


def answer_question(question: str):
    if not question or len(question.strip()) < 3:
        return "Please enter a question (at least 3 characters).", ""
    result = pipeline().answer(question)
    sources = "\n".join(
        f"- **{h.source_title}**  ·  relevance {h.score:.2f}"
        for h in result.contexts
    ) or "_No passages retrieved._"
    tag = "> ⚠️ **Refused — insufficient evidence in the indexed guidelines.**\n\n" if result.refused else ""
    return tag + result.answer, sources


CSS = """
:root {
  --sf: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  --ink: #0b1f2a;
  --muted: #4a5b68;
  --accent: #0d8f86;
}
* { font-family: var(--sf) !important; }

/* Saturated medical backdrop so the frosted glass clearly floats on it */
.gradio-container {
  background:
    radial-gradient(720px 520px at 10% 6%, rgba(45,212,191,0.85) 0%, rgba(45,212,191,0) 60%),
    radial-gradient(760px 560px at 92% 2%, rgba(56,189,248,0.80) 0%, rgba(56,189,248,0) 58%),
    radial-gradient(760px 620px at 78% 102%, rgba(99,102,241,0.55) 0%, rgba(99,102,241,0) 60%),
    radial-gradient(680px 560px at 18% 108%, rgba(16,185,129,0.55) 0%, rgba(16,185,129,0) 60%),
    linear-gradient(155deg, #d2f3ff 0%, #cdf6ec 52%, #e7e9ff 100%) !important;
  background-attachment: fixed !important;
  min-height: 100vh;
}
.gradio-container > .main, .gradio-container .contain,
.gradio-container .wrap, .gradio-container .app { max-width: 860px !important; margin: 0 auto !important; }
.app.gradio-container { padding-top: 28px !important; }

/* Frosted glass panels — higher opacity + crisp edge so they read as glass */
.glass {
  background: rgba(255, 255, 255, 0.58) !important;
  backdrop-filter: blur(26px) saturate(180%);
  -webkit-backdrop-filter: blur(26px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.85) !important;
  border-radius: 24px !important;
  box-shadow: 0 18px 50px rgba(8, 47, 73, 0.18), inset 0 1px 0 rgba(255,255,255,0.6) !important;
  padding: 26px 28px !important;
}

/* Hero — Apple display typography */
#hero { text-align: center; }
#hero h1 {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 46px !important;
  line-height: 1.05 !important;
  letter-spacing: -0.025em !important;
  margin-bottom: 10px !important;
}
#hero p { color: var(--muted) !important; font-size: 19px !important; line-height: 1.5 !important; }
#hero strong { color: var(--ink) !important; font-weight: 600 !important; }
#hero em { color: #5a6b78 !important; font-size: 13px !important; }

/* Warning banner */
#warn {
  background: rgba(255, 251, 235, 0.72) !important;
  border: 1px solid rgba(217, 119, 6, 0.35) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(12px);
  padding: 14px 18px !important;
}
#warn p { color: #92400e !important; margin: 0 !important; font-size: 15px !important; line-height: 1.45; }
#warn strong { color: #7c2d12 !important; }

/* Inputs */
#ask-box textarea, #ask-box input[type=text] {
  background: rgba(255, 255, 255, 0.92) !important;
  border: 1px solid rgba(11, 31, 42, 0.12) !important;
  border-radius: 14px !important;
  color: var(--ink) !important;
  font-size: 16px !important;
}
#ask-box textarea:focus { box-shadow: 0 0 0 4px rgba(13, 143, 134, 0.18) !important; border-color: var(--accent) !important; }

/* Apple-style pill button */
#ask-btn {
  background: var(--accent) !important;
  border: none !important; color: #ffffff !important;
  border-radius: 980px !important;
  font-weight: 500 !important; font-size: 16px !important;
  padding: 13px 22px !important;
  box-shadow: 0 8px 22px rgba(13, 143, 134, 0.32) !important;
  transition: transform .15s ease, filter .15s ease;
}
#ask-btn:hover { filter: brightness(1.06); transform: translateY(-1px); }

/* Answer + sources */
#answer-card, #sources-card { color: #15323d !important; font-size: 16px !important; min-height: 56px; }
#answer-card h1, #answer-card h2, #answer-card h3 { color: var(--ink) !important; }
#answer-card em, #sources-card em { color: #5a6b78 !important; }
.label-wrap span, label span { color: var(--muted) !important; font-weight: 500 !important; }
footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=["-apple-system", "BlinkMacSystemFont", "SF Pro Display", "Helvetica Neue", "sans-serif"],
)

with gr.Blocks(title="Clinical RAG System", theme=theme, css=CSS) as demo:
    with gr.Column(elem_id="hero"):
        gr.Markdown(
            "# 🧬 Clinical RAG System\n"
            "Answers grounded in Ireland's **NCEC National Clinical Guidelines** — "
            "with citations, source passages, and a safe refusal when the evidence is weak."
        )

    with gr.Group(elem_id="warn"):
        gr.Markdown(
            "**Educational demo — not medical advice. Do not use for clinical "
            "decisions, and do not enter real patient information.** "
            f"{DISCLAIMER}"
        )

    with gr.Column(elem_classes=["glass"], elem_id="ask-box"):
        q = gr.Textbox(
            label="Clinical question",
            placeholder="e.g. What should be given within one hour for high-risk sepsis?",
            lines=2,
        )
        btn = gr.Button("Ask", variant="primary", elem_id="ask-btn")

    with gr.Column(elem_classes=["glass"], elem_id="answer-card"):
        out = gr.Markdown(label="Answer", value="_Your grounded answer will appear here._")
    with gr.Column(elem_classes=["glass"], elem_id="sources-card"):
        src = gr.Markdown(label="Retrieved guideline sources", value="_Sources and relevance scores will appear here._")

    gr.Examples(
        [
            "What should be given within one hour for high-risk sepsis?",
            "How is a COPD diagnosis confirmed?",
            "How is diabetic ketoacidosis managed in adults with type 1 diabetes?",
            "What are the core standard precautions for infection prevention and control?",
        ],
        inputs=q,
    )

    gr.Markdown(
        "_Guideline content: NCEC National Clinical Guidelines, © Department of Health, "
        "Ireland — reproduced for non-commercial, educational use with attribution. "
        "Not affiliated with or endorsed by the Department of Health / HSE._",
        elem_id="hero",
    )

    btn.click(answer_question, inputs=q, outputs=[out, src])
    q.submit(answer_question, inputs=q, outputs=[out, src])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
