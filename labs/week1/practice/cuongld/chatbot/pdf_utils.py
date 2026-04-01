from __future__ import annotations

from fpdf import FPDF


def _latin1_safe(s: str) -> str:
    """
    Built-in PDF fonts (Helvetica, etc.) are not Unicode.
    Convert common punctuation and strip/replace characters outside latin-1.
    """
    s = (s or "").replace("\u2014", "-").replace("\u2013", "-")  # em/en dash
    s = s.replace("\u2018", "'").replace("\u2019", "'")         # curly quotes
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return s.encode("latin-1", errors="replace").decode("latin-1")


def plan_markdown_to_pdf_bytes(plan_markdown: str, *, title: str = "Seoul Itinerary") -> bytes:
    """
    Minimal Markdown-to-PDF rendering:
    - Treats the plan as plain text (keeps line breaks).
    - Uses built-in fonts for portability (no OS font dependency).
    """
    text = (plan_markdown or "").strip()
    if not text:
        text = "(empty)"
    title = _latin1_safe(title)
    text = _latin1_safe(text)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=14)
    pdf.multi_cell(0, 8, title)
    pdf.ln(2)

    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, text)

    # fpdf2 returns the PDF as a latin-1 string when dest="S".
    # Encode to bytes for Streamlit download_button.
    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return str(out).encode("latin-1", errors="ignore")

