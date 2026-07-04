from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak


INPUT_PATH = Path("docs/project_report.md")
OUTPUT_PATH = Path("docs/project_report.pdf")


def markdown_line_to_paragraph(line: str, styles):
    stripped = line.strip()

    if not stripped:
        return Spacer(1, 0.25 * cm)

    if stripped.startswith("# "):
        return Paragraph(stripped[2:], styles["Title"])

    if stripped.startswith("## "):
        return Paragraph(stripped[3:], styles["Heading1"])

    if stripped.startswith("### "):
        return Paragraph(stripped[4:], styles["Heading2"])

    if stripped.startswith("- "):
        return Paragraph("• " + stripped[2:], styles["CustomBullet"])

    if stripped.startswith("    "):
        safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(f"<font name='Courier'>{safe}</font>", styles["CodeBlock"])

    safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, styles["BodyText"])


def build_pdf() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input report not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            leftIndent=0.4 * cm,
            spaceAfter=0.15 * cm,
        )
    )

    styles.add(
        ParagraphStyle(
            name="CustomBullet",
            parent=styles["BodyText"],
            leftIndent=0.4 * cm,
            firstLineIndent=-0.2 * cm,
            spaceAfter=0.1 * cm,
        )
    )

    story = []

    for line in INPUT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip() == "---":
            story.append(Spacer(1, 0.3 * cm))
            continue

        story.append(markdown_line_to_paragraph(line, styles))

    doc.build(story)
    print(f"Saved PDF report to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
