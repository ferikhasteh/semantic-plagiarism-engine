"""Build docs/project_spec.tex (Persian, XeLaTeX) from docs/project_spec.md
and compile it to docs/project_spec.pdf.

The Markdown file is kept as the single editable source of truth for the
report content; this script is the reproducible pipeline that turns it into
the exact deliverable filenames required by the course guide
(docs/project_spec.tex + docs/project_spec.pdf).

Requires a TeX Live installation with xelatex, fontspec and babel (the
"persian"/"farsi" babel locale is part of the standard babel distribution).
The Vazirmatn font used for Persian text is bundled in assets/fonts/, and
DejaVu Sans Mono (used for CLI/code blocks) ships with virtually every TeX
Live and Linux installation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

INPUT_PATH = Path("docs/project_spec.md")
TEX_PATH = Path("docs/project_spec.tex")
PDF_PATH = Path("docs/project_spec.pdf")
FONT_DIR = Path("assets/fonts")

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


# Right-to-Left Mark (U+200F). XeLaTeX's babel/bidi handling mis-orders a
# paragraph that starts with a *weak* character -- in particular, our
# manual Persian-digit heading numbers (e.g. "۴. عنوان (English aside)").
# Without this, the trailing English parenthetical gets pulled next to the
# leading digit instead of staying at the end of the line. Prepending an
# invisible RLM anchors the paragraph's base direction as RTL before any
# weak/number character is seen, which fixes it. It's a no-op for lines
# that don't have this problem, so it's applied to every heading/paragraph
# unconditionally rather than only to the ones that start with a digit.
RLM = "‏"


def escape(text: str) -> str:
    out = []
    for ch in text:
        out.append(LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def rtl_text(text: str) -> str:
    """Escape text for LaTeX and anchor it as an RTL paragraph/run."""
    return RLM + escape(text)


def code_line_to_tex(raw_line: str) -> str:
    """Preserve leading indentation (LaTeX collapses runs of spaces) and
    escape LaTeX-special characters, without touching the rest of the
    (already-literal) Unicode content."""
    stripped = raw_line[4:] if raw_line.startswith("    ") else raw_line
    leading = len(stripped) - len(stripped.lstrip(" "))
    content = escape(stripped.lstrip(" "))
    indent = f"\\hspace*{{{leading * 0.55:.2f}em}}" if leading else ""
    return indent + content


def strip_html_comments(markdown_text: str) -> str:
    """Drop <!-- ... --> blocks (used for editor-only notes in the source
    Markdown) before conversion, so they never leak into the PDF."""
    out_lines: list[str] = []
    in_comment = False
    for line in markdown_text.splitlines():
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        if "<!--" in line and "-->" not in line:
            in_comment = True
            continue
        if "<!--" in line and "-->" in line:
            continue  # single-line comment
        out_lines.append(line)
    return "\n".join(out_lines)


def strip_rtl_div_wrapper(markdown_text: str) -> str:
    """Drop the <div dir="rtl" ...> / </div> wrapper around the Markdown
    body. That wrapper exists purely so plain Markdown viewers (GitHub,
    editors) render the report right-to-left -- it is not meaningful
    Markdown content and the PDF is already fully RTL via babel/persian,
    so these lines must not leak into the LaTeX output as literal text."""
    out_lines: list[str] = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<div") and 'dir="rtl"' in stripped:
            continue
        if stripped == "</div>":
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def build_body(markdown_text: str) -> str:
    markdown_text = strip_html_comments(markdown_text)
    markdown_text = strip_rtl_div_wrapper(markdown_text)
    lines = markdown_text.splitlines()
    tex_parts: list[str] = []
    in_list = False
    code_buffer: list[str] = []
    seen_h1 = False
    prev_was_h1 = False

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            tex_parts.append("\\end{itemize}")
            in_list = False

    def flush_code() -> None:
        nonlocal code_buffer
        if code_buffer:
            body = " \\\\\n".join(code_buffer)
            tex_parts.append(
                "{\\mycode\n\\begin{flushleft}\n" + body + "\n\\end{flushleft}}"
            )
            code_buffer = []

    for raw_line in lines:
        stripped = raw_line.strip()
        is_code_line = raw_line.startswith("    ") and stripped != ""

        if is_code_line:
            flush_list()
            code_buffer.append(code_line_to_tex(raw_line))
            continue
        else:
            flush_code()

        if not stripped:
            flush_list()
            prev_was_h1 = False
            continue

        if stripped == "---":
            flush_list()
            tex_parts.append("\\medskip\\noindent\\hrulefill\\medskip")
            prev_was_h1 = False
            continue

        if stripped.startswith("# "):
            flush_list()
            title = rtl_text(stripped[2:])
            tex_parts.append(
                "\\begin{center}\n{\\Large\\bfseries " + title + "}\n\\end{center}"
            )
            seen_h1 = True
            prev_was_h1 = True
            continue

        if stripped.startswith("### ") and prev_was_h1:
            flush_list()
            subtitle = escape(stripped[4:])  # English subtitle line: no RTL anchor
            tex_parts.append(
                "\\begin{center}\n{\\normalfont\\large " + subtitle
                + "}\n\\end{center}\n\\bigskip"
            )
            prev_was_h1 = False
            continue

        prev_was_h1 = False

        if stripped.startswith("## "):
            flush_list()
            heading = rtl_text(stripped[3:])
            tex_parts.append("\\medskip\n{\\large\\bfseries " + heading + "}\\medskip\n")
            continue

        if stripped.startswith("### "):
            flush_list()
            heading = rtl_text(stripped[4:])
            tex_parts.append("\\smallskip\n{\\bfseries " + heading + "}\\smallskip\n")
            continue

        if stripped.startswith("- "):
            if not in_list:
                tex_parts.append("\\begin{itemize}")
                in_list = True
            tex_parts.append("\\item " + rtl_text(stripped[2:]))
            continue

        flush_list()
        tex_parts.append("\\noindent " + rtl_text(stripped) + "\\par")

    flush_list()
    flush_code()

    return "\n\n".join(tex_parts)


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{fontspec}
\usepackage{babel}
\usepackage[a4paper, margin=2.5cm]{geometry}
\usepackage{parskip}

\babelprovide[import, main]{persian}
\babelfont[persian]{rm}[
  Path = %(font_dir)s/,
  Extension = .ttf,
  UprightFont = *-Regular,
  BoldFont = *-Bold,
]{Vazirmatn}
\newfontfamily\mycode{DejaVu Sans Mono}

\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""


def build_tex() -> str:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input report not found: {INPUT_PATH}")

    body = build_body(INPUT_PATH.read_text(encoding="utf-8"))
    preamble = PREAMBLE % {"font_dir": FONT_DIR.as_posix()}
    return preamble + "\n" + body + "\n" + POSTAMBLE


def compile_pdf() -> None:
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    tex_source = build_tex()
    TEX_PATH.write_text(tex_source, encoding="utf-8")

    for _ in range(2):  # run twice for stable cross-references/margins
        result = subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={TEX_PATH.parent.as_posix()}",
                TEX_PATH.as_posix(),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout[-4000:])
            print(result.stderr[-2000:])
            raise RuntimeError("xelatex failed to compile docs/project_spec.tex")

    # Clean up LaTeX auxiliary files, keep the .tex source and the .pdf.
    for ext in (".aux", ".log", ".out"):
        aux_file = TEX_PATH.with_suffix(ext)
        if aux_file.exists():
            try:
                aux_file.unlink()
            except OSError:
                pass  # non-fatal: sandboxed filesystems may restrict deletes

    print(f"Saved LaTeX source to {TEX_PATH}")
    print(f"Saved PDF report to {PDF_PATH}")


if __name__ == "__main__":
    try:
        compile_pdf()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
