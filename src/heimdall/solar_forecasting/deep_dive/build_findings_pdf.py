"""Build a PDF of deep_dive/findings.md with the figure files embedded.

LaTeX is not required. Needs reportlab and pillow, which are not project
dependencies. Run from the repo root with a Python that has both:

    python src/heimdall/solar_forecasting/deep_dive/build_findings_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    CondPageBreak,
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[4]
MD_PATH = Path(__file__).resolve().parent / "findings.md"
OUT_PATH = ROOT / "artifacts" / "solar_variability_findings.pdf"
PNG = re.compile(r"`(artifacts/figures/[^`]+\.png)`")


def xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str) -> str:
    text = xml_escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', text)
    return text


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            spaceBefore=0,
            spaceAfter=10,
            textColor="#1a1a1a",
        )
    )
    base.add(
        ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
            textColor="#1a1a1a",
        )
    )
    base.add(
        ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Times-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            "BulletBody",
            parent=base["Body"],
            alignment=TA_JUSTIFY,
            spaceAfter=2,
        )
    )
    base.add(
        ParagraphStyle(
            "Caption",
            parent=base["Body"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            spaceBefore=2,
            spaceAfter=8,
            textColor="#333333",
        )
    )
    base.add(
        ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11,
        )
    )
    return base


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as im:
        width_px, height_px = im.size
    width = max_width
    height = width * height_px / width_px
    if height > max_height:
        height = max_height
        width = height * width_px / height_px
    return Image(str(path), width=width, height=height)


def parse_table(lines: list[str], cell_style: ParagraphStyle) -> Table:
    rows = []
    for line in lines:
        if re.match(r"^\|\s*-+", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append([Paragraph(inline(c), cell_style) for c in cells])
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), "#eeeeee"),
                ("GRID", (0, 0), (-1, -1), 0.3, "#bbbbbb"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def build() -> None:
    style = styles()
    text = MD_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    story: list = []
    content_width = 7.1 * inch
    max_image_height = 8.2 * inch
    section = ""
    i = 0
    missing: list[str] = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:].strip()), style["H1"]))
            i += 1
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            story.append(Paragraph(inline(section), style["H2"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:].strip()), style["H3"]))
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            story.append(Spacer(1, 4))
            story.append(parse_table(block, style["TableCell"]))
            story.append(Spacer(1, 8))
            continue
        if line.startswith("- "):
            bullets = []
            figure_blocks: list[tuple[str, list[str]]] = []
            while i < len(lines) and lines[i].startswith("- "):
                raw = lines[i][2:].strip()
                paths = PNG.findall(raw)
                caption = PNG.sub("", raw).strip()
                caption = re.sub(r"\s+\.", ".", caption)
                caption = re.sub(r"\s{2,}", " ", caption).strip(" .")
                if section == "Figures" and paths:
                    figure_blocks.append((caption or raw, paths))
                else:
                    bullets.append(
                        ListItem(
                            Paragraph(inline(caption or raw), style["BulletBody"]),
                            leftIndent=12,
                            bulletColor="#222222",
                        )
                    )
                i += 1
            if bullets:
                story.append(
                    ListFlowable(
                        bullets,
                        bulletType="bullet",
                        start="•",
                        leftIndent=16,
                        bulletFontName="Times-Roman",
                        bulletFontSize=10,
                        spaceBefore=2,
                        spaceAfter=6,
                    )
                )
            for caption, paths in figure_blocks:
                story.append(Paragraph("•  " + inline(caption), style["BulletBody"]))
                for rel in paths:
                    path = ROOT / rel
                    if not path.exists():
                        missing.append(rel)
                        story.append(Paragraph(f"Missing figure: {rel}", style["Caption"]))
                        continue
                    story.append(CondPageBreak(3.2 * inch))
                    story.append(Spacer(1, 3))
                    story.append(scaled_image(path, content_width, max_image_height))
                    story.append(Spacer(1, 10))
            continue
        if not line.strip():
            i += 1
            continue
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "- ", "|")):
            para.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(para)), style["Body"]))

    if missing:
        raise SystemExit("Missing figures:\n" + "\n".join(missing))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Where cloud variability justifies a 3D cloud product",
        author="SURFRAD 2025 variability study",
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColorRGB(0.3, 0.3, 0.3)
        canvas.drawString(0.7 * inch, 0.4 * inch, "SURFRAD 2025  ·  solar variability findings")
        canvas.drawRightString(7.8 * inch, 0.4 * inch, str(doc_.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
