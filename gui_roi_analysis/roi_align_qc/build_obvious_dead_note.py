"""Write a self-contained note with the analysis figure stored inside the file."""

from __future__ import annotations

import base64
import os
import zipfile
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURE = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260701\auto1\roi_align_qc\obvious_dead_figure.png"
DOCX = os.path.join(HERE, "OBVIOUS_DEAD_note.docx")
HTML_PATH = os.path.join(HERE, "OBVIOUS_DEAD_note.html")

# A4 landscape, 0.6 inch margins. EMU per twip is 635.
PAGE_W = 16838
PAGE_H = 11906
MARGIN = 864
CONTENT_TWIPS = PAGE_W - 2 * MARGIN
IMAGE_CX = CONTENT_TWIPS * 635
IMAGE_CY = int(IMAGE_CX * 1259 / 1737)


def _run(text: str, size: int = 22, bold: bool = False) -> str:
    weight = "<w:b/>" if bold else ""
    return (
        "<w:r><w:rPr>"
        '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Yu Gothic"/>'
        f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>{weight}"
        "</w:rPr>"
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    )


def paragraph(text: str, size: int = 22, bold: bool = False, before: int = 0, after: int = 120) -> str:
    return (
        "<w:p><w:pPr>"
        f'<w:spacing w:before="{before}" w:after="{after}"/>'
        "</w:pPr>"
        f"{_run(text, size, bold)}</w:p>"
    )


def bullets(items: list[str]) -> str:
    return "".join(paragraph("・ " + item, after=40) for item in items)


def table(rows: list[list[str]], widths: list[int]) -> str:
    borders = "".join(
        f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="C5C5C5"/>'
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    body = []
    for row_index, row in enumerate(rows):
        cells = []
        for text, width in zip(row, widths):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E8EEF4"/>' if row_index == 0 else ""
            cells.append(
                "<w:tc><w:tcPr>"
                f'<w:tcW w:w="{width}" w:type="dxa"/>{shade}'
                "</w:tcPr>"
                f"{paragraph(text, size=20, bold=row_index == 0, after=40)}"
                "</w:tc>"
            )
        body.append("<w:tr>" + "".join(cells) + "</w:tr>")
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    return (
        "<w:tbl><w:tblPr>"
        f'<w:tblW w:w="{sum(widths)}" w:type="dxa"/>'
        f"<w:tblBorders>{borders}</w:tblBorders>"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        + "".join(body)
        + "</w:tbl>"
        + paragraph("", after=80)
    )


def image_paragraph() -> str:
    return f"""<w:p><w:r><w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{IMAGE_CX}" cy="{IMAGE_CY}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="1" name="obvious-dead"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="obvious_dead_figure.png"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="rId1"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{IMAGE_CX}" cy="{IMAGE_CY}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing></w:r></w:p>"""


def document_xml() -> str:
    param_w = [2600, 4200, CONTENT_TWIPS - 6800]
    result_w = [CONTENT_TWIPS - 2200, 2200]
    parts = [
        paragraph("Reject obvious death on the first pre", size=36, bold=True, before=0, after=160),
        paragraph(
            "Use only the first pre, max-projected over Z. "
            "Reject the field when it has a large round bleb and no continuous dendrite shaft. "
            "A bead sitting on a long shaft is kept. "
            "This rule does not try to catch other-Z bleed or a drifted ROI."
        ),
        paragraph("Image and analysis", size=28, bold=True, before=200, after=80),
        paragraph(
            "Left to right: the max projection (white bar is 5 um), the bleb map "
            "(cyan circles are the disks that are erased), the image after those disks are removed, "
            "and the skeleton. Orange is the one piece whose length is measured. "
            "Blue pieces are shorter and are not counted."
        ),
        image_paragraph(),
        paragraph(
            "Top row: reject (bleb 0.60, shaft 2.1 um). "
            "Middle row: keep (bleb 0.52, shaft 7.3 um). "
            "Bottom row: rejected, but labeled keep (bleb 0.53, shaft 2.5 um). "
            "The dendrite is a string of beads, so the longest connected piece is only 2.5 um "
            "even though the eye can still follow it.",
            before=80,
        ),
        paragraph("Parameters, in micrometers", size=28, bold=True, before=200, after=80),
        paragraph(
            "Lengths in obvious_dead.py are micrometers. "
            "Pixel size is taken from the acquisition state, and each length is converted to pixels for that image."
        ),
        paragraph("xy_um = 0.5 * (FOV_x / zoom / pixels_x + FOV_y / zoom / pixels_y)"),
        table(
            [
                ["Name", "Value", "Meaning"],
                ["BLEB_MIN", "0.45", "Bleb score after a 0-1 stretch. Not a length."],
                ["SHAFT_MAX_UM", "4.25 um", "A shaft this short or shorter counts as no shaft."],
                ["BLEB_SIGMA_UM", "0.46, 0.68, 0.91, 1.21, 1.67 um", "Sizes used to find round blebs."],
                ["RIDGE_SIGMA_UM", "0.15, 0.30, 0.46 um", "Widths used to find the bright shaft."],
                ["BLEB_RADIUS_FACTOR", "1.8", "Radius of the disk erased around a bleb, in units of that sigma."],
            ],
            param_w,
        ),
        paragraph(
            "These micrometers match the zoom-14 high-mag sampling the rule was tuned on "
            "(FOV 273 x 271 um, 128 px, 0.152 um/pixel). "
            "At that sampling, 4.25 um is 28 pixels. "
            "At zoom 15 (0.142 um/pixel) the same 4.25 um is 30 pixels."
        ),
        paragraph("Steps", size=28, bold=True, before=200, after=80),
        bullets(
            [
                "Take only the first pre and max-project it over Z.",
                "Stretch intensities from the 1st to the 99.5th percentile onto 0-1.",
                "At each of the five bleb sizes, compute a scale-normalized Laplacian of Gaussian. Bright round objects score high. The bleb score is the strongest peak.",
                "On each scale, erase a disk of radius 1.8 x sigma around peaks that are at least 0.5 and at least 65% of that scale's maximum. This keeps a bleb rim from being counted as a shaft.",
                "On what remains, keep bright ridges (Frangi filter) at or above 15% of the strongest ridge, and thin them to a one-pixel skeleton.",
                "Shaft length is the longest 8-connected path in one skeleton piece, times xy_um. A diagonal step counts as one pixel. Shorter pieces are ignored.",
                "Reject when bleb >= 0.45 and shaft <= 4.25 um. Both must be true.",
            ]
        ),
        paragraph("Result on the 211 labeled fields", size=28, bold=True, before=200, after=80),
        paragraph(
            "Specificity on keeps is 191/195 = 97.9%. "
            "Keep shafts are much longer than the cutoff (median 14.6 um, 10th percentile 6.4 um)."
        ),
        table(
            [
                ["Group", "Called dead"],
                ["User-listed obvious death (10)", "10/10"],
                ["Other category 8, might still be alive (6)", "2/6"],
                ["Category 9 keep (195)", "4/195"],
            ],
            result_w,
        ),
        paragraph("Called dead, but might still be alive", size=28, bold=True, before=160, after=80),
        bullets(
            [
                "20260909_cnt_4_pos1__highmag_7_set1 (bleb 0.53, shaft 3.34 um)",
                "20260623_5_pos1__highmag_4_set2 (bleb 0.56, shaft 3.83 um)",
            ]
        ),
        paragraph("Called dead, but labeled keep", size=28, bold=True, before=160, after=80),
        bullets(
            [
                "20260623_3_pos1__highmag_5_set2 (bleb 0.53, shaft 2.55 um; bottom row of the figure)",
                "20260623_4_pos1__highmag_5_set1 (bleb 0.57, shaft 4.25 um; sits on the cutoff)",
                "20260909_cnt_4_pos1__highmag_4_set1 (bleb 0.46, shaft 3.19 um)",
                "20260909_cnt_4_pos1__highmag_4_set2 (bleb 0.49, shaft 3.64 um)",
            ]
        ),
        paragraph(
            "Writing the sizes in micrometers changes only the zoom-15 fields (20260623). "
            "The same physical sigma covers about 7% more pixels, and 4.25 um is 30 pixels rather than 28. "
            "That is why 20260623_5_pos1__highmag_4_set2 and 20260623_4_pos1__highmag_5_set1 are now called dead. "
            "Zoom-14 calls are unchanged.",
            before=80,
        ),
    ]
    sect = (
        "<w:sectPr>"
        f'<w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}" w:orient="landscape"/>'
        f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" '
        'w:header="0" w:footer="0" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<w:body>"
        + "".join(parts)
        + sect
        + "</w:body></w:document>"
    )


def write_docx(png: bytes) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/figure.png"/>
</Relationships>
"""
    with zipfile.ZipFile(DOCX, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", content_types)
        package.writestr("_rels/.rels", root_rels)
        package.writestr("word/document.xml", document_xml().encode("utf-8"))
        package.writestr("word/_rels/document.xml.rels", doc_rels)
        package.writestr("word/media/figure.png", png)


def write_html(png: bytes) -> None:
    encoded = base64.b64encode(png).decode("ascii")
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Reject obvious death on the first pre</title>
<style>
  body {{ font-family: Calibri, "Segoe UI", sans-serif; max-width: 1100px; margin: 28px auto; color: #1c1c1c; line-height: 1.55; }}
  h1 {{ font-size: 26px; }}
  h2 {{ font-size: 18px; margin-top: 1.4em; }}
  img {{ width: 100%; height: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.6em 0 1em; }}
  th, td {{ border: 1px solid #c5c5c5; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #e8eef4; }}
  code {{ font-family: Consolas, monospace; }}
</style>
</head>
<body>
<h1>Reject obvious death on the first pre</h1>
<p>Use only the first pre, max-projected over Z. Reject the field when it has a large round bleb and no continuous dendrite shaft. A bead sitting on a long shaft is kept. This rule does not try to catch other-Z bleed or a drifted ROI.</p>
<h2>Image and analysis</h2>
<p>Left to right: the max projection (white bar is 5 um), the bleb map (cyan circles are the disks that are erased), the image after those disks are removed, and the skeleton. Orange is the one piece whose length is measured. Blue pieces are shorter and are not counted.</p>
<img alt="First pre, bleb map, bleb removed, and shaft skeleton" src="data:image/png;base64,{encoded}">
<p>Top row: reject (bleb 0.60, shaft 2.1 um). Middle row: keep (bleb 0.52, shaft 7.3 um). Bottom row: rejected, but labeled keep (bleb 0.53, shaft 2.5 um). The dendrite is a string of beads, so the longest connected piece is only 2.5 um even though the eye can still follow it.</p>
<h2>Parameters, in micrometers</h2>
<p>Lengths in obvious_dead.py are micrometers. Pixel size is taken from the acquisition state, and each length is converted to pixels for that image.</p>
<p><code>xy_um = 0.5 * (FOV_x / zoom / pixels_x + FOV_y / zoom / pixels_y)</code></p>
<table>
<tr><th>Name</th><th>Value</th><th>Meaning</th></tr>
<tr><td>BLEB_MIN</td><td>0.45</td><td>Bleb score after a 0-1 stretch. Not a length.</td></tr>
<tr><td>SHAFT_MAX_UM</td><td>4.25 um</td><td>A shaft this short or shorter counts as no shaft.</td></tr>
<tr><td>BLEB_SIGMA_UM</td><td>0.46, 0.68, 0.91, 1.21, 1.67 um</td><td>Sizes used to find round blebs.</td></tr>
<tr><td>RIDGE_SIGMA_UM</td><td>0.15, 0.30, 0.46 um</td><td>Widths used to find the bright shaft.</td></tr>
<tr><td>BLEB_RADIUS_FACTOR</td><td>1.8</td><td>Radius of the disk erased around a bleb, in units of that sigma.</td></tr>
</table>
<p>These micrometers match the zoom-14 high-mag sampling the rule was tuned on (FOV 273 x 271 um, 128 px, 0.152 um/pixel). At that sampling, 4.25 um is 28 pixels. At zoom 15 (0.142 um/pixel) the same 4.25 um is 30 pixels.</p>
<h2>Steps</h2>
<ol>
<li>Take only the first pre and max-project it over Z.</li>
<li>Stretch intensities from the 1st to the 99.5th percentile onto 0-1.</li>
<li>At each of the five bleb sizes, compute a scale-normalized Laplacian of Gaussian. Bright round objects score high. The bleb score is the strongest peak.</li>
<li>On each scale, erase a disk of radius 1.8 x sigma around peaks that are at least 0.5 and at least 65% of that scale's maximum. This keeps a bleb rim from being counted as a shaft.</li>
<li>On what remains, keep bright ridges (Frangi filter) at or above 15% of the strongest ridge, and thin them to a one-pixel skeleton.</li>
<li>Shaft length is the longest 8-connected path in one skeleton piece, times xy_um. A diagonal step counts as one pixel. Shorter pieces are ignored.</li>
<li>Reject when bleb &gt;= 0.45 and shaft &lt;= 4.25 um. Both must be true.</li>
</ol>
<h2>Result on the 211 labeled fields</h2>
<p>Specificity on keeps is 191/195 = 97.9%. Keep shafts are much longer than the cutoff (median 14.6 um, 10th percentile 6.4 um).</p>
<table>
<tr><th>Group</th><th>Called dead</th></tr>
<tr><td>User-listed obvious death (10)</td><td>10/10</td></tr>
<tr><td>Other category 8, might still be alive (6)</td><td>2/6</td></tr>
<tr><td>Category 9 keep (195)</td><td>4/195</td></tr>
</table>
<h2>Called dead, but might still be alive</h2>
<ul>
<li>20260909_cnt_4_pos1__highmag_7_set1 (bleb 0.53, shaft 3.34 um)</li>
<li>20260623_5_pos1__highmag_4_set2 (bleb 0.56, shaft 3.83 um)</li>
</ul>
<h2>Called dead, but labeled keep</h2>
<ul>
<li>20260623_3_pos1__highmag_5_set2 (bleb 0.53, shaft 2.55 um; bottom row of the figure)</li>
<li>20260623_4_pos1__highmag_5_set1 (bleb 0.57, shaft 4.25 um; sits on the cutoff)</li>
<li>20260909_cnt_4_pos1__highmag_4_set1 (bleb 0.46, shaft 3.19 um)</li>
<li>20260909_cnt_4_pos1__highmag_4_set2 (bleb 0.49, shaft 3.64 um)</li>
</ul>
<p>Writing the sizes in micrometers changes only the zoom-15 fields (20260623). The same physical sigma covers about 7% more pixels, and 4.25 um is 30 pixels rather than 28. That is why 20260623_5_pos1__highmag_4_set2 and 20260623_4_pos1__highmag_5_set1 are now called dead. Zoom-14 calls are unchanged.</p>
</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)


def main() -> None:
    with open(FIGURE, "rb") as handle:
        png = handle.read()
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("figure is not a PNG")
    write_docx(png)
    write_html(png)
    with zipfile.ZipFile(DOCX) as package:
        embedded = package.read("word/media/figure.png")
    if embedded != png:
        raise ValueError("docx image does not match the figure")
    print(DOCX)
    print(HTML_PATH)
    print(f"png_bytes={len(png)} docx_bytes={os.path.getsize(DOCX)} html_bytes={os.path.getsize(HTML_PATH)}")


if __name__ == "__main__":
    main()
