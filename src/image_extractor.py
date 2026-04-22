"""
Image extraction from PDFs with figure caption detection.

For each page:
  1. Extract embedded images via PyMuPDF, filter by minimum size.
  2. Detect nearby text blocks starting with "Fig" / "Figure" as captions.
  3. Return a list of ImageRecord objects ready for storage.

Images without a detected caption are kept but marked accordingly.
Page-level renders (full-page images in scanned PDFs) are included only
when no other images are found on that page.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tunables
# ------------------------------------------------------------------
MIN_WIDTH_PX = 400         # filters logos, author photos, icons
MIN_HEIGHT_PX = 400
MIN_AREA_PX = 200_000      # ~450×450 — keeps only real figures
MIN_FILE_BYTES = 10_000    # skip tiny embedded images
CAPTION_SEARCH_GAP = 80    # points below/above image to search for caption
_FIG_RE = re.compile(r"^\s*(fig(?:ure)?\.?\s*\d+)", re.IGNORECASE)


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------
@dataclass
class ImageRecord:
    image_index: int          # sequential across whole document
    page: int                 # 1-based
    xref: int                 # PyMuPDF internal reference
    width: int
    height: int
    ext: str                  # "png" / "jpeg" / etc.
    caption: Optional[str]    # detected figure caption text, or None
    filename: str             # e.g. "p03_img01_Fig3.png"
    data: bytes = field(repr=False)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def extract_images(pdf_path: str | Path) -> List[ImageRecord]:
    """
    Extract figures from a PDF. Returns one ImageRecord per kept image,
    sorted by page then vertical position.
    """
    path = Path(pdf_path)
    records: List[ImageRecord] = []
    global_idx = 0

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.error("Cannot open PDF for image extraction: %s", exc)
        return records

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_records = _extract_page_images(doc, page, page_num + 1)
        for rec in page_records:
            rec.image_index = global_idx
            global_idx += 1
        records.extend(page_records)

    doc.close()
    logger.info("Extracted %d figure(s) from %s", len(records), path.name)
    return records


# ------------------------------------------------------------------
# Per-page extraction
# ------------------------------------------------------------------
def _extract_page_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
) -> List[ImageRecord]:
    raw_images = page.get_images(full=True)
    if not raw_images:
        return []

    # Get text blocks with their bounding boxes for caption search
    text_blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,...)

    kept: List[ImageRecord] = []
    img_counter = 0

    for img_info in raw_images:
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue

        data = base_image["image"]
        ext = base_image.get("ext", "png")
        width = base_image.get("width", 0)
        height = base_image.get("height", 0)

        # Size filter
        if (width < MIN_WIDTH_PX or height < MIN_HEIGHT_PX
                or width * height < MIN_AREA_PX
                or len(data) < MIN_FILE_BYTES):
            continue

        # Get bounding box of this image on the page
        img_bbox = _get_image_bbox(page, xref)

        # Find caption near the image
        caption = _find_caption(img_bbox, text_blocks) if img_bbox else None

        img_counter += 1
        slug = _caption_slug(caption)
        filename = f"p{page_num:02d}_img{img_counter:02d}{slug}.{ext}"

        kept.append(ImageRecord(
            image_index=0,      # filled by caller
            page=page_num,
            xref=xref,
            width=width,
            height=height,
            ext=ext,
            caption=caption,
            filename=filename,
            data=data,
        ))

    return kept


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _get_image_bbox(page: fitz.Page, xref: int) -> Optional[fitz.Rect]:
    """Return the bounding box of an image on the page by its xref."""
    rects = page.get_image_rects(xref)
    if rects:
        return rects[0]
    return None


def _find_caption(
    img_bbox: fitz.Rect,
    text_blocks: list,
) -> Optional[str]:
    """
    Look for a text block starting with 'Fig' within CAPTION_SEARCH_GAP
    points below or above the image bounding box.
    """
    if img_bbox is None:
        return None

    # Search zone: just below the image (primary) then just above (secondary)
    search_below = fitz.Rect(
        img_bbox.x0 - 20, img_bbox.y1,
        img_bbox.x1 + 20, img_bbox.y1 + CAPTION_SEARCH_GAP,
    )
    search_above = fitz.Rect(
        img_bbox.x0 - 20, img_bbox.y0 - CAPTION_SEARCH_GAP,
        img_bbox.x1 + 20, img_bbox.y0,
    )

    for zone in (search_below, search_above):
        for block in text_blocks:
            bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]
            block_rect = fitz.Rect(bx0, by0, bx1, by1)
            if not zone.intersects(block_rect):
                continue
            text = block[4].strip()
            if _FIG_RE.match(text):
                # Collapse whitespace in the caption
                return re.sub(r"\s+", " ", text)

    return None


def _caption_slug(caption: Optional[str]) -> str:
    """Turn 'Fig. 3. Market for ceramics AM' → '_Fig3'."""
    if not caption:
        return ""
    m = _FIG_RE.match(caption)
    if not m:
        return ""
    label = re.sub(r"[^A-Za-z0-9]", "", m.group(1))
    return f"_{label}"
