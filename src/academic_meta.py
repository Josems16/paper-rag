"""
Academic metadata extraction from scientific papers.

Extracts: title, authors, year, DOI, abstract, journal/venue.

Approach:
  - Title:    largest-font text block on page 1
  - Authors:  heuristic zone between title and abstract on page 1
  - DOI:      regex over first 3 pages
  - Year:     regex over first 3 pages (prefers most-recent plausible year)
  - Abstract: text after "Abstract" heading on first 2 pages
  - Journal:  best-effort from header/footer text
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

# ── Patterns ─────────────────────────────────────────────────────────────────

_DOI_RE = re.compile(
    r'\b(10\.\d{4,9}/[^\s,"\]\[<>(){}|\\^`]{3,})',
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r'\b(19[5-9]\d|20[0-2]\d)\b')
_ABSTRACT_START = re.compile(
    r'^\s*abstract\s*[:\-]?\s*$|^\s*abstract\s*[\n\r]',
    re.IGNORECASE | re.MULTILINE,
)
_KEYWORDS_RE = re.compile(
    r'^\s*(?:keywords?|index terms?)\s*[:\-]\s*(.+)',
    re.IGNORECASE | re.MULTILINE,
)
# Lines that look like author affiliations (numbers, emails, university names)
_AFFILIATION_RE = re.compile(
    r'^\s*[\d\*†‡§¶]+\s|university|institute|department|laboratory|lab\b|'
    r'@[a-z]+\.[a-z]{2,}|corresponding author',
    re.IGNORECASE,
)


@dataclass
class AcademicMetadata:
    title: Optional[str] = None
    authors: Optional[str] = None          # comma-separated string
    year: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    journal: Optional[str] = None
    keywords: Optional[str] = None
    citation_key: Optional[str] = None     # e.g. "Smith2023"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "abstract": self.abstract,
            "journal": self.journal,
            "keywords": self.keywords,
            "citation_key": self.citation_key,
        }

    def format_citation(self) -> str:
        """Return a human-readable citation string."""
        parts = []
        if self.authors:
            parts.append(self.authors)
        if self.year:
            parts.append(f"({self.year})")
        if self.title:
            parts.append(f'"{self.title}"')
        if self.journal:
            parts.append(self.journal)
        if self.doi:
            parts.append(f"https://doi.org/{self.doi}")
        return ". ".join(parts) if parts else "Unknown source"


# ── Public API ────────────────────────────────────────────────────────────────

def extract_academic_metadata(pdf_path: str | Path) -> AcademicMetadata:
    """
    Extract academic metadata from the first pages of a PDF.
    Returns best-effort results; fields may be None if not found.
    """
    path = Path(pdf_path)
    meta = AcademicMetadata()

    try:
        doc = fitz.open(str(path))
    except Exception:
        return meta

    try:
        n_sample = min(3, len(doc))

        # Per-page text (plain)
        pages_text = [doc[i].get_text("text") for i in range(n_sample)]
        combined = "\n".join(pages_text)

        # Page 1 dict (has font sizes)
        page1_dict = doc[0].get_text("dict") if len(doc) > 0 else None

        # Extract each field
        meta.title = _extract_title(page1_dict)
        meta.doi = _extract_doi(combined)
        meta.year = _extract_year(combined)
        meta.abstract = _extract_abstract(combined)
        meta.authors = _extract_authors(pages_text[0], meta.title)
        meta.keywords = _extract_keywords(combined)
        meta.citation_key = _make_citation_key(meta.authors, meta.year, meta.title)

    finally:
        doc.close()

    return meta


# ── Field extractors ──────────────────────────────────────────────────────────

def _extract_title(page_dict: Optional[dict]) -> Optional[str]:
    """
    Find the title as the text block(s) with the largest font size on page 1.
    Filters out very short fragments and obvious headers (page numbers, etc.).
    """
    if not page_dict:
        return None

    # Collect (font_size, y_position, text) for every text span
    spans: list[tuple[float, float, str]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            y = line["bbox"][1]
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0.0)
                if text and len(text) > 3 and size > 0:
                    spans.append((size, y, text))

    if not spans:
        return None

    max_size = max(s[0] for s in spans)
    # Accept spans within 85% of max size (accounts for bold vs regular weight)
    threshold = max_size * 0.85
    title_spans = [s for s in spans if s[0] >= threshold]

    # Sort by vertical position so text reads top-to-bottom
    title_spans.sort(key=lambda s: s[1])

    # Join consecutive spans; stop when a gap > 1.5× font size appears
    if not title_spans:
        return None

    title_lines: list[str] = [title_spans[0][2]]
    prev_y = title_spans[0][1]

    for size, y, text in title_spans[1:]:
        if y - prev_y > size * 2.5:  # large gap → title ends here
            break
        title_lines.append(text)
        prev_y = y

    title = " ".join(title_lines).strip()
    title = re.sub(r"\s+", " ", title)

    # Sanity checks
    if len(title) < 5 or title.isdigit():
        return None

    return title[:400]


def _extract_authors(page1_text: str, title: Optional[str]) -> Optional[str]:
    """
    Extract the author line(s) from page 1.

    Heuristic: the lines immediately after the title and before the abstract
    that don't look like affiliations.  Returns a comma-joined string.
    """
    lines = page1_text.splitlines()

    # Find where the title ends
    title_end_idx = 0
    if title:
        title_words = title.lower().split()[:5]
        for i, line in enumerate(lines):
            if any(w in line.lower() for w in title_words if len(w) > 4):
                title_end_idx = i
                break

    # Find where the abstract starts
    abstract_idx = len(lines)
    for i, line in enumerate(lines):
        if re.match(r'^\s*abstract\s*[:\-]?\s*$', line, re.IGNORECASE):
            abstract_idx = i
            break

    # Candidate lines between title and abstract
    candidate_lines = lines[title_end_idx + 1: abstract_idx]

    author_lines: list[str] = []
    for line in candidate_lines[:15]:  # limit search window
        stripped = line.strip()
        if not stripped:
            continue
        # Skip affiliation-like lines
        if _AFFILIATION_RE.search(stripped):
            continue
        # Skip lines that look like section headers or are too long
        if len(stripped) > 200 or stripped.isupper():
            continue
        # Stop at keywords or received-date lines
        if re.match(r'(received|accepted|published|keywords?)', stripped, re.IGNORECASE):
            break
        author_lines.append(stripped)

    if not author_lines:
        return None

    # Join, normalise whitespace
    raw = ", ".join(author_lines)
    raw = re.sub(r"[\d\*†‡§¶]+", "", raw)       # remove affiliation markers
    raw = re.sub(r"\s*,\s*,\s*", ", ", raw)       # double commas
    raw = re.sub(r"\s+", " ", raw).strip(", ")

    return raw[:500] if raw else None


def _extract_year(text: str) -> Optional[str]:
    """Return the most plausible publication year from the text."""
    matches = _YEAR_RE.findall(text)
    if not matches:
        return None
    # Prefer the most recent year found (published date > submitted date)
    years = sorted(set(matches), reverse=True)
    return years[0]


def _extract_doi(text: str) -> Optional[str]:
    """Return the first DOI found in the text."""
    m = _DOI_RE.search(text)
    if not m:
        return None
    doi = m.group(1).rstrip(".")
    return doi


def _extract_abstract(text: str) -> Optional[str]:
    """
    Extract the abstract paragraph.  Looks for the word "Abstract" as a
    heading and takes the following paragraph(s) up to the next heading or
    ~500 words.
    """
    # Try to find "Abstract" as a standalone heading
    m = _ABSTRACT_START.search(text)
    if not m:
        # Try inline "Abstract — text..." or "Abstract: text..."
        inline = re.search(
            r'\babstract\s*[:\-—]\s*(.{50,1500}?)(?=\n\s*\n|\Z)',
            text, re.IGNORECASE | re.DOTALL,
        )
        if inline:
            return _clean_abstract(inline.group(1))
        return None

    after = text[m.end():]

    # Take text until next major heading (all-caps line, numbered section, etc.)
    next_heading = re.search(
        r'\n\s*(?:[1-9IVX]+[\.\s]|[A-Z]{3,}\s*\n|keywords?|introduction)',
        after, re.IGNORECASE,
    )
    if next_heading:
        abstract_text = after[:next_heading.start()]
    else:
        abstract_text = after[:2000]

    return _clean_abstract(abstract_text)


def _extract_keywords(text: str) -> Optional[str]:
    m = _KEYWORDS_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    # Keywords usually end at a blank line or next section
    raw = raw.split("\n")[0].strip(";., ")
    return raw[:300] if raw else None


def _clean_abstract(text: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 30:
        return None
    return text[:2000]


def _make_citation_key(authors: Optional[str], year: Optional[str], title: Optional[str]) -> str:
    """
    Generate a short citation key in BibTeX style: FirstAuthorLastName + Year.
    E.g. "Vaswani2017"
    """
    first_author = ""
    if authors:
        # Take first token that looks like a surname (>3 chars, no digits)
        for token in re.split(r"[,;&\s]+", authors):
            token = token.strip()
            if len(token) > 3 and token.isalpha():
                first_author = token.capitalize()
                break

    yr = year or "????"
    if first_author:
        return f"{first_author}{yr}"

    # Fallback: first word of title
    if title:
        first_word = re.split(r"\s+", title)[0]
        first_word = re.sub(r"[^A-Za-z]", "", first_word).capitalize()
        if first_word:
            return f"{first_word}{yr}"

    return f"Unknown{yr}"
