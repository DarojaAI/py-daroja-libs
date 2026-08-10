"""PDF page layout estimation.

Engine-agnostic heuristics for estimating how much content fits on a
page.  Works with any rendering engine (WeasyPrint, markdown-pdf,
wkhtmltopdf) by counting lines, chars, and section breaks.

Usage::

    from common.llm.pdf_layout import estimate_pages, estimate_lines

    total_lines = estimate_lines(markdown_text)
    pages = estimate_pages(markdown_text, lines_per_page=45)

The defaults are calibrated for US Letter with 0.75" margins.  Adjust
``lines_per_page`` and ``chars_per_line`` for other paper sizes or
margin settings.
"""

from dataclasses import dataclass


@dataclass
class LayoutConfig:
    """Page layout configuration.

    Attributes:
        lines_per_page: Estimated rendered lines per page.
            Default 45 (US Letter, 0.75" margins, 11pt body text).
        chars_per_line: Approximate characters per line.
            Default 120.
        min_lines_for_section: Minimum lines to keep a section together
            (avoid orphan headers at page bottom).  Default 4.
    """

    lines_per_page: int = 45
    chars_per_line: int = 120
    min_lines_for_section: int = 4


def estimate_lines(text: str, config: LayoutConfig | None = None) -> int:
    """Estimate the number of rendered lines for a block of text.

    Accounts for:
    - Empty lines (1 line each)
    - Markdown headings (2 lines — heading + space)
    - Markdown separators (1 line)
    - Long lines wrapping at ``chars_per_line``

    Args:
        text: Text content (markdown or plain).
        config: Layout configuration.  Defaults to :class:`LayoutConfig`.

    Returns:
        Estimated number of rendered lines.
    """
    if config is None:
        config = LayoutConfig()

    lines = text.split("\n")
    total_lines = 0

    for line in lines:
        if not line.strip():
            total_lines += 1
        elif line.startswith("#"):
            total_lines += 2  # heading + space
        elif line.startswith("***"):
            total_lines += 1
        else:
            # Account for line wrapping
            wrapped = max(1, len(line) // config.chars_per_line + (1 if len(line) % config.chars_per_line else 0))
            total_lines += wrapped

    return total_lines


def estimate_pages(text: str, config: LayoutConfig | None = None) -> int:
    """Estimate the number of pages a text block will occupy.

    Args:
        text: Text content (markdown or plain).
        config: Layout configuration.  Defaults to :class:`LayoutConfig`.

    Returns:
        Estimated page count (minimum 1).
    """
    if config is None:
        config = LayoutConfig()

    total = estimate_lines(text, config)
    return max(1, (total + config.lines_per_page - 1) // config.lines_per_page)


def find_page_breaks(text: str, config: LayoutConfig | None = None) -> list[int]:
    """Find optimal page break positions in markdown text.

    Identifies positions where a page break would split a section with
    fewer than ``min_lines_for_section`` lines remaining.  Returns the
    line numbers (0-indexed) where a page break should be inserted.

    Args:
        text: Markdown text.
        config: Layout configuration.

    Returns:
        List of line indices where page breaks are recommended.
    """
    if config is None:
        config = LayoutConfig()

    lines = text.split("\n")
    breaks: list[int] = []
    lines_on_page = 0

    for i, line in enumerate(lines):
        if line.startswith("#"):
            line_count = 2
        elif not line.strip():
            line_count = 1
        else:
            wrapped = max(1, len(line) // config.chars_per_line + (1 if len(line) % config.chars_per_line else 0))
            line_count = wrapped

        if lines_on_page + line_count > config.lines_per_page:
            # Check if we'd orphan a section header
            remaining_lines = config.lines_per_page - lines_on_page
            if remaining_lines < config.min_lines_for_section and line.startswith("#"):
                breaks.append(i)
                lines_on_page = 0
            else:
                breaks.append(i)
                lines_on_page = line_count
        else:
            lines_on_page += line_count

    return breaks
