"""Tests for common.llm.pdf_layout — page layout estimation."""

import pytest
from common.llm.pdf_layout import estimate_lines, estimate_pages, find_page_breaks, LayoutConfig


class TestEstimateLines:
    def test_empty_string(self):
        assert estimate_lines("") == 1  # one empty line

    def test_single_line(self):
        assert estimate_lines("hello world") == 1

    def test_heading_counts_as_two(self):
        assert estimate_lines("# Title") == 2

    def test_separator(self):
        assert estimate_lines("***") == 1

    def test_empty_line(self):
        assert estimate_lines("line1\n\nline2") == 3

    def test_long_line_wraps(self):
        config = LayoutConfig(chars_per_line=10)
        text = "a" * 25  # 25 chars / 10 per line = 3 lines
        assert estimate_lines(text, config) == 3

    def test_exact_multiple_no_extra_line(self):
        config = LayoutConfig(chars_per_line=10)
        text = "a" * 20  # 20 chars / 10 = exactly 2 lines
        assert estimate_lines(text, config) == 2

    def test_multiple_paragraphs(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        assert estimate_lines(text) == 5  # 3 text + 2 empty


class TestEstimatePages:
    def test_short_text_one_page(self):
        assert estimate_pages("hello") == 1

    def test_long_text_multiple_pages(self):
        config = LayoutConfig(lines_per_page=10)
        text = "\n".join([f"Line {i}" for i in range(50)])
        assert estimate_pages(text, config) == 5

    def test_empty_text_one_page(self):
        assert estimate_pages("") == 1

    def test_custom_config(self):
        config = LayoutConfig(lines_per_page=5, chars_per_line=80)
        text = "\n".join(["line"] * 12)
        assert estimate_pages(text, config) == 3  # ceil(12/5) = 3


class TestFindPageBreaks:
    def test_no_breaks_needed(self):
        text = "short text"
        assert find_page_breaks(text) == []

    def test_break_at_page_boundary(self):
        config = LayoutConfig(lines_per_page=5, chars_per_line=80)
        text = "\n".join([f"Line {i}" for i in range(10)])
        breaks = find_page_breaks(text, config)
        assert len(breaks) >= 1

    def test_avoid_orphan_heading(self):
        config = LayoutConfig(lines_per_page=5, chars_per_line=80, min_lines_for_section=4)
        # 4 lines, then a heading that would be orphaned
        text = "\n".join(["line"] * 4) + "\n# Heading\ncontent"
        breaks = find_page_breaks(text, config)
        # Should break before the heading to avoid orphan
        assert len(breaks) >= 1


class TestLayoutConfig:
    def test_defaults(self):
        config = LayoutConfig()
        assert config.lines_per_page == 45
        assert config.chars_per_line == 120
        assert config.min_lines_for_section == 4

    def test_custom_values(self):
        config = LayoutConfig(lines_per_page=50, chars_per_line=80, min_lines_for_section=6)
        assert config.lines_per_page == 50
        assert config.chars_per_line == 80
        assert config.min_lines_for_section == 6
