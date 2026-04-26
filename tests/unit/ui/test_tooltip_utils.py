"""Tests for ``app.ui.widgets.tooltip_utils.wrap_tooltip_html``.

Focused on the pure-Python wrapping helper that backs the application-level
tooltip event filter.  The Qt-level glue (``TooltipVerticalWrapFilter``) is
not exercised here because sibling UI test files inject ``MagicMock`` stubs
for the ``PySide6`` modules via ``sys.modules``; running a real
``QApplication`` from the same process would be order-dependent and flaky.
The wrapping logic itself is where the substantive behaviour lives, so we
keep these tests Qt-free and trust the ~20-line event filter by reading.
"""

from __future__ import annotations

from app.ui.widgets.tooltip_utils import (
    MAX_CHARS_PER_LINE,
    WRAP_THRESHOLD,
    wrap_tooltip_html,
)


class TestWrapTooltipHtml:
    def test_empty_text_returned_unchanged(self):
        assert wrap_tooltip_html("") == ""

    def test_already_rich_text_returned_unchanged(self):
        rich = "<qt>Already <b>wrapped</b></qt>"
        assert wrap_tooltip_html(rich) == rich

    def test_rich_text_with_leading_whitespace_detected(self):
        rich = "   <p>Hello</p>"
        assert wrap_tooltip_html(rich) == rich

    def test_short_plain_text_returned_unchanged(self):
        short = "Short tooltip"
        assert len(short) < WRAP_THRESHOLD
        assert wrap_tooltip_html(short) == short

    def test_long_plain_text_is_wrapped_as_rich_text(self):
        long_text = (
            "This is a very long tooltip that easily exceeds the default "
            "wrap threshold so it should be converted into rich text with "
            "explicit line breaks to force Qt to grow the popup vertically."
        )
        result = wrap_tooltip_html(long_text)

        assert result != long_text
        assert result.startswith("<qt>")
        assert result.endswith("</qt>")
        assert "<br/>" in result

        # No multi-word line exceeds the width budget (allow one-char slack
        # for word-boundary wrapping).
        body = result[len("<qt>"): -len("</qt>")]
        for line in body.split("<br/>"):
            words = line.split()
            if len(words) > 1:
                assert len(line) <= MAX_CHARS_PER_LINE + 1

    def test_html_special_chars_are_escaped(self):
        text = (
            "Use <angle> brackets and ampersands & quotes \"inside\" a "
            "tooltip body — they must be escaped so Qt does not treat them "
            "as tags."
        )
        result = wrap_tooltip_html(text)

        # The literal "<angle>" must not survive as a tag.
        assert "<angle>" not in result
        assert "&lt;angle&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result

    def test_paragraphs_separated_by_blank_lines_preserved(self):
        text = (
            "First paragraph that is reasonably long so it gets kept as-is "
            "after wrapping.\n\n"
            "Second paragraph that is also long enough to force the helper "
            "to emit a rich-text version with multiple blocks."
        )
        result = wrap_tooltip_html(text)

        # Paragraph boundary is encoded as a pair of <br/> (blank line).
        assert "<br/><br/>" in result
        assert result.count("<br/>") >= 2

    def test_single_newlines_collapsed_to_spaces(self):
        text = (
            "Line one which is short enough on its own.\n"
            "Line two which continues the sentence without forming a new "
            "paragraph and together they exceed the threshold."
        )
        result = wrap_tooltip_html(text)

        # After collapsing, no "own.<br/>Line" since the inner newline
        # becomes a space before re-wrapping on word boundaries.
        assert "own.<br/>Line" not in result

    def test_respects_custom_width(self):
        text = "word " * 60  # 300 chars, forces wrapping
        result = wrap_tooltip_html(text.strip(), width_chars=30)

        assert result.startswith("<qt>")
        body = result[len("<qt>"): -len("</qt>")]
        for line in body.split("<br/>"):
            words = line.split()
            if len(words) > 1:
                assert len(line) <= 31  # width + single-char slack

    def test_whitespace_only_text_returned_as_is(self):
        # After collapsing, paragraph list is empty -> original returned.
        text = "   \n\n   \n"
        assert wrap_tooltip_html(text) == text

    def test_short_text_with_newlines_is_wrapped(self):
        # Short-length short-circuit only applies when there are no newlines:
        # an embedded newline means the caller wanted manual breaks, so we
        # still convert to rich text so Qt renders the line break.
        text = "Line one\nLine two"
        result = wrap_tooltip_html(text)
        assert result.startswith("<qt>")
