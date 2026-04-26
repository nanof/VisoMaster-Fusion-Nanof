"""Utilities that keep long tooltips from becoming extremely wide popups.

Problem
-------
Qt's ``QTipLabel`` (the widget that actually renders tooltips) only enables
word-wrap when the tooltip string looks like rich text — internally it calls
``label->setWordWrap(Qt::mightBeRichText(text))``.  Plain-text tooltips grow
horizontally without bound, so long help strings from our settings layout
produce popups that span almost the entire screen with a single line of text.

The fix applied here is cosmetic and non-invasive:

* :func:`wrap_tooltip_html` takes a plain-text tooltip and returns a
  ``<qt>``-wrapped rich-text version with manual line breaks every
  ``width_chars`` characters.  The ``<qt>`` prefix is what flips Qt into
  rich-text mode so the label word-wraps naturally; the pre-wrapping also
  bounds the maximum line width reliably, even when Qt's auto sizing would
  otherwise pick a very wide value on large monitors.
* :func:`install_tooltip_vertical_wrap` constructs a ``QObject`` event filter
  that listens for ``QEvent.ToolTipChange`` on every widget in the process
  and transparently rewrites long plain-text tooltips in place.  The
  rewritten text starts with ``<``, so the recursive re-change is a no-op.

The PySide6 import lives inside :func:`install_tooltip_vertical_wrap` on
purpose: unit tests for :func:`wrap_tooltip_html` must not pull in real Qt,
because sibling UI test files stub ``PySide6`` via ``sys.modules`` and
pre-loading the real library would bypass their stubs and break those tests.
"""

from __future__ import annotations

from html import escape
from textwrap import fill as _textwrap_fill


MAX_CHARS_PER_LINE: int = 75
"""Default soft line length before a long tooltip wraps onto a new line.

Chosen to fit comfortably on a typical UI font without the popup spanning the
full screen.  Lines are broken only on whitespace/word boundaries so the result
still reads naturally.
"""

WRAP_THRESHOLD: int = 60
"""Tooltips shorter than this (and without manual newlines) are left as-is.

Short plain-text tooltips already display acceptably; wrapping them would only
add latency and HTML escaping with no visual benefit.
"""


def wrap_tooltip_html(
    text: str, width_chars: int = MAX_CHARS_PER_LINE
) -> str:
    """Return a rich-text tooltip wrapped at ``width_chars`` columns.

    Parameters
    ----------
    text:
        Source tooltip text.  May already be rich text (in which case it's
        returned unchanged) or plain text.
    width_chars:
        Soft maximum line width in characters.  Words longer than this are
        kept intact (``break_long_words=False``) so URLs and identifiers
        don't get chopped mid-token.

    Returns
    -------
    str
        The wrapped tooltip.  Identical to ``text`` when wrapping would be a
        no-op (already rich, empty, or short enough).

    Notes
    -----
    ``<qt>`` is the minimal marker that forces Qt to treat the string as rich
    text; combined with ``<br/>`` line breaks this is enough to get the
    vertical-growth behaviour we want on every Qt theme and on every platform.
    """
    if not text:
        return text

    stripped = text.lstrip()
    if stripped.startswith("<"):
        # Already rich text; respect whatever the caller set explicitly.
        return text

    # No meaningful gain in wrapping very short single-line tips.
    if len(text) < WRAP_THRESHOLD and "\n" not in text:
        return text

    # Preserve paragraph boundaries (blank line separated), collapse
    # accidental single newlines into spaces, then re-wrap each paragraph.
    paragraphs = text.split("\n\n")
    wrapped_paragraphs = []
    for paragraph in paragraphs:
        collapsed = " ".join(
            line.strip() for line in paragraph.splitlines() if line.strip()
        )
        if not collapsed:
            continue
        wrapped_paragraphs.append(
            _textwrap_fill(
                collapsed,
                width=width_chars,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    if not wrapped_paragraphs:
        return text

    joined = "\n\n".join(wrapped_paragraphs)
    escaped = escape(joined).replace("\n", "<br/>")
    return f"<qt>{escaped}</qt>"


def _needs_wrapping(text: str) -> bool:
    """Return ``True`` when ``text`` would actually be rewritten.

    Centralised so the event filter can short-circuit without calling
    :func:`wrap_tooltip_html` and string-comparing the result, which would
    allocate during every ``QEvent.ToolTipChange`` on every widget.
    """
    if not text:
        return False
    stripped = text.lstrip()
    if stripped.startswith("<"):
        return False
    if len(text) < WRAP_THRESHOLD and "\n" not in text:
        return False
    return True


def install_tooltip_vertical_wrap(
    app, width_chars: int = MAX_CHARS_PER_LINE
):
    """Install an application-level tooltip-wrapping event filter on ``app``.

    The filter listens for ``QEvent.ToolTipChange`` events on any ``QWidget``
    in the process and rewrites long plain-text tooltips in place using
    :func:`wrap_tooltip_html` so Qt renders them as word-wrapped rich text.

    The returned filter is parented to ``app`` so it shares the application's
    lifetime; callers don't need to keep a separate reference alive.

    Parameters
    ----------
    app:
        The ``QApplication`` (or any compatible ``QCoreApplication``) instance.
    width_chars:
        Soft maximum line width forwarded to :func:`wrap_tooltip_html`.
    """
    # Imports are intentionally deferred: keeping PySide6 out of module-level
    # imports means :func:`wrap_tooltip_html` can be unit-tested without a
    # real Qt installation, and without disturbing sibling test modules that
    # stub ``PySide6`` via ``sys.modules``.
    from PySide6 import QtCore, QtWidgets

    class TooltipVerticalWrapFilter(QtCore.QObject):
        """Rewrites plain-text tooltips as bounded rich text on change."""

        def __init__(self, parent=None, _width_chars: int = width_chars):
            super().__init__(parent)
            self._width_chars = _width_chars

        def eventFilter(  # noqa: N802 - Qt API name
            self,
            watched: QtCore.QObject,
            event: QtCore.QEvent,
        ) -> bool:
            try:
                if event.type() != QtCore.QEvent.Type.ToolTipChange:
                    return False
                if not isinstance(watched, QtWidgets.QWidget):
                    return False

                current = watched.toolTip()
                if not _needs_wrapping(current):
                    return False

                wrapped = wrap_tooltip_html(current, self._width_chars)
                if wrapped == current:
                    return False

                # Setting the tooltip fires another ToolTipChange, but the
                # new text starts with "<" so ``_needs_wrapping`` returns
                # False on the next pass and the filter is idempotent.
                watched.setToolTip(wrapped)
            except Exception:
                # Tooltip cosmetics must never break the event loop.
                return False
            return False

    tooltip_filter = TooltipVerticalWrapFilter(app)
    app.installEventFilter(tooltip_filter)
    return tooltip_filter
