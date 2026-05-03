"""Logging and progress-bar infrastructure for imlite.

- A single named logger ``"imlite"`` is used by the whole library.
  Child modules use ``logging.getLogger(__name__)`` which propagates up.
- imlite NEVER adds handlers or sets a level on the root logger.
  That is 100% the caller's responsibility.
- Progress bars use tqdm and respect a global on/off flag.

Public helpers (re-exported from ``imlite.__init__``):
    set_verbosity(level)  - convenience wrapper around the imlite logger level
    set_progress(enabled) - globally enable/disable all tqdm progress bars
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from typing import Any, TypeVar

from tqdm import tqdm

__all__ = ["get_logger", "set_verbosity", "set_progress", "progress"]

# ---------------------------------------------------------------------------
# The one logger for the whole library
# ---------------------------------------------------------------------------
logger = logging.getLogger("imlite")

# A sentinel level above CRITICAL - used for "SILENT" mode.
_SILENT_LEVEL = logging.CRITICAL + 1

# Module-level flag controlling all progress bars.
_show_progress: bool = True

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_logger() -> logging.Logger:
    """Return the ``"imlite"`` logger.

    Useful when users want to attach their own handlers::

        handler = logging.FileHandler("imlite.log")
        imlite.utils.log.get_logger().addHandler(handler)
    """
    return logger


def set_verbosity(level: int | str) -> None:
    """Set the log level for the imlite logger.

    Also attaches a ``StreamHandler`` to *stderr* if no handlers are currently
    configured, so output is immediately visible without extra setup.

    Args:
        level: A :mod:`logging` level integer (e.g. ``logging.DEBUG``) or a
            string name.  The special string ``"SILENT"`` suppresses all
            imlite output.

    Examples:
        >>> import imlite
        >>> imlite.set_verbosity("DEBUG")    # see everything
        >>> imlite.set_verbosity("INFO")     # operation start/end messages
        >>> imlite.set_verbosity("WARNING")  # only warnings and above (default)
        >>> imlite.set_verbosity("SILENT")   # suppress all imlite log output
    """
    if isinstance(level, str):
        upper = level.upper()
        if upper == "SILENT":
            numeric = _SILENT_LEVEL
        else:
            numeric = getattr(logging, upper, None)
            if numeric is None:
                raise ValueError(
                    f"Unknown log level {level!r}. "
                    "Use 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'SILENT'."
                )
    else:
        numeric = level

    logger.setLevel(numeric)

    # Attach a stderr handler if nothing is configured yet so the user
    # actually sees the output without having to call basicConfig().
    if not logger.handlers and not logging.root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
        logger.addHandler(handler)


def set_progress(enabled: bool) -> None:
    """Enable or disable all imlite progress bars globally.

    Args:
        enabled: ``True`` to show progress bars (default), ``False`` to
            suppress them.  Useful in production pipelines, CI environments,
            or unit tests.

    Example:
        >>> import imlite
        >>> imlite.set_progress(False)   # no tqdm output anywhere in imlite
    """
    global _show_progress
    _show_progress = enabled


def progress(
    iterable: Any,
    desc: str = "",
    total: int | None = None,
    unit: str = "frame",
    show: bool = True,
) -> Iterator[Any]:
    """Wrap *iterable* with a tqdm progress bar, respecting the global switch.

    Args:
        iterable: Any iterable to wrap.
        desc: Label shown to the left of the bar.
        total: Total item count for accurate percentage display.  Inferred
            automatically when *iterable* has a ``__len__``.
        unit: Unit name displayed on the bar, e.g. ``"frame"`` or ``"image"``.
        show: Per-call override.  Even if ``True``, bars are still suppressed
            when :func:`set_progress` has been called with ``False``.

    Returns:
        The original iterable wrapped in tqdm, or the iterable unchanged when
        progress output is suppressed.

    Example:
        >>> from imlite.utils.log import progress
        >>> for frame in progress(frames, desc="Encoding", unit="frame"):
        ...     writer.append_data(frame)
    """
    if not (_show_progress and show):
        return iter(iterable)
    return tqdm(iterable, desc=desc, total=total, unit=unit, dynamic_ncols=True)
