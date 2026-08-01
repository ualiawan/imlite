"""FFmpeg discovery and diagnostics.

imlite never asks the user to install ffmpeg.  The ``imageio-ffmpeg``
dependency ships a static ffmpeg binary inside its wheel for every mainstream
platform (Windows x86-64, macOS x86-64 / arm64, manylinux x86-64 / aarch64 /
i686), and ``pip install imlite`` pulls it in automatically.

On the few platforms with no such wheel - musl/Alpine, ppc64le, FreeBSD -
``imageio-ffmpeg`` falls back to ``$IMAGEIO_FFMPEG_EXE`` or an ffmpeg on
``PATH``.  This module turns that fallback's bare ``RuntimeError`` into an
:class:`~imlite.exceptions.ImliteFFmpegError` carrying instructions the user
can actually act on.

Resolution is lazy and cached: importing imlite never touches ffmpeg, and the
lookup happens at most once per process.
"""

import functools
import logging
import platform

from imlite.exceptions import ImliteFFmpegError

log = logging.getLogger(__name__)

__all__ = ["ffmpeg_info", "require_ffmpeg", "resolve_ffmpeg"]

_INSTALL_HINTS = {
    "Linux": "sudo apt install ffmpeg   (or: dnf install ffmpeg / apk add ffmpeg)",
    "Darwin": "brew install ffmpeg",
    "Windows": "winget install Gyan.FFmpeg   (or: choco install ffmpeg)",
}


def _install_hint() -> str:
    """Return a platform-appropriate ffmpeg installation command."""
    return _INSTALL_HINTS.get(platform.system(), "install ffmpeg with your system package manager")


@functools.lru_cache(maxsize=1)
def resolve_ffmpeg() -> str:
    """Return the path to the ffmpeg executable imlite will use.

    The result is cached for the lifetime of the process.

    Returns:
        Absolute path to an ffmpeg binary - normally the static one bundled
        inside the installed ``imageio-ffmpeg`` wheel.

    Raises:
        ImliteFFmpegError: If no ffmpeg binary can be found.

    Example:
        >>> from imlite.utils.ffmpeg import resolve_ffmpeg
        >>> resolve_ffmpeg()  # doctest: +SKIP
        '.../site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.1'
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - imageio-ffmpeg is a hard dependency
        raise ImliteFFmpegError(
            "imageio-ffmpeg is not installed, so imlite cannot read or write video.\n"
            "Reinstall imlite to pull it in:  pip install --force-reinstall imlite"
        ) from exc

    try:
        exe = str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        raise ImliteFFmpegError(
            f"No ffmpeg binary is available on this platform "
            f"({platform.system()} {platform.machine()}).\n"
            f"imageio-ffmpeg normally bundles one, but publishes no wheel for this "
            f"platform. To fix it, either:\n"
            f"  1. Install ffmpeg system-wide:  {_install_hint()}\n"
            f"  2. Or point imlite at an existing binary:\n"
            f"       export IMAGEIO_FFMPEG_EXE=/path/to/ffmpeg\n"
            f"Then run `imlite doctor` to confirm imlite can see it."
        ) from exc

    log.debug("Resolved ffmpeg executable: %s", exe)
    return exe


def require_ffmpeg() -> str:
    """Assert that ffmpeg is usable, returning its path.

    Call this before opening an imageio ffmpeg reader or writer so failures
    surface as an actionable :class:`~imlite.exceptions.ImliteFFmpegError`
    instead of a backend traceback.

    Returns:
        Absolute path to the ffmpeg executable.

    Raises:
        ImliteFFmpegError: If no ffmpeg binary can be found.
    """
    return resolve_ffmpeg()


def ffmpeg_info() -> dict[str, str | bool]:
    """Report which ffmpeg imlite resolved, and whether it is bundled.

    Never raises - if ffmpeg is missing, ``available`` is ``False`` and
    ``error`` explains why.  This is what ``imlite doctor`` prints.

    Returns:
        A dict with keys ``available`` (bool), ``exe``, ``version``,
        ``bundled`` (bool - ``True`` when the binary came from the
        ``imageio-ffmpeg`` wheel rather than the system) and ``error``.

    Example:
        >>> import imlite
        >>> imlite.ffmpeg_info()["available"]  # doctest: +SKIP
        True
    """
    info: dict[str, str | bool] = {
        "available": False,
        "exe": "",
        "version": "",
        "bundled": False,
        "error": "",
    }
    try:
        exe = resolve_ffmpeg()
    except ImliteFFmpegError as exc:
        info["error"] = str(exc)
        return info

    info["available"] = True
    info["exe"] = exe
    info["bundled"] = "imageio_ffmpeg" in exe.replace("\\", "/")

    try:
        import imageio_ffmpeg

        info["version"] = str(imageio_ffmpeg.get_ffmpeg_version())
    except Exception as exc:
        info["version"] = "unknown"
        log.debug("Could not read ffmpeg version: %s", exc)

    return info
