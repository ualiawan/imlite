"""The :func:`load` dispatcher - imlite's single entry point.

``imlite.load(source)`` inspects what it is given and hands back the right
object, so callers do not have to choose between ``read_image``,
``read_video`` and ``read_frames`` up front.

The rules are deliberately boring: the returned type is always predictable
from the input, and unknown extensions are probed rather than guessed at.
"""

import logging
from pathlib import Path
from typing import Any, TypeAlias, Union

import numpy as np

from imlite._typing import Array
from imlite.core.image import Image
from imlite.core.sequence import FrameSequence
from imlite.core.video import Video
from imlite.exceptions import ImliteOpenError
from imlite.utils.path import is_image_file, is_video_file

log = logging.getLogger(__name__)

__all__ = ["load"]

#: Anything :func:`load` accepts.
Source: TypeAlias = Union[str, Path, Array, Image, Video, FrameSequence, list[Any]]  # noqa: UP007


def load(source: Source) -> Image | Video | FrameSequence:
    """Load *source* and return the matching imlite object.

    ======================  ==========================================  ======================
    Input                   Condition                                   Returns
    ======================  ==========================================  ======================
    ``str`` / ``Path``      image extension                             :class:`Image`
    ``str`` / ``Path``      video extension                             :class:`Video`
    ``str`` / ``Path``      is a directory                              :class:`FrameSequence`
    ``list[str]``           image file paths                            :class:`FrameSequence`
    ``list[ndarray]``       raw arrays                                  :class:`FrameSequence`
    ``list[Image]``         ``Image`` objects                           :class:`FrameSequence`
    ``Array``          2-D or 3-D array                            :class:`Image`
    ``Image``/``Video``/    already an imlite object                    unchanged
    ``FrameSequence``
    ======================  ==========================================  ======================

    A file with an unrecognised extension is probed as an image.  Videos
    cannot be probed - the ffmpeg backend picks its decoder from the
    extension - so an exotic video extension raises with instructions rather
    than returning a :class:`Video` that would fail on first use.

    Args:
        source: A file path, directory path, numpy array, list of frames, or
            an existing imlite object.

    Returns:
        An :class:`Image`, :class:`Video` or :class:`FrameSequence`.

    Raises:
        ImliteOpenError: If the source type cannot be determined.
        ImliteReadError: If a file was identified but could not be read.

    Examples:
        >>> img = imlite.load("photo.jpg")            # -> Image
        >>> vid = imlite.load("clip.mp4")             # -> Video
        >>> seq = imlite.load("frames/")              # -> FrameSequence
        >>> seq = imlite.load([img1, img2, img3])     # -> FrameSequence
    """
    log.debug("load() called with %s", type(source).__name__)

    if isinstance(source, (Image, Video, FrameSequence)):
        return source
    if isinstance(source, np.ndarray):
        return Image.from_numpy(source)
    if isinstance(source, list):
        return _load_list(source)
    if isinstance(source, (str, Path)):
        return _load_path(str(source))

    raise ImliteOpenError(
        f"Cannot load a {type(source).__name__!r}. Expected a file path, directory, "
        "numpy array, list of frames, Image, Video or FrameSequence."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_path(path: str) -> Image | Video | FrameSequence:
    """Dispatch a filesystem path to the right imlite type."""
    location = Path(path)

    if location.is_dir():
        log.debug("load(): %r is a directory -> FrameSequence", path)
        return FrameSequence.from_dir(path)

    if is_image_file(path):
        from imlite.ops.io import read_image

        return read_image(path)

    if is_video_file(path):
        if not location.exists():
            raise ImliteOpenError(f"Video file not found: {path!r}")
        return Video(path)

    if not location.exists():
        raise ImliteOpenError(
            f"File not found: {path!r}. "
            "Check the path, or pass a directory to load a frame sequence."
        )

    log.debug("load(): unknown extension %r - probing the file", location.suffix)
    return _probe_image(path)


def _probe_image(path: str) -> Image:
    """Try to decode a file with an unrecognised extension as an image.

    Only images can be probed.  The ffmpeg backend refuses any path whose
    extension it does not recognise, so a video with an exotic extension has to
    be renamed rather than sniffed - the error below says so.
    """
    from imlite.ops.io import read_image

    try:
        return read_image(path)
    except Exception as exc:
        log.debug("Probe as image failed for %r: %s", path, exc)

    raise ImliteOpenError(
        f"Could not determine what {path!r} is: the extension "
        f"{Path(path).suffix!r} is not one imlite recognises, and the file could not be "
        "decoded as an image.\n"
        "If it is a video, rename it to a known extension (.mp4, .mov, .mkv, ...) - "
        "the ffmpeg backend selects its decoder by extension and cannot sniff the contents."
    )


def _load_list(items: list[Any]) -> FrameSequence:
    """Turn a list of paths, arrays or images into a ``FrameSequence``."""
    if not items:
        return FrameSequence.from_images([])

    first = items[0]

    if isinstance(first, (str, Path)):
        from imlite.ops.io import read_image

        if not is_image_file(first):
            raise ImliteOpenError(
                f"A list of paths must contain image files; {str(first)!r} is not a "
                "recognised image format."
            )
        return FrameSequence.from_images([read_image(str(item)) for item in items])

    if isinstance(first, (np.ndarray, Image)):
        return FrameSequence.from_images(items)

    raise ImliteOpenError(
        f"List items must be paths, numpy arrays or Image objects; got {type(first).__name__!r}."
    )
