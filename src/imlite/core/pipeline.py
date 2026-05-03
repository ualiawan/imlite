"""The ``imlite.load()`` smart dispatcher.

Inspects the source and returns the appropriate type:
    Image, Video, or FrameSequence.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from imlite.core.image import Image
from imlite.core.sequence import FrameSequence
from imlite.core.video import Video
from imlite.exceptions import ImliteOpenError
from imlite.utils.path import is_image_file, is_video_file

log = logging.getLogger(__name__)

__all__ = ["load"]

# Union type accepted by load()
_Source = Union[str, Path, np.ndarray, Image, Video, FrameSequence, list]


def load(source: _Source) -> Image | Video | FrameSequence:
    """Load *source* and return the appropriate imlite object.

    This is the primary entry point for the fluent API.

    Dispatch rules:

    ================  ============================================  =================
    Input type        Condition                                     Returns
    ================  ============================================  =================
    ``str`` / Path    extension in :data:`IMAGE_EXTENSIONS`         :class:`Image`
    ``str`` / Path    extension in :data:`VIDEO_EXTENSIONS`         :class:`Video`
    ``str`` / Path    path is a directory                           :class:`FrameSequence`
    ``list[str]``     items are image file paths                    :class:`FrameSequence`
    ``list[ndarray]`` items are numpy arrays                        :class:`FrameSequence`
    ``list[Image]``   items are ``Image`` objects                   :class:`FrameSequence`
    ``np.ndarray``    2-D or 3-D array                              :class:`Image`
    :class:`Image`    passthrough                                   :class:`Image`
    :class:`Video`    passthrough                                   :class:`Video`
    :class:`FrameSequence` passthrough                              :class:`FrameSequence`
    ================  ============================================  =================

    If the extension is unrecognised and the path exists, imlite attempts to
    open it as an image first, then as a video.  An
    :exc:`~imlite.exceptions.ImliteOpenError` is raised if neither succeeds.

    Args:
        source: The resource to load.  Accepts a file path, directory path,
            numpy array, or any of the three core imlite types.

    Returns:
        :class:`Image`, :class:`Video`, or :class:`FrameSequence`.

    Raises:
        ImliteOpenError: If the source type cannot be determined.
        ImliteReadError: If a file path is given but the file cannot be read.

    Examples:
        >>> img   = imlite.load("photo.jpg")          # -> Image
        >>> vid   = imlite.load("clip.mp4")           # -> Video
        >>> seq   = imlite.load("frames/")            # -> FrameSequence
        >>> seq2  = imlite.load([img1, img2, img3])   # -> FrameSequence
        >>> img2  = imlite.load(np.zeros((100,100,3), dtype=np.uint8))
    """
    log.debug("load() called with source type: %s", type(source).__name__)

    # --- Passthrough: already the right type ---
    if isinstance(source, (Image, Video, FrameSequence)):
        return source  # type: ignore[return-value]

    # --- numpy array -> Image ---
    if isinstance(source, np.ndarray):
        return Image.from_numpy(source)

    # --- list ---
    if isinstance(source, list):
        return _open_list(source)

    # --- str / Path ---
    if isinstance(source, (str, Path)):
        return _open_path(str(source))

    raise ImliteOpenError(
        f"Cannot open source of type {type(source).__name__!r}. "
        "Expected a file path, directory, numpy array, list, "
        "Image, Video, or FrameSequence."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_path(path: str) -> Image | Video | FrameSequence:
    p = Path(path)

    if p.is_dir():
        log.debug("open(): path is directory -> FrameSequence")
        return FrameSequence.from_dir(path)

    ext = p.suffix.lower()

    if is_image_file(path):
        log.debug("open(): recognised image extension %r -> Image", ext)
        from imlite.ops.io import read_image  # noqa: PLC0415

        return read_image(path)

    if is_video_file(path):
        log.debug("open(): recognised video extension %r -> Video", ext)
        return Video(path)

    # Unknown extension - probe the file.
    if not p.exists():
        raise ImliteOpenError(f"File not found: {path!r}")

    log.debug("open(): unknown extension %r - probing file content", ext)
    return _probe_file(path)


def _probe_file(path: str) -> Image | Video:
    """Try to open *path* as image then as video; raise if both fail."""
    # Try image first (faster).
    try:
        import imageio.v3 as iio  # noqa: PLC0415

        arr = iio.imread(path)
        if arr is not None:
            arr = arr.astype("uint8")
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[..., ::-1].copy()  # RGB \u2192 BGR
            return Image.from_numpy(arr, color_space="BGR", path=path)
    except Exception:  # noqa: BLE001
        pass

    # Try video.
    try:
        import imageio.v2 as iio2  # noqa: PLC0415

        reader = iio2.get_reader(path, plugin="ffmpeg")
        reader.close()
        return Video(path)
    except Exception:  # noqa: BLE001
        pass

    raise ImliteOpenError(
        f"Could not determine the type of {path!r}. "
        "The file extension is unrecognised and probing as image/video both failed."
    )


def _open_list(items: list) -> FrameSequence:
    if not items:
        return FrameSequence.from_images([])

    first = items[0]

    if isinstance(first, str):
        # List of file paths - check if they're images
        if is_image_file(first):
            frames = []
            from imlite.ops.io import read_image  # noqa: PLC0415

            for p in items:
                frames.append(read_image(p))
            return FrameSequence.from_images(frames)
        raise ImliteOpenError(
            f"List of strings must be image file paths; first item {first!r} "
            "is not a recognised image format."
        )

    if isinstance(first, (np.ndarray, Image)):
        return FrameSequence.from_images(items)

    raise ImliteOpenError(
        f"List items must be str, np.ndarray, or Image; got {type(first).__name__!r}."
    )
