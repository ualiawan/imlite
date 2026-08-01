"""imlite - lightweight image and video processing for Python.

Install it and everything works, including video: there is no OpenCV
dependency and no system ffmpeg to install, because ``imageio-ffmpeg`` ships a
static ffmpeg binary in its wheel.

Start with :func:`load`, then chain::

    import imlite

    # Image pipeline
    imlite.load("photo.jpg").crop(0, 0, 200, 200).rotate(90).save("out.jpg")

    # Video pipeline - streams frame by frame, whatever the video's length
    imlite.load("clip.mp4").extract_frames(step=2).resize(640, 360).merge(25).save("out.mp4")

Every operation is also available as a plain function that accepts an
:class:`Image` **or** a raw ``numpy.ndarray`` and returns the same type::

    out = imlite.crop(array_or_image, 0, 0, 100, 100)
    gray = imlite.to_gray(array, source="RGB")

There is a command line too - run ``imlite --help``, or ``imlite doctor`` to
check that video support is working.
"""

from typing import Any

from imlite._version import __version__
from imlite.core.image import Image
from imlite.core.pipeline import load
from imlite.core.sequence import FrameSequence
from imlite.core.video import Video
from imlite.exceptions import (
    CropOutOfBoundsError,
    ImliteBackendError,
    ImliteColorSpaceError,
    ImliteDtypeError,
    ImliteError,
    ImliteFFmpegError,
    ImliteOpenError,
    ImliteReadError,
    ImliteShapeError,
    ImliteWriteError,
)
from imlite.ops.color import to_bgr, to_gray, to_hsv, to_lab, to_rgb
from imlite.ops.enhance import blur, brightness, contrast, invert, threshold
from imlite.ops.geometry import crop, flip, pad, resize, rotate, thumbnail
from imlite.ops.io import read_image
from imlite.ops.io import write_image as save
from imlite.ops.video_io import extract_frames, merge_frames
from imlite.ops.video_io import get_video_info as video_info
from imlite.utils.ffmpeg import ffmpeg_info
from imlite.utils.log import set_progress, set_verbosity
from imlite.utils.path import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "CropOutOfBoundsError",
    "FrameSequence",
    "Image",
    "ImliteBackendError",
    "ImliteColorSpaceError",
    "ImliteDtypeError",
    "ImliteError",
    "ImliteFFmpegError",
    "ImliteOpenError",
    "ImliteReadError",
    "ImliteShapeError",
    "ImliteWriteError",
    "Video",
    "__version__",
    "blur",
    "brightness",
    "contrast",
    "crop",
    "extract_frames",
    "ffmpeg_info",
    "flip",
    "invert",
    "load",
    "merge_frames",
    "pad",
    "read_frames",
    "read_image",
    "read_video",
    "resize",
    "rotate",
    "save",
    "set_progress",
    "set_verbosity",
    "threshold",
    "thumbnail",
    "to_bgr",
    "to_gray",
    "to_hsv",
    "to_lab",
    "to_rgb",
    "video_info",
]


def read_video(path: str) -> Video:
    """Open a video file and return a :class:`Video` handle.

    No frames are decoded and no metadata is read until you ask for it.

    Args:
        path: Path to the video file.

    Returns:
        A :class:`Video`.

    Example:
        >>> vid = imlite.read_video("clip.mp4")
        >>> vid.extract_frames(step=2).resize(640, 360).merge(fps=25).save("out.mp4")
    """
    return Video(path)


def read_frames(source: str | list[Any]) -> FrameSequence:
    """Load a directory or a list of frames as a :class:`FrameSequence`.

    Args:
        source: Either a directory path, whose image files are loaded in
            natural sort order, or a list of image file paths, ``Array``
            arrays or :class:`Image` objects.  Raw arrays are assumed to be
            BGR.

    Returns:
        A :class:`FrameSequence`.

    Raises:
        TypeError: If *source* is neither a path nor a list.

    Example:
        >>> seq = imlite.read_frames("frames/")
        >>> seq = imlite.read_frames(["a.png", "b.png"])
        >>> seq = imlite.read_frames([img1, img2])
    """
    from pathlib import Path

    if isinstance(source, (str, Path)):
        return FrameSequence.from_dir(str(source))
    if isinstance(source, list):
        from imlite.core.pipeline import _load_list

        return _load_list(source)
    raise TypeError(
        f"read_frames() expects a directory path or a list of frames, "
        f"got {type(source).__name__!r}. To wrap a single array, use read_frames([arr])."
    )
