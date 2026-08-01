"""Custom exception hierarchy for imlite.

All exceptions inherit from :class:`ImliteError` so callers can catch the whole
family with a single ``except ImliteError`` clause::

    ImliteError
    |-- ImliteOpenError
    |-- ImliteReadError
    |-- ImliteWriteError
    |-- ImliteShapeError
    |   `-- CropOutOfBoundsError
    |-- ImliteDtypeError
    |-- ImliteColorSpaceError
    |-- ImliteBackendError
    `-- ImliteFFmpegError
"""

__all__ = [
    "CropOutOfBoundsError",
    "ImliteBackendError",
    "ImliteColorSpaceError",
    "ImliteDtypeError",
    "ImliteError",
    "ImliteFFmpegError",
    "ImliteOpenError",
    "ImliteReadError",
    "ImliteShapeError",
    "ImliteWriteError",
]


class ImliteError(Exception):
    """Base class for all imlite exceptions."""


class ImliteOpenError(ImliteError):
    """Raised when :func:`imlite.load` cannot determine the type of the source."""


class ImliteReadError(ImliteError):
    """Raised when an image or video file cannot be read."""


class ImliteWriteError(ImliteError):
    """Raised when an image or video file cannot be written."""


class ImliteShapeError(ImliteError):
    """Raised when an operation receives an array with an incompatible shape."""


class CropOutOfBoundsError(ImliteShapeError):
    """Raised when a crop rectangle extends beyond the image boundaries.

    Example:
        >>> imlite.crop(img, x=0, y=0, width=9999, height=9999)
        Traceback (most recent call last):
        CropOutOfBoundsError: Crop box (x=0, y=0, w=9999, h=9999) exceeds image size (300x200).
    """


class ImliteDtypeError(ImliteError):
    """Raised when pixel data cannot be interpreted as 8-bit unambiguously.

    imlite stores every image as ``uint8``.  Common input dtypes are converted
    automatically (see :func:`imlite.utils.dtype.as_uint8`); anything whose
    intended range cannot be inferred raises this rather than being silently
    truncated.
    """


class ImliteColorSpaceError(ImliteError):
    """Raised when an image has an unexpected or invalid colour-space tag."""


class ImliteBackendError(ImliteError):
    """Raised when an underlying backend (Pillow, imageio) raises an exception.

    The original exception is always chained via ``raise ... from original``.
    """


class ImliteFFmpegError(ImliteError):
    """Raised when no usable ffmpeg binary can be found.

    ``imageio-ffmpeg`` ships a static ffmpeg for every mainstream platform, so
    this normally only appears on platforms it has no wheel for (musl/Alpine,
    ppc64le, FreeBSD).  The message always includes platform-specific
    installation instructions.
    """
