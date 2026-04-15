"""Custom exception hierarchy for imlite.

All exceptions inherit from ``ImliteError`` so callers can catch the whole
family with a single ``except ImliteError`` clause.
"""


class ImliteError(Exception):
    """Base class for all imlite exceptions."""


class ImliteOpenError(ImliteError):
    """Raised when ``imlite.load()`` cannot determine the type of the source."""


class ImliteReadError(ImliteError):
    """Raised when an image or video file cannot be read."""


class ImliteWriteError(ImliteError):
    """Raised when an image or video file cannot be written."""


class ImliteShapeError(ImliteError):
    """Raised when an operation receives an array with an incompatible shape."""


class CropOutOfBoundsError(ImliteShapeError):
    """Raised when a crop rectangle extends beyond the image boundaries.

    Example::

        imlite.crop(img, x=0, y=0, width=9999, height=9999)
        # -> CropOutOfBoundsError: Crop box (w=9999, h=9999) exceeds image size (300x200)
    """


class ImliteColorSpaceError(ImliteError):
    """Raised when an image has an unexpected or invalid color space tag."""


class ImliteBackendError(ImliteError):
    """Raised when an underlying backend (cv2, imageio) raises an exception.

    The original exception is always chained via ``raise ... from original``.
    """
