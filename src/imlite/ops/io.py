"""Image file reading and writing.

Backend: ``imageio`` v3, which delegates to Pillow for the common formats.
There is no OpenCV dependency anywhere in imlite.

On disk, images are RGB.  Internally, imlite stores BGR (OpenCV's convention,
and what its users expect).  The channel swap happens here so no user-facing
code has to think about it.
"""

import logging
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from imlite.core.image import Image
from imlite.exceptions import ImliteReadError, ImliteWriteError
from imlite.utils.dtype import as_uint8
from imlite.utils.path import ensure_dir

log = logging.getLogger(__name__)

__all__ = ["read_image", "write_image"]

#: Extensions whose plugin accepts a ``quality`` keyword.
_QUALITY_FORMATS = frozenset({".jpg", ".jpeg", ".webp"})


def read_image(path: str) -> Image:
    """Read an image file from disk.

    16-bit and floating-point files are converted to 8-bit by
    :func:`~imlite.utils.dtype.as_uint8` rather than truncated.

    Args:
        path: Path to the image file.

    Returns:
        An :class:`~imlite.core.image.Image` tagged ``"BGR"`` for colour input
        or ``"GRAY"`` for single-channel input, with ``path`` set.

    Raises:
        ImliteReadError: If the file does not exist or cannot be decoded.

    Example:
        >>> img = imlite.read_image("photo.jpg")
    """
    path_str = str(path)
    log.debug("Reading image: %s", path_str)

    if not Path(path_str).exists():
        raise ImliteReadError(f"Image file not found: {path_str!r}")

    try:
        raw = iio.imread(path_str)
    except Exception as exc:
        raise ImliteReadError(
            f"Could not read image {path_str!r}. "
            "Check that it is a supported, non-corrupt image format."
        ) from exc

    if raw is None:
        raise ImliteReadError(f"Could not read image {path_str!r}: the decoder returned no data.")

    pixels = as_uint8(np.asarray(raw))

    if pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        data, color_space = _rgb_to_bgr(pixels), "BGR"
    else:
        data, color_space = pixels, "GRAY"

    log.debug("Read %s: shape=%s space=%s", path_str, data.shape, color_space)
    return Image(data, color_space=color_space, path=path_str, copy=False)


def write_image(img: Image | np.ndarray, path: str, quality: int = 95) -> None:
    """Write *img* to disk.

    Args:
        img: An :class:`~imlite.core.image.Image`, or a raw ``np.ndarray``
            which is assumed to be **BGR** (imlite's internal convention).
        path: Destination path.  The format is inferred from the extension and
            missing parent directories are created.
        quality: JPEG/WebP quality in ``0-100``.  For PNG it is mapped to a
            compression level (higher quality means less compression).

    Raises:
        ImliteWriteError: If the file cannot be written.

    Example:
        >>> imlite.load("photo.jpg").rotate(90).save("rotated.jpg")
    """
    path_str = str(path)
    extension = Path(path_str).suffix.lower()

    if not extension:
        raise ImliteWriteError(
            f"Cannot infer an image format for {path_str!r}: the path has no file extension. "
            "Use something like 'out.png' or 'out.jpg'."
        )

    pixels = _as_rgb_for_disk(img)
    ensure_dir(Path(path_str).parent)

    kwargs: dict[str, Any] = {}
    if extension in _QUALITY_FORMATS:
        kwargs["quality"] = max(0, min(100, quality))
    elif extension == ".png":
        kwargs["compress_level"] = max(0, min(9, 9 - quality // 11))

    log.debug("Writing image to %s (shape=%s)", path_str, pixels.shape)
    try:
        iio.imwrite(path_str, pixels, **kwargs)
    except Exception as exc:
        raise ImliteWriteError(
            f"Could not write image to {path_str!r}. "
            f"Check that the directory is writable and that {extension!r} is a supported format."
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rgb_to_bgr(pixels: np.ndarray) -> np.ndarray:
    """Swap R and B, leaving any alpha channel last.

    A plain ``[..., ::-1]`` would reverse all four channels of an RGBA image
    and yield ABGR, so alpha is handled explicitly.
    """
    if pixels.shape[2] == 4:
        return np.ascontiguousarray(pixels[..., [2, 1, 0, 3]])
    return np.ascontiguousarray(pixels[..., ::-1])


def _as_rgb_for_disk(img: Image | np.ndarray) -> np.ndarray:
    """Return *img* as an RGB (or grayscale) array ready for ``imageio.imwrite``."""
    if isinstance(img, Image):
        # Colour spaces other than BGR/RGB/GRAY have no meaning to an image
        # viewer, so normalise them back to something displayable.
        source = img if img.color_space in ("BGR", "RGB", "GRAY") else img.to_bgr()
        pixels = source.array
        needs_swap = source.color_space == "BGR"
    else:
        pixels = as_uint8(np.asarray(img))
        needs_swap = True  # bare arrays are BGR by imlite convention

    if pixels.ndim == 3 and pixels.shape[2] == 1:
        return pixels[:, :, 0]
    if needs_swap and pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        return _rgb_to_bgr(pixels)  # the swap is its own inverse
    return pixels
