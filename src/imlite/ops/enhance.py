"""Pixel-value operations: blur, brightness, contrast, invert, threshold.

Unlike ``ops/geometry.py`` these change pixel *values* rather than pixel
positions.  They are still colour-space agnostic - brightness and contrast
scale every channel identically, and blur convolves each channel separately -
with one exception: :func:`threshold` needs a notion of intensity, so it
converts colour input to grayscale first using the image's own colour space.

All functions accept an :class:`~imlite.core.image.Image` or a raw
``np.ndarray`` and return the same type.
"""

import numpy as np
from PIL import ImageFilter

from imlite._typing import ImageLike, as_ndarray
from imlite.utils.pil import from_pil, to_pil
from imlite.utils.validate import dispatch_type

__all__ = ["blur", "brightness", "contrast", "invert", "threshold"]


@dispatch_type
def blur(img: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Apply a Gaussian blur to *img*.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        radius: Standard deviation of the Gaussian kernel, in pixels.  Larger
            values blur more.  ``0`` returns the image unchanged.

    Returns:
        The blurred image, in the same type as the input.

    Raises:
        ValueError: If *radius* is negative.

    Example:
        >>> soft = imlite.blur(img, radius=3)
        >>> imlite.load("photo.jpg").blur(5).save("soft.jpg")
    """
    if radius < 0:
        raise ValueError(f"'radius' must be >= 0, got {radius}.")
    if radius == 0:
        return np.array(img)
    return from_pil(to_pil(img).filter(ImageFilter.GaussianBlur(radius)), img)


@dispatch_type
def brightness(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Scale the brightness of *img*.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        factor: Multiplier applied to every channel.  ``1.0`` leaves the image
            unchanged, ``0.0`` produces black, ``2.0`` doubles brightness.
            Results are clipped to ``0-255``.

    Returns:
        The adjusted image, in the same type as the input.

    Raises:
        ValueError: If *factor* is negative.

    Example:
        >>> brighter = imlite.brightness(img, 1.3)
    """
    if factor < 0:
        raise ValueError(f"'factor' must be >= 0, got {factor}.")
    scaled = img.astype(np.float32) * factor
    return as_ndarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8))


@dispatch_type
def contrast(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Scale the contrast of *img* around its own mean intensity.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        factor: Contrast multiplier.  ``1.0`` leaves the image unchanged,
            ``0.0`` flattens it to a uniform grey, values above ``1.0`` push
            pixels away from the mean.  Results are clipped to ``0-255``.

    Returns:
        The adjusted image, in the same type as the input.

    Raises:
        ValueError: If *factor* is negative.

    Example:
        >>> punchy = imlite.contrast(img, 1.5)
    """
    if factor < 0:
        raise ValueError(f"'factor' must be >= 0, got {factor}.")
    pixels = img.astype(np.float32)
    midpoint = float(pixels.mean())
    adjusted = (pixels - midpoint) * factor + midpoint
    return as_ndarray(np.clip(np.rint(adjusted), 0, 255).astype(np.uint8))


@dispatch_type
def invert(img: np.ndarray) -> np.ndarray:
    """Invert *img* photographically (``255 - value``).

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.

    Returns:
        The inverted image, in the same type as the input.  An alpha channel,
        if present, is inverted too - split it off first if that is not what
        you want.

    Example:
        >>> negative = imlite.invert(img)
    """
    return as_ndarray(np.subtract(255, img, dtype=np.uint8))


def threshold(
    img: ImageLike,
    value: int = 128,
    max_value: int = 255,
    invert_output: bool = False,
) -> ImageLike:
    """Binarise *img* at an intensity cut-off.

    Colour input is converted to grayscale first, so the result is always
    single-channel with shape ``(H, W, 1)``.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
            A bare array is assumed to be BGR, matching the rest of imlite.
        value: Cut-off intensity in ``0-255``.  Pixels **above** it become
            *max_value*; the rest become ``0``.
        max_value: Value assigned to pixels above the cut-off.
        invert_output: Swap the two outputs, so pixels above the cut-off become
            ``0`` instead.

    Returns:
        A single-channel binary image, in the same type as the input.  When
        given an :class:`~imlite.core.image.Image` the result is tagged
        ``color_space="GRAY"``.

    Raises:
        ValueError: If *value* or *max_value* is outside ``0-255``.

    Example:
        >>> mask = imlite.load("scan.png").threshold(200)
    """
    if not 0 <= value <= 255:
        raise ValueError(f"'value' must be in 0-255, got {value}.")
    if not 0 <= max_value <= 255:
        raise ValueError(f"'max_value' must be in 0-255, got {max_value}.")

    from imlite.core.image import Image
    from imlite.ops.color import to_gray

    gray = to_gray(img)
    array = gray.array if isinstance(gray, Image) else gray

    above = array > value
    if invert_output:
        above = ~above
    binary = as_ndarray(np.where(above, np.uint8(max_value), np.uint8(0)).astype(np.uint8))

    if isinstance(img, Image):
        return Image.from_numpy(binary, color_space="GRAY")
    return binary
