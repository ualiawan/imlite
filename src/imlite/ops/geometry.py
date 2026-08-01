"""Geometry transforms: crop, rotate, resize, flip, pad, thumbnail.

``crop``, ``flip`` and ``pad`` are pure numpy - they only move or add pixels.
``rotate`` and ``resize`` need real resampling and use Pillow, except for
rotations by exact multiples of 90 degrees, which are an exact ``np.rot90``.

All functions are decorated with :func:`~imlite.utils.validate.dispatch_type`,
so they accept either a raw ``numpy.ndarray`` or an
:class:`~imlite.core.image.Image` and return the same type.  They are
colour-space agnostic: BGR, RGB and grayscale data all pass through unchanged
apart from the geometric change itself.
"""

from collections.abc import Sequence

import numpy as np
from PIL import Image as PILImage

from imlite.exceptions import ImliteShapeError
from imlite.utils.pil import from_pil, to_pil
from imlite.utils.validate import (
    check_axis,
    check_crop_bounds,
    check_positive,
    dispatch_type,
)

__all__ = ["crop", "flip", "pad", "resize", "rotate", "thumbnail"]

Resampling = PILImage.Resampling

#: Names accepted by the ``resample`` argument of :func:`resize`.
RESAMPLE_FILTERS: dict[str, PILImage.Resampling] = {
    "nearest": Resampling.NEAREST,
    "box": Resampling.BOX,
    "bilinear": Resampling.BILINEAR,
    "bicubic": Resampling.BICUBIC,
    "lanczos": Resampling.LANCZOS,
}


@dispatch_type
def crop(img: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop *img* to the rectangle defined by (*x*, *y*, *width*, *height*).

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        x: Left edge of the crop box (pixels from left, 0-indexed).
        y: Top edge of the crop box (pixels from top, 0-indexed).
        width: Width of the crop box in pixels.
        height: Height of the crop box in pixels.

    Returns:
        The cropped image, in the same type as the input.

    Raises:
        CropOutOfBoundsError: If the crop box extends beyond the image.

    Example:
        >>> thumb = imlite.crop(img, x=10, y=10, width=200, height=200)
    """
    img_h, img_w = img.shape[:2]
    check_crop_bounds(img_h, img_w, x, y, width, height)
    return np.array(img[y : y + height, x : x + width])


@dispatch_type
def rotate(img: np.ndarray, angle: float, expand: bool = True) -> np.ndarray:
    """Rotate *img* counter-clockwise by *angle* degrees.

    Exact multiples of 90 degrees take a lossless ``np.rot90`` path - no
    resampling, no interpolation artefacts, and faster than a general warp.
    Every other angle is resampled bicubically by Pillow.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        angle: Rotation angle in degrees, counter-clockwise.
        expand: If ``True`` (default) the output canvas grows to contain the
            whole rotated image.  If ``False`` the output keeps the input size
            and the corners are filled with black.

    Returns:
        The rotated image, in the same type as the input.

    Example:
        >>> upright = imlite.rotate(img, 90)  # lossless
        >>> tilted = imlite.rotate(img, 30, expand=False)
    """
    if angle % 90 == 0:
        quarter_turns = int(angle // 90) % 4
        # .copy() rather than ascontiguousarray: for a zero-turn rotation the
        # latter would hand back the input array itself.
        return np.array(np.rot90(img, quarter_turns))

    rotated = to_pil(img).rotate(angle, resample=Resampling.BICUBIC, expand=expand)
    return from_pil(rotated, img)


@dispatch_type
def resize(
    img: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    keep_aspect: bool = False,
    resample: str = "auto",
) -> np.ndarray:
    """Resize *img* to (*width*, *height*).

    At least one of *width* or *height* must be given.  When only one is
    supplied the other is derived from the original aspect ratio.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        width: Target width in pixels, or ``None`` to infer from *height*.
        height: Target height in pixels, or ``None`` to infer from *width*.
        keep_aspect: If ``True``, scale the image to fit *inside* the
            (*width*, *height*) box without distorting it.  The result may be
            smaller than the box; no padding is added.
        resample: One of ``"nearest"``, ``"box"``, ``"bilinear"``,
            ``"bicubic"``, ``"lanczos"``, or ``"auto"`` (the default).
            ``"auto"`` picks ``"box"`` when shrinking - area averaging, which
            avoids aliasing - and ``"lanczos"`` when enlarging.

    Returns:
        The resized image, in the same type as the input.

    Raises:
        ValueError: If both *width* and *height* are ``None``, if either is
            non-positive, or if *resample* is not a recognised filter name.

    Example:
        >>> small = imlite.resize(img, width=320, height=240)
        >>> thumb = imlite.resize(img, width=128)  # height inferred
        >>> fitted = imlite.resize(img, 640, 480, keep_aspect=True)
    """
    orig_h, orig_w = img.shape[:2]

    if width is None:
        if height is None:
            raise ValueError(
                "resize() needs at least one of 'width' or 'height'. "
                "Pass one to scale proportionally, or both for an exact size."
            )
        check_positive(height, "height")
        width = max(1, round(orig_w * height / orig_h))
    elif height is None:
        check_positive(width, "width")
        height = max(1, round(orig_h * width / orig_w))
    else:
        check_positive(width, "width")
        check_positive(height, "height")

    if keep_aspect:
        scale = min(width / orig_w, height / orig_h)
        width = max(1, round(orig_w * scale))
        height = max(1, round(orig_h * scale))

    if (width, height) == (orig_w, orig_h):
        return np.array(img)

    shrinking = width < orig_w or height < orig_h
    resample_filter = _resolve_resample(resample, shrinking)
    return from_pil(to_pil(img).resize((width, height), resample=resample_filter), img)


@dispatch_type
def thumbnail(img: np.ndarray, size: int, resample: str = "auto") -> np.ndarray:
    """Scale *img* so its longest side is *size* pixels, preserving aspect ratio.

    Unlike :func:`resize` this never enlarges: an image already smaller than
    *size* is returned unchanged, which is what thumbnail generation wants.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        size: Length in pixels of the longest side of the result.
        resample: Resampling filter name - see :func:`resize`.

    Returns:
        The scaled image, in the same type as the input.

    Raises:
        ValueError: If *size* is not positive.

    Example:
        >>> imlite.load("photo.jpg").thumbnail(256).save("thumb.jpg")
    """
    check_positive(size, "size")
    orig_h, orig_w = img.shape[:2]
    longest = max(orig_h, orig_w)
    if longest <= size:
        return np.array(img)

    scale = size / longest
    width = max(1, round(orig_w * scale))
    height = max(1, round(orig_h * scale))
    resample_filter = _resolve_resample(resample, shrinking=True)
    return from_pil(to_pil(img).resize((width, height), resample=resample_filter), img)


@dispatch_type
def flip(img: np.ndarray, axis: str = "h") -> np.ndarray:
    """Flip *img* along the given axis.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        axis: One of ``"h"`` / ``"horizontal"`` (left-right), ``"v"`` /
            ``"vertical"`` (top-bottom), or ``"both"``.

    Returns:
        The flipped image, in the same type as the input.

    Raises:
        ValueError: If *axis* is not one of the recognised values.

    Example:
        >>> mirror = imlite.flip(img, "h")
    """
    check_axis(axis)
    axis_lower = axis.lower()
    if axis_lower in ("h", "horizontal"):
        return np.array(img[:, ::-1])
    if axis_lower in ("v", "vertical"):
        return np.array(img[::-1, :])
    return np.array(img[::-1, ::-1])


@dispatch_type
def pad(
    img: np.ndarray,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    color: int | Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    """Add a constant-colour border around *img*.

    Args:
        img: Input image as ``np.ndarray`` or :class:`~imlite.core.image.Image`.
        top: Pixels to add on the top edge.
        bottom: Pixels to add on the bottom edge.
        left: Pixels to add on the left edge.
        right: Pixels to add on the right edge.
        color: Border fill.  Either a single intensity or a per-channel
            sequence, interpreted in the image's **current** colour space -
            so ``(255, 0, 0)`` is blue for BGR data and red for RGB data.
            A 3-tuple applied to an RGBA image gains a fully opaque alpha.

    Returns:
        The padded image, in the same type as the input.

    Raises:
        ValueError: If any border width is negative, or if *color* has the
            wrong number of components for the image.

    Example:
        >>> boxed = imlite.pad(img, top=10, bottom=10, left=10, right=10)
        >>> white = imlite.pad(img, top=5, bottom=5, color=(255, 255, 255))
    """
    for name, value in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        if value < 0:
            raise ValueError(f"Padding '{name}' must be >= 0, got {value}.")

    height, width = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    new_h, new_w = height + top + bottom, width + left + right

    fill = _fill_value(color, channels)
    if img.ndim == 2:
        out = np.full((new_h, new_w), fill[0], dtype=img.dtype)
    else:
        out = np.empty((new_h, new_w, channels), dtype=img.dtype)
        out[:] = fill
    out[top : top + height, left : left + width] = img
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_resample(name: str, shrinking: bool) -> PILImage.Resampling:
    """Map a resample filter name (possibly ``"auto"``) to a Pillow filter."""
    if name == "auto":
        return Resampling.BOX if shrinking else Resampling.LANCZOS
    try:
        return RESAMPLE_FILTERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown resample filter {name!r}. Choose from {[*sorted(RESAMPLE_FILTERS), 'auto']}."
        ) from None


def _fill_value(color: int | Sequence[int], channels: int) -> np.ndarray:
    """Normalise a fill colour to exactly *channels* components."""
    if isinstance(color, (int, np.integer)):
        return np.full(channels, int(color), dtype=np.uint8)

    values = list(color)
    if len(values) == channels:
        return np.array(values, dtype=np.uint8)
    if len(values) == 1:
        return np.full(channels, values[0], dtype=np.uint8)
    if len(values) == 3 and channels == 4:
        return np.array([*values, 255], dtype=np.uint8)  # opaque alpha
    if len(values) == 3 and channels == 1:
        raise ImliteShapeError(
            f"Cannot pad a single-channel image with the 3-component colour {tuple(values)!r}. "
            "Pass a single intensity, e.g. color=0."
        )
    raise ValueError(
        f"Fill colour {tuple(values)!r} has {len(values)} components but the image has {channels}."
    )
