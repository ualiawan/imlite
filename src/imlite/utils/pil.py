"""Bridge between imlite's numpy arrays and Pillow images.

Pillow is imlite's pixel-resampling backend (resize, rotate, blur).  These
helpers hide the two annoyances of round-tripping through it:

1. Pillow has no concept of a ``(H, W, 1)`` array - it wants ``(H, W)`` for
   single-channel images.  :func:`to_pil` squeezes and :func:`from_pil`
   restores the trailing axis.
2. Pillow labels 3-channel data ``"RGB"``.  imlite usually stores BGR.  Every
   operation routed through here is **channel-order agnostic** (it moves or
   blends pixels without reinterpreting them), so the label is cosmetic and
   the original ordering survives the round trip untouched.
"""

import numpy as np
from PIL import Image as PILImage

from imlite._typing import Array
from imlite.exceptions import ImliteShapeError

__all__ = ["from_pil", "to_pil"]


def to_pil(arr: Array) -> PILImage.Image:
    """Wrap a ``uint8`` numpy array as a :class:`PIL.Image.Image`.

    Args:
        arr: Array of shape ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` or
            ``(H, W, 4)``, dtype ``uint8``.

    Returns:
        A Pillow image in mode ``L``, ``RGB`` or ``RGBA``.

    Raises:
        ImliteShapeError: If *arr* does not have a supported shape.
    """
    if arr.ndim == 2:
        return PILImage.fromarray(arr, mode="L")
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1:
            return PILImage.fromarray(arr[:, :, 0], mode="L")
        if channels == 3:
            return PILImage.fromarray(arr, mode="RGB")
        if channels == 4:
            return PILImage.fromarray(arr, mode="RGBA")
    raise ImliteShapeError(
        f"Cannot convert an array of shape {arr.shape!r} to a Pillow image; "
        "expected (H, W), (H, W, 1), (H, W, 3) or (H, W, 4)."
    )


def from_pil(image: PILImage.Image, like: Array) -> Array:
    """Convert a Pillow image back to a numpy array shaped like *like*.

    Args:
        image: The Pillow image to convert.
        like: The array that was originally passed to :func:`to_pil`.  Only its
            dimensionality is used, so a ``(H, W, 1)`` input yields a
            ``(H, W, 1)`` output rather than a bare ``(H, W)`` one.

    Returns:
        A ``uint8`` array with the same number of axes as *like*.
    """
    out = np.asarray(image, dtype=np.uint8)
    if like.ndim == 3 and out.ndim == 2:
        out = out[:, :, np.newaxis]
    return out
