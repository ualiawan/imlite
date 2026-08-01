"""Pixel dtype normalisation.

imlite stores every image as ``uint8``.  Users arrive with 16-bit TIFFs,
normalised float arrays from ML pipelines, and boolean masks, so a single
``astype(np.uint8)`` would silently corrupt data (``uint16 1000`` wraps to
``232``; ``float 1.0`` collapses to ``1``).

:func:`as_uint8` converts the cases whose intended range is unambiguous and
raises :class:`~imlite.exceptions.ImliteDtypeError` for the rest, so nothing is
ever quietly mangled.

======================  =====================================================
Input dtype             Conversion
======================  =====================================================
``uint8``               returned unchanged
``bool``                ``False`` -> 0, ``True`` -> 255
``uint16``              scaled down by 257 (full 16-bit range -> full 8-bit)
integer, values 0-255   cast directly
float, values 0.0-1.0   scaled by 255 and rounded
float, values 0.0-255.0 rounded
anything else           :class:`~imlite.exceptions.ImliteDtypeError`
======================  =====================================================
"""

import logging

import numpy as np

from imlite._typing import as_ndarray
from imlite.exceptions import ImliteDtypeError

log = logging.getLogger(__name__)

__all__ = ["as_uint8"]

_HINT = (
    "Convert it yourself first, e.g. "
    "`arr = np.clip(arr, 0, 255).astype('uint8')` or "
    "`arr = (arr / arr.max() * 255).astype('uint8')`."
)


def as_uint8(arr: np.ndarray) -> np.ndarray:
    """Return *arr* as a ``uint8`` array, converting by documented rules.

    Args:
        arr: Any numeric or boolean numpy array.

    Returns:
        A ``uint8`` array.  The input is returned unchanged when it is already
        ``uint8`` (no copy is made).

    Raises:
        ImliteDtypeError: If the array's intended 8-bit range cannot be
            inferred - for example a float array containing negative values or
            values above 255, or an integer array outside ``0..255``.

    Example:
        >>> as_uint8(np.ones((2, 2), dtype=np.float32))       # 0.0-1.0 range
        array([[255, 255],
               [255, 255]], dtype=uint8)
    """
    if arr.dtype == np.uint8:
        return arr

    if arr.dtype == np.bool_:
        log.debug("Converting bool array to uint8 (False->0, True->255).")
        return (arr.astype(np.uint8)) * 255

    if arr.size == 0:
        return arr.astype(np.uint8)

    if arr.dtype == np.uint16:
        log.debug("Down-scaling uint16 array to uint8 (divide by 257).")
        return (arr // 257).astype(np.uint8)

    if np.issubdtype(arr.dtype, np.floating):
        return _float_to_uint8(arr)

    if np.issubdtype(arr.dtype, np.integer):
        low, high = int(arr.min()), int(arr.max())
        if low >= 0 and high <= 255:
            return arr.astype(np.uint8)
        raise ImliteDtypeError(
            f"Cannot convert {arr.dtype} image data to uint8: values span "
            f"{low}..{high}, which is outside the 8-bit range 0..255. {_HINT}"
        )

    raise ImliteDtypeError(f"Unsupported image dtype {arr.dtype!r}. imlite works on uint8 pixels.")


def _float_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale a float array into ``uint8``, inferring whether it is 0-1 or 0-255."""
    finite = np.isfinite(arr)
    if not finite.all():
        raise ImliteDtypeError(
            f"Cannot convert {arr.dtype} image data to uint8: it contains NaN or infinity. {_HINT}"
        )

    low, high = float(arr.min()), float(arr.max())
    if low < 0.0:
        raise ImliteDtypeError(
            f"Cannot convert {arr.dtype} image data to uint8: it contains negative "
            f"values (minimum {low:g}). {_HINT}"
        )

    if high <= 1.0:
        log.debug("Treating float array as normalised 0.0-1.0 and scaling by 255.")
        return as_ndarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8))

    if high <= 255.0:
        log.debug("Treating float array as 0.0-255.0 and rounding.")
        return as_ndarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))

    raise ImliteDtypeError(
        f"Cannot convert {arr.dtype} image data to uint8: values reach {high:g}, "
        f"which is above the 8-bit maximum of 255. {_HINT}"
    )
