"""Shared type aliases.

Kept in its own module so the alias can be imported anywhere without dragging
:class:`~imlite.core.image.Image` into a circular import at runtime.
"""

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    from imlite.core.image import Image

__all__ = ["Array", "ImageLike", "as_ndarray"]

#: A numpy array of any dtype. Spelled via ``numpy.typing`` rather than a
#: bare ``np.ndarray`` because only numpy >= 2.4 gives ``ndarray`` PEP 696
#: parameter defaults; on older versions ``mypy --strict`` rejects the bare
#: form with ``type-arg``, and imlite supports numpy >= 1.23.
Array: TypeAlias = npt.NDArray[Any]

#: Anything imlite's operations accept as an image: an :class:`Image` wrapper
#: or a raw ``numpy`` array.  Operations return whichever type they were given.
ImageLike: TypeAlias = "Image | Array"


def as_ndarray(value: Any) -> Array:
    """Narrow a numpy expression that the stubs type as ``Any``.

    numpy types the result of indexing, of arithmetic on ``dtype[Any]`` arrays,
    and of ``astype``/``clip`` as ``Any``.  Under ``mypy --strict`` that trips
    ``warn_return_any`` at every ``return`` in ``imlite.ops``.  Routing those
    returns through here keeps the narrowing in one reviewable place instead of
    scattering ``# type: ignore`` comments across the codebase.

    This is a static-typing no-op: the value is already an ``ndarray`` at
    runtime and nothing is copied or converted.

    Args:
        value: A numpy expression the stubs could not type precisely.

    Returns:
        The same object, typed as ``np.ndarray``.
    """
    return cast("Array", value)
