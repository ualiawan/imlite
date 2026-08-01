"""Input validation utilities and the ops dispatch decorators.

Two decorators let every function in ``imlite.ops`` accept either a raw
``numpy.ndarray`` or an :class:`~imlite.core.image.Image` and return the same
type it was given:

- :func:`dispatch_type` for colour-space-preserving ops (geometry, filters).
- :func:`color_op` for conversions, which additionally re-tag the result.

The validation helpers raise descriptive imlite exceptions *before* the backend
(Pillow, imageio) is ever called.
"""

import functools
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, Protocol, cast

import numpy as np

from imlite._typing import Array, ImageLike
from imlite.exceptions import CropOutOfBoundsError, ImliteShapeError

__all__ = [
    "ColorOp",
    "check_axis",
    "check_crop_bounds",
    "check_ndarray_image_shape",
    "check_positive",
    "color_op",
    "dispatch_type",
    "require_ndarray",
]

_P = ParamSpec("_P")

_VALID_AXES = frozenset({"h", "horizontal", "v", "vertical", "both"})


class ColorOp(Protocol):
    """Call signature of a decorated colour conversion in ``imlite.ops.color``."""

    __name__: str

    def __call__(self, img: ImageLike, *, source: str | None = None) -> ImageLike:
        """Convert *img*, reading its space from its tag or from *source*."""
        ...


# ---------------------------------------------------------------------------
# Dispatch decorators
# ---------------------------------------------------------------------------


def dispatch_type(
    fn: Callable[Concatenate[Array, _P], Array],
) -> Callable[Concatenate[ImageLike, _P], ImageLike]:
    """Make an ops function transparently accept ``Image`` or ``np.ndarray``.

    When called with an :class:`~imlite.core.image.Image`:

    1. The underlying array is passed to *fn* (no copy).
    2. The result is re-wrapped as a new ``Image`` with the **same**
       ``color_space`` and ``path=None``.

    When called with a plain ``Array`` the array is shape-checked and
    *fn*'s result is returned unchanged.

    Args:
        fn: An ops function whose first argument is an ``Array`` and which
            returns a freshly allocated ``Array``.

    Returns:
        The wrapped function.

    Example:
        >>> @dispatch_type
        ... def crop(img: Array, x, y, w, h) -> Array:
        ...     return img[y : y + h, x : x + w].copy()
        >>> crop(my_image, 0, 0, 100, 100)  # doctest: +SKIP  -> Image
        >>> crop(my_array, 0, 0, 100, 100)  # doctest: +SKIP  -> ndarray
    """

    @functools.wraps(fn)
    def wrapper(img: ImageLike, *args: _P.args, **kwargs: _P.kwargs) -> ImageLike:
        from imlite.core.image import Image

        if isinstance(img, Image):
            result = fn(img.array, *args, **kwargs)
            # copy=False: every op allocates its own result, so there is no
            # caller-owned buffer to defend against here.
            return Image(result, color_space=img.color_space, copy=False)
        require_ndarray(img)
        check_ndarray_image_shape(img)
        return fn(img, *args, **kwargs)

    # functools.wraps returns a _Wrapped[...] that mypy will not unify with a
    # plain Callable, even though the call signature is identical.
    return cast("Callable[Concatenate[ImageLike, _P], ImageLike]", wrapper)


def color_op(target: str) -> Callable[[Callable[[Array, str], Array]], ColorOp]:
    """Build a decorator for a colour-space conversion in ``ops/color.py``.

    The wrapped function is called as ``fn(array, source_space)`` and must
    return an array in *target* space.  The decorator handles both call styles:

    - Given an :class:`~imlite.core.image.Image`, the image's own
      ``color_space`` is used as the source and the result is tagged *target*.
    - Given a bare ``Array`` there is no tag to read, so the source space
      comes from the caller's ``source=`` keyword (default ``"BGR"``, matching
      the rest of the library).

    Args:
        target: The colour-space tag the wrapped function produces, e.g.
            ``"RGB"`` or ``"GRAY"``.

    Returns:
        A decorator that adapts the conversion function.

    Example:
        >>> @color_op("GRAY")
        ... def to_gray(arr: Array, source: str) -> Array: ...
        >>> to_gray(my_image)                  # doctest: +SKIP  -> Image(GRAY)
        >>> to_gray(my_array, source="RGB")    # doctest: +SKIP  -> ndarray
    """

    def decorator(fn: Callable[[Array, str], Array]) -> ColorOp:
        @functools.wraps(fn)
        def wrapper(img: ImageLike, *, source: str | None = None) -> ImageLike:
            from imlite.core.image import Image

            if isinstance(img, Image):
                return Image(
                    fn(img.array, source or img.color_space), color_space=target, copy=False
                )
            require_ndarray(img)
            check_ndarray_image_shape(img)
            return fn(img, source or "BGR")

        return cast("ColorOp", wrapper)

    return decorator


# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------


def require_ndarray(arr: Any, name: str = "img") -> None:
    """Raise :exc:`TypeError` if *arr* is not a ``numpy.ndarray``.

    Args:
        arr: Value to check.
        name: Argument name used in the error message.

    Raises:
        TypeError: If *arr* is not an ``Array``.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"Expected an imlite Image or numpy.ndarray for '{name}', got {type(arr).__name__!r}."
        )


# ---------------------------------------------------------------------------
# Bounds / value checks
# ---------------------------------------------------------------------------


def check_crop_bounds(img_h: int, img_w: int, x: int, y: int, w: int, h: int) -> None:
    """Raise :exc:`~imlite.exceptions.CropOutOfBoundsError` if a crop box is invalid.

    Args:
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        x: Left edge of the crop box.
        y: Top edge of the crop box.
        w: Crop box width.
        h: Crop box height.

    Raises:
        CropOutOfBoundsError: If the box extends beyond the image or has
            non-positive dimensions.
    """
    if w <= 0 or h <= 0:
        raise CropOutOfBoundsError(f"Crop dimensions must be positive, got width={w}, height={h}.")
    if x < 0 or y < 0:
        raise CropOutOfBoundsError(f"Crop origin must be non-negative, got x={x}, y={y}.")
    if x + w > img_w or y + h > img_h:
        raise CropOutOfBoundsError(
            f"Crop box (x={x}, y={y}, w={w}, h={h}) exceeds image size ({img_w}x{img_h})."
        )


def check_positive(value: float, name: str) -> None:
    """Raise :exc:`ValueError` if *value* is not strictly positive.

    Args:
        value: Numeric value to check.
        name: Argument name used in the error message.

    Raises:
        ValueError: If *value* is zero or negative.
    """
    if value <= 0:
        raise ValueError(f"'{name}' must be > 0, got {value}.")


def check_axis(axis: str) -> None:
    """Raise :exc:`ValueError` if *axis* is not a recognised flip axis.

    Args:
        axis: Axis string to validate.

    Raises:
        ValueError: If *axis* is not one of ``"h"``, ``"horizontal"``,
            ``"v"``, ``"vertical"`` or ``"both"``.
    """
    if not isinstance(axis, str) or axis.lower() not in _VALID_AXES:
        raise ValueError(f"Invalid flip axis {axis!r}. Choose from: {sorted(_VALID_AXES)}.")


def check_ndarray_image_shape(arr: Array, name: str = "img") -> None:
    """Raise :exc:`~imlite.exceptions.ImliteShapeError` if *arr* is not image-shaped.

    Valid shapes are ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` and ``(H, W, 4)``.

    Args:
        arr: Array to check.
        name: Argument name used in the error message.

    Raises:
        ImliteShapeError: If the shape is not valid.
    """
    if arr.ndim == 2:
        return
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return
    raise ImliteShapeError(
        f"'{name}' has shape {arr.shape!r}; expected (H, W), (H, W, 1), (H, W, 3) or (H, W, 4)."
    )
