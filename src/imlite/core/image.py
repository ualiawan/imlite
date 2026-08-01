"""The :class:`Image` class - imlite's fundamental data unit.

An ``Image`` wraps one ``numpy.ndarray`` and adds three things:

- **A colour-space tag**, so the library always knows whether pixels are BGR,
  RGB, GRAY, HSV or LAB without the user having to remember.
- **Immutable transforms**: every operation returns a new ``Image``; the
  original is never modified.
- **Chaining**: those transforms delegate to ``imlite.ops.*`` and return
  ``Image``, so calls compose left to right.

Pixels are always stored as ``uint8``.  Other dtypes are converted on
construction by :func:`~imlite.utils.dtype.as_uint8`, which handles the
unambiguous cases (bool masks, 16-bit images, normalised floats) and raises
rather than silently truncating anything else.
"""

# PEP 563: Image methods return Image, and to_pil() annotates a
# TYPE_CHECKING-only Pillow import. Both would NameError at class-body
# execution without lazy annotations. Do not remove.
from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from imlite._typing import Array, as_ndarray
from imlite.exceptions import ImliteShapeError
from imlite.utils.dtype import as_uint8

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image as PILImage

log = logging.getLogger(__name__)

__all__ = ["Image"]

#: Colour-space tags an :class:`Image` may carry.
VALID_COLOR_SPACES = frozenset({"BGR", "RGB", "GRAY", "HSV", "LAB"})

FlipAxis = Literal["h", "horizontal", "v", "vertical", "both"]


class Image:
    """A single in-memory image.

    Args:
        data: Pixel data. Accepted shapes are ``(H, W)``, ``(H, W, 1)``,
            ``(H, W, 3)`` and ``(H, W, 4)``.  Non-``uint8`` dtypes are
            converted per :func:`~imlite.utils.dtype.as_uint8`.
        color_space: Colour space of *data* - ``"BGR"`` (the default, matching
            OpenCV and imlite's internal convention), ``"RGB"``, ``"GRAY"``,
            ``"HSV"`` or ``"LAB"``.
        path: Source file path, or ``None`` for images created in memory.
        copy: Take a private copy of *data* (the default), so later writes to
            the caller's array cannot change this ``Image``.  Pass ``False``
            to adopt the buffer instead - faster, but only safe when nothing
            else will ever write to it.  imlite's own operations use
            ``copy=False`` because every one of them allocates a fresh result.

    Raises:
        TypeError: If *data* is not a ``numpy.ndarray``.
        ImliteShapeError: If *data* does not have a supported shape.
        ImliteDtypeError: If *data*'s dtype cannot be mapped to ``uint8``.
        ValueError: If *color_space* is not recognised.

    Example:
        >>> img = imlite.load("photo.jpg")
        >>> img.crop(0, 0, 200, 200).rotate(90).save("thumb.jpg")
    """

    __slots__ = ("_color_space", "_data", "_path")

    def __init__(
        self,
        data: Array,
        color_space: str = "BGR",
        path: str | None = None,
        *,
        copy: bool = True,
    ) -> None:
        _validate_array(data)
        if color_space not in VALID_COLOR_SPACES:
            raise ValueError(
                f"Unknown color_space {color_space!r}. Choose from {sorted(VALID_COLOR_SPACES)}."
            )
        converted = as_uint8(data)
        if copy and converted is data:
            # as_uint8 returned the caller's own buffer untouched; take our own
            # so their later writes cannot reach through into this Image.
            converted = data.copy()
        self._data: Array = np.ascontiguousarray(converted)
        self._color_space: str = color_space
        self._path: str | None = path

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        arr: Array,
        color_space: str = "BGR",
        path: str | None = None,
        *,
        copy: bool = True,
    ) -> Image:
        """Wrap a numpy array as an ``Image``.

        Args:
            arr: Pixel array.
            color_space: Colour-space tag (default ``"BGR"``).
            path: Optional source path.
            copy: Take a private copy of *arr* (the default).  See
                :class:`Image`.

        Returns:
            A new :class:`Image`.

        Example:
            >>> import numpy as np, imlite
            >>> imlite.Image.from_numpy(np.zeros((100, 100, 3), dtype=np.uint8))
            Image(in-memory, shape=(100, 100, 3), color_space='BGR')
        """
        return cls(arr, color_space=color_space, path=path, copy=copy)

    @classmethod
    def from_file(cls, path: str) -> Image:
        """Read an image file from disk.

        A thin wrapper around :func:`~imlite.ops.io.read_image`.

        Args:
            path: Path to the image file.

        Returns:
            A new :class:`Image` with ``color_space="BGR"`` and ``path`` set.

        Example:
            >>> img = imlite.Image.from_file("photo.jpg")
        """
        from imlite.ops.io import read_image

        return read_image(path)

    @classmethod
    def from_pil(cls, image: PILImage.Image) -> Image:
        """Wrap a :class:`PIL.Image.Image`.

        The image is converted to ``RGB`` (or ``RGBA`` when it has
        transparency) and tagged accordingly, so no channel swap is implied.

        Args:
            image: Any Pillow image.

        Returns:
            A new :class:`Image` with ``color_space="RGB"``.

        Example:
            >>> from PIL import Image as PILImage
            >>> imlite.Image.from_pil(PILImage.open("photo.jpg"))  # doctest: +SKIP
        """
        mode = "RGBA" if image.mode in ("RGBA", "LA", "PA") else "RGB"
        return cls(np.asarray(image.convert(mode)), color_space="RGB")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> Array:
        """A copy of the pixel array - safe to mutate without affecting this ``Image``."""
        return as_ndarray(self._data.copy())

    @property
    def array(self) -> Array:
        """A read-only view of the pixel array - no copy.

        Use this in hot loops where :attr:`data`'s copy would be wasted.  The
        view is marked non-writeable so the ``Image``'s immutability holds.
        """
        view = as_ndarray(self._data.view())
        view.flags.writeable = False
        return view

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(height, width, channels)`` - always a 3-tuple.

        A ``(H, W)`` grayscale array is reported as ``(H, W, 1)``.
        """
        height, width = self._data.shape[:2]
        return (height, width, self.channels)

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self._data.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self._data.shape[1])

    @property
    def channels(self) -> int:
        """Number of channels: 1 (gray), 3 (colour) or 4 (colour + alpha)."""
        return 1 if self._data.ndim == 2 else int(self._data.shape[2])

    @property
    def color_space(self) -> str:
        """Current colour-space tag: ``"BGR"``, ``"RGB"``, ``"GRAY"``, ``"HSV"`` or ``"LAB"``."""
        return self._color_space

    @property
    def path(self) -> str | None:
        """Source file path, or ``None`` for in-memory images."""
        return self._path

    @property
    def dtype(self) -> np.dtype[Any]:
        """Underlying array dtype - always ``uint8``."""
        return self._data.dtype

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str, quality: int = 95) -> Image:
        """Write this image to disk.

        Args:
            path: Destination file path.  The format is inferred from the
                extension (``.jpg``, ``.png``, ``.webp``, ...).  Parent
                directories are created if needed.
            quality: JPEG/WebP quality (0-100), or a PNG compression hint.

        Returns:
            ``self``, so ``.save()`` can sit mid-chain.

        Raises:
            ImliteWriteError: If the file cannot be written.

        Example:
            >>> imlite.load("photo.jpg").rotate(90).save("rotated.jpg")
        """
        from imlite.ops.io import write_image

        write_image(self, path, quality=quality)
        return self

    def show(self, title: str = "imlite") -> Image:
        """Display the image.

        Uses matplotlib when it is installed (best in Jupyter and for
        side-by-side comparison) and falls back to Pillow's built-in viewer
        otherwise, so this never fails just because matplotlib is missing.
        Colours are corrected to RGB first, whatever the internal colour space.

        Note:
            In a notebook you rarely need this - an ``Image`` renders itself
            when it is the last expression in a cell.

        Args:
            title: Figure title (matplotlib only).

        Returns:
            ``self``, so ``.show()`` can sit mid-chain.

        Example:
            >>> imlite.load("photo.jpg").crop(0, 0, 200, 200).show()
        """
        rgb = self.to_rgb()
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.debug("matplotlib not installed - falling back to the Pillow viewer.")
            rgb.to_pil().show(title=title)
            return self

        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.axis("off")
        plt.imshow(rgb._data if rgb.channels != 1 else rgb._data[:, :, 0], cmap="gray")
        plt.tight_layout()
        plt.show()
        return self

    def to_numpy(self) -> Array:
        """Return a copy of the pixel array.

        Returns:
            A ``uint8`` ``numpy.ndarray``.

        Example:
            >>> arr = imlite.load("photo.jpg").to_numpy()
        """
        return as_ndarray(self._data.copy())

    def to_pil(self) -> PILImage.Image:
        """Convert to a :class:`PIL.Image.Image` in RGB (or RGBA) mode.

        Pillow is a hard dependency of imlite, so this always works.

        Returns:
            A Pillow image whose channels are in RGB order.

        Example:
            >>> pil = imlite.load("photo.jpg").to_pil()
        """
        from imlite.utils.pil import to_pil as _to_pil

        rgb = self if self._color_space in ("RGB", "GRAY") else self.to_rgb()
        return _to_pil(rgb._data)

    # ------------------------------------------------------------------
    # Geometry transforms (delegate to ops/geometry.py)
    # ------------------------------------------------------------------

    def crop(self, x: int, y: int, width: int, height: int) -> Image:
        """Crop to the rectangle at (*x*, *y*) of size *width* x *height*.

        Args:
            x: Left edge (pixels from left, 0-indexed).
            y: Top edge (pixels from top, 0-indexed).
            width: Crop width in pixels.
            height: Crop height in pixels.

        Returns:
            A new cropped :class:`Image`.

        Raises:
            CropOutOfBoundsError: If the box extends beyond the image.

        Example:
            >>> thumb = img.crop(0, 0, 200, 200)
        """
        from imlite.ops.geometry import crop as _crop

        return _as_image(_crop(self, x, y, width, height))

    def rotate(self, angle: float, expand: bool = True) -> Image:
        """Rotate counter-clockwise by *angle* degrees.

        Args:
            angle: Rotation angle in degrees, counter-clockwise.  Exact
                multiples of 90 are lossless.
            expand: Grow the canvas to fit the whole rotated image (default).

        Returns:
            A new rotated :class:`Image`.

        Example:
            >>> upright = img.rotate(90)
        """
        from imlite.ops.geometry import rotate as _rotate

        return _as_image(_rotate(self, angle, expand))

    def resize(
        self,
        width: int | None = None,
        height: int | None = None,
        keep_aspect: bool = False,
        resample: str = "auto",
    ) -> Image:
        """Resize to (*width*, *height*).

        At least one dimension must be given; the other is derived from the
        aspect ratio when omitted.

        Args:
            width: Target width, or ``None`` to infer.
            height: Target height, or ``None`` to infer.
            keep_aspect: Fit inside the target box without distorting.
            resample: Filter name - see :func:`imlite.resize`.

        Returns:
            A new resized :class:`Image`.

        Example:
            >>> small = img.resize(320, 240)
            >>> narrow = img.resize(width=128)  # height follows
        """
        from imlite.ops.geometry import resize as _resize

        return _as_image(_resize(self, width, height, keep_aspect, resample))

    def thumbnail(self, size: int, resample: str = "auto") -> Image:
        """Scale so the longest side is *size* pixels; never enlarges.

        Args:
            size: Length of the longest side of the result, in pixels.
            resample: Filter name - see :func:`imlite.resize`.

        Returns:
            A new :class:`Image`, or a copy if it was already small enough.

        Example:
            >>> img.thumbnail(256).save("thumb.jpg")
        """
        from imlite.ops.geometry import thumbnail as _thumbnail

        return _as_image(_thumbnail(self, size, resample))

    def flip(self, axis: FlipAxis = "h") -> Image:
        """Flip along *axis*.

        Args:
            axis: ``"h"``/``"horizontal"`` (left-right), ``"v"``/``"vertical"``
                (top-bottom), or ``"both"``.

        Returns:
            A new flipped :class:`Image`.

        Example:
            >>> mirror = img.flip("h")
        """
        from imlite.ops.geometry import flip as _flip

        return _as_image(_flip(self, axis))

    def pad(
        self,
        top: int = 0,
        bottom: int = 0,
        left: int = 0,
        right: int = 0,
        color: int | Sequence[int] = (0, 0, 0),
    ) -> Image:
        """Add a constant-colour border.

        Args:
            top: Pixels to add on the top edge.
            bottom: Pixels to add on the bottom edge.
            left: Pixels to add on the left edge.
            right: Pixels to add on the right edge.
            color: Fill colour, in this image's **current** colour space.

        Returns:
            A new padded :class:`Image`.

        Example:
            >>> boxed = img.pad(top=10, bottom=10, left=10, right=10)
        """
        from imlite.ops.geometry import pad as _pad

        return _as_image(_pad(self, top, bottom, left, right, color))

    # ------------------------------------------------------------------
    # Pixel-value transforms (delegate to ops/enhance.py)
    # ------------------------------------------------------------------

    def blur(self, radius: float = 2.0) -> Image:
        """Apply a Gaussian blur.

        Args:
            radius: Standard deviation of the Gaussian kernel, in pixels.

        Returns:
            A new blurred :class:`Image`.

        Example:
            >>> soft = img.blur(3)
        """
        from imlite.ops.enhance import blur as _blur

        return _as_image(_blur(self, radius))

    def brightness(self, factor: float = 1.0) -> Image:
        """Scale brightness by *factor* (``1.0`` is unchanged).

        Args:
            factor: Multiplier applied to every channel.

        Returns:
            A new :class:`Image`.

        Example:
            >>> brighter = img.brightness(1.3)
        """
        from imlite.ops.enhance import brightness as _brightness

        return _as_image(_brightness(self, factor))

    def contrast(self, factor: float = 1.0) -> Image:
        """Scale contrast by *factor* around the mean (``1.0`` is unchanged).

        Args:
            factor: Contrast multiplier.

        Returns:
            A new :class:`Image`.

        Example:
            >>> punchy = img.contrast(1.5)
        """
        from imlite.ops.enhance import contrast as _contrast

        return _as_image(_contrast(self, factor))

    def invert(self) -> Image:
        """Invert the image photographically (``255 - value``).

        Returns:
            A new :class:`Image`.

        Example:
            >>> negative = img.invert()
        """
        from imlite.ops.enhance import invert as _invert

        return _as_image(_invert(self))

    def threshold(
        self,
        value: int = 128,
        max_value: int = 255,
        invert_output: bool = False,
    ) -> Image:
        """Binarise at an intensity cut-off, returning a grayscale mask.

        Args:
            value: Cut-off intensity in ``0-255``.
            max_value: Value written where the intensity exceeds the cut-off.
            invert_output: Swap the two output values.

        Returns:
            A new single-channel :class:`Image` with ``color_space="GRAY"``.

        Example:
            >>> mask = img.threshold(200)
        """
        from imlite.ops.enhance import threshold as _threshold

        return _as_image(_threshold(self, value, max_value, invert_output))

    # ------------------------------------------------------------------
    # Colour transforms (delegate to ops/color.py)
    # ------------------------------------------------------------------

    def to_rgb(self) -> Image:
        """Convert to RGB.

        Returns:
            A new :class:`Image` with ``color_space="RGB"``.
        """
        from imlite.ops.color import to_rgb as _to_rgb

        return _as_image(_to_rgb(self))

    def to_bgr(self) -> Image:
        """Convert to BGR (imlite's internal default).

        Returns:
            A new :class:`Image` with ``color_space="BGR"``.
        """
        from imlite.ops.color import to_bgr as _to_bgr

        return _as_image(_to_bgr(self))

    def to_gray(self) -> Image:
        """Convert to single-channel grayscale.

        Returns:
            A new :class:`Image` of shape ``(H, W, 1)`` with
            ``color_space="GRAY"``.
        """
        from imlite.ops.color import to_gray as _to_gray

        return _as_image(_to_gray(self))

    def to_hsv(self) -> Image:
        """Convert to HSV (H in 0-179, S and V in 0-255).

        Returns:
            A new :class:`Image` with ``color_space="HSV"``.
        """
        from imlite.ops.color import to_hsv as _to_hsv

        return _as_image(_to_hsv(self))

    def to_lab(self) -> Image:
        """Convert to CIE L*a*b*.

        Returns:
            A new :class:`Image` with ``color_space="LAB"``.
        """
        from imlite.ops.color import to_lab as _to_lab

        return _as_image(_to_lab(self))

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        source = f"path={self._path!r}" if self._path else "in-memory"
        return f"Image({source}, shape={self.shape}, color_space={self._color_space!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return self._color_space == other._color_space and np.array_equal(self._data, other._data)

    def __hash__(self) -> int:
        # Pixel data is mutable in principle, so hashing by identity keeps
        # Image usable as a dict key without pretending to be value-hashable.
        return id(self)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Array:
        """Support ``np.asarray(img)`` and friends."""
        if copy is False:
            return self.array if dtype is None else as_ndarray(self._data.astype(dtype))
        out = as_ndarray(self._data.copy())
        return out if dtype is None else as_ndarray(out.astype(dtype, copy=False))

    def _repr_png_(self) -> bytes:
        """Render inline in Jupyter when an ``Image`` is a cell's last expression."""
        buffer = io.BytesIO()
        self.to_pil().save(buffer, format="PNG")
        return buffer.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_image(result: Any) -> Image:
    """Narrow an ops return value to ``Image``.

    Every op in ``imlite.ops`` returns whatever type it was handed, so passing
    an ``Image`` in always yields an ``Image``.  Static analysis cannot see
    that, and this keeps the assertion in one place instead of scattering
    ``cast`` calls through every method.
    """
    return cast("Image", result)


def _validate_array(arr: Array) -> None:
    """Raise if *arr* is not a valid image array."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__!r}.")
    if arr.ndim == 2:
        return
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return
    raise ImliteShapeError(
        f"Image array must have shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4); got {arr.shape!r}."
    )
