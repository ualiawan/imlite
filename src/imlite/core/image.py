"""The Image class - the fundamental data unit of imlite.

Wraps a single numpy.ndarray and provides:
- Immutable transform methods that return new Image instances.
- A color_space tag so the library always knows whether pixels are BGR,
  RGB, or GRAY without the user having to track it.
- Delegation of all pixel work to imlite.ops.*.

"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from imlite.exceptions import ImliteShapeError

log = logging.getLogger(__name__)

__all__ = ["Image"]

# Valid colour-space tags.
_VALID_COLOR_SPACES = {"BGR", "RGB", "GRAY", "HSV", "LAB"}


class Image:
    """A single in-memory image.

    Args:
        data: Pixel data as a numpy.ndarray. Accepted shapes:
            (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).
            dtype is normalised to uint8 on construction.
        color_space: Colour space of data. One of "BGR" (default,
            OpenCV convention), "RGB", "GRAY", "HSV", or "LAB".
        path: Source file path, or None for in-memory images.

    Note:
        All transform methods (crop, rotate, etc.) return a new
        Image with path=None. The original is never mutated.
    """

    __slots__ = ("_data", "_color_space", "_path")

    def __init__(
        self,
        data: np.ndarray,
        color_space: str = "BGR",
        path: str | None = None,
    ) -> None:
        _validate_array(data)
        if color_space not in _VALID_COLOR_SPACES:
            raise ValueError(
                f"Unknown color_space {color_space!r}. Choose from {sorted(_VALID_COLOR_SPACES)}."
            )
        # Store a copy so callers can't mutate us by holding a reference.
        self._data: np.ndarray = np.ascontiguousarray(data, dtype=np.uint8)
        self._color_space: str = color_space
        self._path: str | None = path

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        color_space: str = "BGR",
        path: str | None = None,
    ) -> Image:
        """Wrap a numpy array as an Image.

        Args:
            arr: Pixel array.  Will be copied to ensure ownership.
            color_space: Colour space tag (default "BGR").
            path: Optional source path.

        Returns:
            A new :class:`Image`.

        Example:
            >>> import numpy as np, imlite
            >>> arr = np.zeros((100, 100, 3), dtype=np.uint8)
            >>> img = imlite.Image.from_numpy(arr)
        """
        return cls(arr, color_space=color_space, path=path)

    @classmethod
    def from_file(cls, path: str) -> Image:
        """Read an image file from disk.

        This is a thin convenience wrapper around
        :func:`~imlite.ops.io.read_image`.

        Args:
            path: Path to the image file.

        Returns:
            A new :class:`Image` with color_space="BGR" and
            path set to the resolved file path.

        Example:
            >>> img = imlite.Image.from_file("photo.jpg")
        """
        from imlite.ops.io import read_image

        return read_image(path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Raw pixel array (a copy - mutations do not affect this Image)."""
        return self._data.copy()

    @property
    def shape(self) -> tuple[int, int, int]:
        """(height, width, channels) - always a 3-tuple.

        Grayscale (H, W) arrays are reported as (H, W, 1).
        """
        if self._data.ndim == 2:
            return (self._data.shape[0], self._data.shape[1], 1)
        return (self._data.shape[0], self._data.shape[1], self._data.shape[2])

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._data.shape[0]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._data.shape[1]

    @property
    def channels(self) -> int:
        """Number of channels: 1 (gray), 3 (colour), or 4 (colour + alpha)."""
        return 1 if self._data.ndim == 2 else self._data.shape[2]

    @property
    def color_space(self) -> str:
        """Current colour space tag: "BGR", "RGB", "GRAY", etc."""
        return self._color_space

    @property
    def path(self) -> str | None:
        """Source file path, or None for in-memory images."""
        return self._path

    @property
    def dtype(self) -> np.dtype:
        """Underlying array dtype (always uint8 after construction)."""
        return self._data.dtype

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str, quality: int = 95) -> Image:
        """Write this image to disk.

        Args:
            path: Destination file path.  Format is inferred from the
                extension (e.g. .jpg, .png).
            quality: JPEG quality (0-100) or PNG compression hint.

        Returns:
            self - allows chaining after save.

        Example:
            >>> imlite.open("photo.jpg").rotate(90).save("rotated.jpg")
        """
        from imlite.ops.io import write_image

        write_image(self, path, quality=quality)
        return self

    def show(self, title: str = "imlite") -> Image:
        """Display the image using matplotlib.

        Works in Jupyter notebooks and non-interactive scripts alike.  The
        image is shown in RGB so colours appear correct regardless of the
        internal color_space.

        Args:
            title: Window/figure title.

        Returns:
            self - allows chaining.

        Example:
            >>> imlite.open("photo.jpg").crop(0, 0, 200, 200).show()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for Image.show(). Install it with: pip install matplotlib"
            ) from exc

        rgb = self.to_rgb()
        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.axis("off")
        plt.imshow(rgb._data)
        plt.tight_layout()
        plt.show()
        return self

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the underlying pixel array.

        Returns:
            numpy.ndarray copy of the image data.

        Example:
            >>> arr = img.to_numpy()
        """
        return self._data.copy()

    def to_pil(self):  # type: ignore[return]
        """Convert to a :class:`PIL.Image.Image`.

        Requires the optional Pillow dependency
        (pip install imlite[pillow]).

        Returns:
            A :class:`PIL.Image.Image` in RGB mode.

        Raises:
            ImportError: If Pillow is not installed.
        """
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for Image.to_pil(). Install it with: pip install imlite[pillow]"
            ) from exc

        rgb = self.to_rgb()
        return PILImage.fromarray(rgb._data)

    # ------------------------------------------------------------------
    # Geometry transforms  (delegate to ops/geometry.py)
    # ------------------------------------------------------------------

    def crop(self, x: int, y: int, width: int, height: int) -> Image:
        """Crop to the rectangle defined by (*x*, *y*, *width*, *height*).

        Args:
            x: Left edge (pixels from left, 0-indexed).
            y: Top edge (pixels from top, 0-indexed).
            width: Crop width in pixels.
            height: Crop height in pixels.

        Returns:
            New :class:`Image` containing the cropped region.

        Example:
            >>> thumb = img.crop(0, 0, 200, 200)
        """
        from imlite.ops.geometry import crop as _crop

        return _crop(self, x, y, width, height)  # type: ignore[return-value]

    def rotate(self, angle: float, expand: bool = True) -> Image:
        """Rotate counter-clockwise by *angle* degrees.

        Args:
            angle: Rotation angle in degrees (counter-clockwise).
            expand: If True the canvas is enlarged to fit the full
                rotated image (default).

        Returns:
            New rotated :class:`Image`.

        Example:
            >>> upright = img.rotate(90)
        """
        from imlite.ops.geometry import rotate as _rotate

        return _rotate(self, angle, expand)  # type: ignore[return-value]

    def resize(
        self,
        width: int | None = None,
        height: int | None = None,
        keep_aspect: bool = False,
    ) -> Image:
        """Resize to (*width*, *height*).

        At least one dimension must be provided.  If only one is given the
        other is inferred to preserve the aspect ratio.

        Args:
            width: Target width, or None to infer.
            height: Target height, or None to infer.
            keep_aspect: Fit within the target box without stretching.

        Returns:
            New resized :class:`Image`.

        Example:
            >>> small = img.resize(320, 240)
            >>> thumb = img.resize(width=128)
        """
        from imlite.ops.geometry import resize as _resize

        return _resize(self, width, height, keep_aspect)  # type: ignore[return-value]

    def flip(self, axis: Literal["h", "horizontal", "v", "vertical", "both"] = "h") -> Image:
        """Flip the image along *axis*.

        Args:
            axis: "h"/"horizontal" (left-right), "v"/"vertical"
                (top-bottom), or "both".

        Returns:
            New flipped :class:`Image`.

        Example:
            >>> mirror = img.flip("h")
        """
        from imlite.ops.geometry import flip as _flip

        return _flip(self, axis)  # type: ignore[return-value]

    def pad(
        self,
        top: int = 0,
        bottom: int = 0,
        left: int = 0,
        right: int = 0,
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image:
        """Add a constant-colour border around the image.

        Args:
            top: Pixels to add on the top edge.
            bottom: Pixels to add on the bottom edge.
            left: Pixels to add on the left edge.
            right: Pixels to add on the right edge.
            color: Fill colour tuple (default black).

        Returns:
            New padded :class:`Image`.

        Example:
            >>> padded = img.pad(top=10, bottom=10, left=10, right=10)
        """
        from imlite.ops.geometry import pad as _pad

        return _pad(self, top, bottom, left, right, color)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Colour transforms  (delegate to ops/color.py)
    # ------------------------------------------------------------------

    def to_rgb(self) -> Image:
        """Convert to RGB colour space.

        Returns:
            New :class:`Image` with color_space="RGB".
        """
        from imlite.ops.color import to_rgb as _to_rgb

        arr = _to_rgb(self._data, _color_space=self._color_space)
        return Image.from_numpy(arr, color_space="RGB")

    def to_bgr(self) -> Image:
        """Convert to BGR colour space (OpenCV convention).

        Returns:
            New :class:`Image` with color_space="BGR".
        """
        from imlite.ops.color import to_bgr as _to_bgr

        arr = _to_bgr(self._data, _color_space=self._color_space)
        return Image.from_numpy(arr, color_space="BGR")

    def to_gray(self) -> Image:
        """Convert to grayscale.

        Returns:
            New :class:`Image` with shape (H, W, 1) and
            color_space="GRAY".
        """
        from imlite.ops.color import to_gray as _to_gray

        arr = _to_gray(self._data, _color_space=self._color_space)
        return Image.from_numpy(arr, color_space="GRAY")

    def to_hsv(self) -> Image:
        """Convert to HSV colour space.

        Returns:
            New :class:`Image` with color_space="HSV".
        """
        from imlite.ops.color import to_hsv as _to_hsv

        arr = _to_hsv(self._data, _color_space=self._color_space)
        return Image.from_numpy(arr, color_space="HSV")

    def to_lab(self) -> Image:
        """Convert to CIE L*a*b* colour space.

        Returns:
            New :class:`Image` with color_space="LAB".
        """
        from imlite.ops.color import to_lab as _to_lab

        arr = _to_lab(self._data, _color_space=self._color_space)
        return Image.from_numpy(arr, color_space="LAB")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        src = f"path={self._path!r}" if self._path else "in-memory"
        return f"Image({src}, shape={self.shape}, color_space={self._color_space!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return self._color_space == other._color_space and np.array_equal(self._data, other._data)

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Allow np.asarray(img) to work directly."""
        arr = self._data
        return arr if dtype is None else arr.astype(dtype)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_array(arr: np.ndarray) -> None:
    """Raise if *arr* is not a valid image array."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__!r}.")
    if arr.ndim == 2:
        return  # grayscale (H, W)
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return
    raise ImliteShapeError(
        f"Image array must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4); "
        f"got {arr.shape!r}."
    )
