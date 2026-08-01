"""Colour-space conversions, implemented in pure numpy.

Every conversion routes through RGB as a hub (``source -> RGB -> target``), so
each pair of spaces needs only a forward and inverse mapping rather than a
combinatorial table.  Round trips such as ``BGR -> LAB -> BGR`` recover the
original image to within 8-bit rounding.

**Conventions.** imlite reproduces OpenCV's 8-bit encodings, because that is
what its users' existing code and thresholds assume:

===========  ===========================================================
Space        8-bit encoding
===========  ===========================================================
``BGR``      blue, green, red - the default for arrays with no tag
``RGB``      red, green, blue
``GRAY``     ITU-R BT.601 luma, always shaped ``(H, W, 1)``
``HSV``      H in 0-179 (degrees / 2), S and V in 0-255
``LAB``      L = L* x 255/100, a = a* + 128, b = b* + 128
===========  ===========================================================

An alpha channel is preserved across ``BGR``/``RGB`` conversions and dropped by
``GRAY``, ``HSV`` and ``LAB``, matching OpenCV.

Each function accepts an :class:`~imlite.core.image.Image` **or** a raw
``Array`` and returns the same type.  For a bare array there is no tag to
read, so the input space comes from the ``source=`` keyword::

    imlite.to_gray(image)                  # uses image.color_space
    imlite.to_gray(array, source="RGB")    # explicit; defaults to "BGR"
"""

import numpy as np

from imlite._typing import Array, as_ndarray
from imlite.exceptions import ImliteColorSpaceError
from imlite.utils.validate import color_op

__all__ = ["to_bgr", "to_gray", "to_hsv", "to_lab", "to_rgb"]

VALID_COLOR_SPACES = frozenset({"BGR", "RGB", "GRAY", "HSV", "LAB"})

# ITU-R BT.601 luma weights, in R, G, B order - the same weights OpenCV's
# COLOR_RGB2GRAY uses.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# sRGB (D65) primaries -> CIE XYZ, and the matching white point.
_RGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE = np.array([0.950456, 1.0, 1.088754], dtype=np.float64)

# CIE L*a*b* piecewise-function constants.
_LAB_EPS = 0.008856
_LAB_KAPPA = 7.787


# ---------------------------------------------------------------------------
# Public conversions
# ---------------------------------------------------------------------------


@color_op("RGB")
def to_rgb(img: Array, source: str = "BGR") -> Array:
    """Convert *img* to RGB.

    Args:
        img: An :class:`~imlite.core.image.Image` or ``Array``.
        source: Colour space of *img*.  Ignored when *img* is an ``Image``
            (its own tag is used).  Defaults to ``"BGR"``.

    Returns:
        The image in RGB, as the same type that was passed in.  An alpha
        channel, if present, is preserved.

    Example:
        >>> rgb = imlite.to_rgb(bgr_array)
        >>> rgb = imlite.load("photo.jpg").to_rgb()
    """
    return _to_rgb_array(img, source)


@color_op("BGR")
def to_bgr(img: Array, source: str = "RGB") -> Array:
    """Convert *img* to BGR (imlite's internal default).

    Args:
        img: An :class:`~imlite.core.image.Image` or ``Array``.
        source: Colour space of *img*.  Ignored when *img* is an ``Image``.
            Defaults to ``"RGB"``.

    Returns:
        The image in BGR, as the same type that was passed in.  An alpha
        channel, if present, is preserved.
    """
    if source == "BGR":
        return np.array(img)
    return _swap_rb(_to_rgb_array(img, source))


@color_op("GRAY")
def to_gray(img: Array, source: str = "BGR") -> Array:
    """Convert *img* to single-channel grayscale.

    Uses the ITU-R BT.601 luma weights ``0.299R + 0.587G + 0.114B``.  The
    result is always shaped ``(H, W, 1)`` rather than ``(H, W)``, so downstream
    code never has to special-case grayscale.

    Args:
        img: An :class:`~imlite.core.image.Image` or ``Array``.
        source: Colour space of *img*.  Ignored when *img* is an ``Image``.

    Returns:
        A grayscale image of shape ``(H, W, 1)``, dtype ``uint8``.  Any alpha
        channel is discarded.
    """
    if source == "GRAY":
        return np.array(_ensure_3d(img))
    rgb, _ = _split_alpha(_to_rgb_array(img, source))
    luma = rgb.astype(np.float32) @ _LUMA
    return _round_u8(luma)[:, :, np.newaxis]


@color_op("HSV")
def to_hsv(img: Array, source: str = "BGR") -> Array:
    """Convert *img* to HSV.

    Args:
        img: An :class:`~imlite.core.image.Image` or ``Array``.
        source: Colour space of *img*.  Ignored when *img* is an ``Image``.

    Returns:
        An HSV image with H in ``0-179`` and S, V in ``0-255`` (OpenCV's 8-bit
        encoding).  Any alpha channel is discarded.

    Example:
        >>> hsv = imlite.load("photo.jpg").to_hsv()
        >>> hue = hsv.data[:, :, 0]  # 0-179
    """
    if source == "HSV":
        return np.array(img)
    rgb, _ = _split_alpha(_to_rgb_array(img, source))
    return _rgb_to_hsv(rgb)


@color_op("LAB")
def to_lab(img: Array, source: str = "BGR") -> Array:
    """Convert *img* to CIE L*a*b*.

    Args:
        img: An :class:`~imlite.core.image.Image` or ``Array``.
        source: Colour space of *img*.  Ignored when *img* is an ``Image``.

    Returns:
        A LAB image using OpenCV's 8-bit encoding: ``L = L* x 255/100``,
        ``a = a* + 128``, ``b = b* + 128``.  Any alpha channel is discarded.
    """
    if source == "LAB":
        return np.array(img)
    rgb, _ = _split_alpha(_to_rgb_array(img, source))
    return _rgb_to_lab(rgb)


# ---------------------------------------------------------------------------
# The RGB hub
# ---------------------------------------------------------------------------


def _to_rgb_array(img: Array, source: str) -> Array:
    """Convert an array in *source* space to RGB (alpha preserved where present)."""
    if source not in VALID_COLOR_SPACES:
        raise ImliteColorSpaceError(
            f"Unknown colour space {source!r}. Choose from {sorted(VALID_COLOR_SPACES)}."
        )
    if source == "RGB":
        return np.array(img)
    if source == "BGR":
        return _swap_rb(img)
    if source == "GRAY":
        return np.repeat(_ensure_3d(img), 3, axis=2)
    if source == "HSV":
        return _hsv_to_rgb(img)
    return _lab_to_rgb(img)


# ---------------------------------------------------------------------------
# Array plumbing
# ---------------------------------------------------------------------------


def _ensure_3d(img: Array) -> Array:
    """Return *img* with an explicit channel axis: ``(H, W)`` becomes ``(H, W, 1)``."""
    return np.asarray(img[:, :, np.newaxis]) if img.ndim == 2 else img


def _split_alpha(img: Array) -> tuple[Array, Array | None]:
    """Split a 4-channel array into its colour channels and its alpha channel."""
    arr = _ensure_3d(img)
    if arr.shape[2] == 4:
        return np.asarray(arr[:, :, :3]), np.asarray(arr[:, :, 3:])
    return arr, None


def _swap_rb(img: Array) -> Array:
    """Swap the red and blue channels, leaving any alpha channel in place.

    ``img[..., ::-1]`` would reverse all four channels of an RGBA image and
    produce ABGR, so the alpha channel is split off first.
    """
    arr = _ensure_3d(img)
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    rgb, alpha = _split_alpha(arr)
    swapped = rgb[:, :, ::-1]
    if alpha is None:
        return np.ascontiguousarray(swapped)
    return np.ascontiguousarray(np.concatenate([swapped, alpha], axis=2))


def _round_u8(arr: Array) -> Array:
    """Round a float array and clip it into ``uint8``."""
    return as_ndarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# HSV
# ---------------------------------------------------------------------------


def _rgb_to_hsv(rgb: Array) -> Array:
    """Convert an 8-bit RGB array to OpenCV-encoded 8-bit HSV."""
    scaled = rgb.astype(np.float32) / 255.0
    red, green, blue = scaled[..., 0], scaled[..., 1], scaled[..., 2]

    value = scaled.max(axis=-1)
    delta = value - scaled.min(axis=-1)
    safe_delta = np.where(delta == 0, 1.0, delta)

    hue = np.select(
        [delta == 0, value == red, value == green],
        [
            np.zeros_like(value),
            60.0 * (green - blue) / safe_delta,
            120.0 + 60.0 * (blue - red) / safe_delta,
        ],
        default=240.0 + 60.0 * (red - green) / safe_delta,
    )
    hue = np.mod(hue, 360.0)

    saturation = np.where(value == 0, 0.0, delta / np.where(value == 0, 1.0, value))

    # Hue lives in 2-degree buckets 0-179. A hue just under 360 rounds up to
    # bucket 180, which is off the end - wrap it to 0, since 360 degrees and
    # 0 degrees are the same colour.
    hue_u8 = (np.rint(hue / 2.0) % 180).astype(np.uint8)

    return np.stack([hue_u8, _round_u8(saturation * 255.0), _round_u8(value * 255.0)], axis=-1)


def _hsv_to_rgb(hsv: Array) -> Array:
    """Convert an OpenCV-encoded 8-bit HSV array back to 8-bit RGB."""
    arr = _ensure_3d(hsv).astype(np.float32)
    hue = arr[..., 0] * 2.0
    saturation = arr[..., 1] / 255.0
    value = arr[..., 2] / 255.0

    sector = np.floor(hue / 60.0)
    offset = hue / 60.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * offset)
    t = value * (1.0 - saturation * (1.0 - offset))

    index = (sector.astype(np.int32) % 6)[..., np.newaxis]
    table = np.stack(
        [
            np.stack([value, t, p], axis=-1),
            np.stack([q, value, p], axis=-1),
            np.stack([p, value, t], axis=-1),
            np.stack([p, q, value], axis=-1),
            np.stack([t, p, value], axis=-1),
            np.stack([value, p, q], axis=-1),
        ],
        axis=-2,
    )
    rgb = np.take_along_axis(table, index[..., np.newaxis], axis=-2)[..., 0, :]
    return _round_u8(rgb * 255.0)


# ---------------------------------------------------------------------------
# L*a*b*
# ---------------------------------------------------------------------------


def _srgb_to_linear(channel: Array) -> Array:
    """Undo the sRGB transfer function, mapping 0-1 gamma values to linear light."""
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(channel: Array) -> Array:
    """Apply the sRGB transfer function, mapping linear light back to 0-1 gamma values."""
    return np.where(channel <= 0.0031308, channel * 12.92, 1.055 * channel ** (1 / 2.4) - 0.055)


def _rgb_to_lab(rgb: Array) -> Array:
    """Convert an 8-bit sRGB array to OpenCV-encoded 8-bit L*a*b*."""
    linear = _srgb_to_linear(rgb.astype(np.float64) / 255.0)
    xyz = (linear @ _RGB_TO_XYZ.T) / _WHITE

    fxyz = np.where(xyz > _LAB_EPS, np.cbrt(xyz), _LAB_KAPPA * xyz + 16.0 / 116.0)
    fx, fy, fz = fxyz[..., 0], fxyz[..., 1], fxyz[..., 2]

    y = xyz[..., 1]
    lightness = np.where(y > _LAB_EPS, 116.0 * fy - 16.0, 903.3 * y)
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)

    return np.stack(
        [
            _round_u8(lightness * 255.0 / 100.0),
            _round_u8(a_star + 128.0),
            _round_u8(b_star + 128.0),
        ],
        axis=-1,
    )


def _lab_to_rgb(lab: Array) -> Array:
    """Convert an OpenCV-encoded 8-bit L*a*b* array back to 8-bit sRGB."""
    arr = _ensure_3d(lab).astype(np.float64)
    lightness = arr[..., 0] * 100.0 / 255.0
    a_star = arr[..., 1] - 128.0
    b_star = arr[..., 2] - 128.0

    fy = (lightness + 16.0) / 116.0
    fx = fy + a_star / 500.0
    fz = fy - b_star / 200.0

    def _finv(f: Array) -> Array:
        cubed = f**3
        return np.where(cubed > _LAB_EPS, cubed, (f - 16.0 / 116.0) / _LAB_KAPPA)

    xyz = np.stack([_finv(fx), _finv(fy), _finv(fz)], axis=-1) * _WHITE
    linear = np.clip(xyz @ _XYZ_TO_RGB.T, 0.0, 1.0)
    return _round_u8(_linear_to_srgb(linear) * 255.0)
