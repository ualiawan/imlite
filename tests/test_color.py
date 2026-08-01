"""Tests for colour-space operations (ops/color.py).

Reference values come from the CIE definitions and OpenCV's documented 8-bit
encodings, so these tests still pin the numbers now that imlite computes them
in numpy instead of calling cv2.
"""

import numpy as np
import pytest

import imlite
from imlite import Image
from imlite.exceptions import ImliteColorSpaceError
from imlite.ops.color import to_bgr, to_gray, to_hsv, to_lab, to_rgb


def _pixel(red: int, green: int, blue: int) -> np.ndarray:
    """Return a 1x1 BGR array for the given RGB triple."""
    return np.array([[[blue, green, red]]], dtype=np.uint8)


def _make_image(color_space: str = "BGR") -> Image:
    rng = np.random.default_rng(1234)
    return Image.from_numpy(
        rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), color_space=color_space
    )


# ---------------------------------------------------------------------------
# The dual API: every colour op must accept Image *and* ndarray
# ---------------------------------------------------------------------------

ALL_COLOR_OPS = [to_rgb, to_bgr, to_gray, to_hsv, to_lab]


class TestDualApi:
    @pytest.mark.parametrize("op", ALL_COLOR_OPS)
    def test_accepts_image_and_returns_image(self, op, bgr_image):
        out = op(bgr_image)
        assert isinstance(out, Image)

    @pytest.mark.parametrize("op", ALL_COLOR_OPS)
    def test_accepts_ndarray_and_returns_ndarray(self, op, bgr_array):
        out = op(bgr_array)
        assert isinstance(out, np.ndarray)

    @pytest.mark.parametrize("op", ALL_COLOR_OPS)
    def test_exported_at_top_level(self, op):
        assert getattr(imlite, op.__name__) is op

    @pytest.mark.parametrize("op", ALL_COLOR_OPS)
    def test_image_and_array_paths_agree(self, op, bgr_image):
        from_image = op(bgr_image).data
        from_array = op(bgr_image.data, source="BGR")
        assert np.array_equal(from_image, from_array)

    def test_image_tag_wins_over_default_source(self, rgb_array):
        img = Image.from_numpy(rgb_array, color_space="RGB")
        # to_rgb on an RGB-tagged Image must be a no-op, not a channel swap.
        assert np.array_equal(to_rgb(img).data, rgb_array)

    def test_explicit_source_overrides_image_tag(self, bgr_image):
        # Deliberately lie about the space; the override must be honoured.
        assert np.array_equal(to_rgb(bgr_image, source="RGB").data, bgr_image.data)

    def test_unknown_source_raises(self, bgr_array):
        with pytest.raises(ImliteColorSpaceError):
            to_rgb(bgr_array, source="CMYK")

    def test_rejects_non_image_input(self):
        with pytest.raises(TypeError):
            to_gray([[1, 2, 3]])


# ---------------------------------------------------------------------------
# to_rgb / to_bgr
# ---------------------------------------------------------------------------


class TestToRgb:
    def test_channel_swap(self, bgr_array):
        rgb = to_rgb(bgr_array, source="BGR")
        assert np.array_equal(rgb[:, :, 0], bgr_array[:, :, 2])
        assert np.array_equal(rgb[:, :, 2], bgr_array[:, :, 0])

    def test_idempotent(self, bgr_array):
        rgb = to_rgb(bgr_array, source="BGR")
        assert np.array_equal(rgb, to_rgb(rgb, source="RGB"))

    def test_image_method_tags_rgb(self, bgr_image):
        out = bgr_image.to_rgb()
        assert out.color_space == "RGB"
        assert out.shape == bgr_image.shape

    def test_gray_expands_to_three_channels(self, gray_array):
        rgb = to_rgb(gray_array, source="GRAY")
        assert rgb.shape == (128, 128, 3)

    def test_rgba_keeps_alpha_last(self):
        # BGRA (1, 2, 3, 9) must become RGBA (3, 2, 1, 9), not ABGR.
        bgra = np.array([[[1, 2, 3, 9]]], dtype=np.uint8)
        assert tuple(to_rgb(bgra, source="BGR")[0, 0]) == (3, 2, 1, 9)


class TestToBgr:
    def test_roundtrip_is_lossless(self, bgr_image):
        assert np.array_equal(bgr_image.to_rgb().to_bgr().data, bgr_image.data)

    def test_idempotent(self, bgr_image):
        assert np.array_equal(bgr_image.to_bgr().data, bgr_image.data)

    def test_rgba_keeps_alpha_last(self):
        rgba = np.array([[[1, 2, 3, 9]]], dtype=np.uint8)
        assert tuple(to_bgr(rgba, source="RGB")[0, 0]) == (3, 2, 1, 9)


# ---------------------------------------------------------------------------
# to_gray
# ---------------------------------------------------------------------------


class TestToGray:
    def test_shape_is_always_h_w_1(self, bgr_image):
        gray = bgr_image.to_gray()
        assert gray.channels == 1
        assert len(gray.shape) == 3
        assert gray.color_space == "GRAY"

    @pytest.mark.parametrize(
        ("rgb", "expected"),
        [((255, 0, 0), 76), ((0, 255, 0), 150), ((0, 0, 255), 29), ((255, 255, 255), 255)],
    )
    def test_bt601_luma_weights(self, rgb, expected):
        """0.299R + 0.587G + 0.114B - the same weights cv2.COLOR_BGR2GRAY uses."""
        assert int(to_gray(_pixel(*rgb), source="BGR")[0, 0, 0]) == expected

    def test_idempotent(self, bgr_image):
        gray = bgr_image.to_gray()
        assert np.array_equal(gray.data, gray.to_gray().data)

    def test_accepts_2d_gray_array(self):
        flat = np.full((8, 8), 100, dtype=np.uint8)
        assert to_gray(flat, source="GRAY").shape == (8, 8, 1)

    def test_alpha_is_dropped(self):
        rgba = np.array([[[255, 255, 255, 0]]], dtype=np.uint8)
        assert to_gray(rgba, source="BGR").shape == (1, 1, 1)


# ---------------------------------------------------------------------------
# to_hsv
# ---------------------------------------------------------------------------


class TestToHsv:
    @pytest.mark.parametrize(
        ("rgb", "expected"),
        [
            ((255, 0, 0), (0, 255, 255)),
            ((0, 255, 0), (60, 255, 255)),
            ((0, 0, 255), (120, 255, 255)),
            ((128, 128, 128), (0, 0, 128)),
            ((0, 0, 0), (0, 0, 0)),
        ],
    )
    def test_opencv_8bit_encoding(self, rgb, expected):
        """H is halved into 0-179; S and V fill 0-255, as OpenCV does."""
        assert tuple(int(v) for v in to_hsv(_pixel(*rgb), source="BGR")[0, 0]) == expected

    def test_hue_never_exceeds_179(self, bgr_image):
        assert bgr_image.to_hsv().data[:, :, 0].max() <= 179

    def test_roundtrip_within_quantisation_error(self, bgr_image):
        back = bgr_image.to_hsv().to_bgr()
        error = np.abs(bgr_image.data.astype(int) - back.data.astype(int)).max()
        assert error <= 6, f"HSV roundtrip drifted by {error}"

    def test_idempotent(self, bgr_image):
        hsv = bgr_image.to_hsv()
        assert np.array_equal(hsv.data, hsv.to_hsv().data)

    def test_from_rgb_tagged_image(self):
        assert _make_image("RGB").to_hsv().color_space == "HSV"


# ---------------------------------------------------------------------------
# to_lab
# ---------------------------------------------------------------------------


class TestToLab:
    @pytest.mark.parametrize(
        ("rgb", "expected"),
        [
            ((255, 255, 255), (255, 128, 128)),  # L*=100, a*=b*=0
            ((0, 0, 0), (0, 128, 128)),
            ((255, 0, 0), (136, 208, 195)),  # L*=53.24 a*=80.09 b*=67.20
            ((0, 255, 0), (224, 42, 211)),  # L*=87.74 a*=-86.18 b*=83.18
            ((0, 0, 255), (82, 207, 20)),  # L*=32.30 a*=79.19 b*=-107.86
        ],
    )
    def test_matches_cie_reference(self, rgb, expected):
        """L = L* x 255/100, a = a* + 128, b = b* + 128 (OpenCV's 8-bit encoding)."""
        got = tuple(int(v) for v in to_lab(_pixel(*rgb), source="BGR")[0, 0])
        assert got == expected

    def test_roundtrip_within_quantisation_error(self, bgr_image):
        back = bgr_image.to_lab().to_bgr()
        error = np.abs(bgr_image.data.astype(int) - back.data.astype(int)).max()
        # a* and b* quantise to whole units, which is a few RGB counts on
        # saturated colours; this is inherent to the 8-bit encoding.
        assert error <= 20, f"LAB roundtrip drifted by {error}"

    def test_idempotent(self, bgr_image):
        lab = bgr_image.to_lab()
        assert np.array_equal(lab.data, lab.to_lab().data)

    def test_from_rgb_tagged_image(self):
        assert _make_image("RGB").to_lab().color_space == "LAB"
