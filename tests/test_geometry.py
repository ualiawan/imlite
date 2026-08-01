"""Tests for geometry operations (ops/geometry.py)."""

import numpy as np
import pytest

import imlite
from imlite import Image
from imlite.exceptions import CropOutOfBoundsError, ImliteShapeError
from imlite.ops.geometry import crop, flip, pad, resize, rotate

# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------


class TestCrop:
    def test_basic_ndarray(self, bgr_array):
        out = crop(bgr_array, x=10, y=20, width=50, height=60)
        assert out.shape == (60, 50, 3)

    def test_basic_image(self, bgr_image):
        out = crop(bgr_image, x=0, y=0, width=100, height=100)
        assert isinstance(out, Image)
        assert out.shape == (100, 100, 3)

    def test_out_of_bounds_raises(self, bgr_image):
        with pytest.raises(CropOutOfBoundsError):
            crop(bgr_image, x=200, y=200, width=200, height=200)

    def test_zero_width_raises(self, bgr_image):
        with pytest.raises((CropOutOfBoundsError, ValueError)):
            crop(bgr_image, x=0, y=0, width=0, height=50)

    def test_zero_height_raises(self, bgr_image):
        with pytest.raises((CropOutOfBoundsError, ValueError)):
            crop(bgr_image, x=0, y=0, width=50, height=0)

    def test_exact_fit(self, bgr_image):
        # Crop exactly the full image should be lossless
        out = crop(bgr_image, x=0, y=0, width=256, height=256)
        assert np.array_equal(out.data, bgr_image.data)

    def test_preserves_color_space(self, bgr_image):
        out = crop(bgr_image, 0, 0, 50, 50)
        assert out.color_space == "BGR"


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------


class TestRotate:
    @pytest.mark.parametrize("angle", [90, 180, 270, -90, -180, -270])
    def test_fast_paths_square(self, bgr_image, angle):
        out = rotate(bgr_image, angle)
        assert isinstance(out, Image)
        assert out.shape == (256, 256, 3)

    def test_90_rect(self, bgr_array):
        # Non-square array - dimensions should swap after 90 degrees
        arr = bgr_array[:100, :200]  # 100x200
        out = rotate(arr, 90, expand=True)
        assert out.shape[:2] == (200, 100)

    def test_180_pixel_roundtrip(self, bgr_image):
        out = rotate(rotate(bgr_image, 180), 180)
        assert np.array_equal(out.data, bgr_image.data)

    def test_arbitrary_angle(self, bgr_image):
        out = rotate(bgr_image, 45, expand=True)
        assert isinstance(out, Image)
        # Canvas expands so output is larger than 256x256
        assert out.height >= 256

    def test_no_expand(self, bgr_image):
        out = rotate(bgr_image, 45, expand=False)
        assert out.shape == bgr_image.shape

    def test_preserves_color_space(self, bgr_image):
        out = rotate(bgr_image, 90)
        assert out.color_space == "BGR"


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------


class TestResize:
    def test_downscale(self, bgr_image):
        out = resize(bgr_image, width=64, height=64)
        assert out.shape == (64, 64, 3)

    def test_upscale(self, bgr_image):
        out = resize(bgr_image, width=512, height=512)
        assert out.shape == (512, 512, 3)

    def test_infer_height(self, bgr_image):
        # Keep aspect: 256x256 -> width=128 -> height inferred as 128
        out = resize(bgr_image, width=128, keep_aspect=True)
        assert out.width == 128
        assert out.height == 128

    def test_infer_width(self, bgr_image):
        out = resize(bgr_image, height=64, keep_aspect=True)
        assert out.height == 64

    def test_ndarray_input(self, bgr_array):
        out = resize(bgr_array, width=32, height=32)
        assert out.shape == (32, 32, 3)

    def test_preserves_color_space(self, bgr_image):
        out = resize(bgr_image, 64, 64)
        assert out.color_space == "BGR"

    def test_missing_both_dims_raises(self, bgr_image):
        with pytest.raises(ValueError):
            resize(bgr_image)


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------


class TestFlip:
    @pytest.mark.parametrize("axis", ["h", "horizontal", "v", "vertical", "both"])
    def test_valid_axes(self, bgr_image, axis):
        out = flip(bgr_image, axis)
        assert isinstance(out, Image)
        assert out.shape == bgr_image.shape

    def test_invalid_axis_raises(self, bgr_image):
        with pytest.raises(ValueError):
            flip(bgr_image, "diagonal")

    def test_flip_h_pixel_check(self, bgr_array):
        out = flip(bgr_array, "h")
        assert np.array_equal(out[:, 0], bgr_array[:, -1])

    def test_flip_v_pixel_check(self, bgr_array):
        out = flip(bgr_array, "v")
        assert np.array_equal(out[0, :], bgr_array[-1, :])

    def test_double_flip_identity(self, bgr_image):
        out = flip(flip(bgr_image, "h"), "h")
        assert np.array_equal(out.data, bgr_image.data)

    def test_ndarray_input(self, bgr_array):
        out = flip(bgr_array, "v")
        assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# pad
# ---------------------------------------------------------------------------


class TestPad:
    def test_basic(self, bgr_image):
        out = pad(bgr_image, top=10, bottom=10, left=20, right=20)
        assert out.height == 256 + 20
        assert out.width == 256 + 40

    def test_color_kwarg(self, bgr_image):
        out = pad(bgr_image, top=5, bottom=5, left=5, right=5, color=(255, 0, 0))
        # Check top-left corner is the pad colour (BGR blue = (255,0,0))
        assert tuple(out.data[0, 0]) == (255, 0, 0)

    def test_zero_pad_identity(self, bgr_image):
        out = pad(bgr_image, 0, 0, 0, 0)
        assert np.array_equal(out.data, bgr_image.data)

    def test_ndarray_input(self, bgr_array):
        out = pad(bgr_array, top=1, bottom=1, left=1, right=1)
        assert isinstance(out, np.ndarray)
        assert out.shape[0] == bgr_array.shape[0] + 2


# ---------------------------------------------------------------------------
# thumbnail
# ---------------------------------------------------------------------------


class TestThumbnail:
    def test_longest_side_becomes_size(self, bgr_array):
        wide = bgr_array[:100, :200]  # 100 tall, 200 wide
        out = imlite.thumbnail(wide, 50)
        assert out.shape[:2] == (25, 50)

    def test_never_enlarges(self, bgr_array):
        small = bgr_array[:32, :32]
        assert imlite.thumbnail(small, 512).shape == small.shape

    def test_preserves_color_space(self, bgr_image):
        assert bgr_image.thumbnail(64).color_space == "BGR"

    def test_zero_size_raises(self, bgr_image):
        with pytest.raises(ValueError):
            bgr_image.thumbnail(0)


# ---------------------------------------------------------------------------
# Backend behaviour after dropping OpenCV
# ---------------------------------------------------------------------------


class TestRotateExactMultiples:
    def test_90_is_lossless_four_times_round(self, bgr_image):
        out = bgr_image
        for _ in range(4):
            out = out.rotate(90)
        assert np.array_equal(out.data, bgr_image.data)

    def test_90_matches_numpy_rot90(self, bgr_array):
        assert np.array_equal(rotate(bgr_array, 90), np.rot90(bgr_array, 1))

    def test_270_matches_numpy_rot90(self, bgr_array):
        assert np.array_equal(rotate(bgr_array, 270), np.rot90(bgr_array, 3))

    def test_negative_90_is_clockwise(self, bgr_array):
        assert np.array_equal(rotate(bgr_array, -90), np.rot90(bgr_array, 3))

    def test_360_is_identity(self, bgr_array):
        assert np.array_equal(rotate(bgr_array, 360), bgr_array)

    def test_non_multiple_near_90_is_not_snapped(self, bgr_image):
        """int(90.5) used to be 90, silently turning a tilt into a quarter turn."""
        tilted = bgr_image.rotate(90.5, expand=False)
        square = bgr_image.rotate(90, expand=False)
        assert not np.array_equal(tilted.data, square.data)


class TestResizeFilters:
    @pytest.mark.parametrize("name", ["nearest", "box", "bilinear", "bicubic", "lanczos", "auto"])
    def test_all_named_filters_work(self, bgr_image, name):
        assert bgr_image.resize(64, 64, resample=name).shape == (64, 64, 3)

    def test_unknown_filter_raises(self, bgr_image):
        with pytest.raises(ValueError, match="resample"):
            bgr_image.resize(64, 64, resample="magic")

    def test_same_size_is_identity(self, bgr_image):
        assert np.array_equal(bgr_image.resize(256, 256).data, bgr_image.data)

    def test_grayscale_survives_resize(self, gray_array):
        assert resize(gray_array, 32, 32).shape == (32, 32, 1)

    def test_rgba_survives_resize(self):
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        assert resize(rgba, 8, 8).shape == (8, 8, 4)


class TestPadEdgeCases:
    def test_scalar_colour(self, gray_array):
        out = pad(gray_array, top=2, color=200)
        assert out[0, 0, 0] == 200

    def test_three_tuple_on_rgba_adds_opaque_alpha(self):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        out = pad(rgba, top=1, color=(10, 20, 30))
        assert tuple(out[0, 0]) == (10, 20, 30, 255)

    def test_wrong_component_count_raises(self, bgr_image):
        with pytest.raises(ValueError, match="components"):
            bgr_image.pad(top=1, color=(1, 2))

    def test_negative_padding_raises(self, bgr_image):
        with pytest.raises(ValueError, match=">= 0"):
            bgr_image.pad(top=-5)

    def test_2d_grayscale_stays_2d(self):
        flat = np.zeros((8, 8), dtype=np.uint8)
        assert pad(flat, top=1, bottom=1, color=5).shape == (10, 8)


class TestOpsRejectBadInput:
    @pytest.mark.parametrize("op", [crop, rotate, resize, flip, pad])
    def test_non_array_input_raises_typeerror(self, op):
        with pytest.raises(TypeError, match=r"Image or numpy\.ndarray"):
            op([[1, 2, 3]])

    def test_wrong_shape_raises_shape_error(self):
        with pytest.raises(ImliteShapeError):
            flip(np.zeros((4, 4, 7), dtype=np.uint8))
