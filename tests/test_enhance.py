"""Tests for pixel-value operations (ops/enhance.py)."""

import numpy as np
import pytest

import imlite
from imlite import Image
from imlite.ops.enhance import blur, brightness, contrast, invert, threshold


@pytest.fixture()
def flat_image() -> Image:
    """64x64 uniform mid-grey BGR image - easy to reason about numerically."""
    return Image.from_numpy(np.full((64, 64, 3), 100, dtype=np.uint8))


class TestBlur:
    def test_preserves_shape_and_space(self, bgr_image):
        out = blur(bgr_image, radius=2)
        assert out.shape == bgr_image.shape
        assert out.color_space == "BGR"

    def test_zero_radius_is_identity(self, bgr_image):
        assert np.array_equal(blur(bgr_image, 0).data, bgr_image.data)

    def test_uniform_image_survives_blur(self, flat_image):
        assert np.array_equal(blur(flat_image, 3).data, flat_image.data)

    def test_reduces_local_variance(self, bgr_image):
        assert float(blur(bgr_image, 4).data.std()) < float(bgr_image.data.std())

    def test_negative_radius_raises(self, bgr_image):
        with pytest.raises(ValueError, match="radius"):
            blur(bgr_image, -1)

    def test_ndarray_input(self, bgr_array):
        assert isinstance(blur(bgr_array, 1), np.ndarray)

    def test_grayscale_input(self, gray_array):
        assert blur(gray_array, 2).shape == gray_array.shape


class TestBrightness:
    def test_factor_one_is_identity(self, flat_image):
        assert np.array_equal(brightness(flat_image, 1.0).data, flat_image.data)

    def test_scales_linearly(self, flat_image):
        assert brightness(flat_image, 1.5).data[0, 0, 0] == 150

    def test_zero_produces_black(self, bgr_image):
        assert brightness(bgr_image, 0.0).data.max() == 0

    def test_clips_at_255(self, flat_image):
        assert brightness(flat_image, 10.0).data.max() == 255

    def test_negative_factor_raises(self, bgr_image):
        with pytest.raises(ValueError, match="factor"):
            brightness(bgr_image, -1)


class TestContrast:
    def test_factor_one_is_identity(self, bgr_image):
        assert np.array_equal(contrast(bgr_image, 1.0).data, bgr_image.data)

    def test_zero_flattens_to_the_mean(self, bgr_image):
        flattened = contrast(bgr_image, 0.0).data
        assert flattened.min() == flattened.max()

    def test_increases_spread(self, bgr_image):
        assert float(contrast(bgr_image, 2.0).data.std()) > float(bgr_image.data.std())

    def test_negative_factor_raises(self, bgr_image):
        with pytest.raises(ValueError, match="factor"):
            contrast(bgr_image, -0.5)


class TestInvert:
    def test_complements_values(self, flat_image):
        assert invert(flat_image).data[0, 0, 0] == 155

    def test_is_its_own_inverse(self, bgr_image):
        assert np.array_equal(invert(invert(bgr_image)).data, bgr_image.data)

    def test_ndarray_input(self, bgr_array):
        assert isinstance(invert(bgr_array), np.ndarray)


class TestThreshold:
    def test_returns_single_channel_gray(self, bgr_image):
        out = threshold(bgr_image, 100)
        assert out.channels == 1
        assert out.color_space == "GRAY"

    def test_output_is_binary(self, bgr_image):
        assert set(np.unique(threshold(bgr_image, 128).data).tolist()) <= {0, 255}

    def test_cutoff_is_exclusive_below(self):
        ramp = Image.from_numpy(np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8))
        assert threshold(ramp, 128).data[0, :, 0].tolist() == [0, 255]

    def test_invert_output_swaps_the_result(self):
        ramp = Image.from_numpy(np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8))
        assert threshold(ramp, 128, invert_output=True).data[0, :, 0].tolist() == [255, 0]

    def test_max_value_is_honoured(self, bgr_image):
        assert threshold(bgr_image, 10, max_value=200).data.max() <= 200

    def test_ndarray_input_returns_ndarray(self, bgr_array):
        out = threshold(bgr_array, 128)
        assert isinstance(out, np.ndarray)
        assert out.shape == (256, 256, 1)

    @pytest.mark.parametrize("bad", [-1, 256])
    def test_out_of_range_value_raises(self, bgr_image, bad):
        with pytest.raises(ValueError, match="0-255"):
            threshold(bgr_image, bad)

    def test_out_of_range_max_value_raises(self, bgr_image):
        with pytest.raises(ValueError, match="max_value"):
            threshold(bgr_image, 128, max_value=300)


class TestChainingAndExports:
    @pytest.mark.parametrize(
        "name", ["blur", "brightness", "contrast", "invert", "threshold", "thumbnail"]
    )
    def test_exported_at_top_level(self, name):
        assert callable(getattr(imlite, name))

    def test_image_methods_chain(self, bgr_image):
        out = bgr_image.blur(1).brightness(1.1).contrast(1.2).invert().threshold(100)
        assert isinstance(out, Image)
        assert out.color_space == "GRAY"
