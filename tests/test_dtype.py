"""Tests for pixel dtype normalisation (utils/dtype.py).

The rule imlite commits to: convert what is unambiguous, raise on what is not,
never silently truncate.  A plain ``astype(np.uint8)`` would turn ``uint16
1000`` into ``232`` and ``float 1.0`` into ``1``.
"""

import numpy as np
import pytest

from imlite import Image
from imlite.exceptions import ImliteDtypeError
from imlite.utils.dtype import as_uint8


class TestUnambiguousConversions:
    def test_uint8_passes_through_without_copying(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        assert as_uint8(arr) is arr

    def test_bool_maps_to_0_and_255(self):
        arr = np.array([[True, False]], dtype=bool)
        assert as_uint8(arr).tolist() == [[255, 0]]

    def test_uint16_scales_by_257_not_modulo(self):
        # The old astype() path wrapped 1000 to 232.
        arr = np.full((2, 2), 1000, dtype=np.uint16)
        assert as_uint8(arr)[0, 0] == 3

    def test_uint16_full_range_maps_to_full_range(self):
        arr = np.array([[0, 65535]], dtype=np.uint16)
        assert as_uint8(arr).tolist() == [[0, 255]]

    def test_normalised_float_scales_to_255(self):
        arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        assert as_uint8(arr).tolist() == [[0, 128, 255]]

    def test_float_already_in_0_255_is_rounded(self):
        arr = np.array([[0.4, 128.6, 254.5]], dtype=np.float64)
        assert as_uint8(arr).tolist() == [[0, 129, 254]]

    def test_small_int_range_casts_directly(self):
        arr = np.array([[0, 128, 255]], dtype=np.int32)
        assert as_uint8(arr).tolist() == [[0, 128, 255]]

    def test_empty_array_is_handled(self):
        assert as_uint8(np.zeros((0, 0), dtype=np.float32)).dtype == np.uint8


class TestAmbiguousInputRaises:
    def test_out_of_range_integers(self):
        with pytest.raises(ImliteDtypeError, match=r"0\.\.255"):
            as_uint8(np.array([[1000]], dtype=np.int32))

    def test_negative_integers(self):
        with pytest.raises(ImliteDtypeError):
            as_uint8(np.array([[-5]], dtype=np.int16))

    def test_negative_floats(self):
        with pytest.raises(ImliteDtypeError, match="negative"):
            as_uint8(np.array([[-0.5, 0.5]], dtype=np.float32))

    def test_floats_above_255(self):
        with pytest.raises(ImliteDtypeError, match="above the 8-bit maximum"):
            as_uint8(np.array([[1000.0]], dtype=np.float32))

    def test_nan(self):
        with pytest.raises(ImliteDtypeError, match="NaN"):
            as_uint8(np.array([[np.nan]], dtype=np.float32))

    def test_infinity(self):
        with pytest.raises(ImliteDtypeError, match="NaN or infinity"):
            as_uint8(np.array([[np.inf]], dtype=np.float32))

    def test_complex_dtype(self):
        with pytest.raises(ImliteDtypeError, match="Unsupported"):
            as_uint8(np.array([[1 + 2j]], dtype=np.complex128))


class TestImageAppliesThePolicy:
    def test_normalised_float_image_is_scaled_not_floored(self):
        img = Image.from_numpy(np.ones((4, 4, 3), dtype=np.float32))
        assert img.data[0, 0].tolist() == [255, 255, 255]

    def test_uint16_image_is_scaled(self):
        img = Image.from_numpy(np.full((4, 4, 3), 1000, dtype=np.uint16))
        assert img.dtype == np.uint8
        assert img.data[0, 0].tolist() == [3, 3, 3]

    def test_bool_mask_becomes_a_visible_image(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        assert Image.from_numpy(mask).data[0, 0] == 255

    def test_ambiguous_input_raises_rather_than_corrupting(self):
        with pytest.raises(ImliteDtypeError):
            Image.from_numpy(np.full((4, 4, 3), 5000, dtype=np.int32))
