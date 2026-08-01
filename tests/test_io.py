"""Tests for image I/O (ops/io.py)."""

import numpy as np
import pytest

import imlite
from imlite import Image
from imlite.exceptions import ImliteReadError
from imlite.ops.io import read_image, write_image


class TestReadImage:
    def test_reads_png(self, sample_png):
        img = read_image(sample_png)
        assert isinstance(img, Image)
        assert img.shape[2] == 3

    def test_reads_jpg(self, sample_jpg):
        img = read_image(sample_jpg)
        assert isinstance(img, Image)

    def test_color_space_is_bgr(self, sample_png):
        img = read_image(sample_png)
        assert img.color_space == "BGR"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((ImliteReadError, FileNotFoundError)):
            read_image(str(tmp_path / "nonexistent.png"))

    def test_path_attribute_set(self, sample_png):
        img = read_image(sample_png)
        assert img.path == sample_png


class TestWriteImage:
    def test_write_png(self, bgr_image, tmp_path):
        out = str(tmp_path / "out.png")
        write_image(bgr_image, out)
        assert (tmp_path / "out.png").exists()

    def test_write_jpg(self, bgr_image, tmp_path):
        out = str(tmp_path / "out.jpg")
        write_image(bgr_image, out)
        assert (tmp_path / "out.jpg").exists()

    def test_write_ndarray(self, bgr_array, tmp_path):
        out = str(tmp_path / "arr.png")
        write_image(bgr_array, out)
        assert (tmp_path / "arr.png").exists()

    def test_roundtrip_png_lossless(self, bgr_image, tmp_path):
        out = str(tmp_path / "rt.png")
        write_image(bgr_image, out)
        loaded = read_image(out)
        assert np.array_equal(bgr_image.data, loaded.data)

    def test_creates_parent_dir(self, bgr_image, tmp_path):
        out = str(tmp_path / "deep" / "dir" / "out.png")
        write_image(bgr_image, out)
        assert (tmp_path / "deep" / "dir" / "out.png").exists()


class TestLoadDispatcher:
    def test_load_image_path(self, sample_png):
        obj = imlite.load(sample_png)
        assert isinstance(obj, Image)

    def test_load_ndarray(self, bgr_array):
        obj = imlite.load(bgr_array)
        assert isinstance(obj, Image)

    def test_load_image_passthrough(self, bgr_image):
        obj = imlite.load(bgr_image)
        assert obj is bgr_image

    def test_load_list_of_arrays(self, bgr_array):
        from imlite import FrameSequence

        seq = imlite.load([bgr_array, bgr_array])
        assert isinstance(seq, FrameSequence)
