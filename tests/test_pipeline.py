"""Tests for the load() dispatcher (core/pipeline.py) and the top-level surface."""

from pathlib import Path

import numpy as np
import pytest

import imlite
from imlite import FrameSequence, Image, Video
from imlite.exceptions import ImliteOpenError


class TestLoadDispatch:
    def test_image_path_returns_image(self, sample_png):
        assert isinstance(imlite.load(sample_png), Image)

    def test_pathlib_path_works(self, sample_png):
        assert isinstance(imlite.load(Path(sample_png)), Image)

    def test_video_path_returns_video(self, sample_video):
        assert isinstance(imlite.load(sample_video), Video)

    def test_directory_returns_sequence(self, sample_frames_dir):
        seq = imlite.load(sample_frames_dir)
        assert isinstance(seq, FrameSequence)
        assert len(seq) == 10

    def test_ndarray_returns_image(self, bgr_array):
        assert isinstance(imlite.load(bgr_array), Image)

    def test_list_of_arrays_returns_sequence(self, bgr_array):
        assert len(imlite.load([bgr_array, bgr_array])) == 2

    def test_list_of_images_returns_sequence(self, bgr_image):
        assert len(imlite.load([bgr_image] * 3)) == 3

    def test_list_of_paths_returns_sequence(self, sample_png):
        assert len(imlite.load([sample_png, sample_png])) == 2

    def test_empty_list_returns_empty_sequence(self):
        assert len(imlite.load([])) == 0

    @pytest.mark.parametrize("factory", ["image", "video", "sequence"])
    def test_imlite_objects_pass_through_unchanged(
        self, factory, bgr_image, sample_video, sample_frames_dir
    ):
        original = {
            "image": bgr_image,
            "video": Video(sample_video),
            "sequence": FrameSequence.from_dir(sample_frames_dir),
        }[factory]
        assert imlite.load(original) is original


class TestLoadErrors:
    def test_missing_image_file(self):
        with pytest.raises((ImliteOpenError, imlite.ImliteReadError)):
            imlite.load("no-such-file.png")

    def test_missing_video_file(self):
        with pytest.raises(ImliteOpenError, match="not found"):
            imlite.load("no-such-file.mp4")

    def test_unknown_extension_that_is_not_media(self, tmp_path):
        junk = tmp_path / "notes.xyz"
        junk.write_text("this is not an image")
        with pytest.raises(ImliteOpenError, match="Could not determine"):
            imlite.load(str(junk))

    def test_unsupported_type(self):
        with pytest.raises(ImliteOpenError, match="Cannot load"):
            imlite.load(42)

    def test_list_of_non_media_paths(self, tmp_path):
        with pytest.raises(ImliteOpenError, match="image files"):
            imlite.load([str(tmp_path / "a.txt")])

    def test_list_of_unsupported_items(self):
        with pytest.raises(ImliteOpenError, match="List items"):
            imlite.load([{"not": "a frame"}])


class TestUnknownExtensionProbing:
    def test_png_bytes_with_a_wrong_extension_load_as_an_image(self, tmp_path, sample_png):
        disguised = tmp_path / "mystery.dat"
        disguised.write_bytes(Path(sample_png).read_bytes())
        assert isinstance(imlite.load(str(disguised)), Image)

    def test_video_bytes_with_a_wrong_extension_explain_the_limit(self, tmp_path, sample_video):
        """ffmpeg picks its decoder by extension, so a disguised video cannot be sniffed."""
        disguised = tmp_path / "mystery.dat"
        disguised.write_bytes(Path(sample_video).read_bytes())
        with pytest.raises(ImliteOpenError, match="rename it to a known extension"):
            imlite.load(str(disguised))

    @pytest.mark.parametrize("extension", [".mpg", ".3gp", ".ogv", ".mts", ".vob"])
    def test_common_video_extensions_are_recognised(self, extension):
        assert extension in imlite.VIDEO_EXTENSIONS

    @pytest.mark.parametrize("extension", [".jfif", ".tga", ".pnm", ".jp2"])
    def test_common_image_extensions_are_recognised(self, extension):
        assert extension in imlite.IMAGE_EXTENSIONS


class TestExplicitConstructors:
    def test_read_image(self, sample_png):
        assert isinstance(imlite.read_image(sample_png), Image)

    def test_read_video(self, sample_video):
        assert isinstance(imlite.read_video(sample_video), Video)

    def test_read_frames_from_directory(self, sample_frames_dir):
        assert len(imlite.read_frames(sample_frames_dir)) == 10

    def test_read_frames_from_list(self, bgr_image):
        assert len(imlite.read_frames([bgr_image, bgr_image])) == 2

    def test_read_frames_rejects_other_types(self, bgr_array):
        with pytest.raises(TypeError, match="read_frames"):
            imlite.read_frames(bgr_array)


class TestPublicSurface:
    def test_all_names_are_importable(self):
        missing = [name for name in imlite.__all__ if not hasattr(imlite, name)]
        assert missing == []

    def test_all_has_no_duplicates(self):
        assert len(imlite.__all__) == len(set(imlite.__all__))

    def test_package_ships_a_py_typed_marker(self):
        marker = Path(imlite.__file__).parent / "py.typed"
        assert marker.exists(), "PEP 561 marker missing - downstream type checkers will ignore us"

    def test_opencv_is_not_a_dependency(self):
        import sys

        imlite.load(np.zeros((8, 8, 3), dtype=np.uint8)).resize(4, 4).to_gray()
        assert "cv2" not in sys.modules, "imlite must not import OpenCV"

    def test_extension_registries_are_exported(self):
        assert ".png" in imlite.IMAGE_EXTENSIONS
        assert ".mp4" in imlite.VIDEO_EXTENSIONS
