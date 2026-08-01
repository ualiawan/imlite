"""Tests for the Image class."""

import numpy as np
import pytest

from imlite import Image
from imlite.exceptions import ImliteShapeError


class TestImageConstruction:
    def test_from_numpy_bgr(self, bgr_array):
        img = Image.from_numpy(bgr_array, color_space="BGR")
        assert img.color_space == "BGR"
        assert img.shape == (256, 256, 3)

    def test_from_numpy_rgb(self, rgb_array):
        img = Image.from_numpy(rgb_array, color_space="RGB")
        assert img.color_space == "RGB"

    def test_from_numpy_rejects_bad_dims(self):
        # 1-D arrays should not be valid images
        with pytest.raises((ImliteShapeError, TypeError, ValueError)):
            Image.from_numpy(np.zeros((100,), dtype=np.uint8))

    def test_from_numpy_rejects_4d_array(self):
        # 4-D arrays are never valid
        with pytest.raises((ImliteShapeError, TypeError, ValueError)):
            Image.from_numpy(np.zeros((10, 10, 3, 1), dtype=np.uint8))

    def test_from_numpy_float_cast_to_uint8(self, bgr_array):
        arr_f = bgr_array.astype(np.float32) / 255.0
        img = Image.from_numpy((arr_f * 255).astype(np.uint8))
        assert img.data.dtype == np.uint8

    def test_from_file(self, sample_png):
        img = Image.from_file(sample_png)
        assert img.shape[2] == 3
        assert img.color_space == "BGR"

    def test_path_property(self, sample_png):
        img = Image.from_file(sample_png)
        assert img.path == sample_png


class TestImageProperties:
    def test_shape_is_3tuple(self, bgr_image):
        h, w, c = bgr_image.shape
        assert (h, w, c) == (256, 256, 3)

    def test_height_width(self, bgr_image):
        assert bgr_image.height == 256
        assert bgr_image.width == 256

    def test_channels(self, bgr_image):
        assert bgr_image.channels == 3

    def test_data_returns_copy(self, bgr_image):
        d1 = bgr_image.data
        d1[:] = 0
        d2 = bgr_image.data
        assert not np.all(d2 == 0), "data property should return a copy, not a view"

    def test_to_numpy_equals_data(self, bgr_image):
        assert np.array_equal(bgr_image.to_numpy(), bgr_image.data)

    def test_numpy_protocol(self, bgr_image):
        arr = np.asarray(bgr_image)
        assert arr.shape == (256, 256, 3)

    def test_repr(self, bgr_image):
        r = repr(bgr_image)
        assert "Image" in r
        assert "BGR" in r
        assert "256" in r


class TestImageEquality:
    def test_equal_images(self, bgr_array):
        a = Image.from_numpy(bgr_array, color_space="BGR")
        b = Image.from_numpy(bgr_array, color_space="BGR")
        assert a == b

    def test_different_color_space_not_equal(self, bgr_array, rgb_array):
        a = Image.from_numpy(bgr_array, color_space="BGR")
        b = Image.from_numpy(bgr_array, color_space="RGB")
        assert a != b


class TestImageGeometryMethods:
    def test_crop_returns_image(self, bgr_image):
        out = bgr_image.crop(10, 10, 50, 50)
        assert isinstance(out, Image)
        assert out.shape == (50, 50, 3)

    def test_rotate_90(self, bgr_image):
        out = bgr_image.rotate(90)
        assert isinstance(out, Image)
        # 90 degrees on a square stays 256x256
        assert out.shape == (256, 256, 3)

    def test_resize(self, bgr_image):
        out = bgr_image.resize(64, 64)
        assert out.shape == (64, 64, 3)

    def test_flip_h(self, bgr_image):
        out = bgr_image.flip("h")
        assert isinstance(out, Image)
        assert out.shape == bgr_image.shape

    def test_pad(self, bgr_image):
        out = bgr_image.pad(top=5, bottom=5, left=5, right=5)
        assert out.height == 266
        assert out.width == 266

    def test_chaining(self, bgr_image):
        out = bgr_image.crop(0, 0, 100, 100).rotate(180).resize(50, 50)
        assert out.shape == (50, 50, 3)


class TestImageColorMethods:
    def test_to_rgb(self, bgr_image):
        rgb = bgr_image.to_rgb()
        assert rgb.color_space == "RGB"

    def test_to_gray(self, bgr_image):
        gray = bgr_image.to_gray()
        assert gray.color_space == "GRAY"
        assert gray.channels == 1

    def test_idempotent_to_rgb(self, bgr_image):
        rgb = bgr_image.to_rgb()
        rgb2 = rgb.to_rgb()
        assert np.array_equal(rgb.data, rgb2.data)

    def test_to_hsv(self, bgr_image):
        hsv = bgr_image.to_hsv()
        assert hsv.color_space == "HSV"

    def test_to_lab(self, bgr_image):
        lab = bgr_image.to_lab()
        assert lab.color_space == "LAB"


class TestImageSave:
    def test_save_png(self, bgr_image, tmp_path):
        out = tmp_path / "out.png"
        bgr_image.save(str(out))
        assert out.exists()

    def test_save_jpg(self, bgr_image, tmp_path):
        out = tmp_path / "out.jpg"
        bgr_image.save(str(out))
        assert out.exists()

    def test_save_roundtrip(self, bgr_image, tmp_path):
        out = tmp_path / "rt.png"
        bgr_image.save(str(out))
        loaded = Image.from_file(str(out))
        # PNG should be lossless
        assert np.array_equal(bgr_image.data, loaded.data)


class TestImagePil:
    def test_to_pil(self, bgr_image):
        pytest.importorskip("PIL", reason="Pillow not installed")
        pil = bgr_image.to_pil()
        assert pil.size == (256, 256)


class TestImageOwnership:
    def test_construction_copies_by_default(self, bgr_array):
        img = Image.from_numpy(bgr_array)
        bgr_array[:] = 0
        assert not np.all(img.data == 0), "Image must not alias the caller's array"

    def test_copy_false_adopts_the_buffer(self, bgr_array):
        img = Image.from_numpy(bgr_array, copy=False)
        assert np.shares_memory(img.array, bgr_array)

    def test_array_property_is_a_read_only_view(self, bgr_image):
        view = bgr_image.array
        assert not view.flags.writeable
        with pytest.raises(ValueError):
            view[0, 0, 0] = 1

    def test_array_avoids_the_copy_data_makes(self, bgr_image):
        assert np.shares_memory(bgr_image.array, bgr_image.array)
        assert not np.shares_memory(bgr_image.data, bgr_image.array)

    def test_asarray_returns_a_copy(self, bgr_image):
        arr = np.asarray(bgr_image)
        arr[:] = 0
        assert not np.all(bgr_image.data == 0)

    def test_transforms_do_not_mutate_the_original(self, bgr_image):
        before = bgr_image.data
        bgr_image.crop(0, 0, 10, 10).rotate(45).blur(2).invert()
        assert np.array_equal(bgr_image.data, before)


class TestImagePilInterop:
    def test_from_pil_roundtrip(self, bgr_image):
        restored = Image.from_pil(bgr_image.to_pil())
        assert restored.color_space == "RGB"
        assert np.array_equal(restored.data, bgr_image.to_rgb().data)

    def test_to_pil_is_rgb_ordered(self):
        # BGR (255, 0, 0) is blue, so Pillow must report (0, 0, 255).
        blue = Image.from_numpy(np.full((2, 2, 3), (255, 0, 0), dtype=np.uint8))
        assert blue.to_pil().getpixel((0, 0)) == (0, 0, 255)

    def test_to_pil_from_grayscale(self, bgr_image):
        assert bgr_image.to_gray().to_pil().mode == "L"

    def test_from_pil_keeps_alpha(self):
        from PIL import Image as PILImage

        assert Image.from_pil(PILImage.new("RGBA", (4, 4))).channels == 4


class TestImageNotebookRendering:
    def test_repr_png_emits_a_png(self, bgr_image):
        payload = bgr_image._repr_png_()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")

    def test_repr_png_works_for_every_color_space(self, bgr_image):
        for image in (bgr_image.to_rgb(), bgr_image.to_gray(), bgr_image.to_hsv()):
            assert image._repr_png_().startswith(b"\x89PNG")


class TestImageShow:
    def test_uses_matplotlib_when_available(self, bgr_image, monkeypatch):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        assert bgr_image.show() is bgr_image
        plt.close("all")

    def test_falls_back_to_pillow_without_matplotlib(self, bgr_image, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_matplotlib(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("matplotlib is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_matplotlib)

        shown = []
        from PIL import Image as PILImage

        monkeypatch.setattr(PILImage.Image, "show", lambda self, **kw: shown.append(kw))
        assert bgr_image.show() is bgr_image
        assert shown, "show() should fall back to the Pillow viewer, not raise"


class TestImageNewMethods:
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("thumbnail", (64,)),
            ("blur", (1.0,)),
            ("brightness", (1.2,)),
            ("contrast", (1.2,)),
            ("invert", ()),
        ],
    )
    def test_returns_image_and_preserves_space(self, bgr_image, method, args):
        out = getattr(bgr_image, method)(*args)
        assert isinstance(out, Image)
        assert out.color_space == "BGR"

    def test_threshold_returns_gray(self, bgr_image):
        assert bgr_image.threshold(120).color_space == "GRAY"


class TestImageHashing:
    def test_is_hashable(self, bgr_image):
        assert {bgr_image: "value"}[bgr_image] == "value"
