"""Tests for ffmpeg discovery and diagnostics (utils/ffmpeg.py).

The point of this module is that users never install ffmpeg themselves, and
that when the bundled binary is somehow missing they get instructions instead
of a bare RuntimeError.
"""

import pytest

import imlite
from imlite.exceptions import ImliteFFmpegError
from imlite.utils import ffmpeg as ffmpeg_utils


@pytest.fixture(autouse=True)
def _clear_resolver_cache():
    """resolve_ffmpeg is process-cached; reset it around tests that monkeypatch it."""
    ffmpeg_utils.resolve_ffmpeg.cache_clear()
    yield
    ffmpeg_utils.resolve_ffmpeg.cache_clear()


class TestResolveFfmpeg:
    def test_finds_the_bundled_binary(self):
        exe = ffmpeg_utils.resolve_ffmpeg()
        assert exe
        assert "ffmpeg" in exe.lower()

    def test_result_is_cached(self):
        assert ffmpeg_utils.resolve_ffmpeg() == ffmpeg_utils.resolve_ffmpeg()
        assert ffmpeg_utils.resolve_ffmpeg.cache_info().hits >= 1

    def test_require_ffmpeg_agrees(self):
        assert ffmpeg_utils.require_ffmpeg() == ffmpeg_utils.resolve_ffmpeg()


class TestMissingBinaryIsActionable:
    def test_error_names_the_env_var_and_a_package_manager(self, monkeypatch):
        import imageio_ffmpeg

        def _explode() -> str:
            raise RuntimeError("No ffmpeg exe could be found.")

        monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", _explode)

        with pytest.raises(ImliteFFmpegError) as excinfo:
            ffmpeg_utils.resolve_ffmpeg()

        message = str(excinfo.value)
        assert "IMAGEIO_FFMPEG_EXE" in message
        assert "imlite doctor" in message
        assert any(hint in message for hint in ("apt", "brew", "winget"))

    def test_ffmpeg_info_reports_rather_than_raises(self, monkeypatch):
        import imageio_ffmpeg

        def _explode() -> str:
            raise RuntimeError("No ffmpeg exe could be found.")

        monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", _explode)

        report = ffmpeg_utils.ffmpeg_info()
        assert report["available"] is False
        assert report["error"]


class TestFfmpegInfo:
    def test_reports_a_working_backend(self):
        report = imlite.ffmpeg_info()
        assert report["available"] is True
        assert report["exe"]
        assert report["version"]
        assert report["error"] == ""

    def test_detects_the_bundled_binary(self):
        # In a normal pip install the binary comes from the imageio-ffmpeg wheel,
        # which is the whole "no system install needed" promise.
        assert imlite.ffmpeg_info()["bundled"] is True

    def test_exported_at_top_level(self):
        assert imlite.ffmpeg_info is ffmpeg_utils.ffmpeg_info


class TestImportIsLazy:
    def test_importing_imlite_does_not_probe_ffmpeg(self):
        """`import imlite` must stay fast and must not shell out to ffmpeg."""
        import subprocess
        import sys

        code = (
            "import imlite, imlite.utils.ffmpeg as f; print(f.resolve_ffmpeg.cache_info().currsize)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "0"
