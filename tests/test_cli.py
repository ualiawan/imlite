"""Tests for the ``imlite`` command-line interface (cli.py)."""

from pathlib import Path

import pytest

from imlite.cli import _parse_size, main


class TestTopLevel:
    def test_no_command_prints_help(self, capsys):
        assert main([]) == 0
        assert "usage: imlite" in capsys.readouterr().out

    def test_version_flag(self, capsys):
        import imlite

        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert imlite.__version__ in capsys.readouterr().out

    def test_unknown_command_exits_nonzero(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["nonsense"])
        assert excinfo.value.code != 0


class TestDoctor:
    def test_reports_a_working_install(self, capsys):
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "imlite" in out
        assert "ffmpeg" in out
        assert "Video support is working." in out

    def test_lists_every_backend(self, capsys):
        main(["doctor"])
        out = capsys.readouterr().out
        for backend in ("numpy", "pillow", "imageio", "ffmpeg", "python"):
            assert backend in out

    def test_never_mentions_opencv(self, capsys):
        main(["doctor"])
        assert "opencv" not in capsys.readouterr().out.lower()


class TestInfo:
    def test_image_metadata(self, sample_png, capsys):
        assert main(["info", sample_png]) == 0
        out = capsys.readouterr().out
        assert "width" in out
        assert "256" in out

    def test_video_metadata(self, sample_video, capsys):
        assert main(["info", sample_video]) == 0
        out = capsys.readouterr().out
        assert "fps" in out
        assert "frame_count" in out

    def test_directory_of_frames(self, sample_frames_dir, capsys):
        assert main(["info", sample_frames_dir]) == 0
        assert "frames" in capsys.readouterr().out

    def test_missing_file_is_a_handled_error(self, capsys):
        assert main(["info", "does-not-exist.png"]) == 1
        assert "error:" in capsys.readouterr().err


class TestExtract:
    def test_writes_frames(self, sample_video, tmp_path, capsys):
        out_dir = tmp_path / "frames"
        assert main(["-q", "extract", sample_video, str(out_dir)]) == 0
        assert len(list(out_dir.glob("*.png"))) == 30
        assert "Extracted 30 frames" in capsys.readouterr().out

    def test_step_reduces_the_count(self, sample_video, tmp_path):
        out_dir = tmp_path / "frames"
        main(["-q", "extract", sample_video, str(out_dir), "--step", "3"])
        assert len(list(out_dir.glob("*.png"))) == 10

    def test_format_flag(self, sample_video, tmp_path):
        out_dir = tmp_path / "frames"
        main(["-q", "extract", sample_video, str(out_dir), "--step", "10", "--format", "jpg"])
        assert len(list(out_dir.glob("*.jpg"))) == 3

    def test_start_and_end(self, sample_video, tmp_path):
        out_dir = tmp_path / "frames"
        main(["-q", "extract", sample_video, str(out_dir), "--start", "5", "--end", "15"])
        assert len(list(out_dir.glob("*.png"))) == 10


class TestMerge:
    def test_encodes_a_video(self, sample_frames_dir, tmp_path, capsys):
        out = tmp_path / "out.mp4"
        assert main(["-q", "merge", sample_frames_dir, str(out), "--fps", "10"]) == 0
        assert out.exists()
        assert "Encoded 10 frames" in capsys.readouterr().out

    def test_resize_flag(self, sample_frames_dir, tmp_path):
        import imlite

        out = tmp_path / "small.mp4"
        main(["-q", "merge", sample_frames_dir, str(out), "--fps", "10", "--resize", "32x32"])
        assert imlite.video_info(str(out))["width"] == 32

    def test_empty_directory_is_a_handled_error(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main(["-q", "merge", str(empty), str(tmp_path / "out.mp4")]) == 1
        assert "no image files found" in capsys.readouterr().err


class TestConvert:
    def test_changes_format(self, sample_png, tmp_path, capsys):
        out = tmp_path / "out.jpg"
        assert main(["-q", "convert", sample_png, str(out)]) == 0
        assert out.exists()
        assert "Wrote" in capsys.readouterr().out

    def test_resize_both_dimensions(self, sample_png, tmp_path):
        import imlite

        out = tmp_path / "small.png"
        main(["-q", "convert", sample_png, str(out), "--resize", "64x32"])
        assert imlite.read_image(str(out)).shape[:2] == (32, 64)

    def test_resize_width_only_keeps_aspect(self, sample_png, tmp_path):
        import imlite

        out = tmp_path / "narrow.png"
        main(["-q", "convert", sample_png, str(out), "--resize", "64x"])
        assert imlite.read_image(str(out)).shape[:2] == (64, 64)

    def test_rotate_flip_and_gray(self, sample_png, tmp_path):
        import imlite

        out = tmp_path / "t.png"
        main(["-q", "convert", sample_png, str(out), "--rotate", "90", "--flip", "h", "--gray"])
        assert imlite.read_image(str(out)).channels == 1

    def test_missing_input_is_a_handled_error(self, tmp_path, capsys):
        assert main(["-q", "convert", "nope.png", str(tmp_path / "o.png")]) == 1
        assert "error:" in capsys.readouterr().err


class TestParseSize:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("640x360", (640, 360)), ("640x", (640, None)), ("x360", (None, 360))],
    )
    def test_valid_forms(self, text, expected):
        assert _parse_size(text) == expected

    @pytest.mark.parametrize("text", ["640", "axb", "640x360x2", "x"])
    def test_invalid_forms_exit(self, text):
        with pytest.raises(SystemExit):
            _parse_size(text)


class TestEntryPointIsRegistered:
    def test_console_script_declared(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        assert 'imlite = "imlite.cli:main"' in pyproject.read_text(encoding="utf-8")
