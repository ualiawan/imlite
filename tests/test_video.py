"""Tests for the Video class and video I/O helpers."""

from pathlib import Path

import numpy as np
import pytest

import imlite
from imlite import FrameSequence, Image, Video
from imlite.ops.video_io import get_video_info

# ---------------------------------------------------------------------------
# Video construction / metadata
# ---------------------------------------------------------------------------


class TestVideoMetadata:
    def test_open_video(self, sample_video):
        vid = Video(sample_video)
        assert isinstance(vid, Video)

    def test_path_property(self, sample_video):
        vid = Video(sample_video)
        assert vid.path == sample_video

    def test_fps(self, sample_video):
        vid = Video(sample_video)
        assert vid.fps > 0

    def test_frame_count(self, sample_video):
        vid = Video(sample_video)
        assert vid.frame_count > 0

    def test_width_height(self, sample_video):
        vid = Video(sample_video)
        assert vid.width == 64
        assert vid.height == 64

    def test_duration(self, sample_video):
        vid = Video(sample_video)
        assert vid.duration > 0


# ---------------------------------------------------------------------------
# get_video_info helper
# ---------------------------------------------------------------------------


class TestGetVideoInfo:
    def test_returns_dict(self, sample_video):
        info = get_video_info(sample_video)
        assert isinstance(info, dict)

    def test_required_keys(self, sample_video):
        info = get_video_info(sample_video)
        for key in ("fps", "frame_count", "duration", "width", "height"):
            assert key in info, f"missing key: {key}"

    def test_width_height_match(self, sample_video):
        info = get_video_info(sample_video)
        assert info["width"] == 64
        assert info["height"] == 64


# ---------------------------------------------------------------------------
# Video.extract_frames
# ---------------------------------------------------------------------------


class TestVideoExtractFrames:
    def test_returns_frame_sequence(self, sample_video):
        vid = Video(sample_video)
        seq = vid.extract_frames()
        assert isinstance(seq, FrameSequence)

    def test_step_reduces_count(self, sample_video):
        vid = Video(sample_video)
        seq_full = vid.extract_frames(step=1)
        seq_half = vid.extract_frames(step=2)
        assert len(seq_half) <= len(seq_full)

    def test_extract_to_disk(self, sample_video, tmp_path):
        vid = Video(sample_video)
        vid.extract_frames(output_dir=str(tmp_path))
        frames = list(tmp_path.glob("*.png"))
        assert len(frames) > 0

    def test_start_end(self, sample_video):
        vid = Video(sample_video)
        seq = vid.extract_frames(start=5, end=15)
        frames = list(seq)
        # Should have roughly 10 frames (end-start)
        assert 1 <= len(frames) <= 30


# ---------------------------------------------------------------------------
# Video.from_frames / save
# ---------------------------------------------------------------------------


class TestVideoFromFrames:
    def test_from_frames_and_save(self, bgr_image, tmp_path):
        seq = FrameSequence.from_images([bgr_image] * 10)
        vid = Video.from_frames(seq, fps=5.0)
        out = str(tmp_path / "out.mp4")
        vid.save(out)
        import os

        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# imlite.load() with video path
# ---------------------------------------------------------------------------


class TestLoadDispatcherVideo:
    def test_load_returns_video(self, sample_video):
        obj = imlite.load(sample_video)
        assert isinstance(obj, Video)


# ---------------------------------------------------------------------------
# Re-encoding an existing file (used to raise NotImplementedError)
# ---------------------------------------------------------------------------


class TestVideoReEncode:
    def test_save_to_a_new_path_transcodes(self, sample_video, tmp_path):
        out = tmp_path / "transcoded.mp4"
        result = imlite.load(sample_video).save(str(out), show_progress=False)
        assert out.exists()
        assert result.path == str(out)

    def test_transcode_preserves_dimensions_and_count(self, sample_video, tmp_path):
        source = imlite.load(sample_video)
        expected = (source.width, source.height, source.frame_count)

        out = tmp_path / "copy.mp4"
        source.save(str(out), show_progress=False)

        info = imlite.video_info(str(out))
        assert (info["width"], info["height"], info["frame_count"]) == expected

    def test_fps_override(self, sample_video, tmp_path):
        out = tmp_path / "fast.mp4"
        imlite.load(sample_video).save(str(out), fps=30, show_progress=False)
        assert imlite.video_info(str(out))["fps"] == 30

    def test_saving_to_the_same_path_is_a_no_op(self, sample_video):
        before = Path(sample_video).stat().st_mtime_ns
        imlite.load(sample_video).save(sample_video)
        assert Path(sample_video).stat().st_mtime_ns == before


class TestPendingVideo:
    def test_is_pending_flag(self, bgr_image, tmp_path):
        video = FrameSequence.from_images([bgr_image] * 3).merge(fps=10)
        assert video.is_pending is True
        video.save(str(tmp_path / "out.mp4"), show_progress=False)
        assert video.is_pending is False

    def test_pending_reports_its_own_fps_and_codec(self, bgr_image):
        video = FrameSequence.from_images([bgr_image] * 3).merge(fps=12.5, codec="libx264")
        assert video.fps == 12.5
        assert video.codec == "libx264"
        assert video.frame_count == 3

    def test_pending_size_is_an_explicit_error(self, bgr_image):
        video = FrameSequence.from_images([bgr_image]).merge(fps=10)
        with pytest.raises(ValueError, match="not been encoded yet"):
            _ = video.width

    def test_pending_extract_frames_is_an_explicit_error(self, bgr_image):
        video = FrameSequence.from_images([bgr_image]).merge(fps=10)
        with pytest.raises(ValueError, match="no frames to extract"):
            video.extract_frames()

    def test_repr_shows_pending_state(self, bgr_image):
        assert "pending" in repr(FrameSequence.from_images([bgr_image]).merge(fps=10))


# ---------------------------------------------------------------------------
# Errors and edge cases
# ---------------------------------------------------------------------------


class TestVideoErrors:
    def test_missing_file_raises_read_error(self):
        with pytest.raises(imlite.ImliteReadError, match="not found"):
            imlite.video_info("no-such-video.mp4")

    def test_empty_sequence_will_not_encode(self, tmp_path):
        with pytest.raises(imlite.ImliteWriteError, match="empty"):
            FrameSequence.from_images([]).merge(fps=10).save(str(tmp_path / "empty.mp4"))

    def test_zero_fps_raises(self, bgr_image, tmp_path):
        with pytest.raises(ValueError, match="fps"):
            FrameSequence.from_images([bgr_image]).merge(fps=0).save(str(tmp_path / "o.mp4"))

    def test_bad_codec_raises_write_error(self, bgr_image, tmp_path):
        with pytest.raises((imlite.ImliteWriteError, imlite.ImliteError)):
            FrameSequence.from_images([bgr_image] * 2).merge(fps=10, codec="not-a-codec").save(
                str(tmp_path / "bad.mp4"), show_progress=False
            )


class TestMetadataCaching:
    def test_repeated_probes_reuse_the_cache(self, sample_video):
        from imlite.ops.video_io import _probe_video

        imlite.video_info(sample_video)
        before = _probe_video.cache_info().hits
        imlite.video_info(sample_video)
        assert _probe_video.cache_info().hits == before + 1

    def test_returned_dict_is_a_copy(self, sample_video):
        info = imlite.video_info(sample_video)
        info["fps"] = -1
        assert imlite.video_info(sample_video)["fps"] != -1

    def test_rewriting_the_file_invalidates_the_cache(self, sample_video, tmp_path):
        import shutil

        target = tmp_path / "clip.mp4"
        shutil.copy(sample_video, target)
        first = imlite.video_info(str(target))["frame_count"]

        FrameSequence.from_images([Image.from_numpy(np.zeros((64, 64, 3), np.uint8))] * 5).merge(
            fps=10
        ).save(str(target), show_progress=False)

        assert imlite.video_info(str(target))["frame_count"] != first


class TestColourFidelityThroughVideo:
    def test_bgr_survives_the_round_trip(self, tmp_path):
        # A distinctly blue frame must still be blue after encode + decode.
        blue = Image.from_numpy(np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8))
        out = tmp_path / "blue.mp4"
        FrameSequence.from_images([blue] * 5).merge(fps=10).save(str(out), show_progress=False)

        decoded = imlite.load(str(out)).extract_frames()[0]
        b, g, r = (int(v) for v in decoded.data[32, 32])
        assert b > 200 and g < 60 and r < 60, f"expected blue, got BGR ({b}, {g}, {r})"

    def test_raw_rgb_arrays_warn_about_channel_order(self, tmp_path, caplog):
        frames = [np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8)] * 3
        with caplog.at_level("WARNING"):
            imlite.merge_frames(frames, str(tmp_path / "raw.mp4"), fps=10, show_progress=False)
        assert any("BGR" in record.message for record in caplog.records)


class TestOutputDimensionsAreExact:
    """imageio defaults to macro_block_size=16, which turns 640x360 into 640x368."""

    @pytest.mark.parametrize(("width", "height"), [(640, 360), (100, 100), (80, 60), (320, 240)])
    def test_requested_size_is_the_encoded_size(self, tmp_path, width, height):
        frames = [np.full((height, width, 3), (i * 40 % 256, 90, 200), np.uint8) for i in range(4)]
        out = tmp_path / f"{width}x{height}.mp4"
        imlite.read_frames(frames).merge(fps=10).save(str(out), show_progress=False)

        info = imlite.video_info(str(out))
        assert (info["width"], info["height"]) == (width, height)

    def test_full_pipeline_honours_the_resize(self, sample_video, tmp_path):
        out = tmp_path / "resized.mp4"
        imlite.load(sample_video).extract_frames().resize(48, 36).merge(fps=10).save(
            str(out), show_progress=False
        )
        info = imlite.video_info(str(out))
        assert (info["width"], info["height"]) == (48, 36)

    def test_odd_dimensions_are_padded_and_reported(self, tmp_path, caplog):
        frames = [np.zeros((61, 41, 3), np.uint8) for _ in range(3)]
        out = tmp_path / "odd.mp4"
        with caplog.at_level("WARNING"):
            imlite.read_frames(frames).merge(fps=10).save(str(out), show_progress=False)

        info = imlite.video_info(str(out))
        assert (info["width"], info["height"]) == (42, 62)  # yuv420p needs even dimensions
        assert any("macro_block_size" in record.message for record in caplog.records)

    def test_macro_block_size_is_configurable(self, tmp_path):
        frames = [np.zeros((360, 640, 3), np.uint8) for _ in range(3)]
        out = tmp_path / "aligned.mp4"
        imlite.read_frames(frames).merge(fps=10).save(
            str(out), macro_block_size=16, show_progress=False
        )
        assert imlite.video_info(str(out))["height"] == 368  # opt-in alignment

    def test_invalid_macro_block_size_raises(self, tmp_path, bgr_image):
        with pytest.raises(ValueError, match="macro_block_size"):
            imlite.read_frames([bgr_image]).merge(fps=10).save(
                str(tmp_path / "x.mp4"), macro_block_size=0
            )
