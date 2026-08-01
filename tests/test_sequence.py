"""Tests for FrameSequence (core/sequence.py)."""

import numpy as np
import pytest

import imlite
from imlite import FrameSequence, Image

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFrameSequenceConstruction:
    def test_from_images_list_of_arrays(self, bgr_array):
        seq = FrameSequence.from_images([bgr_array, bgr_array, bgr_array])
        assert len(seq) == 3

    def test_from_images_list_of_image(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image, bgr_image])
        assert len(seq) == 2

    def test_from_dir(self, sample_frames_dir):
        seq = FrameSequence.from_dir(sample_frames_dir)
        assert len(seq) == 10

    def test_from_video(self, sample_video):
        seq = FrameSequence.from_video(sample_video)
        # Length may be approximate; just check > 0
        assert len(seq) > 0

    def test_from_video_with_step(self, sample_video):
        seq_full = FrameSequence.from_video(sample_video)
        seq_step = FrameSequence.from_video(sample_video, step=2)
        assert len(seq_step) <= len(seq_full)


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class TestFrameSequenceIteration:
    def test_iter_yields_images(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image, bgr_image])
        frames = list(seq)
        assert all(isinstance(f, Image) for f in frames)

    def test_iter_count(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 5)
        assert sum(1 for _ in seq) == 5

    def test_getitem_int(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image, bgr_image, bgr_image])
        frame = seq[0]
        assert isinstance(frame, Image)

    def test_getitem_slice(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 6)
        sub = seq[1:4]
        assert isinstance(sub, FrameSequence)
        assert len(sub) == 3


# ---------------------------------------------------------------------------
# Lazy transforms
# ---------------------------------------------------------------------------


class TestFrameSequenceLazyTransforms:
    def test_crop_deferred(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 3)
        seq2 = seq.crop(0, 0, 50, 50)
        # crop should return a new FrameSequence without immediately processing
        assert isinstance(seq2, FrameSequence)
        frames = list(seq2)
        assert all(f.shape == (50, 50, 3) for f in frames)

    def test_rotate_deferred(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 2)
        seq2 = seq.rotate(180)
        frames = list(seq2)
        assert all(isinstance(f, Image) for f in frames)

    def test_resize_deferred(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 2)
        seq2 = seq.resize(32, 32)
        frames = list(seq2)
        assert all(f.shape == (32, 32, 3) for f in frames)

    def test_chained_transforms(self, bgr_image):
        seq = (
            FrameSequence.from_images([bgr_image] * 3)
            .crop(0, 0, 100, 100)
            .rotate(90)
            .resize(50, 50)
        )
        frames = list(seq)
        assert all(f.shape == (50, 50, 3) for f in frames)

    def test_original_unchanged(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 3)
        _ = seq.crop(0, 0, 50, 50)
        # Original seq should still yield full-size frames
        frames = list(seq)
        assert all(f.shape == (256, 256, 3) for f in frames)


# ---------------------------------------------------------------------------
# save_frames
# ---------------------------------------------------------------------------


class TestFrameSequenceSaveFrames:
    def test_saves_pngs(self, bgr_image, tmp_path):
        seq = FrameSequence.from_images([bgr_image] * 4)
        seq.save_frames(str(tmp_path), fmt="png")
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) == 4

    def test_custom_prefix(self, bgr_image, tmp_path):
        seq = FrameSequence.from_images([bgr_image] * 2)
        seq.save_frames(str(tmp_path), fmt="png", prefix="img_")
        files = sorted(tmp_path.glob("img_*.png"))
        assert len(files) == 2


# ---------------------------------------------------------------------------
# to_list / merge
# ---------------------------------------------------------------------------


class TestFrameSequenceToList:
    def test_to_list_returns_list_of_images(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image, bgr_image])
        result = seq.to_list()
        assert isinstance(result, list)
        assert all(isinstance(f, Image) for f in result)

    def test_merge_returns_video(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 5)
        vid = seq.merge(fps=10.0)
        assert isinstance(vid, imlite.Video)


# ---------------------------------------------------------------------------
# Laziness (DG-5) - a long video must never be materialised in RAM
# ---------------------------------------------------------------------------


class TestFrameSequenceLaziness:
    def test_extract_frames_without_output_dir_is_lazy(self, sample_video):
        seq = imlite.extract_frames(sample_video, step=2)
        assert seq._source_type == "video", "extract_frames() must not materialise frames"
        assert seq._eager_frames is None

    def test_video_extract_frames_is_lazy(self, sample_video):
        seq = imlite.load(sample_video).extract_frames()
        assert seq._eager_frames is None

    def test_queued_transforms_decode_nothing(self, sample_video, monkeypatch):
        from imlite.ops import video_io

        decoded = []
        original = video_io.iter_video_frames

        def _spy(*args, **kwargs):
            decoded.append(1)
            yield from original(*args, **kwargs)

        monkeypatch.setattr(video_io, "iter_video_frames", _spy)

        seq = imlite.load(sample_video).extract_frames(step=2).resize(32, 32).flip("h")
        assert decoded == [], "building a chain must not start decoding"

        list(seq)
        assert decoded == [1], "iterating should decode exactly once"

    def test_getitem_zero_stops_early(self, sample_video):
        """seq[0] must not decode the whole video just to return frame 0."""
        seq = imlite.load(sample_video).extract_frames()
        consumed = 0
        original_iter = type(seq).__iter__

        def _counting_iter(self):
            nonlocal consumed
            for frame in original_iter(self):
                consumed += 1
                yield frame

        type(seq).__iter__ = _counting_iter
        try:
            assert isinstance(seq[0], Image)
        finally:
            type(seq).__iter__ = original_iter
        assert consumed == 1, f"seq[0] decoded {consumed} frames, expected 1"

    def test_len_of_video_sequence_needs_no_decode(self, sample_video):
        assert len(imlite.load(sample_video).extract_frames(step=3)) == 10

    def test_extract_to_disk_returns_a_dir_backed_sequence(self, sample_video, tmp_path):
        seq = imlite.extract_frames(sample_video, str(tmp_path), step=5)
        assert seq._source_type == "dir"
        assert len(seq) == 6


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestFrameSequenceIndexing:
    def test_negative_index(self, bgr_image):
        frames = [bgr_image.brightness(f) for f in (0.5, 1.0, 1.5)]
        seq = FrameSequence.from_images(frames)
        assert np.array_equal(seq[-1].data, frames[-1].data)

    def test_out_of_range_raises_indexerror(self, bgr_image):
        with pytest.raises(IndexError):
            FrameSequence.from_images([bgr_image])[5]

    def test_very_negative_index_raises(self, bgr_image):
        with pytest.raises(IndexError):
            FrameSequence.from_images([bgr_image])[-5]

    def test_non_integer_index_raises_typeerror(self, bgr_image):
        with pytest.raises(TypeError, match="integers or slices"):
            FrameSequence.from_images([bgr_image])["first"]

    def test_index_applies_pending_ops(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 3).resize(32, 32)
        assert seq[1].shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# Construction guards and new transforms
# ---------------------------------------------------------------------------


class TestFrameSequenceValidation:
    def test_from_images_rejects_other_types(self):
        with pytest.raises(TypeError, match="item 1"):
            FrameSequence.from_images([np.zeros((4, 4, 3), np.uint8), "not a frame"])

    @pytest.mark.parametrize(("kwarg", "value"), [("step", 0), ("start", -1)])
    def test_from_video_validates_arguments(self, sample_video, kwarg, value):
        with pytest.raises(ValueError):
            FrameSequence.from_video(sample_video, **{kwarg: value})

    def test_repr_mentions_pending_ops(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image]).resize(8, 8).flip("h")
        assert "pending_ops=2" in repr(seq)


class TestFrameSequenceNewTransforms:
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("blur", (1.0,)),
            ("brightness", (1.2,)),
            ("contrast", (1.2,)),
            ("thumbnail", (32,)),
            ("to_gray", ()),
            ("pad", (2, 2, 2, 2)),
        ],
    )
    def test_transform_applies_to_every_frame(self, bgr_image, method, args):
        seq = getattr(FrameSequence.from_images([bgr_image] * 2), method)(*args)
        frames = list(seq)
        assert len(frames) == 2
        assert all(isinstance(f, Image) for f in frames)

    def test_to_gray_changes_channel_count(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 2).to_gray()
        assert all(f.channels == 1 for f in seq)


class TestSaveFramesReturnsSequence:
    def test_returns_a_dir_backed_sequence_for_chaining(self, bgr_image, tmp_path):
        out = FrameSequence.from_images([bgr_image] * 3).save_frames(str(tmp_path))
        assert isinstance(out, FrameSequence)
        assert len(out) == 3

    def test_empty_sequence_writes_nothing_and_does_not_crash(self, tmp_path):
        imlite.set_verbosity("INFO")
        try:
            out = FrameSequence.from_images([]).save_frames(str(tmp_path))
        finally:
            imlite.set_verbosity("WARNING")
        assert len(out) == 0


class TestFrameSequenceColourTransforms:
    @pytest.mark.parametrize(("method", "space"), [("to_rgb", "RGB"), ("to_bgr", "BGR")])
    def test_retags_every_frame(self, bgr_image, method, space):
        seq = getattr(FrameSequence.from_images([bgr_image] * 3), method)()
        assert [f.color_space for f in seq] == [space] * 3

    def test_to_rgb_then_to_bgr_round_trips(self, bgr_image):
        seq = FrameSequence.from_images([bgr_image] * 2).to_rgb().to_bgr()
        assert all(np.array_equal(f.data, bgr_image.data) for f in seq)

    def test_merge_fixes_colour_order_regardless(self, tmp_path):
        """merge() converts to RGB itself, so an explicit to_rgb() must not double-swap."""
        blue = Image.from_numpy(np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8))
        out = tmp_path / "blue.mp4"
        FrameSequence.from_images([blue] * 5).to_rgb().merge(fps=10).save(
            str(out), show_progress=False
        )
        b, g, r = (int(v) for v in imlite.load(str(out)).extract_frames()[0].data[32, 32])
        assert b > 200 and g < 60 and r < 60, f"expected blue, got BGR ({b}, {g}, {r})"
