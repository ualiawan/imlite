"""The :class:`Video` class - a handle to a video file, or to a pending encode.

A ``Video`` is one of two things:

- **File-backed** (``Video("clip.mp4")``): metadata is read lazily on first
  access and then cached.  No frames are ever loaded on construction.
- **Pending** (``Video.from_frames(seq, fps)``): no file exists yet; the
  frames are encoded when :meth:`save` is called.

Either way the whole pipeline stays streaming - a two-hour 4K video is
processed one frame at a time.
"""

# PEP 563: methods return Video, and FrameSequence is a
# TYPE_CHECKING-only import. Do not remove.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from imlite.core.sequence import FrameSequence

log = logging.getLogger(__name__)

__all__ = ["Video"]


class Video:
    """A video file handle, or a video waiting to be encoded.

    Args:
        path: Path to an existing video file.

    Note:
        Metadata properties (:attr:`fps`, :attr:`frame_count`, ...) hit the
        file on first access and are cached from then on.

    Example:
        >>> vid = imlite.load("clip.mp4")
        >>> vid.fps, vid.frame_count  # doctest: +SKIP
        (25.0, 300)
        >>> vid.extract_frames(step=2).resize(640, 360).merge(12.5).save("small.mp4")
    """

    __slots__ = ("_meta", "_path", "_pending_codec", "_pending_fps", "_pending_frames")

    def __init__(self, path: str) -> None:
        self._path: str = str(path)
        self._meta: dict[str, Any] | None = None
        self._pending_frames: FrameSequence | None = None
        self._pending_fps: float = 30.0
        self._pending_codec: str = "libx264"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_frames(
        cls,
        frames: FrameSequence,
        fps: float = 30.0,
        codec: str = "libx264",
    ) -> Video:
        """Create a ``Video`` backed by a :class:`~imlite.core.sequence.FrameSequence`.

        Nothing is encoded until :meth:`save` is called, so the source sequence
        may still be lazy at this point.

        Args:
            frames: Source frames.
            fps: Frame rate for the output.
            codec: FFmpeg codec name (default ``"libx264"``).

        Returns:
            A new pending :class:`Video` with no file path yet.

        Example:
            >>> Video.from_frames(my_sequence, fps=25).save("out.mp4")
        """
        video = cls.__new__(cls)
        video._path = ""
        video._meta = None
        video._pending_frames = frames
        video._pending_fps = fps
        video._pending_codec = codec
        return video

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_pending(self) -> bool:
        """``True`` when this video has not been encoded to disk yet."""
        return self._pending_frames is not None

    @property
    def path(self) -> str:
        """File path, or an empty string for a video that is still pending."""
        return self._path

    @property
    def fps(self) -> float:
        """Frames per second."""
        if self._pending_frames is not None:
            return self._pending_fps
        return float(self._load_meta()["fps"])

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        if self._pending_frames is not None:
            return len(self._pending_frames)
        return int(self._load_meta()["frame_count"])

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self._pending_frames is not None:
            return self.frame_count / self._pending_fps if self._pending_fps else 0.0
        return float(self._load_meta()["duration"])

    @property
    def width(self) -> int:
        """Frame width in pixels.

        Raises:
            ValueError: If this video is still pending - the size is not known
                until the first frame has been produced.
        """
        return int(self._require_file_meta("width")["width"])

    @property
    def height(self) -> int:
        """Frame height in pixels.

        Raises:
            ValueError: If this video is still pending.
        """
        return int(self._require_file_meta("height")["height"])

    @property
    def codec(self) -> str:
        """Codec name, e.g. ``"h264"``."""
        if self._pending_frames is not None:
            return self._pending_codec
        return str(self._load_meta()["codec"])

    @property
    def info(self) -> dict[str, Any]:
        """All file metadata as a flat dict.

        Keys: ``path``, ``fps``, ``frame_count``, ``duration``, ``width``,
        ``height``, ``codec``, ``size_bytes``.

        Raises:
            ValueError: If this video is still pending.
        """
        return self._require_file_meta("info")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        output_dir: str | None = None,
        step: int = 1,
        start: int = 0,
        end: int | None = None,
        fmt: str = "png",
        show_progress: bool = True,
    ) -> FrameSequence:
        """Extract frames from this video.

        With no *output_dir* the result is lazy - nothing is decoded until you
        iterate it.

        Args:
            output_dir: Directory to write frame images into, or ``None``
                (default) to stream frames on demand without writing anything.
            step: Take every *step*-th frame.
            start: First frame index (inclusive).
            end: Index to stop before (exclusive), or ``None`` for all.
            fmt: Image format for written frames, e.g. ``"png"``.
            show_progress: Show a progress bar while writing frames.

        Returns:
            A :class:`~imlite.core.sequence.FrameSequence`.

        Raises:
            ValueError: If this video is still pending.

        Example:
            >>> seq = imlite.load("clip.mp4").extract_frames(step=2)
        """
        if self._pending_frames is not None:
            raise ValueError(
                "This Video has not been encoded yet, so there are no frames to extract. "
                "Use the FrameSequence you built it from, or call save() first."
            )

        from imlite.ops.video_io import extract_frames as _extract

        return _extract(
            self._path,
            output_dir=output_dir,
            step=step,
            start=start,
            end=end,
            fmt=fmt,
            show_progress=show_progress,
        )

    def save(
        self,
        path: str,
        fps: float | None = None,
        codec: str | None = None,
        macro_block_size: int = 2,
        show_progress: bool = True,
    ) -> Video:
        """Write this video to disk.

        For a pending video, the frames are decoded, transformed and encoded
        now - one at a time.  For a file-backed video, this re-encodes the file
        at *path*, which is how you transcode or change frame rate.

        Args:
            path: Destination file path.
            fps: Frame-rate override.  Defaults to the pending rate, or the
                source rate when re-encoding.
            codec: Codec override (default ``"libx264"``).
            macro_block_size: Round frame dimensions up to a multiple of this.
                The default ``2`` satisfies ``yuv420p`` without inflating the
                size - see :func:`~imlite.ops.video_io.merge_frames`.
            show_progress: Show a progress bar while encoding.

        Returns:
            ``self``, with :attr:`path` updated to *path*.

        Raises:
            ImliteWriteError: If the output cannot be written.
            ImliteFFmpegError: If no ffmpeg binary is available.

        Example:
            >>> seq.merge(fps=25).save("out.mp4")
            >>> imlite.load("in.mov").save("out.mp4")  # transcode
        """
        from imlite.ops.video_io import merge_frames

        if self._pending_frames is not None:
            source: FrameSequence = self._pending_frames
            out_fps = fps if fps is not None else self._pending_fps
            out_codec = codec if codec is not None else self._pending_codec
        else:
            if str(path) == self._path:
                log.debug("save() called with the source path - nothing to do.")
                return self
            source = self.extract_frames()
            out_fps = fps if fps is not None else (self.fps or 30.0)
            out_codec = codec if codec is not None else "libx264"
            log.info("Re-encoding %s -> %s", self._path, path)

        merge_frames(
            source,
            output_path=str(path),
            fps=out_fps,
            codec=out_codec,
            macro_block_size=macro_block_size,
            show_progress=show_progress,
        )

        self._path = str(path)
        self._pending_frames = None
        self._meta = None  # the file changed; drop cached metadata
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict[str, Any]:
        """Read and cache this file's metadata."""
        if self._meta is None:
            from imlite.ops.video_io import get_video_info

            self._meta = get_video_info(self._path)
        return self._meta

    def _require_file_meta(self, attribute: str) -> dict[str, Any]:
        """Return file metadata, refusing when this video has not been encoded yet."""
        if self._pending_frames is not None:
            raise ValueError(
                f"'{attribute}' is not known for a Video that has not been encoded yet. "
                "Call save() first, or inspect the source frames directly."
            )
        return self._load_meta()

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._pending_frames is not None:
            return (
                f"Video(pending, fps={self._pending_fps}, codec={self._pending_codec!r}, "
                f"frames={len(self._pending_frames)})"
            )
        return f"Video(path={self._path!r})"
