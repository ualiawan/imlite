"""The ``Video`` class - a handle to a video file or a pending encode.

- Constructed from a file path: metadata is loaded lazily on first access.
- Constructed from a ``FrameSequence`` via ``from_frames()``: no file exists
  yet; encoding happens when ``.save()`` is called.
- Does NOT load any frames into memory on construction.
"""

from __future__ import annotations

import logging

from imlite.core.sequence import FrameSequence

log = logging.getLogger(__name__)

__all__ = ["Video"]


class Video:
    """A video file handle or a pending video encode.

    Args:
        path: Path to an existing video file.  Pass ``None`` only when
            constructing via :meth:`from_frames`.

    Note:
        Metadata properties (``fps``, ``frame_count``, etc.) are loaded
        lazily from the file on first access - no I/O on construction.
    """

    __slots__ = ("_path", "_meta", "_pending_frames", "_pending_fps", "_pending_codec")

    def __init__(self, path: str) -> None:
        self._path: str = str(path)
        self._meta: dict | None = None
        self._pending_frames: FrameSequence | None = None
        self._pending_fps: float = 30.0
        self._pending_codec: str = "libx264"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_frames(
        cls,
        frames: "FrameSequence",
        fps: float = 30.0,
        codec: str = "libx264",
    ) -> "Video":
        """Create a ``Video`` backed by a :class:`~imlite.core.sequence.FrameSequence`.

        The video is **not** encoded until :meth:`save` is called.

        Args:
            frames: Source frames (may be lazy - no frames decoded yet).
            fps: Frame rate for the output video.
            codec: FFmpeg codec (default ``"libx264"``).

        Returns:
            A new :class:`Video` with no file path yet.

        Example:
            >>> video = Video.from_frames(my_sequence, fps=25)
            >>> video.save("output.mp4")
        """
        # Placeholder path - will be set by save().
        vid = cls.__new__(cls)
        vid._path = ""
        vid._meta = None
        vid._pending_frames = frames
        vid._pending_fps = fps
        vid._pending_codec = codec
        return vid

    # ------------------------------------------------------------------
    # Properties  (lazy-loaded from file)
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        if self._meta is None:
            from imlite.ops.video_io import get_video_info  # noqa: PLC0415

            self._meta = get_video_info(self._path)
        return self._meta

    @property
    def path(self) -> str:
        """File path (empty string if not yet saved)."""
        return self._path

    @property
    def fps(self) -> float:
        """Frames per second."""
        if self._pending_frames is not None:
            return self._pending_fps
        return float(self._load_meta().get("fps", 0.0))

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        if self._pending_frames is not None:
            return len(self._pending_frames)
        return int(self._load_meta().get("frame_count", 0))

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self._pending_frames is not None:
            fc = self.frame_count
            fps = self._pending_fps
            return fc / fps if fps else 0.0
        return float(self._load_meta().get("duration", 0.0))

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self._load_meta().get("width", 0))

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self._load_meta().get("height", 0))

    @property
    def codec(self) -> str:
        """Codec string (e.g. ``"h264"``)."""
        if self._pending_frames is not None:
            return self._pending_codec
        return str(self._load_meta().get("codec", ""))

    @property
    def info(self) -> dict:
        """All metadata as a flat dictionary.

        Keys: ``path``, ``fps``, ``frame_count``, ``duration``, ``width``,
        ``height``, ``codec``, ``size_bytes``.
        """
        return self._load_meta()

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
    ) -> "FrameSequence":
        """Extract frames from this video file.

        Args:
            output_dir: If given, frames are saved here as image files.
                If ``None``, frames are returned in-memory.
            step: Take every *step*-th frame.
            start: First frame index (inclusive).
            end: Last frame index (exclusive).  ``None`` = until end.
            fmt: Image format for saved files (e.g. ``"png"``).
            show_progress: Show a tqdm progress bar.

        Returns:
            A :class:`~imlite.core.sequence.FrameSequence`.

        Example:
            >>> seq = imlite.open("video.mp4").extract_frames(step=2)
        """
        from imlite.ops.video_io import extract_frames as _extract  # noqa: PLC0415

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
        show_progress: bool = True,
    ) -> "Video":
        """Write this video to disk.

        If the video was created via :meth:`from_frames`, frames are encoded
        now (lazy decoding + transforms happen here).

        If the video was opened from a file this is a no-op unless a new
        *path* is provided (copy / re-encode not yet supported - raises
        ``NotImplementedError``).

        Args:
            path: Destination file path.
            fps: Frame rate override.  Defaults to the value set at
                construction.
            codec: Codec override.
            show_progress: Show a tqdm progress bar during encoding.

        Returns:
            ``self`` with ``path`` updated to *path*.

        Example:
            >>> seq.merge(fps=25).save("out.mp4")
        """
        from imlite.ops.video_io import merge_frames  # noqa: PLC0415

        if self._pending_frames is not None:
            out_fps = fps if fps is not None else self._pending_fps
            out_codec = codec if codec is not None else self._pending_codec
            merge_frames(
                self._pending_frames,
                output_path=str(path),
                fps=out_fps,
                codec=out_codec,
                show_progress=show_progress,
            )
            self._path = str(path)
            self._pending_frames = None
            self._meta = None  # invalidate cached metadata
            return self

        if str(path) == self._path:
            log.debug("save() called with same path - nothing to do.")
            return self

        raise NotImplementedError(
            "Re-encoding an existing video file is not yet supported. "
            "Use extract_frames() -> transform -> merge() -> save() instead."
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._pending_frames is not None:
            return (
                f"Video(pending, fps={self._pending_fps}, "
                f"codec={self._pending_codec!r}, "
                f"frames={len(self._pending_frames)})"
            )
        return f"Video(path={self._path!r})"
