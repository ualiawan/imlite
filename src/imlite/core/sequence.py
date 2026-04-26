"""The ``FrameSequence`` class - an ordered, iterable collection of frames.

- Can be **lazy** (backed by a video file; frames decoded on demand) or
  **eager** (backed by an in-memory list of ``Image`` objects).
- Transforms (``rotate``, ``crop``, …) are **deferred** - stored as lambdas
  in ``_pending_ops`` and applied frame-by-frame during iteration.
- Peak memory usage is O(1) regardless of sequence length.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from imlite.core.video import Video

import numpy as np

from imlite.core.image import Image
from imlite.utils.path import sorted_frame_paths

log = logging.getLogger(__name__)

__all__ = ["FrameSequence"]

# Type alias for the internal source
_Source = Union[str, list]  # str = video path or directory; list = eager frames


class FrameSequence:
    """An ordered, iterable collection of image frames.

    Frames are always yielded as :class:`~imlite.core.image.Image` objects.

    Construct via the class methods rather than ``__init__`` directly:

    - :meth:`from_video` - lazy stream from a video file.
    - :meth:`from_dir` - lazy stream from a directory of image files.
    - :meth:`from_images` - eager list of ``Image`` or ``np.ndarray``.
    """

    __slots__ = (
        "_source",
        "_source_type",  # "video" | "dir" | "list"
        "_step",
        "_start",
        "_end",
        "_pending_ops",
        "_eager_frames",  # populated by to_list() or from_images()
    )

    def __init__(self) -> None:
        # Use class methods for construction.
        self._source: str | None = None
        self._source_type: str = "list"
        self._step: int = 1
        self._start: int = 0
        self._end: int | None = None
        self._pending_ops: list[Callable[[Image], Image]] = []
        self._eager_frames: list[Image] | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_video(
        cls,
        path: str,
        step: int = 1,
        start: int = 0,
        end: int | None = None,
    ) -> "FrameSequence":
        """Create a lazy ``FrameSequence`` backed by a video file.

        Frames are decoded one at a time during iteration - no frames are
        loaded into memory on construction.

        Args:
            path: Path to the video file.
            step: Take every *step*-th frame.
            start: First frame index (inclusive, 0-based).
            end: Last frame index (exclusive).  ``None`` = until end.

        Returns:
            A new lazy :class:`FrameSequence`.
        """
        seq = cls()
        seq._source = str(path)
        seq._source_type = "video"
        seq._step = step
        seq._start = start
        seq._end = end
        return seq

    @classmethod
    def from_dir(cls, directory: str) -> "FrameSequence":
        """Create a lazy ``FrameSequence`` backed by a directory of images.

        Image files are discovered in natural sort order.

        Args:
            directory: Path to a directory containing image files.

        Returns:
            A new lazy :class:`FrameSequence`.
        """
        seq = cls()
        seq._source = str(directory)
        seq._source_type = "dir"
        return seq

    @classmethod
    def from_images(
        cls,
        images: list,
    ) -> "FrameSequence":
        """Create an eager ``FrameSequence`` from a list of frames.

        Args:
            images: A list of :class:`~imlite.core.image.Image` objects or
                ``np.ndarray`` arrays.  Arrays are wrapped in ``Image``
                automatically.

        Returns:
            A new eager :class:`FrameSequence`.
        """
        from imlite.core.image import Image  # noqa: PLC0415

        seq = cls()
        seq._source_type = "list"
        wrapped: list[Image] = []
        for item in images:
            if isinstance(item, Image):
                wrapped.append(item)
            elif isinstance(item, np.ndarray):
                wrapped.append(Image.from_numpy(item))
            else:
                raise TypeError(
                    f"from_images() expects Image or np.ndarray, got {type(item).__name__!r}."
                )
        seq._eager_frames = wrapped
        return seq

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of frames.

        For lazy sources this is computed from metadata (video) or directory
        listing - no frames are decoded.
        """
        if self._source_type == "list":
            return len(self._eager_frames or [])

        if self._source_type == "dir":
            assert self._source is not None
            return len(sorted_frame_paths(self._source))

        if self._source_type == "video":
            from imlite.ops.video_io import get_video_info  # noqa: PLC0415

            assert self._source is not None
            info = get_video_info(self._source)
            total = info["frame_count"]
            stop = self._end if self._end is not None else total
            return max(0, len(range(self._start, min(stop, total), self._step)))

        return 0

    def __iter__(self) -> Iterator[Image]:
        """Yield frames one at a time, applying any pending transforms."""
        for frame in self._iter_source():
            for op in self._pending_ops:
                frame = op(frame)
            yield frame

    def __getitem__(self, idx: int | slice) -> "Image | FrameSequence":
        """Access frames by index or slice.

        - Integer index -> :class:`~imlite.core.image.Image`.
        - Slice -> new :class:`FrameSequence` (eager, pending ops applied).
        """
        if isinstance(idx, int):
            frames = self.to_list()
            return frames[idx]
        # Slice -> materialise then re-wrap.
        frames = self.to_list()
        return FrameSequence.from_images(frames[idx])

    def __repr__(self) -> str:
        try:
            n = len(self)
        except Exception:  # noqa: BLE001
            n = "?"
        src = self._source or "in-memory"
        ops = len(self._pending_ops)
        return f"FrameSequence(source={src!r}, frames={n}, pending_ops={ops})"

    # ------------------------------------------------------------------
    # Batch transforms  (deferred - ops queued, not executed yet)
    # ------------------------------------------------------------------

    def crop(self, x: int, y: int, width: int, height: int) -> "FrameSequence":
        """Queue a crop transform for every frame.

        Args:
            x: Left edge of the crop box.
            y: Top edge of the crop box.
            width: Crop width in pixels.
            height: Crop height in pixels.

        Returns:
            New :class:`FrameSequence` with the crop queued.
        """
        from imlite.ops.geometry import crop as _crop  # noqa: PLC0415

        new_seq = self._clone()
        new_seq._pending_ops.append(lambda img: _crop(img, x, y, width, height))  # type: ignore[arg-type]
        return new_seq

    def rotate(self, angle: float, expand: bool = True) -> "FrameSequence":
        """Queue a rotation transform for every frame.

        Args:
            angle: Rotation angle in degrees (counter-clockwise).
            expand: Expand canvas to fit (default ``True``).

        Returns:
            New :class:`FrameSequence` with the rotation queued.
        """
        from imlite.ops.geometry import rotate as _rotate  # noqa: PLC0415

        new_seq = self._clone()
        new_seq._pending_ops.append(lambda img: _rotate(img, angle, expand))  # type: ignore[arg-type]
        return new_seq

    def resize(
        self,
        width: int | None = None,
        height: int | None = None,
        keep_aspect: bool = False,
    ) -> "FrameSequence":
        """Queue a resize transform for every frame.

        Args:
            width: Target width, or ``None`` to infer.
            height: Target height, or ``None`` to infer.
            keep_aspect: Preserve aspect ratio.

        Returns:
            New :class:`FrameSequence` with the resize queued.
        """
        from imlite.ops.geometry import resize as _resize  # noqa: PLC0415

        new_seq = self._clone()
        new_seq._pending_ops.append(lambda img: _resize(img, width, height, keep_aspect))  # type: ignore[arg-type]
        return new_seq

    def flip(self, axis: str = "h") -> "FrameSequence":
        """Queue a flip transform for every frame.

        Args:
            axis: ``"h"`` / ``"horizontal"``, ``"v"`` / ``"vertical"``,
                or ``"both"``.

        Returns:
            New :class:`FrameSequence` with the flip queued.
        """
        from imlite.ops.geometry import flip as _flip  # noqa: PLC0415

        new_seq = self._clone()
        new_seq._pending_ops.append(lambda img: _flip(img, axis))  # type: ignore[arg-type]
        return new_seq

    def apply(self, fn: Callable[[Image], Image]) -> "FrameSequence":
        """Queue a custom per-frame function.

        Args:
            fn: A callable that accepts an :class:`~imlite.core.image.Image`
                and returns a transformed :class:`~imlite.core.image.Image`.

        Returns:
            New :class:`FrameSequence` with *fn* queued.

        Example:
            >>> seq.apply(lambda img: img.to_gray())
        """
        new_seq = self._clone()
        new_seq._pending_ops.append(fn)
        return new_seq

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_frames(
        self,
        output_dir: str,
        fmt: str = "png",
        prefix: str = "frame",
        zero_pad: int = 5,
        show_progress: bool = True,
    ) -> None:
        """Write all frames to *output_dir* as image files.

        Pending transforms are applied during this call.

        Args:
            output_dir: Directory to write frames into (created if needed).
            fmt: Image format (e.g. ``"png"``, ``"jpg"``).
            prefix: Filename prefix (e.g. ``"frame"`` -> ``"frame_00001.png"``).
            zero_pad: Number of digits for the zero-padded index.
            show_progress: Show a tqdm progress bar.

        Example:
            >>> seq.save_frames("frames/", fmt="jpg")
        """
        from imlite.ops.io import write_image  # noqa: PLC0415
        from imlite.utils.log import progress  # noqa: PLC0415
        from imlite.utils.path import ensure_dir  # noqa: PLC0415

        ensure_dir(output_dir)
        total = len(self) if hasattr(self, "__len__") else None
        log.info("Saving frames to %s (fmt=%s)", output_dir, fmt)

        for i, frame in enumerate(
            progress(self, desc="Saving frames", total=total, unit="frame", show=show_progress)
        ):
            dest = f"{output_dir}/{prefix}_{i:0{zero_pad}d}.{fmt}"
            write_image(frame, dest)

        log.info("Done. %d frames saved.", i + 1 if total else "?")

    def merge(self, fps: float = 30.0, codec: str = "libx264") -> "Video":
        """Assemble this sequence into a :class:`~imlite.core.video.Video`.

        The video is **not** written to disk yet - call ``.save("out.mp4")``
        on the returned :class:`~imlite.core.video.Video` to encode it.

        Args:
            fps: Output frame rate.
            codec: FFmpeg codec name (default ``"libx264"``).

        Returns:
            A :class:`~imlite.core.video.Video` with this sequence as its
            pending frame source.

        Example:
            >>> seq.rotate(90).merge(fps=25).save("out.mp4")
        """
        from imlite.core.video import Video  # noqa: PLC0415

        return Video.from_frames(self, fps=fps, codec=codec)

    def to_list(self) -> list[Image]:
        """Force eager evaluation and return all frames as a list.

        This loads every frame into RAM - use with caution on long videos.

        Returns:
            List of :class:`~imlite.core.image.Image` objects.
        """
        if self._source_type == "list" and not self._pending_ops:
            return list(self._eager_frames or [])
        return list(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clone(self) -> "FrameSequence":
        """Return a shallow copy of this sequence (same source, same pending ops copy)."""
        new_seq = FrameSequence()
        new_seq._source = self._source
        new_seq._source_type = self._source_type
        new_seq._step = self._step
        new_seq._start = self._start
        new_seq._end = self._end
        new_seq._pending_ops = list(self._pending_ops)  # shallow copy of op list
        new_seq._eager_frames = self._eager_frames  # shared reference - immutable
        return new_seq

    def _iter_source(self) -> Iterator[Image]:
        """Yield raw (un-transformed) frames from the underlying source."""
        from imlite.core.image import Image  # noqa: PLC0415

        if self._source_type == "list":
            yield from (self._eager_frames or [])
            return

        if self._source_type == "dir":
            assert self._source is not None
            paths = sorted_frame_paths(self._source)
            for p in paths:
                from imlite.ops.io import read_image  # noqa: PLC0415

                yield read_image(p)
            return

        if self._source_type == "video":
            assert self._source is not None
            import cv2  # noqa: PLC0415
            import imageio.v2 as iio2  # noqa: PLC0415

            try:
                reader = iio2.get_reader(self._source, plugin="ffmpeg")
                meta = reader.get_meta_data()
                total = int(meta.get("nframes", 0)) or None
                stop = self._end if self._end is not None else (total or 10**9)
                indices = range(self._start, stop, self._step)

                for frame_idx in indices:
                    try:
                        rgb_arr = reader.get_data(frame_idx)
                    except IndexError:
                        break
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Frame %d unreadable, skipping: %s", frame_idx, exc)
                        continue
                    # imageio yields RGB -> convert to BGR.
                    if rgb_arr.ndim == 3 and rgb_arr.shape[2] == 3:
                        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                    else:
                        bgr = rgb_arr
                    yield Image.from_numpy(bgr, color_space="BGR")
            finally:
                reader.close()
