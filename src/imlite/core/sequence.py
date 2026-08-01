"""The :class:`FrameSequence` class - an ordered, iterable collection of frames.

A sequence is either **lazy** (backed by a video file or a directory of
images, decoding one frame at a time) or **eager** (backed by a list already in
memory).  Either way it yields :class:`~imlite.core.image.Image` objects.

Transforms are **deferred**.  ``seq.resize(640, 360).flip("h")`` queues two
functions and returns immediately; they run per frame during iteration.  Peak
memory therefore stays at one frame no matter how long the video is::

    imlite.load("4k-2hour.mp4").extract_frames(step=5).resize(640, 360).merge(25).save("out.mp4")
"""

# PEP 563: methods return FrameSequence, and merge() annotates Video,
# which is a TYPE_CHECKING-only import. Do not remove.
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from imlite._typing import Array
from imlite.core.image import Image
from imlite.utils.path import sorted_frame_paths

if TYPE_CHECKING:  # pragma: no cover
    from imlite.core.video import Video

log = logging.getLogger(__name__)

__all__ = ["FrameSequence"]

SourceType = Literal["video", "dir", "list"]
FlipAxis = Literal["h", "horizontal", "v", "vertical", "both"]


class FrameSequence:
    """An ordered, iterable collection of image frames.

    Build one with a class method rather than calling ``FrameSequence()``:

    - :meth:`from_video` - lazy stream from a video file.
    - :meth:`from_dir` - lazy stream from a directory of image files.
    - :meth:`from_images` - eager list of ``Image`` or ``Array`` frames.

    Example:
        >>> seq = imlite.load("clip.mp4").extract_frames(step=2)
        >>> seq.resize(640, 360).merge(fps=25).save("small.mp4")
    """

    __slots__ = (
        "_eager_frames",
        "_end",
        "_pending_ops",
        "_source",
        "_source_type",
        "_start",
        "_step",
    )

    def __init__(self) -> None:
        self._source: str | None = None
        self._source_type: SourceType = "list"
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
    ) -> FrameSequence:
        """Create a lazy sequence backed by a video file.

        No frames are decoded until the sequence is iterated.

        Args:
            path: Path to the video file.
            step: Take every *step*-th frame.
            start: First frame index (inclusive, 0-based).
            end: Index to stop before (exclusive), or ``None`` for the whole file.

        Returns:
            A new lazy :class:`FrameSequence`.

        Raises:
            ValueError: If *step* is not positive or *start* is negative.
        """
        if step < 1:
            raise ValueError(f"'step' must be >= 1, got {step}.")
        if start < 0:
            raise ValueError(f"'start' must be >= 0, got {start}.")

        seq = cls()
        seq._source = str(path)
        seq._source_type = "video"
        seq._step = step
        seq._start = start
        seq._end = end
        return seq

    @classmethod
    def from_dir(cls, directory: str) -> FrameSequence:
        """Create a lazy sequence backed by a directory of image files.

        Files are discovered in natural sort order, so ``frame_2.png`` comes
        before ``frame_10.png``.

        Args:
            directory: Directory containing image files.

        Returns:
            A new lazy :class:`FrameSequence`.
        """
        seq = cls()
        seq._source = str(directory)
        seq._source_type = "dir"
        return seq

    @classmethod
    def from_images(cls, images: Sequence[Image | Array]) -> FrameSequence:
        """Create an eager sequence from frames already in memory.

        Args:
            images: :class:`~imlite.core.image.Image` objects or ``Array``
                arrays.  Arrays are wrapped automatically and assumed to be BGR.

        Returns:
            A new eager :class:`FrameSequence`.

        Raises:
            TypeError: If any item is neither an ``Image`` nor an ``Array``.
        """
        wrapped: list[Image] = []
        for position, item in enumerate(images):
            if isinstance(item, Image):
                wrapped.append(item)
            elif isinstance(item, np.ndarray):
                wrapped.append(Image.from_numpy(item))
            else:
                raise TypeError(
                    f"from_images() expects Image or Array frames; "
                    f"item {position} is {type(item).__name__!r}."
                )

        seq = cls()
        seq._source_type = "list"
        seq._eager_frames = wrapped
        return seq

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of frames, without decoding any of them."""
        if self._source_type == "list":
            return len(self._eager_frames or [])

        if self._source_type == "dir":
            return len(sorted_frame_paths(self._require_source()))

        from imlite.ops.video_io import get_video_info

        total = int(get_video_info(self._require_source())["frame_count"])
        stop = total if self._end is None else min(self._end, total)
        return max(0, len(range(self._start, stop, self._step)))

    def __iter__(self) -> Iterator[Image]:
        """Yield frames one at a time, applying any queued transforms."""
        for frame in self._iter_source():
            transformed = frame
            for op in self._pending_ops:
                transformed = op(transformed)
            yield transformed

    def __getitem__(self, index: int | slice) -> Image | FrameSequence:
        """Access frames by index or slice.

        An integer index streams only as far as that frame instead of
        materialising the whole sequence, so ``seq[0]`` on a two-hour video
        decodes one frame.  Negative indices and slices do need the length, and
        for a video source that means a full pass.

        Args:
            index: Integer frame index (negative counts from the end) or a slice.

        Returns:
            An :class:`~imlite.core.image.Image` for an integer index, or a new
            eager :class:`FrameSequence` for a slice.

        Raises:
            IndexError: If an integer index is out of range.
            TypeError: If *index* is neither an ``int`` nor a ``slice``.
        """
        if isinstance(index, slice):
            return FrameSequence.from_images(self.to_list()[index])

        if not isinstance(index, (int, np.integer)):
            raise TypeError(
                f"FrameSequence indices must be integers or slices, got {type(index).__name__!r}."
            )

        position = int(index)
        if position < 0:
            position += len(self)
            if position < 0:
                raise IndexError(f"Frame index {index} is out of range.")

        for current, frame in enumerate(self):
            if current == position:
                return frame
        raise IndexError(f"Frame index {index} is out of range.")

    def __repr__(self) -> str:
        try:
            count: int | str = len(self)
        except Exception:
            count = "?"
        source = self._source or "in-memory"
        return (
            f"FrameSequence(source={source!r}, frames={count}, "
            f"pending_ops={len(self._pending_ops)})"
        )

    # ------------------------------------------------------------------
    # Deferred transforms - queued now, applied during iteration
    # ------------------------------------------------------------------

    def crop(self, x: int, y: int, width: int, height: int) -> FrameSequence:
        """Queue a crop for every frame.

        Args:
            x: Left edge of the crop box.
            y: Top edge of the crop box.
            width: Crop width in pixels.
            height: Crop height in pixels.

        Returns:
            A new :class:`FrameSequence` with the crop queued.
        """
        return self.apply(lambda img: img.crop(x, y, width, height))

    def rotate(self, angle: float, expand: bool = True) -> FrameSequence:
        """Queue a rotation for every frame.

        Args:
            angle: Rotation angle in degrees, counter-clockwise.
            expand: Grow the canvas to fit the rotated frame (default).

        Returns:
            A new :class:`FrameSequence` with the rotation queued.
        """
        return self.apply(lambda img: img.rotate(angle, expand))

    def resize(
        self,
        width: int | None = None,
        height: int | None = None,
        keep_aspect: bool = False,
        resample: str = "auto",
    ) -> FrameSequence:
        """Queue a resize for every frame.

        Args:
            width: Target width, or ``None`` to infer.
            height: Target height, or ``None`` to infer.
            keep_aspect: Fit inside the target box without distorting.
            resample: Filter name - see :func:`imlite.resize`.

        Returns:
            A new :class:`FrameSequence` with the resize queued.

        Note:
            Video encoders need every frame to be the same size.  With
            ``keep_aspect=True`` and mixed input sizes, follow the resize with
            :meth:`pad` before :meth:`merge`.
        """
        return self.apply(lambda img: img.resize(width, height, keep_aspect, resample))

    def thumbnail(self, size: int, resample: str = "auto") -> FrameSequence:
        """Queue a thumbnail scale for every frame.

        Args:
            size: Length of the longest side, in pixels.
            resample: Filter name - see :func:`imlite.resize`.

        Returns:
            A new :class:`FrameSequence` with the scale queued.
        """
        return self.apply(lambda img: img.thumbnail(size, resample))

    def flip(self, axis: FlipAxis = "h") -> FrameSequence:
        """Queue a flip for every frame.

        Args:
            axis: ``"h"``/``"horizontal"``, ``"v"``/``"vertical"`` or ``"both"``.

        Returns:
            A new :class:`FrameSequence` with the flip queued.
        """
        return self.apply(lambda img: img.flip(axis))

    def pad(
        self,
        top: int = 0,
        bottom: int = 0,
        left: int = 0,
        right: int = 0,
        color: int | Sequence[int] = (0, 0, 0),
    ) -> FrameSequence:
        """Queue a constant-colour border for every frame.

        Args:
            top: Pixels to add on the top edge.
            bottom: Pixels to add on the bottom edge.
            left: Pixels to add on the left edge.
            right: Pixels to add on the right edge.
            color: Fill colour in each frame's current colour space.

        Returns:
            A new :class:`FrameSequence` with the padding queued.
        """
        return self.apply(lambda img: img.pad(top, bottom, left, right, color))

    def blur(self, radius: float = 2.0) -> FrameSequence:
        """Queue a Gaussian blur for every frame.

        Args:
            radius: Standard deviation of the Gaussian kernel, in pixels.

        Returns:
            A new :class:`FrameSequence` with the blur queued.
        """
        return self.apply(lambda img: img.blur(radius))

    def brightness(self, factor: float = 1.0) -> FrameSequence:
        """Queue a brightness change for every frame.

        Args:
            factor: Multiplier applied to every channel.

        Returns:
            A new :class:`FrameSequence` with the adjustment queued.
        """
        return self.apply(lambda img: img.brightness(factor))

    def contrast(self, factor: float = 1.0) -> FrameSequence:
        """Queue a contrast change for every frame.

        Args:
            factor: Contrast multiplier.

        Returns:
            A new :class:`FrameSequence` with the adjustment queued.
        """
        return self.apply(lambda img: img.contrast(factor))

    def to_gray(self) -> FrameSequence:
        """Queue a grayscale conversion for every frame.

        Returns:
            A new :class:`FrameSequence` with the conversion queued.
        """
        return self.apply(lambda img: img.to_gray())

    def to_rgb(self) -> FrameSequence:
        """Queue an RGB conversion for every frame.

        Returns:
            A new :class:`FrameSequence` with the conversion queued.

        Note:
            :meth:`merge` handles colour order for you, so this is only needed
            when you consume the frames yourself.
        """
        return self.apply(lambda img: img.to_rgb())

    def to_bgr(self) -> FrameSequence:
        """Queue a BGR conversion for every frame.

        Returns:
            A new :class:`FrameSequence` with the conversion queued.
        """
        return self.apply(lambda img: img.to_bgr())

    def apply(self, fn: Callable[[Image], Image]) -> FrameSequence:
        """Queue an arbitrary per-frame function.

        Args:
            fn: A callable taking an :class:`~imlite.core.image.Image` and
                returning a transformed one.

        Returns:
            A new :class:`FrameSequence` with *fn* queued.

        Example:
            >>> seq.apply(lambda img: img.to_gray().threshold(180))
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
    ) -> FrameSequence:
        """Write every frame to *output_dir* as an image file.

        Queued transforms are applied as frames are written.

        Args:
            output_dir: Directory to write into; created if missing.
            fmt: Image format, e.g. ``"png"`` or ``"jpg"``.
            prefix: Filename prefix, so ``"frame"`` gives ``frame_00001.png``.
            zero_pad: Number of digits in the zero-padded index.
            show_progress: Show a progress bar.

        Returns:
            A new :class:`FrameSequence` backed by *output_dir*, so writing can
            sit mid-chain.

        Example:
            >>> imlite.load("clip.mp4").extract_frames(step=5).save_frames("frames/")
        """
        from imlite.ops.io import write_image
        from imlite.utils.log import progress
        from imlite.utils.path import ensure_dir

        ensure_dir(output_dir)
        total = self._safe_len()
        log.info("Saving frames to %s (fmt=%s)", output_dir, fmt)

        written = 0
        for frame in progress(
            self, desc="Saving frames", total=total, unit="frame", show=show_progress
        ):
            write_image(frame, f"{output_dir}/{prefix}_{written:0{zero_pad}d}.{fmt}")
            written += 1

        log.info("Done. %d frames saved to %s", written, output_dir)
        return FrameSequence.from_dir(str(output_dir))

    def merge(self, fps: float = 30.0, codec: str = "libx264") -> Video:
        """Assemble this sequence into a :class:`~imlite.core.video.Video`.

        Nothing is encoded yet - call ``.save("out.mp4")`` on the result.
        Frames are decoded and transformed during that call, one at a time.

        Args:
            fps: Output frame rate.
            codec: FFmpeg codec name (default ``"libx264"``).

        Returns:
            A :class:`~imlite.core.video.Video` with this sequence pending.

        Example:
            >>> seq.rotate(90).merge(fps=25).save("out.mp4")
        """
        from imlite.core.video import Video

        return Video.from_frames(self, fps=fps, codec=codec)

    def to_list(self) -> list[Image]:
        """Force eager evaluation and return every frame.

        Warning:
            This loads the whole sequence into RAM.  Iterate the sequence
            instead when it is backed by a long video.

        Returns:
            A list of :class:`~imlite.core.image.Image` objects.
        """
        if self._source_type == "list" and not self._pending_ops:
            return list(self._eager_frames or [])
        return list(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_source(self) -> str:
        """Return the source path, or fail loudly if this sequence has none."""
        if self._source is None:
            raise ValueError(
                f"FrameSequence has source_type={self._source_type!r} but no source path. "
                "Build sequences with from_video(), from_dir() or from_images()."
            )
        return self._source

    def _safe_len(self) -> int | None:
        """Return the frame count when it is cheap to know, else ``None``."""
        try:
            return len(self)
        except Exception as exc:
            log.debug("Sequence length unavailable: %s", exc)
            return None

    def _clone(self) -> FrameSequence:
        """Copy this sequence: same source, an independent list of queued ops."""
        new_seq = FrameSequence()
        new_seq._source = self._source
        new_seq._source_type = self._source_type
        new_seq._step = self._step
        new_seq._start = self._start
        new_seq._end = self._end
        new_seq._pending_ops = list(self._pending_ops)
        new_seq._eager_frames = self._eager_frames  # Image is immutable; sharing is safe
        return new_seq

    def _iter_source(self) -> Iterator[Image]:
        """Yield raw, untransformed frames from the underlying source."""
        if self._source_type == "list":
            yield from self._eager_frames or []
            return

        if self._source_type == "dir":
            from imlite.ops.io import read_image

            for path in sorted_frame_paths(self._require_source()):
                yield read_image(path)
            return

        from imlite.ops.video_io import iter_video_frames

        yield from iter_video_frames(
            self._require_source(), step=self._step, start=self._start, end=self._end
        )
