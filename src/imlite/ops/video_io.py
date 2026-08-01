"""Video I/O: metadata, frame extraction and encoding.

Backend: ``imageio`` driving the ffmpeg binary that ``imageio-ffmpeg`` bundles
in its wheel.  Users never install ffmpeg themselves; see
:mod:`imlite.utils.ffmpeg` for how the binary is located and what happens on
the rare platform that has no bundled build.

Colour handling: ffmpeg yields and expects **RGB**, while imlite stores
**BGR**.  Every conversion happens in this module so no user-facing code has
to think about it.

Decoding is strictly sequential - frames are read in order and unwanted ones
discarded.  Random-access seeking makes ffmpeg re-initialise, which is far
slower than skipping for the strides real pipelines use.
"""

# PEP 563: Image, FrameSequence and Video are TYPE_CHECKING-only imports
# used in signatures, kept out of the runtime import graph. Do not remove.
from __future__ import annotations

import functools
import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import imageio.v2 as iio2
import numpy as np

from imlite.exceptions import ImliteReadError, ImliteWriteError
from imlite.utils.ffmpeg import require_ffmpeg
from imlite.utils.log import progress
from imlite.utils.path import ensure_dir

if TYPE_CHECKING:  # pragma: no cover
    from imlite.core.image import Image
    from imlite.core.sequence import FrameSequence
    from imlite.core.video import Video

log = logging.getLogger(__name__)

# imageio-ffmpeg is chatty about harmless encoder details.
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

__all__ = ["extract_frames", "get_video_info", "iter_video_frames", "merge_frames"]

# imageio v2 names its plugins in upper case; pinning it stops imageio from
# picking a non-ffmpeg plugin for containers such as .gif.
_FFMPEG: Any = "FFMPEG"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def get_video_info(path: str) -> dict[str, Any]:
    """Return metadata for a video file.

    Results are cached per (path, size, modification time), so repeatedly
    asking a :class:`~imlite.core.video.Video` for its properties does not
    re-launch ffmpeg.

    Args:
        path: Path to the video file.

    Returns:
        A dict with keys ``path``, ``fps``, ``frame_count``, ``duration``
        (seconds), ``width``, ``height``, ``codec`` and ``size_bytes``.

    Raises:
        ImliteReadError: If the file is missing or cannot be opened.
        ImliteFFmpegError: If no ffmpeg binary is available.

    Example:
        >>> info = imlite.video_info("clip.mp4")
        >>> info["fps"], info["frame_count"]  # doctest: +SKIP
        (25.0, 300)
    """
    file = Path(path)
    if not file.is_file():
        raise ImliteReadError(f"Video file not found: {str(path)!r}")
    stat = file.stat()
    return dict(_probe_video(str(path), stat.st_size, stat.st_mtime))


@functools.lru_cache(maxsize=32)
def _probe_video(path: str, _size: int, _mtime: float) -> dict[str, Any]:
    """Read metadata from disk. Cache key includes size and mtime so edits invalidate it."""
    require_ffmpeg()
    log.debug("Probing video metadata: %s", path)

    try:
        reader = iio2.get_reader(path, format=_FFMPEG)
        meta = reader.get_meta_data()
        reader.close()
    except Exception as exc:
        raise ImliteReadError(
            f"Could not open video {path!r}. Check that it is a supported, non-corrupt video file."
        ) from exc

    fps = float(meta.get("fps", 0.0) or 0.0)
    width, height = meta.get("size", (0, 0))
    frame_count = _resolve_frame_count(path, meta.get("nframes", 0), fps, meta.get("duration"))
    duration: float = 0.0
    reported = meta.get("duration")
    if _is_finite(reported):
        duration = float(reported)
    elif fps:
        duration = frame_count / fps

    return {
        "path": path,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": int(width),
        "height": int(height),
        "codec": str(meta.get("codec", "")),
        "size_bytes": Path(path).stat().st_size,
    }


def _is_finite(value: object) -> TypeGuard[float]:
    """Return ``True`` if *value* is a real, finite number."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _resolve_frame_count(path: str, nframes: object, fps: float, duration: object) -> int:
    """Determine a usable frame count, falling back through cheaper estimates first."""
    if _is_finite(nframes) and float(nframes) > 0:
        return int(nframes)

    # Some containers (notably streamed webm and certain MKVs) report an
    # infinite or zero frame count. Estimate from the duration if we can,
    # because counting means decoding the whole file.
    if _is_finite(duration) and fps:
        estimated = int(duration * fps)
        if estimated > 0:
            log.debug(
                "Frame count unavailable for %s; estimated %d from duration.", path, estimated
            )
            return estimated

    log.warning("Frame count unavailable for %s - counting by decoding the whole file.", path)
    try:
        reader = iio2.get_reader(path, format=_FFMPEG)
        try:
            return sum(1 for _ in reader.iter_data())
        finally:
            reader.close()
    except Exception as exc:
        log.debug("Could not count frames in %s: %s", path, exc)
        return 0


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def iter_video_frames(
    path: str,
    step: int = 1,
    start: int = 0,
    end: int | None = None,
) -> Iterator[Image]:
    """Yield frames from a video file one at a time, as BGR images.

    Peak memory is a single frame regardless of how long the video is.

    Args:
        path: Path to the video file.
        step: Yield every *step*-th frame, counting from *start*.
        start: Index of the first frame to yield (inclusive, 0-based).
        end: Index to stop before (exclusive), or ``None`` for the whole file.

    Yields:
        :class:`~imlite.core.image.Image` frames tagged ``"BGR"``.

    Raises:
        ValueError: If *step* is not positive or *start* is negative.
        ImliteReadError: If the video cannot be opened.
        ImliteFFmpegError: If no ffmpeg binary is available.

    Example:
        >>> for frame in iter_video_frames("clip.mp4", step=10):  # doctest: +SKIP
        ...     print(frame.shape)
    """
    from imlite.core.image import Image

    if step < 1:
        raise ValueError(f"'step' must be >= 1, got {step}.")
    if start < 0:
        raise ValueError(f"'start' must be >= 0, got {start}.")

    require_ffmpeg()
    try:
        reader = iio2.get_reader(str(path), format=_FFMPEG)
    except Exception as exc:
        raise ImliteReadError(f"Could not open video {str(path)!r} for reading.") from exc

    try:
        for index, rgb in enumerate(reader.iter_data()):
            if index < start:
                continue
            if end is not None and index >= end:
                break
            if (index - start) % step:
                continue
            yield Image(_rgb_to_bgr(rgb), color_space="BGR", copy=False)
    finally:
        reader.close()


def extract_frames(
    video_path: str,
    output_dir: str | None = None,
    step: int = 1,
    start: int = 0,
    end: int | None = None,
    fmt: str = "png",
    show_progress: bool = True,
) -> FrameSequence:
    """Extract frames from a video.

    With no *output_dir* this returns a **lazy** sequence: nothing is decoded
    until you iterate it, so ``extract_frames("4k.mp4").resize(640, 360)``
    costs one frame of memory rather than the whole video.  Pass *output_dir*
    to decode now and write each frame to disk.

    Args:
        video_path: Path to the source video.
        output_dir: Directory to write frame images into.  When ``None``
            (default) no files are written and the returned sequence streams
            from the video on demand.
        step: Take every *step*-th frame.
        start: Index of the first frame (inclusive, 0-based).
        end: Index to stop before (exclusive), or ``None`` for the whole file.
        fmt: Image format for written frames, e.g. ``"png"`` or ``"jpg"``.
            Ignored when *output_dir* is ``None``.
        show_progress: Show a progress bar while writing frames.  Only applies
            when *output_dir* is given; there is nothing to measure in the
            lazy case.

    Returns:
        A :class:`~imlite.core.sequence.FrameSequence` - lazy over the video,
        or backed by *output_dir* when frames were written.

    Raises:
        ImliteReadError: If the video cannot be opened.
        ImliteWriteError: If a frame cannot be written.
        ImliteFFmpegError: If no ffmpeg binary is available.

    Example:
        >>> seq = imlite.extract_frames("clip.mp4", step=2)          # lazy
        >>> imlite.extract_frames("clip.mp4", "frames/", step=5)     # to disk
    """
    from imlite.core.sequence import FrameSequence
    from imlite.ops.io import write_image

    if output_dir is None:
        log.debug("Building lazy frame sequence over %s (step=%d)", video_path, step)
        return FrameSequence.from_video(str(video_path), step=step, start=start, end=end)

    log.info("Extracting frames from %s to %s (step=%d)", video_path, output_dir, step)
    ensure_dir(output_dir)

    total = _expected_frame_count(str(video_path), step, start, end)
    frames = iter_video_frames(str(video_path), step=step, start=start, end=end)

    saved = 0
    for frame in progress(frames, desc="Extracting", total=total, unit="frame", show=show_progress):
        write_image(frame, str(Path(output_dir) / f"frame_{saved:05d}.{fmt}"))
        saved += 1

    log.info("Done. %d frames saved to %s", saved, output_dir)
    return FrameSequence.from_dir(str(output_dir))


def _expected_frame_count(path: str, step: int, start: int, end: int | None) -> int | None:
    """Best-effort count of how many frames a stride will yield, for progress bars."""
    try:
        total = int(get_video_info(path)["frame_count"])
    except Exception as exc:
        log.debug("Could not determine frame count for %s: %s", path, exc)
        return None
    if total <= 0:
        return None
    stop = total if end is None else min(end, total)
    return max(0, len(range(start, stop, step)))


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def merge_frames(
    source: str | list[Any] | FrameSequence,
    output_path: str,
    fps: float = 30.0,
    codec: str = "libx264",
    quality: int | None = None,
    macro_block_size: int = 2,
    show_progress: bool = True,
) -> Video:
    """Encode frames into a video file.

    Args:
        source: A directory of image files, a list of
            :class:`~imlite.core.image.Image` or ``np.ndarray`` frames, or a
            :class:`~imlite.core.sequence.FrameSequence`.  Raw arrays are
            assumed to be **BGR**, matching the rest of imlite - pass
            ``Image`` objects if yours are RGB.
        output_path: Destination path, e.g. ``"out.mp4"``.
        fps: Output frame rate.
        codec: FFmpeg codec name (default ``"libx264"``).
        quality: Optional ffmpeg quality, ``0`` (worst) to ``10`` (best).
            ``None`` leaves imageio's default.
        macro_block_size: Frame dimensions are rounded **up** to a multiple of
            this before encoding.  The default of ``2`` is the smallest value
            that satisfies ``yuv420p``, which needs even dimensions, so a
            640x360 request really produces 640x360.  Raise it to ``16`` for
            old hardware decoders that need macroblock alignment, or drop it to
            ``1`` to disable rounding entirely.
        show_progress: Show a progress bar while encoding.

    Returns:
        A :class:`~imlite.core.video.Video` pointing at *output_path*.

    Raises:
        ImliteWriteError: If the output cannot be opened or no frames were
            written.
        ImliteFFmpegError: If no ffmpeg binary is available.

    Note:
        imageio's own default is ``macro_block_size=16``, which silently turns a
        640x360 render into 640x368 and 1920x1080 into 1920x1088.  imlite lowers
        it and logs a warning whenever a frame really does get resized, so the
        output size is never a surprise.

    Example:
        >>> imlite.merge_frames("frames/", "out.mp4", fps=25)
    """
    from imlite.core.image import Image
    from imlite.core.video import Video

    if fps <= 0:
        raise ValueError(f"'fps' must be > 0, got {fps}.")

    sequence = _as_sequence(source)

    require_ffmpeg()
    total = _safe_len(sequence)
    log.info(
        "Encoding %s frames -> %s (fps=%s codec=%s)",
        total if total is not None else "?",
        output_path,
        fps,
        codec,
    )
    ensure_dir(Path(output_path).parent)

    if macro_block_size < 1:
        raise ValueError(f"'macro_block_size' must be >= 1, got {macro_block_size}.")

    writer_kwargs: dict[str, Any] = {
        "fps": fps,
        "codec": codec,
        "macro_block_size": macro_block_size,
    }
    if quality is not None:
        writer_kwargs["quality"] = quality

    try:
        writer = iio2.get_writer(str(output_path), **writer_kwargs)
    except Exception as exc:
        raise ImliteWriteError(
            f"Could not open {str(output_path)!r} for writing with codec {codec!r}."
        ) from exc

    # ffmpeg validates the codec and frame geometry lazily, when the first
    # frame reaches it - so encoding failures surface here, not at get_writer,
    # and arrive as a bare OSError from a broken pipe.
    count = 0
    try:
        try:
            for frame in progress(
                sequence, desc="Encoding", total=total, unit="frame", show=show_progress
            ):
                pixels = frame.to_rgb().array if isinstance(frame, Image) else np.asarray(frame)
                pixels = _drop_single_channel(pixels)
                if count == 0:
                    _warn_if_padded(pixels.shape[:2], macro_block_size, output_path)
                writer.append_data(pixels)
                count += 1
        finally:
            writer.close()
    except Exception as exc:
        raise ImliteWriteError(
            f"Encoding to {str(output_path)!r} failed after {count} frame(s). "
            f"Check that the codec {codec!r} is available (run `imlite doctor`) and that "
            "every frame has the same width and height."
        ) from exc

    if count == 0:
        raise ImliteWriteError(
            f"No frames were written to {str(output_path)!r} - the source sequence was empty."
        )

    log.info("Done. %d frames encoded to %s", count, output_path)
    return Video(str(output_path))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rgb_to_bgr(pixels: np.ndarray) -> np.ndarray:
    """Swap the R and B channels, keeping any alpha channel last."""
    array = np.asarray(pixels)
    if array.ndim != 3:
        return array
    if array.shape[2] == 4:
        return np.ascontiguousarray(array[..., [2, 1, 0, 3]])
    if array.shape[2] == 3:
        return np.ascontiguousarray(array[..., ::-1])
    return array


def _as_sequence(source: str | list[Any] | FrameSequence) -> FrameSequence:
    """Normalise anything ``merge_frames`` accepts into a ``FrameSequence``."""
    from imlite.core.sequence import FrameSequence

    if isinstance(source, str):
        return FrameSequence.from_dir(source)
    if isinstance(source, list):
        # Warn before wrapping: from_images() tags raw arrays "BGR", so RGB
        # input would silently produce a colour-swapped video.
        if any(isinstance(item, np.ndarray) for item in source):
            log.warning(
                "Encoding raw numpy frames: imlite assumes they are BGR. If yours are RGB "
                "the output colours will be swapped - wrap them first with "
                "Image.from_numpy(arr, color_space='RGB')."
            )
        return FrameSequence.from_images(source)
    return source


def _warn_if_padded(shape: tuple[int, int], macro_block_size: int, output_path: str) -> None:
    """Log a warning if the encoder will round the frame size up.

    ffmpeg's ``yuv420p`` needs even dimensions, so odd ones get padded no matter
    what.  Saying so out loud beats letting the user discover that their 641px
    render came back 642px wide.
    """
    if macro_block_size <= 1:
        return
    height, width = shape
    padded = tuple(-(-value // macro_block_size) * macro_block_size for value in (width, height))
    if padded != (width, height):
        log.warning(
            "Frames are %dx%d, which is not a multiple of macro_block_size=%d, so %s will be "
            "encoded at %dx%d. Resize to a multiple of %d to keep the exact size.",
            width,
            height,
            macro_block_size,
            output_path,
            padded[0],
            padded[1],
            macro_block_size,
        )


def _drop_single_channel(pixels: np.ndarray) -> np.ndarray:
    """Collapse ``(H, W, 1)`` to ``(H, W)``, which is what ffmpeg wants for gray frames."""
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        return pixels[:, :, 0]
    return pixels


def _safe_len(sequence: FrameSequence) -> int | None:
    """Return ``len(sequence)`` when it is knowable without decoding, else ``None``."""
    try:
        return len(sequence)
    except Exception as exc:
        log.debug("Sequence length unavailable: %s", exc)
        return None
