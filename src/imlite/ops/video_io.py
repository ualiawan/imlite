"""Video I/O - frame extraction and video encoding.

Backend: ``imageio`` with the ``ffmpeg`` plugin (provided by ``imageio-ffmpeg``).
The ffmpeg binary is bundled in the wheel - no system install required.

- imageio-ffmpeg yields and expects **RGB** frames.
- imlite stores images as **BGR** internally.
- Conversions happen here, not in user-facing code.
"""

from __future__ import annotations

import logging
from pathlib import Path

import imageio.v2 as iio2
import numpy as np

from imlite.core.sequence import FrameSequence
from imlite.core.video import Video
from imlite.exceptions import ImliteReadError, ImliteWriteError
from imlite.utils.log import progress
from imlite.utils.path import ensure_dir

log = logging.getLogger(__name__)

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

__all__ = ["extract_frames", "merge_frames", "get_video_info"]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def get_video_info(path: str) -> dict:
    """Return metadata for a video file.

    Args:
        path: Path to the video file.

    Returns:
        A dictionary with keys: ``path``, ``fps``, ``frame_count``,
        ``duration``, ``width``, ``height``, ``codec``, ``size_bytes``.

    Raises:
        ImliteReadError: If the file cannot be opened.

    Example:
        >>> info = imlite.video_info("clip.mp4")
        >>> print(info["fps"], info["frame_count"])
    """
    log.debug("Reading video metadata: %s", path)
    try:
        reader = iio2.get_reader(str(path))
        meta = reader.get_meta_data()
        reader.close()
    except Exception as exc:  # noqa: BLE001
        raise ImliteReadError(f"Could not open video {path!r} to read metadata.") from exc

    fps: float = float(meta.get("fps", 0.0))
    size: tuple[int, int] = meta.get("size", (0, 0))  # (W, H)
    nframes = meta.get("nframes", 0)
    # Some imageio versions report nframes=inf for certain containers;
    # count by iterating if so.
    import math  # noqa: PLC0415

    if isinstance(nframes, float) and (math.isinf(nframes) or math.isnan(nframes)):
        log.debug("nframes=inf for %s - counting by iteration", path)
        try:
            count_reader = iio2.get_reader(str(path))
            nframes = sum(1 for _ in count_reader)
            count_reader.close()
        except Exception:  # noqa: BLE001
            nframes = 0
    duration_raw = meta.get("duration", int(nframes) / fps if fps and nframes else 0.0)
    if isinstance(duration_raw, float) and (math.isinf(duration_raw) or math.isnan(duration_raw)):
        duration_raw = int(nframes) / fps if fps and nframes else 0.0

    file_size = Path(path).stat().st_size if Path(path).is_file() else 0

    return {
        "path": str(path),
        "fps": fps,
        "frame_count": int(nframes),
        "duration": float(duration_raw),
        "width": int(size[0]),
        "height": int(size[1]),
        "codec": str(meta.get("codec", "")),
        "size_bytes": file_size,
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str,
    output_dir: str | None = None,
    step: int = 1,
    start: int = 0,
    end: int | None = None,
    fmt: str = "png",
    show_progress: bool = True,
) -> "FrameSequence":
    """Extract frames from a video file.

    Args:
        video_path: Path to the source video.
        output_dir: If given, frames are saved as image files inside this
            directory and a :class:`~imlite.core.sequence.FrameSequence`
            backed by that directory is returned.  If ``None``, frames are
            held in memory.
        step: Take every *step*-th frame (1 = every frame, 2 = every other
            frame, …).
        start: Index of the first frame to extract (inclusive, 0-based).
        end: Index of the last frame to extract (exclusive).  ``None`` means
            extract until the end of the video.
        fmt: Image format for saved frames (e.g. ``"png"``, ``"jpg"``).
            Ignored when *output_dir* is ``None``.
        show_progress: Show a tqdm progress bar.  Controlled globally by
            :func:`~imlite.utils.log.set_progress`.

    Returns:
        A :class:`~imlite.core.sequence.FrameSequence` containing the
        extracted frames.

    Raises:
        ImliteReadError: If the video cannot be opened.
        ImliteWriteError: If frame files cannot be written to *output_dir*.

    Example:
        >>> seq = imlite.extract_frames("video.mp4", step=2)
        >>> imlite.extract_frames("video.mp4", output_dir="frames/", step=5)
    """
    from imlite.core.image import Image  # noqa: PLC0415
    from imlite.core.sequence import FrameSequence  # noqa: PLC0415
    from imlite.ops.io import write_image  # noqa: PLC0415

    log.info("Extracting frames from %s (step=%d)", video_path, step)

    # Use get_video_info for a reliable frame count (handles nframes=inf/0).
    try:
        info = get_video_info(video_path)
        total_frames: int | None = info["frame_count"] or None
        reader = iio2.get_reader(str(video_path))
    except Exception as exc:  # noqa: BLE001
        raise ImliteReadError(f"Could not open video {video_path!r} for frame extraction.") from exc

    # Build the list of frame indices to extract.
    stop = end if end is not None else (total_frames or 10**9)
    indices = range(start, stop, step)

    frames: list[Image] = []

    if output_dir is not None:
        ensure_dir(output_dir)

    saved = 0
    try:
        for frame_idx in progress(
            indices,
            desc="Extracting",
            total=len(indices) if isinstance(indices, range) else None,
            unit="frame",
            show=show_progress,
        ):
            try:
                rgb_arr = reader.get_data(frame_idx)
            except IndexError:
                log.debug("Frame %d out of range - stopping early.", frame_idx)
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("Frame %d is corrupt or unreadable, skipping: %s", frame_idx, exc)
                continue

            # imageio returns RGB -> convert to BGR for internal storage.
            if rgb_arr.ndim == 3 and rgb_arr.shape[2] == 3:
                bgr_arr: np.ndarray = rgb_arr[..., ::-1].copy()
            elif rgb_arr.ndim == 3 and rgb_arr.shape[2] == 4:
                bgr_arr = rgb_arr[..., [2, 1, 0, 3]].copy()  # RGBA → BGRA
            else:
                bgr_arr = rgb_arr  # grayscale - keep as-is

            img = Image.from_numpy(bgr_arr, color_space="BGR")

            if output_dir is not None:
                dest = str(Path(output_dir) / f"frame_{saved:05d}.{fmt}")
                write_image(img, dest)
                saved += 1
            else:
                frames.append(img)

    finally:
        reader.close()

    if output_dir is not None:
        log.info("Done. %d frames saved to %s", saved, output_dir)
        return FrameSequence.from_dir(str(output_dir))
    else:
        log.info("Done. %d frames extracted (in-memory).", len(frames))
        return FrameSequence.from_images(frames)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_frames(
    source: "str | list | FrameSequence",
    output_path: str,
    fps: float = 30.0,
    codec: str = "libx264",
    show_progress: bool = True,
) -> "Video":
    """Assemble frames into a video file.

    Args:
        source: One of:

            - A directory path (``str``) - image files are loaded in natural order.
            - A ``list`` of :class:`~imlite.core.image.Image` or ``np.ndarray``.
            - A :class:`~imlite.core.sequence.FrameSequence`.

        output_path: Destination ``.mp4`` (or other ffmpeg-supported) path.
        fps: Output frame rate.
        codec: FFmpeg codec name (default ``"libx264"``).
        show_progress: Show a tqdm progress bar.

    Returns:
        A :class:`~imlite.core.video.Video` pointing to *output_path*.

    Raises:
        ImliteWriteError: If the output file cannot be written.

    Example:
        >>> imlite.merge_frames("frames/", "out.mp4", fps=25)
        >>> imlite.merge_frames(my_sequence, "out.mp4", fps=30)
    """
    from imlite.core.image import Image  # noqa: PLC0415
    from imlite.core.sequence import FrameSequence  # noqa: PLC0415
    from imlite.core.video import Video  # noqa: PLC0415

    # Normalise source to FrameSequence.
    if isinstance(source, str):
        seq: FrameSequence = FrameSequence.from_dir(source)
    elif isinstance(source, list):
        seq = FrameSequence.from_images(source)
    else:
        seq = source  # type: ignore[assignment]

    total = len(seq) if hasattr(seq, "__len__") else None

    log.info(
        "Encoding %s frames -> %s (fps=%s codec=%s)",
        total if total else "?",
        output_path,
        fps,
        codec,
    )

    ensure_dir(Path(output_path).parent)

    try:
        writer = iio2.get_writer(
            str(output_path),
            fps=fps,
            codec=codec,
        )
    except Exception as exc:  # noqa: BLE001
        raise ImliteWriteError(f"Could not open {output_path!r} for writing.") from exc

    count = 0
    try:
        for frame in progress(seq, desc="Encoding", total=total, unit="frame", show=show_progress):
            arr: np.ndarray = frame.data if isinstance(frame, Image) else frame  # type: ignore[union-attr]
            # imageio-ffmpeg expects RGB; convert from BGR internal convention.
            if isinstance(frame, Image) and frame.color_space == "BGR":
                arr = arr[..., ::-1].copy()
            elif not isinstance(frame, Image) and arr.ndim == 3 and arr.shape[2] == 3:
                # Raw ndarray - assume BGR (OpenCV default).
                arr = arr[..., ::-1].copy()
            try:
                writer.append_data(arr)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to encode frame %d: %s", count, exc)
                continue
            count += 1
    finally:
        writer.close()

    log.info("Done. %d frames encoded to %s", count, output_path)
    return Video(str(output_path))
