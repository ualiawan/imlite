"""Video pipeline: video -> frames -> transforms -> video, lazily throughout.

Nothing is decoded until the final save(), and only one frame is ever in memory.

Run:
    python examples/02_video_pipeline.py
"""

import tempfile
from pathlib import Path

import numpy as np

import imlite


def make_sample(path: Path, frames: int = 60, fps: int = 30) -> None:
    """Write a short synthetic clip: a bright square sliding across the frame."""
    size = 240
    sequence = []
    for i in range(frames):
        frame = np.full((size, size, 3), 40, dtype=np.uint8)
        x = int((size - 40) * i / (frames - 1))
        frame[100:140, x : x + 40] = (60, 200, 255)  # BGR: orange
        sequence.append(frame)
    imlite.read_frames(sequence).merge(fps=fps).save(str(path), show_progress=False)


def main() -> None:
    """Run the video pipeline example."""
    workdir = Path(tempfile.mkdtemp(prefix="imlite-example-"))
    source = workdir / "slide.mp4"
    make_sample(source)

    # --- metadata is read lazily and then cached --------------------------
    video = imlite.load(str(source))
    print(f"loaded      {video!r}")
    for key, value in video.info.items():
        print(f"  {key:<12} {value}")

    # --- building a pipeline decodes nothing ------------------------------
    pipeline = (
        video.extract_frames(step=2)  # -> FrameSequence, lazy
        .resize(160, 160)  # queued
        .flip("h")  # queued
        .brightness(1.2)  # queued
    )
    print(f"\npipeline    {pipeline!r}")
    print(f"            {len(pipeline)} frames, still nothing decoded")

    # Indexing streams only as far as it must, so this decodes one frame.
    print(f"first frame {pipeline[0].shape}")

    # --- save() is where the work happens ---------------------------------
    out = workdir / "processed.mp4"
    pipeline.merge(fps=video.fps / 2).save(str(out), show_progress=False)
    print(f"\nencoded  -> {out.name}: {imlite.video_info(str(out))}")

    # --- frames on disk, then back to video -------------------------------
    frames_dir = workdir / "frames"
    saved = video.extract_frames(str(frames_dir), step=10, fmt="jpg", show_progress=False)
    print(f"\nextracted   {len(saved)} jpg frames to {frames_dir.name}/")

    contact = workdir / "contact.mp4"
    imlite.read_frames(str(frames_dir)).thumbnail(120).merge(fps=2).save(
        str(contact), show_progress=False
    )
    print(f"reassembled {contact.name} at 2 fps")

    # --- transcoding is a one-liner ---------------------------------------
    transcoded = workdir / "transcoded.mp4"
    imlite.load(str(source)).save(str(transcoded), fps=15, show_progress=False)
    print(f"transcoded  {transcoded.name} at {imlite.video_info(str(transcoded))['fps']} fps")

    print(f"\noutput in {workdir}")


if __name__ == "__main__":
    main()
