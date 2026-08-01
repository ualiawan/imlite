"""Batch-process a directory of images into thumbnails.

The same job from the shell:
    for f in photos/*; do imlite convert "$f" "thumbs/$(basename "$f")" --resize 200x; done

Run:
    python examples/03_batch_thumbnails.py
"""

import tempfile
from pathlib import Path

import numpy as np

import imlite


def make_gallery(directory: Path, count: int = 8) -> None:
    """Write a handful of images at deliberately mixed sizes."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(count):
        height, width = int(rng.integers(200, 600)), int(rng.integers(200, 600))
        arr = np.full((height, width, 3), (i * 30 % 256, 120, 200), dtype=np.uint8)
        arr[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = (20, 20, 220)
        imlite.Image.from_numpy(arr).save(str(directory / f"photo_{i:02d}.png"))


def main() -> None:
    """Run the batch thumbnail example."""
    workdir = Path(tempfile.mkdtemp(prefix="imlite-example-"))
    gallery, thumbs = workdir / "photos", workdir / "thumbs"
    make_gallery(gallery)

    # --- one at a time, with per-file control -----------------------------
    # load() on a directory gives a FrameSequence, but here we want the paths
    # so each output keeps its own name.
    from imlite.utils.path import sorted_frame_paths

    for source in sorted_frame_paths(gallery):
        name = Path(source).stem
        image = imlite.read_image(source)
        image.thumbnail(200).save(str(thumbs / f"{name}.jpg"), quality=85)

    written = sorted(thumbs.glob("*.jpg"))
    print(f"wrote {len(written)} thumbnails to {thumbs}")
    for path in written[:3]:
        thumb = imlite.read_image(str(path))
        print(f"  {path.name:<16} {thumb.width}x{thumb.height}")

    # --- or as a sequence, when a uniform pipeline is enough --------------
    # save_frames() returns a sequence over what it wrote, so it chains.
    processed = (
        imlite.read_frames(str(gallery))
        .resize(160, 160)  # uniform size, so this could feed a video
        .to_gray()
        .brightness(1.1)
        .save_frames(str(workdir / "processed"), fmt="png", show_progress=False)
    )
    print(f"\nprocessed {len(processed)} frames uniformly -> {workdir / 'processed'}")

    # A uniform sequence can go straight to a contact-sheet video.
    imlite.read_frames(str(workdir / "processed")).merge(fps=2).save(
        str(workdir / "contact.mp4"), show_progress=False
    )
    print(f"contact sheet -> {workdir / 'contact.mp4'}")

    print(f"\noutput in {workdir}")


if __name__ == "__main__":
    main()
