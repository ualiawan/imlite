"""Image pipeline: load, chain transforms, convert colour spaces, save.

Run:
    python examples/01_image_pipeline.py
"""

import tempfile
from pathlib import Path

import numpy as np

import imlite


def make_sample(path: Path) -> None:
    """Write a 512x512 BGR gradient so the example needs no external files."""
    height, width = 512, 512
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)  # blue ramps right
    arr[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]  # green ramps down
    arr[:, :, 2] = 90
    imlite.Image.from_numpy(arr).save(str(path))


def main() -> None:
    """Run the image pipeline example."""
    workdir = Path(tempfile.mkdtemp(prefix="imlite-example-"))
    source = workdir / "gradient.png"
    make_sample(source)

    # --- load() picks the type from the input -----------------------------
    img = imlite.load(str(source))
    print(f"loaded      {img!r}")
    print(f"            {img.width}x{img.height}, {img.channels} channels, {img.dtype}")

    # --- chaining: each step returns a new Image --------------------------
    result = (
        img.crop(x=64, y=64, width=384, height=384)
        .rotate(90)  # exact multiples of 90 are lossless
        .resize(width=256, height=256)
        .brightness(1.15)
        .blur(radius=1.5)
    )
    result.save(str(workdir / "chained.png"))
    print(f"chained  -> {result.shape}")

    # The original is untouched: transforms never mutate in place.
    assert img.shape == (512, 512, 3)

    # --- the same operations as plain functions ---------------------------
    same = imlite.blur(
        imlite.brightness(
            imlite.resize(imlite.rotate(imlite.crop(img, 64, 64, 384, 384), 90), 256, 256),
            1.15,
        ),
        1.5,
    )
    assert np.array_equal(same.data, result.data), "method and function forms must agree"
    print("functional  identical to the chained form")

    # --- colour spaces round-trip ----------------------------------------
    for space in ("rgb", "hsv", "lab"):
        converted = getattr(img, f"to_{space}")()
        back = converted.to_bgr()
        drift = int(np.abs(img.data.astype(int) - back.data.astype(int)).max())
        print(f"BGR->{space.upper():<4}->BGR  max drift {drift:>3} (8-bit quantisation)")

    gray = img.to_gray()
    print(f"grayscale   {gray.shape} tagged {gray.color_space!r}")

    # --- thumbnails never enlarge ----------------------------------------
    thumb = img.thumbnail(128)
    print(f"thumbnail   {thumb.width}x{thumb.height}")
    assert imlite.load(np.zeros((32, 32, 3), np.uint8)).thumbnail(512).shape == (32, 32, 3)

    # --- a binary mask ----------------------------------------------------
    mask = img.to_gray().threshold(128)
    coverage = float((mask.data > 0).mean())
    print(f"threshold   {coverage:.1%} of pixels above the cut-off")

    print(f"\noutput in {workdir}")


if __name__ == "__main__":
    main()
