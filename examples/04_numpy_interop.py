"""Dropping imlite into an existing numpy pipeline.

Every operation is a plain function that takes an Image *or* a raw ndarray and
returns the same type, so imlite can handle one step without owning the rest.

Run:
    python examples/04_numpy_interop.py
"""

import numpy as np

import imlite


def main() -> None:
    """Run the numpy interop example."""
    rng = np.random.default_rng(0)

    # --- arrays in, arrays out --------------------------------------------
    arr = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)

    cropped = imlite.crop(arr, x=20, y=20, width=200, height=160)
    resized = imlite.resize(cropped, width=100, height=80)
    gray = imlite.to_gray(resized, source="BGR")

    print("ndarray pipeline")
    for label, value in [("input", arr), ("crop", cropped), ("resize", resized), ("gray", gray)]:
        print(f"  {label:<8} {type(value).__name__:<8} {value.shape}")

    # --- a bare array has no colour tag, so conversions ask ---------------
    # These give different answers, which is exactly why source= exists.
    as_bgr = imlite.to_gray(arr, source="BGR")
    as_rgb = imlite.to_gray(arr, source="RGB")
    print(
        f"\nsource matters: BGR and RGB luma differ by up to {np.abs(as_bgr.astype(int) - as_rgb.astype(int)).max()}"
    )

    # --- wrap once and the tag travels with the data ----------------------
    image = imlite.Image.from_numpy(arr, color_space="RGB")
    print(f"\nwrapped     {image!r}")
    print(f"  to_gray() needs no source= - it reads {image.color_space!r} from the Image")

    # --- dtype conversion is explicit, never silent -----------------------
    normalised = rng.random((64, 64, 3)).astype(np.float32)  # 0.0 - 1.0
    from_float = imlite.Image.from_numpy(normalised)
    print(f"\nfloat 0-1   scaled to 0-255: max {from_float.data.max()}")

    sixteen_bit = np.full((8, 8, 3), 30000, dtype=np.uint16)
    print(
        f"uint16      30000 -> {imlite.Image.from_numpy(sixteen_bit).data[0, 0, 0]} (scaled, not wrapped)"
    )

    try:
        imlite.Image.from_numpy(np.full((8, 8, 3), 5000, dtype=np.int32))
    except imlite.ImliteDtypeError as exc:
        print(
            f"ambiguous   refused rather than corrupted:\n              {str(exc).splitlines()[0]}"
        )

    # --- zero-copy views for hot loops ------------------------------------
    view = image.array  # read-only, no copy
    copy = image.data  # a copy you own
    print(
        f"\n.array      shares memory: {np.shares_memory(view, image.array)}, writeable: {view.flags.writeable}"
    )
    print(f".data       independent copy, writeable: {copy.flags.writeable}")

    # --- handing frames to another library --------------------------------
    frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
    sequence = imlite.read_frames(frames)  # raw arrays are assumed BGR
    stacked = np.stack([f.to_rgb().data for f in sequence.resize(32, 32)])
    print(f"\nstacked     {stacked.shape} RGB array, ready for a model")


if __name__ == "__main__":
    main()
