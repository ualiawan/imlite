# Quick Start

```bash
pip install imlite
```

Nothing else. Check that video support came along for the ride:

```bash
imlite doctor
```

## Loading things

`imlite.load()` looks at what you give it and returns the matching type.

```python
import imlite

img = imlite.load("photo.jpg")     # -> Image
vid = imlite.load("clip.mp4")      # -> Video
seq = imlite.load("frames/")       # -> FrameSequence
seq = imlite.load([arr1, arr2])    # -> FrameSequence
```

Use the explicit constructors when you want the type stated in the code:

```python
img = imlite.read_image("photo.jpg")
vid = imlite.read_video("clip.mp4")
seq = imlite.read_frames("frames/")
```

## Transforming an image

Every transform returns a **new** `Image`; the original is untouched, so chains
are safe to build.

```python
result = (
    imlite.load("photo.jpg")
    .crop(x=100, y=50, width=400, height=300)
    .rotate(90)                      # multiples of 90 are lossless
    .resize(width=256, height=256)
    .brightness(1.1)
)
result.save("out.png")
```

| Group | Methods |
|---|---|
| Geometry | `crop` `rotate` `resize` `thumbnail` `flip` `pad` |
| Pixels | `blur` `brightness` `contrast` `invert` `threshold` |
| Colour | `to_rgb` `to_bgr` `to_gray` `to_hsv` `to_lab` |

## Working with numpy

Every operation is also a function, and functions take arrays:

```python
import numpy as np

arr = np.zeros((480, 640, 3), dtype=np.uint8)
cropped = imlite.crop(arr, x=0, y=0, width=100, height=100)   # -> ndarray
gray = imlite.to_gray(arr, source="RGB")                      # -> ndarray, (H, W, 1)
```

A bare array has no colour-space tag, so conversions take `source=` (defaulting
to `"BGR"`, OpenCV's convention). An `Image` carries its own tag and needs no
`source`.

Going the other way is free:

```python
img = imlite.Image.from_numpy(arr, color_space="RGB")
arr = img.to_numpy()          # a copy
arr = np.asarray(img)         # also a copy - Image is immutable
```

## Video to frames

```python
vid = imlite.load("clip.mp4")
print(vid.fps, vid.frame_count, vid.width, vid.height)

# Lazy: nothing is decoded until you iterate
seq = vid.extract_frames(step=2)
for frame in seq:
    ...

# Eager: write the frames to disk
vid.extract_frames("frames/", step=5, fmt="jpg")
```

`extract_frames()` without an `output_dir` costs one frame of memory, regardless
of the video's length.

## Frames to video

```python
imlite.read_frames("frames/").resize(640, 360).merge(fps=25).save("out.mp4")
```

`merge()` builds a `Video` but does not encode; `save()` does the work. So an
entire pipeline stays lazy from decode to encode:

```python
(
    imlite.load("clip.mp4")
    .extract_frames(step=2)
    .resize(640, 360)
    .flip("h")
    .merge(fps=12.5)
    .save("small.mp4")
)
```

## Transcoding

```python
imlite.load("clip.mov").save("clip.mp4")           # container / codec change
imlite.load("clip.mp4").save("slow.mp4", fps=12)   # change the frame rate
```

## Colour spaces

Images load as BGR, matching OpenCV. Conversions round-trip.

```python
img = imlite.load("photo.jpg")   # BGR

img.to_rgb()      # RGB
img.to_gray()     # GRAY, shape (H, W, 1)
img.to_hsv()      # HSV, H in 0-179 and S/V in 0-255, as OpenCV encodes it
img.to_lab()      # LAB, L = L* x 255/100 with +128 offsets on a and b
```

## In a notebook

An `Image` renders itself, with no `show()` and no matplotlib:

```python
imlite.load("photo.jpg").thumbnail(400).blur(2)
```

Outside a notebook, `img.show()` opens a window - matplotlib if it is installed,
otherwise Pillow's viewer.

## Logging and progress

```python
imlite.set_verbosity("INFO")     # operation start/end messages
imlite.set_verbosity("DEBUG")    # everything
imlite.set_verbosity("SILENT")   # nothing

imlite.set_progress(False)       # no progress bars - use this in CI
```

imlite uses the standard `logging` module and never installs a handler of its own
unless you call `set_verbosity()`.

## From the shell

The same operations, no Python required:

```bash
imlite info clip.mp4
imlite extract clip.mp4 frames/ --step 2
imlite merge frames/ out.mp4 --fps 30
imlite convert photo.png thumb.jpg --resize 320x --quality 85
```

See the [CLI guide](guides/cli.md) for the full set.
