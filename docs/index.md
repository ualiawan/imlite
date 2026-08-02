# imlite

**Lightweight image and video processing for Python - one-liners and chainable syntax.**

[![PyPI](https://img.shields.io/pypi/v/py-imlite?cacheSeconds=3600)](https://pypi.org/project/py-imlite/)
[![Python](https://img.shields.io/pypi/pyversions/py-imlite?cacheSeconds=3600)](https://pypi.org/project/py-imlite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

```bash
pip install py-imlite      # the import name is just `imlite`
```

That is the whole setup. **No OpenCV, no system ffmpeg.** Video works immediately
on Linux, macOS and Windows.

## What is imlite?

Everyday computer-vision work is mostly boilerplate: open a capture, loop, check a
return flag, remember which library wants RGB and which wants BGR, close the
handle. imlite collapses that into one line, without hiding what happened.

```python
import imlite

# Image pipeline
imlite.load("photo.jpg").crop(0, 0, 300, 300).rotate(90).thumbnail(256).save("out.jpg")

# Video pipeline - lazy from end to end
(
    imlite.load("clip.mp4")      # -> Video
    .extract_frames(step=2)      # -> FrameSequence, nothing decoded yet
    .resize(640, 360)            # queued
    .brightness(1.1)             # queued
    .merge(fps=25)               # -> Video, still not encoded
    .save("out.mp4")             # decode, transform and encode, one frame at a time
)
```

Peak memory in that video pipeline is **one frame**, whether the clip runs for
5 seconds or 5 hours.

## Two APIs, one implementation

Every operation is both a chained method and a plain function. The functional form
takes an `Image` *or* a raw numpy array and gives back whichever you passed:

```python
imlite.crop(image, 0, 0, 100, 100)      # -> Image
imlite.crop(array, 0, 0, 100, 100)      # -> ndarray
imlite.to_gray(array, source="RGB")     # bare arrays declare their colour space
```

imlite therefore drops into an existing numpy pipeline instead of taking it over.

## Features

- **Smart [`load()`](api/index.md)** - pass a path, directory, array, list of
  frames, or an existing imlite object and get the right type back.
- **Lazy `FrameSequence`** - transforms queue up and run per frame during
  iteration, so long videos never fill RAM.
- **Geometry**: `crop` `rotate` `resize` `thumbnail` `flip` `pad`.
- **Pixels**: `blur` `brightness` `contrast` `invert` `threshold`.
- **Colour**: `to_rgb` `to_bgr` `to_gray` `to_hsv` `to_lab` - round-trippable,
  using OpenCV's 8-bit encodings so existing thresholds keep working.
- **[Command line](guides/cli.md)**: `imlite doctor / info / extract / merge / convert`.
- **Notebook-native** - an `Image` renders itself as a cell's last expression.
- **Safe dtypes** - 16-bit and float input is
  [converted, never truncated](guides/dtypes.md).
- **Typed** - ships `py.typed` and passes `mypy --strict`.
- **Progress bars and standard logging** on every long operation.

## Backends

| Job | Backend |
|---|---|
| Arrays | `numpy` |
| Resampling: resize, rotate, blur | `pillow` |
| Image files | `imageio` (Pillow underneath) |
| Video | `imageio` + the ffmpeg binary bundled in the `imageio-ffmpeg` wheel |
| Colour conversion | pure `numpy` |

There is no OpenCV dependency. Dropping it removed roughly 118 MB from the install
and lifted the `numpy>=2` pin that OpenCV 5 imposes. See
[Installation](guides/install.md) for the details, and run `imlite doctor` to
check your own environment.

## Next steps

- [Quick Start](quickstart.md) - the five-minute tour
- [Installation](guides/install.md) - what gets installed, and the ffmpeg story
- [Command Line](guides/cli.md) - using imlite without writing Python
- [Limitations](guides/limitations.md) - what imlite does not do, and what to use instead
- [API Reference](api/index.md)
- [Changelog](changelog.md)
