# imlite

**Image and video processing for Python, without the boilerplate.**

[![PyPI](https://img.shields.io/pypi/v/py-imlite?cacheSeconds=3600)](https://pypi.org/project/py-imlite/)
[![Python](https://img.shields.io/pypi/pyversions/py-imlite?cacheSeconds=3600)](https://pypi.org/project/py-imlite/)
[![CI](https://github.com/ualiawan/imlite/actions/workflows/ci.yml/badge.svg)](https://github.com/ualiawan/imlite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

```bash
pip install py-imlite      # the import name is just `imlite`
```

**No OpenCV. No system ffmpeg.**

---

## Quick start

```python
import imlite

# Image: load -> transform -> save
imlite.load("photo.jpg").crop(0, 0, 300, 300).rotate(90).thumbnail(256).save("out.jpg")

# Video: streams one frame at a time, so a two-hour 4K clip costs one frame of RAM
(
    imlite.load("clip.mp4")
    .extract_frames(step=2)     # lazy - nothing decoded yet
    .resize(640, 360)           # queued
    .merge(fps=25)              # still not encoded
    .save("out.mp4")            # decode, transform, encode
)
```

`imlite.load()` returns whichever type fits the input:

| You give it | You get |
|---|---|
| `"photo.jpg"` | `Image` |
| `"clip.mp4"` | `Video` |
| `"frames/"`, or a list of frames | `FrameSequence` |
| a numpy array | `Image` |

Every operation is **also** a plain function that accepts an `Image` *or* a raw
numpy array, and returns whichever you passed:

```python
imlite.crop(array, 0, 0, 100, 100)      # -> ndarray
imlite.to_gray(array, source="RGB")     # -> ndarray
imlite.crop(image, 0, 0, 100, 100)      # -> Image
```

So imlite drops into an existing numpy pipeline instead of taking it over.

## From the shell

```bash
imlite doctor                             # check the install and its backends
imlite info clip.mp4                      # metadata
imlite extract clip.mp4 frames/ --step 2  # video -> frames
imlite merge frames/ out.mp4 --fps 30     # frames -> video
imlite convert photo.png thumb.jpg --resize 320x --gray
```

## Operations

| Group | Available as a method and as a function |
|---|---|
| Geometry | `crop` `rotate` `resize` `thumbnail` `flip` `pad` |
| Pixels | `blur` `brightness` `contrast` `invert` `threshold` |
| Colour | `to_rgb` `to_bgr` `to_gray` `to_hsv` `to_lab` |
| Video | `extract_frames` `merge_frames` `video_info` |

Colour conversions reproduce OpenCV's 8-bit encodings exactly (BT.601 luma, hue
in 0-179, `L*255/100`), so thresholds written against `cv2` keep working.

## Backends

Pixel work is numpy and Pillow. Video uses the static ffmpeg binary bundled
inside the `imageio-ffmpeg` wheel. Check yours any time:

```console
$ imlite doctor
imlite          0.1.0
numpy           2.4.6
pillow          12.3.0
imageio         2.37.4
ffmpeg          7.1  (bundled with imageio-ffmpeg)

Video support is working.
```

## Documentation

Full docs: **<https://ualiawan.github.io/imlite/>**

| | |
|---|---|
| [Quick Start](docs/quickstart.md) | The five-minute tour |
| [Installation](docs/guides/install.md) | What gets installed, and the ffmpeg story |
| [Command Line](docs/guides/cli.md) | Every subcommand, with recipes |
| [Video & Frames](docs/guides/video.md) | Extraction, encoding, transcoding |
| [Colour](docs/guides/color.md) | Encodings and round-trip behaviour |
| [Pixel dtypes](docs/guides/dtypes.md) | How 16-bit and float input is converted |
| [API Reference](https://ualiawan.github.io/imlite/api/) | Every public symbol |

Runnable scripts live in [examples/](examples/).

## Limitations

The full list, with workarounds, is in
**[docs/guides/limitations.md](docs/guides/limitations.md)**. The ones that bite
most often:

- **8-bit only.** 16-bit files scale down on read; float input is converted by
  documented rules, never truncated.
- **Raw numpy arrays are assumed BGR.** Wrap RGB arrays with
  `Image.from_numpy(arr, color_space="RGB")`. `merge_frames()` warns when it
  sees bare arrays.
- **Videos are found by file extension**, not by content, because ffmpeg picks
  its decoder from the suffix. Images *are* sniffed.
- **Audio is dropped on transcode**, and `save()` re-encodes rather than
  stream-copying.
- **Decoding is sequential**, so a large `start=` decodes and discards
  everything before it.
- **Single-threaded** per-frame processing.

## Roadmap

Each item below removes one of the limitations above.

**0.2.0**

- [ ] Audio passthrough on transcode
- [ ] Stream-copy when no frames were transformed
- [ ] Parallel `save_frames()`
- [ ] `FrameSequence.batch(n)` for model inference
- [ ] More ops: `sharpen`, `equalize`, `autocontrast`, `posterize`
- [ ] Annotation: `draw_box`, `draw_text` (blocked on a font-bundling decision)

**Later**

- [ ] Keyframe-aware seeking for large `start` offsets
- [ ] Async video reading
- [ ] Lazy slicing of `FrameSequence`
- [ ] A `Mask` type for segmentation work
- [ ] Optional `pillow-simd` backend

**Not planned:** reintroducing OpenCV, a 16-bit data model, content-sniffing
video files, or a GUI. Each of those would undo something the library is built
around; see [limitations](docs/guides/limitations.md) for why.

## Development

```bash
git clone https://github.com/ualiawan/imlite.git
cd imlite
uv sync --extra dev --extra docs --extra show
pre-commit install
```

```bash
pytest                           # tests, with an 85% coverage gate
ruff check src tests examples    # lint
ruff format src tests examples   # format
mypy                             # strict type check
mkdocs serve                     # docs on localhost:8000
```

Python 3.10+. CI runs the suite on Linux, macOS and Windows across 3.10-3.13.

A few house rules for contributions: no OpenCV and no direct `subprocess`
ffmpeg calls; every public operation must accept an `Image` **and** an
`ndarray` and be tested on both paths; wrap backend exceptions in an imlite
one; and keep `import imlite` cheap.

## License

MIT - see [LICENSE](LICENSE).
