# Changelog

All notable changes to imlite are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
imlite uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [0.1.0] - 2026-08-02

First release.

### Core

- `imlite.load()` - one entry point that returns an `Image`, `Video` or
  `FrameSequence` depending on what it is given: a file path, a directory, a
  numpy array, a list of frames, or an existing imlite object.
- `Image` - immutable `uint8` wrapper with a colour-space tag. Every transform
  returns a new instance; the original is never modified.
- `Video` - lazy handle to a video file, or a pending encode built from frames.
- `FrameSequence` - ordered frames, lazy by default. Transforms are queued and
  applied per frame during iteration, so peak memory is one frame regardless of
  the video's length.
- Explicit constructors: `read_image()`, `read_video()`, `read_frames()`.

### Operations

Each is available as a chained method and as a plain function that accepts an
`Image` **or** a raw `np.ndarray`, returning the same type it was given.

- Geometry: `crop`, `rotate`, `resize`, `thumbnail`, `flip`, `pad`.
- Pixels: `blur`, `brightness`, `contrast`, `invert`, `threshold`.
- Colour: `to_rgb`, `to_bgr`, `to_gray`, `to_hsv`, `to_lab`.
- Video: `extract_frames`, `merge_frames`, `video_info`.

Colour conversions are pure numpy and reproduce OpenCV's 8-bit encodings exactly:
BT.601 luma for grayscale, hue in 0-179 for HSV, and `L*x255/100` with `+128`
offsets for LAB. All five spaces round-trip through an RGB hub.

### Backends: no OpenCV, no system ffmpeg

- `opencv-python-headless` is **not** a dependency. Resampling uses Pillow, which
  `imageio` already required, so this removes roughly **118 MB** from the install
  (~246 MB down to ~128 MB) at no cost. It also lifts the `numpy>=2` pin that
  OpenCV 5 imposed, so numpy 1.x works again.
- Video uses the static ffmpeg binary bundled in the `imageio-ffmpeg` wheel.
  `pip install imlite` is the entire setup on Linux, macOS and Windows.
- New `imlite.ffmpeg_info()` and `imlite doctor` report the resolved binary and
  its origin. On a platform with no bundled build, `ImliteFFmpegError` names the
  right package-manager command and the `IMAGEIO_FFMPEG_EXE` override, rather
  than surfacing a bare `RuntimeError`.
- ffmpeg resolution is lazy and cached - `import imlite` never launches it.

### Command line

`imlite doctor`, `info`, `extract`, `merge` and `convert`, with `--verbose` and
`--quiet`. No dependencies beyond the standard library.

### Pixel dtype policy

Images are stored as `uint8`. Other dtypes are converted by documented rules
rather than truncated: `bool` maps to 0/255, `uint16` scales by 257, floats in
0.0-1.0 scale by 255, floats in 0.0-255.0 round. Anything ambiguous - negative
values, floats above 255, NaN, out-of-range integers - raises `ImliteDtypeError`
with instructions. Previously `float 1.0` silently became `1` and `uint16 1000`
silently became `232`.

### Quality of life

- `Image` renders inline in Jupyter via `_repr_png_()`; no `show()` call and no
  matplotlib needed.
- `Image.show()` uses matplotlib when installed and falls back to Pillow's viewer,
  instead of raising on a dependency that was never declared.
- `Image.from_pil()` / `Image.to_pil()` for Pillow interop.
- `Image(..., copy=False)` adopts a buffer without copying, for hot loops.
- `Image.array` gives a read-only, zero-copy view; `Image.data` still copies.
- Progress bars (`tqdm`) on long operations, controlled by
  `imlite.set_progress()`.
- Standard `logging` throughout, controlled by `imlite.set_verbosity()`.
- Ships `py.typed` - the package is fully typed and passes `mypy --strict`.

### Notes on behaviour worth knowing

- **Raw numpy arrays are assumed to be BGR** wherever a colour space is not
  stated, matching OpenCV's convention. `merge_frames()` logs a warning when it
  receives raw arrays, because RGB input would silently produce colour-swapped
  video. Wrap them in `Image.from_numpy(arr, color_space="RGB")` to be explicit.
- **`resize(resample=...)`** accepts `"nearest"`, `"box"`, `"bilinear"`,
  `"bicubic"`, `"lanczos"` or `"auto"` (the default: box when shrinking, lanczos
  when enlarging).
- **Encoded videos come out the size you asked for.** imageio's writer defaults
  to `macro_block_size=16`, which silently rounds frame dimensions up - a 640x360
  render becomes 640x368 and 1920x1080 becomes 1920x1088. imlite defaults to
  `2` instead, the smallest value `yuv420p` actually requires, and logs a warning
  on the rare frame size that still needs padding. Pass
  `save(..., macro_block_size=16)` if an old hardware decoder needs macroblock
  alignment.
- **Videos are identified by extension.** The ffmpeg backend selects its decoder
  from the file extension and cannot sniff content, so a video with an
  unrecognised extension raises with instructions instead of failing later.
  Images *are* sniffed, so a mislabelled PNG still loads.
- **Rotation by an exact multiple of 90 degrees is lossless** and uses `np.rot90`.

[Unreleased]: https://github.com/ualiawan/imlite/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ualiawan/imlite/releases/tag/v0.1.0
