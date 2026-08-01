# Limitations

Everything imlite does not do, and why. Each entry says what to do instead, and
links to the roadmap item that would remove it.

Nothing here is a bug. If you hit something not on this page,
[open an issue](https://github.com/ualiawan/imlite/issues).

---

## Scope

### imlite is an 8-bit library

Pixels are stored as `uint8`. A 16-bit TIFF or PNG is scaled down on read
(divided by 257), and floating-point input is converted by
[documented rules](dtypes.md).

**If you need the precision:** read the file with `imageio` directly and work on
the array yourself. imlite's functional API still accepts it, as long as you
convert to `uint8` before wrapping it in an `Image`.

```python
import imageio.v3 as iio
raw = iio.imread("scan-16bit.tif")   # stays uint16
```

*Not planned.* A dual-depth data model would complicate every operation for a
minority of users.

### It is not a replacement for OpenCV's breadth

imlite covers the common 90%: geometry, pixel values, colour, and video frame
handling. There is no feature detection, no contours, no morphology, no
calibration, no ML.

**If you need those:** use imlite for I/O and framing, hand `img.to_numpy()` to
whatever library does the specialist work, and wrap the result back with
`imlite.Image.from_numpy(...)`. The two coexist fine - imlite just does not
*depend* on OpenCV.

### No GUI

`Image.show()` opens a matplotlib or Pillow window, and images render inline in
Jupyter. There is no interactive viewer, no event loop, no drawing tools.

---

## Colour and channels

### Raw numpy arrays are assumed to be BGR

A bare `np.ndarray` carries no colour-space tag, so imlite assumes OpenCV's
convention. If your arrays are RGB, output colours will be swapped.

```python
imlite.merge_frames(rgb_frames, "out.mp4")   # logs a warning; colours swapped
```

**Do this instead** - say what the data is, once:

```python
frames = [imlite.Image.from_numpy(a, color_space="RGB") for a in rgb_frames]
imlite.merge_frames(frames, "out.mp4")       # correct
```

Functional colour ops take the same information through `source=`:

```python
imlite.to_gray(array, source="RGB")
```

`merge_frames()` logs a warning whenever it receives raw arrays, so this fails
loudly rather than silently.

### Alpha is dropped by some conversions

`to_rgb()` and `to_bgr()` preserve an alpha channel. `to_gray()`, `to_hsv()` and
`to_lab()` discard it, matching OpenCV. Split it off first if you need it back:

```python
alpha = img.data[:, :, 3]
gray = img.to_gray()
```

### Colour round trips are lossy at 8 bits

`BGR -> LAB -> BGR` recovers the original to within a few counts on saturated
colours. That is inherent to the 8-bit encoding - `a*` and `b*` quantise to whole
units - not an imlite defect. HSV drifts less, typically 1-4 counts.

---

## Video

### Videos are identified by extension, not content

The ffmpeg backend selects its decoder from the file extension and cannot sniff
the bytes, so a video with an unrecognised extension raises rather than being
probed:

```
ImliteOpenError: Could not determine what 'clip.dat' is ...
If it is a video, rename it to a known extension (.mp4, .mov, .mkv, ...)
```

`imlite.VIDEO_EXTENSIONS` lists the 29 recognised ones. **Images are sniffed**,
so a mislabelled PNG loads fine - the asymmetry is a backend constraint.

**Workaround:** rename, or symlink to a recognised extension.

### Decoding is sequential

Frames are read in order and unwanted ones discarded. A large `start=` therefore
decodes and throws away everything before it.

```python
seq = video.extract_frames(start=50_000)   # decodes 50,000 frames first
```

This is deliberate: random access makes ffmpeg re-initialise, which costs more
than skipping at the strides real pipelines use. It only bites when you want a
small window late in a long file.

*Roadmap: keyframe-aware seeking for large `start` offsets.*

### Audio is dropped

ffmpeg is invoked with `-an`. Extracting frames and re-encoding produces a
silent video.

**Workaround:** mux the original audio back with ffmpeg directly.

```bash
ffmpeg -i processed.mp4 -i original.mp4 -c copy -map 0:v:0 -map 1:a:0 final.mp4
```

*Roadmap: audio passthrough on transcode.*

### `save()` re-encodes, it does not stream-copy

`imlite.load("in.mov").save("out.mp4")` fully decodes and re-encodes. That is
what you want when transforming frames, but it is slow and generation-lossy when
you only meant to change the container.

*Roadmap: detect the no-transform case and stream-copy.*

### Frame dimensions are rounded up to even numbers

`yuv420p` requires even width and height, so odd dimensions are padded by one
pixel. imlite logs a warning when this happens.

imageio's own default rounds up to a multiple of **16** - which silently turns
640x360 into 640x368. imlite defaults to `2` instead. Pass
`save(..., macro_block_size=16)` if an old hardware decoder needs the alignment.

### Frame counts can be approximate

Some containers (streamed WebM, certain MKVs) report no frame count. imlite
estimates from duration x fps, and only decodes the whole file to count when
there is no other option. `len(sequence)` on such a file may be off by one or
two.

---

## Performance

### Not competing with hand-tuned C++

imlite is a convenience layer over numpy, Pillow and ffmpeg. Per-frame Python
overhead is real. For throughput-critical work, hand `to_numpy()` to a vectorised
or compiled pipeline.

### Frame processing is single-threaded

Transforms run one frame at a time in the calling thread. On a multi-core machine
a batch image job leaves most of the CPU idle.

**Workaround:** drive it yourself.

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor() as pool:
    pool.map(process_one, paths)
```

*Roadmap: parallel `save_frames()`, which is I/O bound and should scale well.*

### `to_list()` and slicing materialise everything

`seq[1:10]` and `seq.to_list()` load every frame into RAM. Integer indexing
(`seq[0]`) streams and stops early; slices cannot, because they need the length.

**For long videos:** iterate, or use `start`/`end`/`step` on
`extract_frames()`.

---

## Platform

### A few platforms have no bundled ffmpeg

`imageio-ffmpeg` publishes no wheel for musl/Alpine, ppc64le or FreeBSD. Images
work everywhere; video needs a system ffmpeg on those. imlite tells you exactly
what to run - see [Installation](install.md#platforms-with-no-bundled-ffmpeg).

### Python 3.10+

3.9 reached end of life in October 2025.

---

## See also

- [Roadmap](https://github.com/ualiawan/imlite/blob/main/PLAN.md#11-roadmap) -
  which of these are planned to change
- [Pixel dtypes](dtypes.md) - the full conversion table
- [Colour operations](color.md) - encodings and round-trip behaviour
