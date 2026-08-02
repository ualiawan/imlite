# Installation

```bash
pip install py-imlite
```

```python
import imlite            # note: the import name has no `py-` prefix
```

The distribution is called `py-imlite` because PyPI reserves `imlite`: it
normalises names by folding `i` and `l` together, which makes it collide with
the unrelated `lmlite` project. The import name, the CLI and every symbol are
unaffected. This split is common - `pillow` imports as `PIL`, `beautifulsoup4`
as `bs4`, `scikit-learn` as `sklearn`.

That is the whole thing. Video works immediately - **there is no system ffmpeg to
install and no OpenCV to compile**.

## What you get

| Package | Why | Size |
|---|---|---|
| `numpy` | The array type everything speaks | ~23 MB |
| `pillow` | Resampling backend for resize, rotate and blur | ~16 MB |
| `imageio` | Image and video container handling | ~2 MB |
| `imageio-ffmpeg` | A static ffmpeg binary, shipped in the wheel | ~30-90 MB |
| `tqdm` | Progress bars on long operations | <1 MB |

No OpenCV. imlite used to depend on `opencv-python-headless`, which is on its own
larger than everything above combined and pins `numpy>=2`. Pixel work is now numpy
and Pillow, and Pillow was already installed anyway - `imageio` requires it.

## Checking the install

```bash
imlite doctor
```

```
imlite          0.1.0
python          3.12.3
numpy           2.4.6
pillow          12.3.0
imageio         2.37.4
ffmpeg          7.1  (bundled with imageio-ffmpeg)
                .../site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.1

Video support is working.
```

Paste that output into any bug report and half the diagnosis is already done.

## Optional extras

```bash
pip install "py-imlite[show]"   # matplotlib, for Image.show() figures
pip install "py-imlite[dev]"    # pytest, ruff, mypy, pre-commit
pip install "py-imlite[docs]"   # mkdocs and friends
```

`Image.show()` works without the `show` extra - it falls back to Pillow's viewer.
In a notebook you do not need either: an `Image` renders itself as the last
expression in a cell.

## Platforms

Wheels exist for every platform imlite supports, so nothing is built from source:

| Platform | Bundled ffmpeg |
|---|---|
| Linux x86-64, aarch64, i686 (manylinux) | Yes |
| macOS x86-64 and Apple Silicon | Yes |
| Windows x86-64 | Yes |
| musl/Alpine, ppc64le, FreeBSD | No - see below |

Python 3.10 through 3.14.

### Platforms with no bundled ffmpeg

`imageio-ffmpeg` publishes no wheel for a handful of platforms. Images still work
everywhere; only video needs the binary. `imlite doctor` will tell you, and so will
the first video call:

```
ImliteFFmpegError: No ffmpeg binary is available on this platform (Linux ppc64le).
imageio-ffmpeg normally bundles one, but publishes no wheel for this platform. To fix it, either:
  1. Install ffmpeg system-wide:  sudo apt install ffmpeg   (or: dnf install ffmpeg / apk add ffmpeg)
  2. Or point imlite at an existing binary:
       export IMAGEIO_FFMPEG_EXE=/path/to/ffmpeg
Then run `imlite doctor` to confirm imlite can see it.
```

Either fix works; imlite uses whichever binary `imageio-ffmpeg` resolves.

## Using an ffmpeg you already have

Set `IMAGEIO_FFMPEG_EXE` before importing imlite. This is also how you opt into a
build with extra codecs:

```bash
export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg
imlite doctor          # confirms which binary was picked up
```

```python
import imlite
imlite.ffmpeg_info()
# {'available': True, 'exe': '/usr/local/bin/ffmpeg', 'version': '7.1',
#  'bundled': False, 'error': ''}
```

Resolution is lazy and cached: importing imlite never launches ffmpeg, and the
lookup happens at most once per process.

## Installing from source

```bash
git clone https://github.com/ualiawan/imlite.git
cd imlite
uv sync --extra dev --extra docs --extra show   # or: pip install -e ".[dev,docs,show]"
pre-commit install
pytest
```
