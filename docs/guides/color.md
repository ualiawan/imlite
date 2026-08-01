# Colour Operations

imlite tracks the colour space of every `Image` in its `color_space` attribute.
All colour conversions are **idempotent** - converting to the current space returns a copy unchanged.

## Supported colour spaces

| Space | Tag | Channels |
|-------|-----|----------|
| BGR (OpenCV default) | `"BGR"` | 3 |
| RGB | `"RGB"` | 3 |
| Grayscale | `"GRAY"` | 1 |
| HSV | `"HSV"` | 3 |
| CIE L\*a\*b\* | `"LAB"` | 3 |

## Converting via method chaining

```python
img = imlite.load("photo.jpg")   # color_space = "BGR"

rgb  = img.to_rgb()              # color_space = "RGB"
gray = img.to_gray()             # color_space = "GRAY", shape (H, W, 1)
hsv  = img.to_hsv()              # color_space = "HSV"
lab  = img.to_lab()              # color_space = "LAB"
bgr  = rgb.to_bgr()              # back to BGR
```

## Functional form

```python
import imlite
import numpy as np

arr_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
arr_rgb = imlite.to_rgb(arr_bgr)                 # ndarray in, ndarray out
arr_gray = imlite.to_gray(arr_rgb, source="RGB") # declare the input space
```

An `Image` carries its own `color_space`, so the conversion reads it and
`source=` is unnecessary. A bare array has no tag, so `source=` says what it is -
defaulting to `"BGR"`, matching the rest of imlite.

## 8-bit encodings

Conversions are pure numpy and reproduce OpenCV's 8-bit encodings, so thresholds
written against `cv2` keep working:

| Space | Encoding |
|---|---|
| `GRAY` | ITU-R BT.601 luma, `0.299R + 0.587G + 0.114B` |
| `HSV` | H in `0-179` (degrees / 2), S and V in `0-255` |
| `LAB` | `L = L* x 255/100`, `a = a* + 128`, `b = b* + 128` |

Every conversion routes through RGB as a hub, so all pairs round-trip:

```python
img.to_lab().to_bgr()   # recovers the original to within 8-bit quantisation
```

The residual error is inherent to the encoding - `a*` and `b*` quantise to whole
units, which is a few RGB counts on saturated colours.

## Alpha channels

`to_rgb()` and `to_bgr()` preserve an alpha channel and keep it last, so BGRA
`(1, 2, 3, 9)` becomes RGBA `(3, 2, 1, 9)`. `to_gray()`, `to_hsv()` and
`to_lab()` drop it, matching OpenCV.

## Grayscale shape

`to_gray()` always returns a 3-D array with shape `(H, W, 1)` to keep the API consistent.
If you need a 2-D array for a third-party function:

```python
gray_2d = img.to_gray().data[:, :, 0]
```
