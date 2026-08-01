# Pixel dtypes

imlite stores every image as `uint8`. Input in another dtype is converted on
construction, by rules that are documented and never silent.

## Why this matters

A plain `astype(np.uint8)` corrupts data quietly, and that is what most quick
scripts do:

```python
np.uint16(1000).astype(np.uint8)        # 232 - wrapped around, not scaled
np.float32(1.0).astype(np.uint8)        # 1 - a white pixel became near-black
```

Both produce a valid-looking array with wrong pixels. imlite converts what is
unambiguous and raises on the rest.

## The rules

| Input dtype | Conversion |
|---|---|
| `uint8` | Used as-is |
| `bool` | `False` -> 0, `True` -> 255 |
| `uint16` | Divided by 257, so the full 16-bit range maps to the full 8-bit range |
| Integer, values within 0-255 | Cast directly |
| Float, values within 0.0-1.0 | Scaled by 255 and rounded |
| Float, values within 0.0-255.0 | Rounded |
| Anything else | [`ImliteDtypeError`](../api/exceptions.md) |

```python
import numpy as np, imlite

imlite.Image.from_numpy(np.ones((4, 4, 3), np.float32)).data[0, 0]
# array([255, 255, 255], dtype=uint8)     not [1, 1, 1]

imlite.Image.from_numpy(np.full((4, 4, 3), 1000, np.uint16)).data[0, 0]
# array([3, 3, 3], dtype=uint8)           not [232, 232, 232]

mask = np.zeros((4, 4), bool); mask[0, 0] = True
imlite.Image.from_numpy(mask).data[0, 0]
# 255                                     a visible mask, not an invisible 1
```

## What raises

Anything whose intended range cannot be inferred:

```python
imlite.Image.from_numpy(np.full((4, 4, 3), 5000, np.int32))
# ImliteDtypeError: Cannot convert int32 image data to uint8: values span
# 5000..5000, which is outside the 8-bit range 0..255. Convert it yourself
# first, e.g. `arr = np.clip(arr, 0, 255).astype('uint8')` or
# `arr = (arr / arr.max() * 255).astype('uint8')`.
```

Also rejected: negative values, floats above 255, and NaN or infinity. The message
always names the offending range and shows two ways to fix it, because only you
know whether your data wants clipping or rescaling.

## Float in 0-1 vs 0-255

Both are common, so imlite picks by looking at the data: a float array whose
maximum is at or below `1.0` is treated as normalised and scaled by 255, otherwise
values are rounded in place.

The ambiguous case is a genuinely dark 0-255 float image whose maximum happens to
be <= 1.0. It is indistinguishable from a normalised one, and imlite will brighten
it. If your pipeline produces such data, convert explicitly:

```python
img = imlite.Image.from_numpy(np.rint(arr).astype(np.uint8))
```

## Reading 16-bit files

The same rules apply to files, so a 16-bit TIFF or PNG scales down rather than
wrapping:

```python
img = imlite.read_image("scan-16bit.tif")
img.dtype     # dtype('uint8')
```

imlite is an 8-bit library. If you need to keep 16-bit precision, read the file
with `imageio` directly and do the maths yourself - see
[Limitations](limitations.md#imlite-is-an-8-bit-library).

## Converting explicitly

The helper is public if you want it on a bare array:

```python
from imlite.utils.dtype import as_uint8

as_uint8(np.array([[0.0, 0.5, 1.0]], np.float32))
# array([[  0, 128, 255]], dtype=uint8)
```
