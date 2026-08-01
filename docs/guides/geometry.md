# Geometry Operations

All geometry functions accept both `np.ndarray` and `Image` objects and return the
same type as input. Resampling uses Pillow; there is no OpenCV dependency.

## crop

```python
# As a method
out = img.crop(x=50, y=50, width=200, height=200)

# Functional form
out = imlite.crop(arr, x=50, y=50, width=200, height=200)
```

Raises `CropOutOfBoundsError` if the crop rectangle extends beyond the image bounds.

## rotate

```python
out = img.rotate(90)         # 90 degrees counter-clockwise, canvas expands
out = img.rotate(45)         # arbitrary angle, canvas expands by default
out = img.rotate(45, expand=False)   # fixed canvas (corners clipped)
```

Exact multiples of 90 degrees take a lossless `np.rot90` path - no resampling, no
interpolation artefacts, and faster than a general warp. Every other angle is
resampled bicubically by Pillow, with the corners filled black.

```python
img.rotate(90).rotate(90).rotate(90).rotate(90) == img   # exactly equal
```

## resize

```python
out = img.resize(width=640, height=480)        # explicit size
out = img.resize(width=320, keep_aspect=True)  # infer height from aspect ratio
out = img.resize(height=240, keep_aspect=True) # infer width
```

Pick the filter with `resample=`:

| Value | Pillow filter |
|---|---|
| `"auto"` (default) | `BOX` when shrinking, `LANCZOS` when enlarging |
| `"nearest"` | `NEAREST` - fastest, blocky; right for label masks |
| `"box"` | `BOX` - area averaging, the equivalent of OpenCV's `INTER_AREA` |
| `"bilinear"` | `BILINEAR` |
| `"bicubic"` | `BICUBIC` |
| `"lanczos"` | `LANCZOS` - highest quality, slowest |

```python
out = img.resize(320, 240, resample="nearest")   # segmentation masks
```

## thumbnail

Scale so the longest side is `size` pixels, preserving the aspect ratio. Unlike
`resize`, it never enlarges - an image already smaller than `size` comes back
unchanged, which is what thumbnail generation wants.

```python
out = img.thumbnail(256)
imlite.load("photo.jpg").thumbnail(256).save("thumb.jpg")
```

## flip

```python
out = img.flip("h")           # horizontal mirror
out = img.flip("v")           # vertical mirror
out = img.flip("both")        # flip both axes (same as a 180 degree rotation)
```

## pad

```python
out = img.pad(top=10, bottom=10, left=20, right=20)
out = img.pad(top=10, bottom=10, left=20, right=20, color=(255, 255, 255))
```

`color` is interpreted in the image's **current** colour space, so `(255, 0, 0)`
is blue on BGR data and red on RGB data. A single intensity works too, and a
3-tuple on an RGBA image gains a fully opaque alpha:

```python
img.pad(top=10, color=128)                  # grey border, any channel count
rgba.pad(top=10, color=(10, 20, 30))        # alpha becomes 255
```

Passing the wrong number of components raises `ValueError` rather than
broadcasting something unexpected.
