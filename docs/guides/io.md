# Reading & Writing

## Read an image

```python
import imlite

img = imlite.load("photo.jpg")       # smart loader
img = imlite.read_image("photo.jpg") # explicit constructor
print(img.color_space)               # "BGR" - OpenCV convention
print(img.shape)                     # (H, W, 3)
```

`read_image()` uses `imageio` (which uses Pillow underneath). There is no OpenCV
dependency. 16-bit and floating-point files are converted to 8-bit by documented
rules rather than truncated - see [dtype handling](dtypes.md).

## Write / save an image

```python
img.save("out.png")                 # via Image method
imlite.save(img, "out.png")         # same, functional form
imlite.save(img, "out.jpg", quality=85)   # JPEG quality 0-100
```

Parent directories are created automatically.

## Supported formats

| Format  | Read | Write |
|---------|------|-------|
| JPEG    | yes  | yes (quality param) |
| PNG     | yes  | yes (lossless) |
| BMP     | yes  | yes |
| TIFF    | yes  | yes |
| WebP    | yes  | yes |
| PPM/PGM | yes  | yes |

Any format `imageio`/Pillow can decode works. imlite recognises the extensions in
`imlite.IMAGE_EXTENSIONS`; a file with an unrecognised extension is sniffed by
content, so a mislabelled PNG still loads.
