# Chaining Syntax

imlite supports a fully chainable dot-syntax on `Image`, `Video`, and `FrameSequence`.
Every transform method returns a **new** instance - the original is never mutated.

On a `FrameSequence` the transforms are *deferred*: each call queues a function and
returns immediately. Nothing is decoded until the sequence is iterated, or until
`save()`, `save_frames()` or `to_list()` forces it. That is what keeps memory flat
across a long video.

The rule is simple: **always create an object first**, then chain transforms on it.

## Image chaining

```python
import imlite

result = (
    imlite.load("photo.jpg")           # -> Image
    .crop(x=0, y=0, width=400, height=400)
    .rotate(90)
    .resize(width=256, height=256)
    .to_gray()
    .save("out.png")
)
```

## Video -> FrameSequence -> Video pipeline

```python
(
    imlite.load("input.mp4")          # -> Video
    .extract_frames(step=2)           # -> FrameSequence (lazy)
    .resize(640, 360)                 # -> FrameSequence (deferred)
    .brightness(1.1)                  # -> FrameSequence (deferred)
    .merge(fps=25)                    # -> Video (pending encode)
    .save("output.mp4")               # encode to disk
)
```

## Directory -> FrameSequence -> Video

```python
(
    imlite.load("frames/")            # -> FrameSequence
    .crop(0, 0, 1280, 720)
    .merge(fps=30)                    # -> Video
    .save("assembled.mp4")
)
```

## Mixing functional and chained APIs

You can mix both styles freely:

```python
arr = imlite.crop(some_ndarray, 0, 0, 100, 100)   # returns ndarray
img = imlite.load("photo.jpg").rotate(45)          # returns Image
combined = imlite.load(arr).resize(64, 64)         # ndarray -> Image, then resize
```
