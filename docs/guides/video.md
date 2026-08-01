# Video & Frames

!!! note
    Audio is dropped on transcode, videos are identified by file extension, and
    decoding is sequential. See [Limitations](limitations.md#video) for the
    details and the workarounds.

## Loading a video

Always start by creating a `Video` object:

```python
import imlite

# Smart loader
vid = imlite.load("clip.mp4")

# Explicit constructor
vid = imlite.read_video("clip.mp4")

print(vid.fps)          # 29.97
print(vid.frame_count)  # 900
print(vid.duration)     # 30.03 seconds
print(vid.width, vid.height)  # 1920 1080
```

## Extracting frames to disk

```python
vid = imlite.read_video("clip.mp4")

# All frames
vid.extract_frames(output_dir="frames/")

# Every 5th frame only
vid.extract_frames(output_dir="frames/", step=5)

# Frames 100-200
vid.extract_frames(output_dir="frames/", start=100, end=200)
```

A `tqdm` progress bar is shown automatically.

## Lazy frame iteration (memory-efficient)

```python
seq = imlite.read_video("clip.mp4").extract_frames(step=2)
for frame in seq:           # frames are decoded one-at-a-time
    do_something(frame)
```

## Applying transforms to all frames

`FrameSequence` transforms are **deferred** - applied lazily during iteration,
so no extra copies are held in memory.

```python
seq = (
    imlite.read_video("clip.mp4")
    .extract_frames(step=2)
    .crop(0, 0, 640, 360)
    .resize(320, 180)
    .to_gray()
)
for frame in seq:
    print(frame.shape)   # (180, 320, 1)
```

## Saving processed frames to disk

```python
seq.save_frames("processed/", fmt="png")
```

## Rebuilding a video from frames

```python
# From a FrameSequence
seq.merge(fps=25).save("output.mp4")

# From a directory of images
imlite.load("frames/").merge(fps=25).save("output.mp4")
```

## Getting video metadata

```python
vid = imlite.read_video("clip.mp4")
print(vid.fps, vid.frame_count, vid.duration, vid.width, vid.height)
```
