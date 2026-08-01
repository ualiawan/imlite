# Command Line

Installing imlite puts an `imlite` command on your PATH. It covers the jobs people
otherwise write a throwaway script for.

```bash
imlite --help
imlite <command> --help
```

Two global flags apply everywhere: `-v/--verbose` turns on progress log messages,
and `-q/--quiet` silences logs and progress bars (use it in scripts and CI).

## `imlite doctor`

Reports the resolved backends and whether video support works. Run it first when
anything is off, and paste the output into bug reports.

```bash
imlite doctor
```

Exits `0` when ffmpeg was found and `1` when it was not, so it works in a health
check:

```bash
imlite doctor || echo "video support is broken"
```

## `imlite info`

Prints metadata for an image, a video, or a directory of frames.

```bash
imlite info clip.mp4
```

```
path         clip.mp4
fps          25.0
frame_count  300
duration     12.0
width        1920
height       1080
codec        h264
size_bytes   4194304
```

```bash
imlite info photo.jpg
imlite info frames/
```

## `imlite extract`

Video to frames.

```bash
imlite extract clip.mp4 frames/
imlite extract clip.mp4 frames/ --step 5              # every 5th frame
imlite extract clip.mp4 frames/ --start 100 --end 200 # a range
imlite extract clip.mp4 frames/ --format jpg          # jpg instead of png
```

| Flag | Default | Meaning |
|---|---|---|
| `--step N` | 1 | Take every Nth frame |
| `--start N` | 0 | First frame index, inclusive |
| `--end N` | end of file | Stop *before* this index |
| `--format EXT` | `png` | Output image format |

Frames are written as `frame_00000.png`, `frame_00001.png`, ... so they sort
correctly and feed straight back into `imlite merge`.

## `imlite merge`

Frames to video. Files are read in natural sort order, so `frame_2.png` comes
before `frame_10.png`.

```bash
imlite merge frames/ out.mp4 --fps 30
imlite merge frames/ out.mp4 --fps 24 --resize 1280x720
imlite merge frames/ out.webm --codec libvpx-vp9
```

## `imlite convert`

One-shot transforms on a single image. Steps apply in the order listed below,
regardless of the order you pass the flags.

```bash
imlite convert photo.png photo.jpg --quality 85
imlite convert photo.jpg thumb.jpg --resize 320x
imlite convert scan.png out.png --rotate 90 --gray
```

| Flag | Meaning |
|---|---|
| `--resize WxH` | Resize. `640x360` exact, `640x` or `x360` keeps the aspect ratio |
| `--rotate DEG` | Rotate counter-clockwise; multiples of 90 are lossless |
| `--flip {h,v,both}` | Flip along an axis |
| `--gray` | Convert to grayscale |
| `--quality N` | JPEG/WebP quality, 0-100 (default 95) |

## Recipes

Halve a video's frame rate:

```bash
imlite extract clip.mp4 /tmp/f --step 2 -q && imlite merge /tmp/f out.mp4 --fps 12.5 -q
```

Contact sheet of thumbnails from a video:

```bash
imlite extract clip.mp4 /tmp/f --step 30 --format jpg -q
for f in /tmp/f/*.jpg; do imlite convert "$f" "thumbs/$(basename "$f")" --resize 160x -q; done
```

Transcode a folder of clips:

```bash
for v in *.mov; do imlite extract "$v" /tmp/f -q && imlite merge /tmp/f "${v%.mov}.mp4" -q; done
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | A handled error - the message is on stderr, prefixed `error:` |
| 2 | Bad arguments (from `argparse`) |
| 130 | Interrupted with Ctrl-C |
