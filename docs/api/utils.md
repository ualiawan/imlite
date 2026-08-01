# Utilities

Helpers that are part of the public contract: ffmpeg discovery, dtype
normalisation, path handling, and the decorators that give every operation its
dual `Image`/`ndarray` API.

## ffmpeg discovery

::: imlite.utils.ffmpeg

## Pixel dtype normalisation

::: imlite.utils.dtype

## Paths and extension registries

::: imlite.utils.path
    options:
      members:
        - IMAGE_EXTENSIONS
        - VIDEO_EXTENSIONS
        - is_image_file
        - is_video_file
        - sorted_frame_paths
        - ensure_dir

## Logging and progress

::: imlite.utils.log

## Dispatch decorators

::: imlite.utils.validate
    options:
      members:
        - dispatch_type
        - color_op
        - ColorOp
