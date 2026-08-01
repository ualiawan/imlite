"""The ``imlite`` command-line interface.

Five subcommands cover the tasks people otherwise write a throwaway script
for::

    imlite doctor                              # is video support working?
    imlite info clip.mp4                       # metadata for an image or video
    imlite extract clip.mp4 frames/ --step 2   # video -> frames
    imlite merge frames/ out.mp4 --fps 30      # frames -> video
    imlite convert in.png out.jpg --resize 640x

Only the standard library is used here; the heavy imports happen inside each
handler so ``imlite --help`` stays instant.
"""

import argparse
import logging
import sys
from collections.abc import Sequence

from imlite._version import __version__

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the imlite command-line interface.

    Args:
        argv: Argument list to parse.  Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit status: ``0`` on success, ``1`` on a handled error,
        ``130`` if interrupted.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    _configure_logging(args)

    from imlite.exceptions import ImliteError

    try:
        return int(args.handler(args))
    except ImliteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted", file=sys.stderr)
        return 130


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser and all subcommands."""
    parser = argparse.ArgumentParser(
        prog="imlite",
        description="Lightweight image and video processing. No OpenCV, no system ffmpeg.",
    )
    parser.add_argument("--version", action="version", version=f"imlite {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress log messages")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide progress bars and logs")
    parser.set_defaults(command=None)

    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = subcommands.add_parser(
        "doctor", help="check that imlite's backends are working", description=_doctor.__doc__
    )
    doctor.set_defaults(handler=_doctor)

    info = subcommands.add_parser(
        "info", help="print metadata for an image or video", description=_info.__doc__
    )
    info.add_argument("path", help="image or video file")
    info.set_defaults(handler=_info)

    extract = subcommands.add_parser(
        "extract", help="extract frames from a video", description=_extract.__doc__
    )
    extract.add_argument("video", help="source video file")
    extract.add_argument("output_dir", help="directory to write frames into")
    extract.add_argument("--step", type=int, default=1, help="take every Nth frame (default: 1)")
    extract.add_argument("--start", type=int, default=0, help="first frame index (default: 0)")
    extract.add_argument("--end", type=int, default=None, help="stop before this frame index")
    extract.add_argument("--format", default="png", help="frame image format (default: png)")
    extract.set_defaults(handler=_extract)

    merge = subcommands.add_parser(
        "merge", help="encode a directory of frames into a video", description=_merge.__doc__
    )
    merge.add_argument("input_dir", help="directory of frame images, in natural sort order")
    merge.add_argument("output", help="output video file, e.g. out.mp4")
    merge.add_argument("--fps", type=float, default=30.0, help="frame rate (default: 30)")
    merge.add_argument("--codec", default="libx264", help="ffmpeg codec (default: libx264)")
    merge.add_argument("--resize", metavar="WxH", help="resize frames first, e.g. 640x360 or 640x")
    merge.set_defaults(handler=_merge)

    convert = subcommands.add_parser(
        "convert", help="convert and transform a single image", description=_convert.__doc__
    )
    convert.add_argument("input", help="source image file")
    convert.add_argument("output", help="destination image file")
    convert.add_argument("--resize", metavar="WxH", help="target size, e.g. 640x360, 640x or x360")
    convert.add_argument("--rotate", type=float, default=None, help="rotate counter-clockwise")
    convert.add_argument("--flip", choices=["h", "v", "both"], help="flip along an axis")
    convert.add_argument("--gray", action="store_true", help="convert to grayscale")
    convert.add_argument("--quality", type=int, default=95, help="JPEG/WebP quality (default: 95)")
    convert.set_defaults(handler=_convert)

    return parser


def _configure_logging(args: argparse.Namespace) -> None:
    """Apply the global --verbose / --quiet flags."""
    import imlite

    if args.quiet:
        imlite.set_progress(False)
        imlite.set_verbosity("ERROR")
    elif args.verbose:
        imlite.set_verbosity(logging.INFO)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _doctor(_args: argparse.Namespace) -> int:
    """Report imlite's version, its backends, and whether video support works."""
    import numpy
    import PIL

    import imlite

    print(f"imlite          {imlite.__version__}")
    print(f"python          {sys.version.split()[0]}")
    print(f"numpy           {numpy.__version__}")
    print(f"pillow          {PIL.__version__}")

    try:
        import imageio

        print(f"imageio         {imageio.__version__}")
    except ImportError:  # pragma: no cover - imageio is a hard dependency
        print("imageio         MISSING")

    report = imlite.ffmpeg_info()
    if report["available"]:
        origin = "bundled with imageio-ffmpeg" if report["bundled"] else "system install"
        print(f"ffmpeg          {report['version']}  ({origin})")
        print(f"                {report['exe']}")
        print("\nVideo support is working.")
        return 0

    print("ffmpeg          NOT FOUND")
    print(f"\n{report['error']}")
    return 1


def _info(args: argparse.Namespace) -> int:
    """Print metadata for an image or a video file."""
    import imlite

    obj = imlite.load(args.path)

    if isinstance(obj, imlite.Video):
        for key, value in obj.info.items():
            print(f"{key:<12} {value}")
        return 0

    if isinstance(obj, imlite.Image):
        print(f"{'path':<12} {obj.path}")
        print(f"{'width':<12} {obj.width}")
        print(f"{'height':<12} {obj.height}")
        print(f"{'channels':<12} {obj.channels}")
        print(f"{'color_space':<12} {obj.color_space}")
        print(f"{'dtype':<12} {obj.dtype}")
        return 0

    print(f"{'source':<12} {args.path}")
    print(f"{'frames':<12} {len(obj)}")
    return 0


def _extract(args: argparse.Namespace) -> int:
    """Decode a video and write its frames into a directory as image files."""
    import imlite

    sequence = imlite.extract_frames(
        args.video,
        output_dir=args.output_dir,
        step=args.step,
        start=args.start,
        end=args.end,
        fmt=args.format,
    )
    print(f"Extracted {len(sequence)} frames to {args.output_dir}")
    return 0


def _merge(args: argparse.Namespace) -> int:
    """Encode a directory of frame images into a video file."""
    import imlite

    sequence = imlite.read_frames(args.input_dir)
    frame_count = len(sequence)
    if frame_count == 0:
        print(f"error: no image files found in {args.input_dir!r}", file=sys.stderr)
        return 1

    if args.resize:
        width, height = _parse_size(args.resize)
        sequence = sequence.resize(width, height)

    sequence.merge(fps=args.fps, codec=args.codec).save(args.output)
    print(f"Encoded {frame_count} frames to {args.output} at {args.fps} fps")
    return 0


def _convert(args: argparse.Namespace) -> int:
    """Read one image, apply the requested transforms, and write it out."""
    import imlite

    image = imlite.read_image(args.input)

    if args.resize:
        width, height = _parse_size(args.resize)
        image = image.resize(width, height)
    if args.rotate is not None:
        image = image.rotate(args.rotate)
    if args.flip:
        image = image.flip(args.flip)
    if args.gray:
        image = image.to_gray()

    image.save(args.output, quality=args.quality)
    print(f"Wrote {args.output} ({image.width}x{image.height})")
    return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_size(text: str) -> tuple[int | None, int | None]:
    """Parse a ``WxH`` size argument, where either side may be omitted.

    Args:
        text: ``"640x360"``, ``"640x"`` (height follows the aspect ratio) or
            ``"x360"`` (width follows).

    Returns:
        A ``(width, height)`` tuple in which the omitted side is ``None``.

    Raises:
        SystemExit: If *text* is not a valid size specification.
    """
    parts = text.lower().split("x")
    if len(parts) != 2 or not any(parts):
        raise SystemExit(f"error: invalid size {text!r}; expected WxH, e.g. 640x360, 640x or x360")

    try:
        width = int(parts[0]) if parts[0] else None
        height = int(parts[1]) if parts[1] else None
    except ValueError:
        raise SystemExit(
            f"error: invalid size {text!r}; width and height must be whole numbers"
        ) from None

    return width, height


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
