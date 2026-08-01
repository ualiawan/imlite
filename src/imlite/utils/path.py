"""Path helpers, extension registries, and file-system utilities for imlite."""

import re
from pathlib import Path

__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "ensure_dir",
    "is_image_file",
    "is_video_file",
    "sorted_frame_paths",
]

# ---------------------------------------------------------------------------
# Extension registries
# ---------------------------------------------------------------------------

# Kept deliberately broad. imageio refuses to open a file whose extension it
# does not recognise, so an extension missing from these sets cannot be
# recovered by sniffing the file's contents - it just fails.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".jpe",
        ".jfif",
        ".png",
        ".bmp",
        ".dib",
        ".tiff",
        ".tif",
        ".webp",
        ".ico",
        ".ppm",
        ".pgm",
        ".pbm",
        ".pnm",
        ".tga",
        ".jp2",
        ".j2k",
        ".exr",
        ".hdr",
        ".pcx",
        ".sgi",
        ".im",
        ".msp",
        ".xbm",
    }
)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".m4v",
        ".avi",
        ".mov",
        ".qt",
        ".mkv",
        ".webm",
        ".wmv",
        ".asf",
        ".flv",
        ".f4v",
        ".ts",
        ".mts",
        ".m2ts",
        ".mpg",
        ".mpeg",
        ".m1v",
        ".m2v",
        ".mpv",
        ".3gp",
        ".3g2",
        ".ogv",
        ".ogg",
        ".vob",
        ".divx",
        ".mxf",
        ".rm",
        ".rmvb",
        ".gif",
    }
)


# ---------------------------------------------------------------------------
# Type-detection helpers
# ---------------------------------------------------------------------------


def is_image_file(path: str | Path) -> bool:
    """Return ``True`` if *path* has a recognised image extension.

    Args:
        path: File path to test.

    Returns:
        ``True`` when the extension is in :data:`IMAGE_EXTENSIONS`.
    """
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: str | Path) -> bool:
    """Return ``True`` if *path* has a recognised video extension.

    Args:
        path: File path to test.

    Returns:
        ``True`` when the extension is in :data:`VIDEO_EXTENSIONS`.
    """
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def sorted_frame_paths(
    directory: str | Path,
    extensions: frozenset[str] | None = None,
) -> list[str]:
    """Return image file paths inside *directory*, sorted in natural order.

    Natural sort means ``frame_2.png`` comes before ``frame_10.png``,
    unlike lexicographic sort which would put ``frame_10`` first.

    Args:
        directory: Directory to scan.
        extensions: Set of lowercase extensions to include.  Defaults to
            :data:`IMAGE_EXTENSIONS`.

    Returns:
        List of absolute file paths as strings, naturally sorted by filename.

    Raises:
        NotADirectoryError: If *directory* does not exist or is not a directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    exts = extensions if extensions is not None else IMAGE_EXTENSIONS
    paths = [p for p in directory.iterdir() if p.suffix.lower() in exts]

    def _natural_key(p: Path) -> list[int | str]:
        parts: list[int | str] = []
        for chunk in re.split(r"(\d+)", p.name):
            parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
        return parts

    return [str(p) for p in sorted(paths, key=_natural_key)]


def ensure_dir(path: str | Path) -> Path:
    """Create *path* as a directory (including parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved :class:`~pathlib.Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
