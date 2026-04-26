"""Path helpers, extension registries, and file-system utilities for imlite."""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "is_image_file",
    "is_video_file",
    "sorted_frame_paths",
    "ensure_dir",
    "stem",
]

# ---------------------------------------------------------------------------
# Extension registries
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".ico", ".ppm", ".pgm"}
)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".gif"}
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


def stem(path: str | Path) -> str:
    """Return the filename without its extension.

    Args:
        path: Any file path.

    Returns:
        The stem (e.g. ``"photo"`` from ``"/some/dir/photo.jpg"``).
    """
    return Path(path).stem


def resolve(path: str | Path) -> str:
    """Return the absolute, resolved string path.

    Args:
        path: Any file path.

    Returns:
        Absolute path as a string.
    """
    return str(Path(path).resolve())


def file_size_bytes(path: str | Path) -> int:
    """Return the size of *path* in bytes.

    Args:
        path: Path to an existing file.

    Returns:
        File size in bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return os.path.getsize(p)
