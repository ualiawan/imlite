"""Shared pytest fixtures for the imlite test suite.

All fixtures that need real files on disk use ``tmp_path`` (pytest built-in)
so every test gets an isolated temporary directory that is cleaned up after the
session ends.

Slow fixtures (video creation) are session-scoped so they are only built once.
"""

import numpy as np
import pytest

import imlite

# ---------------------------------------------------------------------------
# Suppress progress bars globally - keeps CI output clean
# ---------------------------------------------------------------------------

imlite.set_progress(False)


# ---------------------------------------------------------------------------
# Raw array fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rgb_array() -> np.ndarray:
    """256x256 uint8 RGB gradient array (H, W, 3)."""
    h, w = 256, 256
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.arange(w, dtype=np.uint8), (h, 1))  # R ramps left->right
    arr[:, :, 1] = np.tile(
        np.arange(h, dtype=np.uint8).reshape(-1, 1), (1, w)
    )  # G ramps top->bottom
    arr[:, :, 2] = 128
    return arr


@pytest.fixture()
def bgr_array(rgb_array: np.ndarray) -> np.ndarray:
    """Same gradient but in BGR channel order (as OpenCV would store it)."""
    return rgb_array[:, :, ::-1].copy()


@pytest.fixture()
def gray_array() -> np.ndarray:
    """128x128 uint8 grayscale array (H, W, 1) - single channel."""
    arr = np.arange(128 * 128, dtype=np.uint8).reshape(128, 128, 1)
    return arr


# ---------------------------------------------------------------------------
# Image object fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bgr_image(bgr_array: np.ndarray) -> imlite.Image:
    """``Image`` wrapping the 256x256 BGR gradient array."""
    return imlite.Image.from_numpy(bgr_array, color_space="BGR")


# ---------------------------------------------------------------------------
# On-disk image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_png(tmp_path_factory: pytest.TempPathFactory) -> str:
    """256x256 RGB PNG saved to a session-scoped temp directory."""
    h, w = 256, 256
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    arr[:, :, 1] = np.tile(np.arange(h, dtype=np.uint8).reshape(-1, 1), (1, w))
    arr[:, :, 2] = 128
    # imageio expects RGB
    import imageio.v3 as iio

    path = tmp_path_factory.mktemp("imgs") / "sample.png"
    iio.imwrite(str(path), arr)
    return str(path)


@pytest.fixture(scope="session")
def sample_jpg(tmp_path_factory: pytest.TempPathFactory) -> str:
    """256x256 RGB JPEG saved to a session-scoped temp directory."""
    h, w = 256, 256
    arr = np.full((h, w, 3), 200, dtype=np.uint8)
    import imageio.v3 as iio

    path = tmp_path_factory.mktemp("imgs") / "sample.jpg"
    iio.imwrite(str(path), arr, quality=90)
    return str(path)


# ---------------------------------------------------------------------------
# On-disk video fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> str:
    """30-frame synthetic MP4 (64x64, 10 fps) - created with imageio-ffmpeg."""
    import imageio.v2 as iio

    path = tmp_path_factory.mktemp("videos") / "sample.mp4"
    fps = 10
    n_frames = 30
    writer = iio.get_writer(str(path), fps=fps)
    for i in range(n_frames):
        # Solid-colour frames: colour cycles through 30 shades
        frame = np.full((64, 64, 3), (i * 8 % 256, 100, 200), dtype=np.uint8)
        writer.append_data(frame)
    writer.close()
    return str(path)


# ---------------------------------------------------------------------------
# On-disk frames directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_frames_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Directory with 10 PNG frames named frame_0001.png ... frame_0010.png."""
    import imageio.v3 as iio

    d = tmp_path_factory.mktemp("frames")
    for i in range(1, 11):
        frame = np.full((64, 64, 3), (i * 25 % 256, 128, 64), dtype=np.uint8)
        iio.imwrite(str(d / f"frame_{i:04d}.png"), frame)
    return str(d)
