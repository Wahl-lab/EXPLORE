"""Tests for utility modules (video IO, filesystem IO)."""

from __future__ import annotations

import pandas as pd
import pytest

from explore.utils.io import save_results
from explore.utils.video import VideoReader

# ---------------------------------------------------------------------------
# VideoReader
# ---------------------------------------------------------------------------


def test_video_reader_fps(fake_video_path):
    reader = VideoReader(fake_video_path)
    assert reader.fps > 0


def test_video_reader_frame_count(fake_video_path):
    reader = VideoReader(fake_video_path)
    assert reader.frame_count == 30


def test_video_reader_read_frame(fake_video_path):
    reader = VideoReader(fake_video_path)
    frame = reader.read_frame(0)
    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_video_reader_iter_frames_count(fake_video_path):
    reader = VideoReader(fake_video_path)
    frames = list(reader.iter_frames(start=0, end=10))
    assert len(frames) == 10
    idx, f = frames[0]
    assert idx == 0
    assert f.shape[2] == 3


def test_video_reader_iter_frames_with_step(fake_video_path):
    reader = VideoReader(fake_video_path)
    frames = list(reader.iter_frames(start=0, end=20, step=2))
    assert all(idx % 2 == 0 for idx, _ in frames)


def test_video_reader_context_manager(fake_video_path):
    with VideoReader(fake_video_path) as reader:
        frame = reader.read_frame(0)
    assert frame is not None


def test_video_reader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4").read_frame(0)


def test_video_reader_iter_with_crop(fake_video_path):
    reader = VideoReader(fake_video_path)
    crop = (10, 10, 100, 100)
    for _, frame in reader.iter_frames(start=0, end=3, crop=crop):
        assert frame.shape[:2] == (90, 90)


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


def test_save_results_creates_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = save_results(df, tmp_path / "out.csv")
    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == ["a", "b"]


def test_save_results_avoids_overwrite(tmp_path):
    df = pd.DataFrame({"x": [1]})
    path = tmp_path / "res.csv"
    out1 = save_results(df, path)
    out2 = save_results(df, path)
    assert out1 != out2
    assert out2.name == "res_1.csv"


def test_save_results_overwrite_disabled(tmp_path):
    df = pd.DataFrame({"x": [1]})
    path = tmp_path / "res.csv"
    save_results(df, path, avoid_overwrite=False)
    save_results(df, path, avoid_overwrite=False)
    # Both writes go to same path — no error, file exists
    assert path.exists()
