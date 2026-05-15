"""Tests for BehaviorAnalyzer and statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from explore.pipeline.analysis import BehaviorAnalyzer

# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


def test_discrimination_index_novel_preference():
    di = BehaviorAnalyzer.discrimination_index(t_novel=30.0, t_familiar=10.0)
    assert pytest.approx(di, abs=1e-6) == 0.5


def test_discrimination_index_no_preference():
    di = BehaviorAnalyzer.discrimination_index(t_novel=20.0, t_familiar=20.0)
    assert pytest.approx(di, abs=1e-6) == 0.0


def test_discrimination_index_zero_total():
    di = BehaviorAnalyzer.discrimination_index(0.0, 0.0)
    assert di == 0.0


def test_recognition_index_novel_preference():
    ri = BehaviorAnalyzer.recognition_index(t_novel=30.0, t_familiar=10.0)
    assert pytest.approx(ri, abs=1e-6) == 0.75


def test_recognition_index_zero_total():
    ri = BehaviorAnalyzer.recognition_index(0.0, 0.0)
    assert ri == 0.5


# ---------------------------------------------------------------------------
# BehaviorAnalyzer.__init__
# ---------------------------------------------------------------------------


def test_invalid_fps_raises():
    with pytest.raises(ValueError, match="fps"):
        BehaviorAnalyzer(fps=0)


def test_invalid_bin_raises():
    with pytest.raises(ValueError, match="bin_duration"):
        BehaviorAnalyzer(fps=25, bin_duration_seconds=0)


# ---------------------------------------------------------------------------
# Bout filtering
# ---------------------------------------------------------------------------


def test_filter_bouts_removes_short_runs():
    analyzer = BehaviorAnalyzer(fps=10.0, min_bout_seconds=1.0)
    # 5 frames of exploration at 10 fps = 0.5 s → below min (1 s), should be removed
    arr = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32)
    filtered = analyzer._filter_bouts(arr)
    assert filtered.sum() == 0


def test_filter_bouts_keeps_long_runs():
    analyzer = BehaviorAnalyzer(fps=10.0, min_bout_seconds=1.0)
    # 15 frames = 1.5 s → above min, should be kept
    arr = np.array([1] * 15 + [0] * 5, dtype=np.int32)
    filtered = analyzer._filter_bouts(arr)
    assert filtered[:15].sum() == 15


def test_filter_bouts_zero_minimum_passes_everything():
    analyzer = BehaviorAnalyzer(fps=25.0, min_bout_seconds=0.0)
    arr = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    filtered = analyzer._filter_bouts(arr)
    np.testing.assert_array_equal(filtered, arr)


# ---------------------------------------------------------------------------
# Bout counting
# ---------------------------------------------------------------------------


def test_count_bouts_basic():
    analyzer = BehaviorAnalyzer(fps=25.0)
    arr = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int32)
    assert analyzer._count_bouts(arr) == 2


def test_count_bouts_empty():
    analyzer = BehaviorAnalyzer(fps=25.0)
    assert analyzer._count_bouts(np.array([], dtype=np.int32)) == 0


def test_count_bouts_all_zeros():
    analyzer = BehaviorAnalyzer(fps=25.0)
    assert analyzer._count_bouts(np.zeros(10, dtype=np.int32)) == 0


def test_count_bouts_all_ones():
    analyzer = BehaviorAnalyzer(fps=25.0)
    assert analyzer._count_bouts(np.ones(10, dtype=np.int32)) == 1


# ---------------------------------------------------------------------------
# BehaviorAnalyzer.compute
# ---------------------------------------------------------------------------


def test_compute_basic(tmp_path):
    fps = 25.0
    analyzer = BehaviorAnalyzer(
        fps=fps, bin_duration_seconds=60.0, min_bout_seconds=0.0
    )

    # Simulate 90 s of data: first 30 s exploring object1, rest not
    n_frames = int(90 * fps)
    obj1 = np.zeros(n_frames, dtype=np.int32)
    obj1[: int(30 * fps)] = 1
    obj2 = np.zeros(n_frames, dtype=np.int32)

    df = analyzer.compute({"obj1": obj1, "obj2": obj2}, animal_id="animal_1")

    assert "obj1_time_s" in df.columns
    assert "obj2_time_s" in df.columns
    # First bin (0-60s): all 30 s of exploration is in bin 1
    assert (
        pytest.approx(df.loc[df["minute"] == 1.0, "obj1_time_s"].values[0], abs=1)
        == 30.0
    )


def test_compute_empty_raises():
    analyzer = BehaviorAnalyzer(fps=25.0)
    with pytest.raises(ValueError, match="must not be empty"):
        analyzer.compute({}, animal_id="x")


def test_compute_adds_experiment_column():
    fps = 10.0
    analyzer = BehaviorAnalyzer(
        fps=fps, bin_duration_seconds=10.0, min_bout_seconds=0.0
    )
    obj = np.zeros(50, dtype=np.int32)
    df = analyzer.compute({"obj": obj}, animal_id="a1", experiment_id="exp01")
    assert (df["experiment"] == "exp01").all()


# ---------------------------------------------------------------------------
# DI / RI addition
# ---------------------------------------------------------------------------


def test_add_di_ri_columns():
    fps = 25.0
    analyzer = BehaviorAnalyzer(
        fps=fps, bin_duration_seconds=60.0, min_bout_seconds=0.0
    )
    n = int(120 * fps)
    novel = np.zeros(n, dtype=np.int32)
    novel[: int(30 * fps)] = 1
    familiar = np.zeros(n, dtype=np.int32)
    familiar[int(30 * fps) : int(40 * fps)] = 1

    df = analyzer.compute({"novel": novel, "familiar": familiar}, animal_id="a1")
    df = analyzer.add_di_ri(
        df, novel_col="novel_time_s", familiar_col="familiar_time_s"
    )

    assert "DI" in df.columns
    assert "RI" in df.columns
    assert df["RI"].between(0, 1).all()


def test_add_di_ri_missing_column_raises():
    fps = 25.0
    analyzer = BehaviorAnalyzer(fps=fps, min_bout_seconds=0.0)
    obj = np.zeros(25, dtype=np.int32)
    df = analyzer.compute({"obj": obj}, animal_id="a")
    with pytest.raises(KeyError):
        analyzer.add_di_ri(df, "MISSING_time_s", "obj_time_s")


# ---------------------------------------------------------------------------
# Aggregate and summary
# ---------------------------------------------------------------------------


def test_aggregate_multiple_animals():
    fps = 10.0
    analyzer = BehaviorAnalyzer(
        fps=fps, bin_duration_seconds=10.0, min_bout_seconds=0.0
    )
    obj = np.zeros(50, dtype=np.int32)

    df1 = analyzer.compute({"obj": obj}, animal_id="a1")
    df2 = analyzer.compute({"obj": obj}, animal_id="a2")
    combined = BehaviorAnalyzer.aggregate([df1, df2])

    assert len(combined) == len(df1) + len(df2)
    assert set(combined["animal"]) == {"a1", "a2"}


def test_aggregate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        BehaviorAnalyzer.aggregate([])
