import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from morse_lidar.face import rotation_z
from morse_lidar.face_cli import _calibration, _matrix_summary, _settings
from morse_lidar.face_regions import locate_regions


def frontal_face(nose=True):
    """Analytic convex skin with a known nasal bump, Z up, front +X (mm)."""
    x, y = np.meshgrid(np.arange(-68.0, 69.0, 2.0), np.arange(-90.0, 111.0, 2.0))
    depth = 80.0 - 0.004 * x**2 - 0.0008 * y**2
    if nose:
        depth += 23.0 * np.exp(-(x / 10.0)**2 - (y / 13.0)**2)
        depth += 5.0 * np.exp(-(x / 7.0)**2 - ((y - 25.0) / 25.0)**2)
    return np.column_stack((depth.ravel(), x.ravel(), y.ravel()))


def test_finds_nose_and_forehead_on_known_geometry():
    points = frontal_face()
    region = locate_regions(points)
    assert np.linalg.norm(region.nose_tip - np.array([104.8, 0.0, 0.0])) < 5.0
    assert region.mask.dtype == bool and region.mask.shape == (len(points),)
    assert np.all(points[region.forehead, 2] > 50.0)
    assert not region.mask[points[:, 2] < -15.0].any()
    assert region.nose.sum() >= 50 and region.forehead.sum() >= 50


def test_landmark_is_equivariant_to_yaw_and_translation():
    points = frontal_face()
    original = locate_regions(points)
    rotation = rotation_z(np.radians(123.0))
    shift = np.array([310.0, -120.0, 600.0])
    moved = locate_regions(points @ rotation.T + shift)
    assert np.allclose(moved.nose_tip, original.nose_tip @ rotation.T + shift, atol=1e-6)
    # Points exactly on a metric window edge can differ by roundoff.
    assert (moved.mask & original.mask).sum() / (moved.mask | original.mask).sum() > 0.99


def test_noise_and_an_isolated_depth_spike_do_not_replace_the_nose():
    points = frontal_face()
    points += np.random.default_rng(41).normal(0.0, 0.15, points.shape)
    spike = np.argmin(np.linalg.norm(points[:, 1:] - [12.0, 55.0], axis=1))
    points[spike, 0] += 70.0
    region = locate_regions(points)
    assert np.linalg.norm(region.nose_tip[1:]) < 6.0


def test_smooth_skin_without_a_nose_is_rejected():
    with pytest.raises(ValueError, match="nose prominence"):
        locate_regions(frontal_face(nose=False))


def test_missing_forehead_is_an_explicit_failure():
    points = frontal_face()
    with pytest.raises(ValueError, match="forehead"):
        locate_regions(points[points[:, 2] < 45.0])


@pytest.mark.parametrize("points", [np.zeros((10, 3)), np.zeros((200, 2)), np.full((200, 3), np.nan),
                                    frontal_face() / 1000.0])
def test_invalid_cloud_or_units_are_rejected(points):
    with pytest.raises(ValueError):
        locate_regions(points)


def test_multiface_vertex_labels_cannot_change_auto_selection(tmp_path, monkeypatch):
    from morse_lidar import expression_bench as bench
    from morse_lidar.virtual_scan import Scan

    points = frontal_face()
    monkeypatch.setattr(bench, "extract_head", lambda *a, **k: SimpleNamespace(points=points, offset=(0, 0, 0)))
    labels = SimpleNamespace(rigid=np.array([True, False]), nose_tip=0)
    selections = []
    reports = []
    for vertex_label in (0, 1):
        scan = Scan(points[:, [1, 2, 0]], np.full(len(points), vertex_label))
        shots = {"A": bench.Shots("A", {"neutral": scan}, {})}
        heads, report = bench._automatic_heads(shots, labels, tmp_path, False)
        selections.append(heads[("A", "gallery", "neutral")][0])
        reports.append(report["scans"]["A/gallery/neutral"])
    assert np.array_equal(*selections)
    assert reports[0]["oracle_precision"] == 1.0
    assert reports[1]["oracle_precision"] == 0.0


def test_full_face_calibration_cannot_be_reused_for_auto(tmp_path):
    args = argparse.Namespace(up_axis="y", units="mm", region="full")
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"settings": _settings(args), "geometric_mm": {}}))
    args.region, args.calibration = "auto", str(path)
    with pytest.raises(SystemExit, match="calibration was made"):
        _calibration(args)


def test_failed_scans_remain_in_matrix_coverage():
    args = argparse.Namespace(up_axis="y", units="mm", region="auto", min_area=0.0)
    scans = {name: {"error": "no forehead" if name == "A2" else None} for name in ("A1", "A2", "B1")}
    rows = [{"scan_a": a, "scan_b": b, "genuine": a[0] == b[0], "error": "no forehead",
             "reliable": False} for a, b in (("A1", "A2"), ("A1", "B1"), ("A2", "B1"))]
    summary = _matrix_summary(rows, scans, list(scans), {n: n[0] for n in scans}, args)
    assert summary["coverage"]["requested_pairs"] == 3
    assert summary["coverage"]["eer_is_conditional_on_success"]
    assert summary["identification"]["requested_scans"] == 3
    assert summary["identification"]["rank1_complete_correct"] == 0


def test_matrix_does_not_drop_pairs_of_a_failed_detector(tmp_path, monkeypatch):
    from morse_lidar import face_cli as cli

    points = frontal_face()
    region = locate_regions(points)
    monkeypatch.setattr(cli, "_upright", lambda path, args: path.stem)
    monkeypatch.setattr(cli, "extract_head", lambda name, **kw:
                        SimpleNamespace(points=points if name != "A2" else points[:100], dropped=0.0))

    def locate(cloud):
        if len(cloud) == 100:
            raise ValueError("missing forehead")
        return region

    monkeypatch.setattr(cli, "locate_regions", locate)
    monkeypatch.setattr(cli, "_match_pair", lambda task:
                        {"geometric_mm": 2.0, "shared_cm2": 30.0, "error": None})
    args = cli._parser().parse_args(["matrix", "A1.ply", "A2.ply", "B1.ply", "--up-axis", "z",
                                     "--region", "auto", "--no-plots", "--no-topology", "--jobs", "1",
                                     "--out", str(tmp_path)])
    summary = cli.matrix(args)
    assert summary["pairs"]["failed"] == 2
    assert summary["coverage"]["requested_pairs"] == 3
    assert summary["coverage"]["scored_pairs"] == 1
    assert "A2" in summary["scans"]
    assert (tmp_path / "face_pairs.csv").read_text().count("\n") == 4


@pytest.mark.parametrize("plot_fails", [False, True])
def test_region_inspection_never_matches_pairs_and_keeps_report(tmp_path, monkeypatch, plot_fails):
    from morse_lidar import face_cli as cli

    points = frontal_face()
    monkeypatch.setattr(cli, "_upright", lambda path, args: points if path.stem == "A1" else points[:10])
    monkeypatch.setattr(cli, "extract_head", lambda cloud, **kw:
                        SimpleNamespace(points=cloud, dropped=0.0, offset=(0, 0, 0)))

    def forbidden_match(*args, **kwargs):
        raise AssertionError("regions-only must not compare scans")

    def broken_plot(*args):
        raise RuntimeError("plot unavailable")

    monkeypatch.setattr(cli, "match", forbidden_match)
    monkeypatch.setattr(cli, "_match_pair", forbidden_match)
    monkeypatch.setattr(cli, "plot_regions", broken_plot)
    options = ["regions", "A1.ply", "B1.ply", "--up-axis", "z", "--out", str(tmp_path)]
    if not plot_fails:
        options.append("--no-plots")
    args = cli._parser().parse_args(options)
    if plot_fails:
        with pytest.raises(RuntimeError, match="plot unavailable"):
            cli.inspect_regions(args)
    else:
        cli.inspect_regions(args)
    report = json.loads((tmp_path / "auto_regions.json").read_text())
    assert report["mode"] == "regions_only" and report["total"] == 2
    assert report["detected"] == 1 and report["failed"] == 1
    assert report["scans"]["B1"]["error"] is not None
    with np.load(tmp_path / "auto_regions.npz") as masks:
        assert "A1/nose" in masks and "B1/points" in masks
