import json
import struct
from pathlib import Path

import numpy as np
import pytest

from morse_lidar.pointcloud import load_point_cloud
from morse_lidar.watch import Watcher, main


def _surface(seed, bump=0.0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-60, 60, (2, 4000))  # millimetres, like Revo Scan exports
    z = 25 * np.exp(-(x**2 + y**2) / 800) + bump * np.exp(-((x - 30) ** 2 + y**2) / 80) + 0.2 * rng.normal(size=x.size)
    return np.column_stack((x, y, z))


def _write_binary_ply(path, points):
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    path.write_bytes(header.encode() + points.astype("<f4").tobytes())


OPTIONS = ["--geometry", "delaunay", "--voxel-size", "2", "--persistence-threshold", "2"]


def test_obj_and_stl_loaders(tmp_path):
    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [0, 1, 1]]], dtype=float)
    obj = tmp_path / "m.obj"
    obj.write_text("# mesh\nv 0 0 0\nv 1 0 0\nv 0 1 0 0.5 0.5 0.5\nvn 0 0 1\nf 1 2 3\n")
    assert load_point_cloud(obj).points.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    binary = tmp_path / "b.stl"
    body = b"".join(struct.pack("<3f9fH", 0, 0, 1, *tri.ravel(), 0) for tri in triangles)
    binary.write_bytes(b"\0" * 80 + struct.pack("<I", 2) + body)
    assert len(load_point_cloud(binary).points) == 5
    ascii_stl = tmp_path / "a.stl"
    ascii_stl.write_text(
        "solid t\n" + "".join("facet normal 0 0 1\nouter loop\n" + "".join(f"vertex {a} {b} {c}\n" for a, b, c in tri) + "endloop\nendfacet\n" for tri in triangles) + "endsolid t\n"
    )
    assert len(load_point_cloud(ascii_stl).points) == 5


def test_watcher_processes_new_scans_logs_and_compares(tmp_path, capsys):
    export = tmp_path / "export"
    export.mkdir()
    _write_binary_ply(export / "vase_01.ply", _surface(1))
    _write_binary_ply(export / "vase_02.ply", _surface(2))
    (export / "notes.txt.tmp").write_text("ignored")
    main([str(export), "--once", "--", *OPTIONS])
    out = capsys.readouterr().out
    assert "vase_01.ply" in out and "это эталон" in out and "до эталона" in out
    log = (export / "morse-lidar" / "scans.csv").read_text().splitlines()
    assert len(log) == 3 and all(",ok," in line for line in log[1:])
    manifest = json.loads((export / "morse-lidar" / "manifest.draft.json").read_text())
    assert manifest["units"] == "mm" and [scan["object"] for scan in manifest["scans"]] == ["vase", "vase"]
    assert manifest["parameters"]["persistence_threshold"] == 2.0

    # A second pass only picks up the new file; a changed object is further away than a repeat.
    _write_binary_ply(export / "vase_changed.ply", _surface(3, bump=15))
    watcher = Watcher(export, export / "morse-lidar", OPTIONS, reference=export / "morse-lidar" / "vase_01.ply.json", settle=0)
    rows = watcher.run_once()
    assert [row["file"] for row in rows] == ["vase_changed.ply"]
    first_two = [line.split(",") for line in log[1:]]
    repeat_distance = float(first_two[1][9])
    assert rows[0]["bottleneck_vs_reference"] > repeat_distance


def test_watcher_survives_a_broken_file(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "broken.ply").write_text("not a ply")
    _write_binary_ply(export / "good_01.ply", _surface(4))
    rows = Watcher(export, tmp_path / "out", OPTIONS, reference=None, settle=0).run_once()
    status = {row["file"]: row["status"] for row in rows}
    assert status == {"broken.ply": "error", "good_01.ply": "ok"}


def test_watcher_rejects_input_output_options(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--once", "--", "--input", "x"])


def test_restart_keeps_reference_and_full_manifest(tmp_path, capsys):
    export = tmp_path / "export"
    export.mkdir()
    _write_binary_ply(export / "vase_01.ply", _surface(1))
    _write_binary_ply(export / "vase_02.ply", _surface(2))
    main([str(export), "--once", "--", *OPTIONS])
    _write_binary_ply(export / "vase_03.ply", _surface(5))
    main([str(export), "--once", "--", *OPTIONS])
    out = capsys.readouterr().out
    assert "Эталон: vase_01.ply" in out
    last = out.strip().splitlines()
    assert any("vase_03.ply" in line and "до эталона (vase_01.ply)" in line for line in last)
    manifest = json.loads((export / "morse-lidar" / "manifest.draft.json").read_text())
    assert [Path(scan["path"]).name for scan in manifest["scans"]] == ["vase_01.ply", "vase_02.ply", "vase_03.ply"]


def test_reexported_file_is_listed_once_and_same_stem_formats_do_not_collide(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    points = _surface(1)
    _write_binary_ply(export / "cup_01.ply", points)
    np.savetxt(export / "cup_01.xyz", _surface(2))
    watcher = Watcher(export, tmp_path / "out", OPTIONS, reference=None, settle=0)
    watcher.run_once()
    assert (tmp_path / "out" / "cup_01.ply.json").exists() and (tmp_path / "out" / "cup_01.xyz.json").exists()
    import os, time as _time
    os.utime(export / "cup_01.ply", ns=(_time.time_ns(), _time.time_ns() - 10**10))
    assert [row["file"] for row in watcher.run_once()] == ["cup_01.ply"]
    from morse_lidar.watch import write_manifest_draft
    manifest = json.loads(write_manifest_draft(watcher, "mm").read_text())
    assert sorted(Path(scan["path"]).name for scan in manifest["scans"]) == ["cup_01.ply", "cup_01.xyz"]


def test_reference_with_other_options_and_output_inside_export_are_rejected(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    _write_binary_ply(export / "vase_01.ply", _surface(1))
    main([str(export), "--once", "--", *OPTIONS])
    reference = export / "morse-lidar" / "vase_01.ply.json"
    with pytest.raises(SystemExit):
        Watcher(export, tmp_path / "other", ["--geometry", "delaunay", "--voxel-size", "3", "--persistence-threshold", "2"], reference, 0)
    with pytest.raises(SystemExit):
        Watcher(export, export, OPTIONS, None, 0)


def test_vanishing_files_do_not_stop_the_watcher(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "gone.ply").symlink_to(tmp_path / "missing.ply")
    _write_binary_ply(export / "vase_01.ply", _surface(1))
    rows = Watcher(export, tmp_path / "out", OPTIONS, reference=None, settle=0).run_once()
    assert [row["file"] for row in rows] == ["vase_01.ply"]
