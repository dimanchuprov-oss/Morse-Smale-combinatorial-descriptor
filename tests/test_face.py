import json
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from morse_lidar.face import (
    HEAD_CENTRE,
    _head_axis,
    _morphology,
    enroll,
    equal_error_rate,
    estimate_normals,
    extract_head,
    match,
    radial_bottleneck,
    radial_map,
    register,
    rotation_z,
    simulate_view,
    visible_from,
)
from morse_lidar.face_cli import _person, main


def synthetic_head(depth=95.0, width=75.0, height=115.0, nose=20.0, chin=0.0, centre=(40.0, -30.0)) -> np.ndarray:
    """Ellipsoid head with a nose (and optional chin) bump facing +x, on a neck, in mm, Z up."""
    azimuth, polar = np.meshgrid(np.linspace(-np.pi, np.pi, 360, endpoint=False), np.linspace(0.02, np.pi - 0.3, 170))
    azimuth, polar = azimuth.ravel(), polar.ravel()
    radius = 1.0 / np.sqrt(
        (np.sin(polar) * np.cos(azimuth) / depth) ** 2
        + (np.sin(polar) * np.sin(azimuth) / width) ** 2
        + (np.cos(polar) / height) ** 2
    )
    radius += nose * np.exp(-(azimuth**2) / 0.03 - (polar - 1.75) ** 2 / 0.02)
    radius += chin * np.exp(-(azimuth**2) / 0.15 - (polar - 2.45) ** 2 / 0.02)
    head = np.column_stack(
        (radius * np.sin(polar) * np.cos(azimuth), radius * np.sin(polar) * np.sin(azimuth), radius * np.cos(polar))
    )
    angle, level = np.meshgrid(np.linspace(-np.pi, np.pi, 120, endpoint=False), np.linspace(-200.0, -95.0, 50))
    neck = np.column_stack((45 * np.cos(angle.ravel()), 45 * np.sin(angle.ravel()), level.ravel()))
    return np.vstack((head, neck)) + np.array([centre[0], centre[1], 1000.0])


PERSON_A = synthetic_head()
PERSON_B = synthetic_head(depth=102.0, width=80.0, height=108.0, nose=13.0, chin=10.0)


def test_head_axis_is_recovered_from_a_visible_arc():
    angle = np.linspace(-1.0, 1.0, 200)  # about 115 degrees of a 80 mm circle
    arc = np.column_stack((12.0 + 80.0 * np.cos(angle), -7.0 + 80.0 * np.sin(angle)))
    assert np.allclose(_head_axis(arc), [12.0, -7.0], atol=0.5)
    assert np.linalg.norm(np.median(arc, axis=0) - [12.0, -7.0]) > 60.0  # what the median would give


def test_extract_head_moves_axis_to_origin_and_top_to_zero():
    head = extract_head(PERSON_A)
    assert head.points[:, 2].max() == pytest.approx(0.0, abs=2.5)
    assert head.points[:, 2].min() >= -250.0
    assert np.allclose(head.offset[:2], (40.0, -30.0), atol=2.0)
    yz_up = PERSON_A[:, [1, 2, 0]]  # the same scan exported Y-up with the face along +z
    assert len(extract_head(yz_up, up_axis="y").points) == pytest.approx(len(head.points), rel=0.05)


def _with_body(head: np.ndarray) -> np.ndarray:
    angle, level = np.meshgrid(np.linspace(-np.pi, np.pi, 200, endpoint=False), np.linspace(-1500.0, -200.0, 260))
    body = np.column_stack((160 * np.cos(angle.ravel()), 110 * np.sin(angle.ravel()), level.ravel()))
    return np.vstack((head, body + np.array([40.0, -30.0, 1000.0])))


def test_extract_head_rejects_a_wrong_up_axis_on_a_body_scan():
    body = _with_body(PERSON_A)
    assert len(extract_head(body).points) > 1000
    for wrong in ("x", "y"):
        with pytest.raises(ValueError, match="up axis"):
            extract_head(body, up_axis=wrong)


def test_extract_head_accepts_a_face_only_scan():
    face = PERSON_A[PERSON_A[:, 2] > PERSON_A[:, 2].max() - 170.0]
    assert len(extract_head(face).points) > 1000


def test_visible_from_keeps_the_side_facing_the_camera():
    head = extract_head(PERSON_A).points
    seen = head[visible_from(head, np.array([1.0, 0.0, 0.0]))]
    assert 0.2 * len(head) < len(seen) < 0.65 * len(head)
    assert np.quantile(seen[:, 0], 0.05) > -20.0  # the back of the head is gone


def test_visible_from_does_not_leak_the_back_of_a_sparse_head():
    # Random thinning leaves empty image cells on the front surface; the back
    # of the head must not show through them.
    rng = np.random.default_rng(4)
    head = extract_head(PERSON_A).points
    sparse = head[rng.random(len(head)) < 0.35]
    front = np.array([1.0, 0.0, 0.0])
    depth_only = sparse[visible_from(sparse, front)]
    with_normals = sparse[visible_from(sparse, front, estimate_normals(sparse, HEAD_CENTRE))]
    assert np.mean(depth_only[:, 0] < -30.0) < 0.01
    assert np.mean(with_normals[:, 0] < -30.0) == 0.0


def test_visible_from_keeps_steep_but_visible_surfaces():
    # Dense sphere seen along +x: with normals, surfaces tilted up to ~80 degrees stay visible.
    azimuth, polar = np.meshgrid(np.linspace(-np.pi, np.pi, 900, endpoint=False), np.linspace(0.01, np.pi - 0.01, 450))
    normals = np.column_stack(
        (np.sin(polar.ravel()) * np.cos(azimuth.ravel()), np.sin(polar.ravel()) * np.sin(azimuth.ravel()),
         np.cos(polar.ravel()))
    )  # fmt: skip
    sphere = 100.0 * normals
    seen = visible_from(sphere, np.array([1.0, 0.0, 0.0]), normals)
    for low, high in ((0.2, 0.5), (0.5, 1.0)):
        band = (normals[:, 0] > low) & (normals[:, 0] <= high)
        assert seen[band].mean() > 0.9
    assert not seen[normals[:, 0] < 0.0].any()


def test_extract_head_drops_body_parts_that_do_not_touch_the_head():
    # A plate in front of the neck, 70-110 mm under the chin, as a chest seen below a raised chin.
    side, level = np.meshgrid(np.arange(-90.0, 30.0, 2.0), np.arange(890.0, 930.0, 2.0))
    plate = np.column_stack((np.full(side.size, 165.0), side.ravel(), level.ravel()))
    head = extract_head(np.vstack((PERSON_A, plate))).points
    assert not ((head[:, 0] > 115.0) & (head[:, 2] < -170.0)).any()
    assert len(head) == pytest.approx(len(extract_head(PERSON_A).points), rel=0.01)


def test_extract_head_reports_the_dropped_share():
    side, level = np.meshgrid(np.arange(-90.0, 30.0, 2.0), np.arange(890.0, 930.0, 2.0))
    plate = np.column_stack((np.full(side.size, 165.0), side.ravel(), level.ravel()))
    assert extract_head(PERSON_A).dropped < 0.01
    assert 0.02 < extract_head(np.vstack((PERSON_A, plate))).dropped < 0.2


def test_shared_area_does_not_depend_on_the_sampling_density():
    # The same 80 mm hemisphere (402 cm^2) sampled at 1.5 and at 3.5 mm spacing.
    rng = np.random.default_rng(0)
    areas = []
    for spacing in (1.5, 3.5):
        directions = rng.normal(size=(int(2 * np.pi * 80**2 / spacing**2), 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        directions[:, 0] = np.abs(directions[:, 0])
        areas.append(enroll(80.0 * directions).areas.sum() / 100.0)
    assert areas == pytest.approx([402.0, 402.0], rel=0.08)


@pytest.mark.parametrize("yaw", [0.0, 60.0, -90.0, 150.0])
def test_registration_recovers_a_scrambled_partial_view(yaw):
    gallery = enroll(extract_head(PERSON_A[0::2]))
    probe = extract_head(simulate_view(PERSON_A[1::2], math.radians(yaw), seed=3))
    registration = register(probe.points, gallery)
    distance, _ = gallery.tree.query(registration.apply(probe.points))
    assert registration.inliers > 0.95 and np.median(distance) < 1.5  # 2 mm voxels: ~1 mm to the nearest point


def test_two_partial_views_with_a_small_overlap_are_registered_without_bias():
    # Front halves of views from 40 degrees left and right, like two frontal
    # snapshots of a turned head: they share about 55% of their points, less than
    # the 80% a fixed trimming ratio kept, which pulled the pose towards the parts
    # only one view has. (Whole synthetic heads are nearly symmetric under a turn
    # by 180 degrees, so the test keeps the face side as real snapshots do.)
    views = [extract_head(simulate_view(PERSON_A, math.radians(yaw), seed=7, scramble=False)) for yaw in (-40.0, 40.0)]
    probe, gallery = (view.points[view.points[:, 0] > 0.0] for view in views)
    result = match(probe, gallery, topology=False)
    truth = probe + np.subtract(views[0].offset, views[1].offset)
    assert np.median(np.linalg.norm(result.registration.apply(probe) - truth, axis=1)) < 1.5
    assert result.geometric < 0.6 and result.shared_cm2 > 50.0


def test_match_scores_same_person_below_other_person():
    galleries = {"A": enroll(extract_head(PERSON_A[0::2])), "B": enroll(extract_head(PERSON_B[0::2]))}
    for person, cloud in (("A", PERSON_A), ("B", PERSON_B)):
        probe = extract_head(simulate_view(cloud[1::2], math.radians(45.0), seed=5))
        scores = {label: match(probe.points, head, topology=False).geometric for label, head in galleries.items()}
        other = "B" if person == "A" else "A"
        assert scores[person] < 0.5 and scores[other] > 3 * scores[person]


def test_radial_map_ignores_points_outside_the_head_band():
    points = np.array([[80.0, 0.0, -10.0], [80.0, 0.0, 5.0], [80.0, 0.0, -400.0]])
    field = radial_map(points)
    assert np.isfinite(field).sum() == 1


def test_morphology_keeps_the_border_of_a_full_mask():
    assert _morphology(np.ones((6, 8), dtype=bool)).all()


def test_radial_bottleneck_is_zero_for_identical_heads_and_positive_for_different():
    pytest.importorskip("gudhi")
    head = extract_head(PERSON_A).points
    assert radial_bottleneck(head, head)[0] == pytest.approx(0.0, abs=1e-9)
    other = extract_head(PERSON_B).points
    assert radial_bottleneck(other, head)[0] > 1.0


def test_radial_bottleneck_does_not_depend_on_where_the_face_points():
    pytest.importorskip("gudhi")
    gallery = extract_head(PERSON_A).points
    probe = extract_head(simulate_view(PERSON_B, 0.0, seed=2, scramble=False)).points
    turn = rotation_z(np.pi).T
    first = radial_bottleneck(probe, gallery)
    turned = radial_bottleneck(probe @ turn, gallery @ turn)
    assert turned[0] == pytest.approx(first[0], rel=0.15)
    assert turned[1] == pytest.approx(first[1], rel=0.05)


def test_equal_error_rate_takes_the_middle_of_a_clean_gap():
    rate, threshold = equal_error_rate([1.0, 1.5, 2.0], [4.0, 5.0])
    assert rate == 0.0 and threshold == pytest.approx(3.0)
    rate, _ = equal_error_rate([1.0, 2.0, 6.0, 7.0], [3.0, 4.0, 5.0, 8.0])
    assert 0.0 < rate < 1.0


def _write_xyz(path, points):
    np.savetxt(path, points, fmt="%.3f")
    return path


def test_cli_experiment_then_calibrated_compare(tmp_path, capsys):
    a = _write_xyz(tmp_path / "a.xyz", PERSON_A)
    b = _write_xyz(tmp_path / "b.xyz", PERSON_B)
    probe = _write_xyz(tmp_path / "probe.xyz", simulate_view(PERSON_B, 0.7, seed=1, scramble=False))
    out = tmp_path / "faces"
    main(["experiment", "--person", f"A={a}", "--person", f"B={b}", "--probe", str(probe), "--up-axis", "z",
          "--yaws=-60,0,60", "--pitches", "0", "--no-topology", "--no-plots", "--out", str(out)])  # fmt: skip
    summary = json.loads((out / "face_summary.json").read_text(encoding="utf-8"))
    assert summary["genuine_pairs"] == 6 and summary["impostor_pairs"] == 6 and summary["failed_pairs"] == 0
    assert summary["geometric_mm"]["eer"] == 0.0
    assert summary["settings"] == {"up_axis": "z", "units": "mm", "method": 2}
    assert summary["real_probes"]["probe.xyz"]["best_match"] == "B"
    assert (out / "face_scores.csv").read_text(encoding="utf-8").count("\n") == 1 + 12 + 2
    capsys.readouterr()
    calibration = str(out / "face_summary.json")
    main(["compare", str(probe), "--gallery", f"A={a}", "--gallery", f"B={b}", "--up-axis", "z", "--no-topology",
          "--calibration", calibration])  # fmt: skip
    result = json.loads(capsys.readouterr().out)
    assert result["best_match"] == "B" and result["decision"] == "same"
    main(["compare", str(probe), "--gallery", f"A={a}", "--up-axis", "z", "--no-topology"])
    assert json.loads(capsys.readouterr().out)["decision"] is None
    main(["compare", str(probe), "--gallery", f"A={a}", "--gallery", f"B={b}", "--up-axis", "z", "--no-topology",
          "--calibration", calibration, "--min-area", "100000"])  # fmt: skip
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "insufficient overlap" and result["best_match"] is None
    with pytest.raises(SystemExit, match="calibration was made"):
        main(["compare", str(probe), "--gallery", f"A={a}", "--up-axis", "z", "--units", "cm", "--no-topology",
              "--calibration", calibration])  # fmt: skip
    old = json.loads((out / "face_summary.json").read_text(encoding="utf-8"))
    old["settings"].pop("method")  # a summary of the previous method version
    (out / "old.json").write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="older version"):
        main(["compare", str(probe), "--gallery", f"A={a}", "--up-axis", "z", "--no-topology",
              "--calibration", str(out / "old.json")])  # fmt: skip


def test_cli_experiment_survives_a_bad_real_probe(tmp_path):
    a = _write_xyz(tmp_path / "a.xyz", PERSON_A)
    b = _write_xyz(tmp_path / "b.xyz", PERSON_B)
    bad = _write_xyz(tmp_path / "bad.xyz", PERSON_A[:40])
    out = tmp_path / "faces"
    main(["experiment", "--person", f"A={a}", "--person", f"B={b}", "--probe", str(bad), "--up-axis", "z",
          "--yaws=0", "--pitches", "0", "--no-topology", "--no-plots", "--out", str(out)])  # fmt: skip
    summary = json.loads((out / "face_summary.json").read_text(encoding="utf-8"))
    assert summary["real_probes"]["bad.xyz"]["decision"] is None
    assert set(summary["real_probes"]["bad.xyz"]["failed"]) == {"A", "B"}
    assert summary["failed_pairs"] == 2


def test_person_label_is_the_file_name_without_the_scan_number():
    from pathlib import Path

    assert _person(Path("Ваня1.ply")) == "Ваня"
    assert _person(Path("scan_12.xyz")) == "scan"
    assert _person(Path("007.ply")) == "007"


def test_cli_matrix_compares_every_pair_of_named_scans(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    paths = []
    for name, cloud, yaw in (("A1", PERSON_A, -30.0), ("A2", PERSON_A, 30.0), ("B1", PERSON_B, -30.0),
                             ("B2", PERSON_B, 30.0)):  # fmt: skip
        view = simulate_view(cloud, math.radians(yaw), seed=len(paths))
        paths.append(str(_write_xyz(tmp_path / f"{name}.xyz", view)))
    out = tmp_path / "matrix"
    main(["matrix", *paths, "--up-axis", "z", "--no-topology", "--jobs", "1", "--min-area", "10", "--out", str(out)])
    summary = json.loads((out / "face_matrix_summary.json").read_text(encoding="utf-8"))
    assert summary["people"] == {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    assert summary["pairs"] == {"genuine": 2, "impostor": 4, "failed": 0, "reliable": 6}
    assert summary["identification"]["rank1_correct"] == 4
    assert summary["reliable_pairs"]["geometric_mm"]["eer"] == 0.0
    decisions = summary["decisions"]
    assert decisions["same_person_same"] == 2 and decisions["different_people_different"] == 4
    assert summary["identification"]["with_area_limit"]["identified"] == 4
    assert all(scan["dropped_fraction"] is not None for scan in summary["scans"].values())
    assert (out / "face_pairs.csv").read_text(encoding="utf-8").count("\n") == 1 + 6
    for name in ("face_matrix.png", "face_score_vs_area.png", "face_nearest.png", "face_examples.png"):
        assert (out / name).stat().st_size > 10_000
    capsys.readouterr()
    calibration = str(out / "face_matrix_summary.json")
    main(["compare", paths[1], "--gallery", f"A={paths[0]}", "--gallery", f"B={paths[2]}", "--up-axis", "z",
          "--no-topology", "--calibration", calibration])  # fmt: skip
    result = json.loads(capsys.readouterr().out)
    assert result["best_match"] == "A" and result["decision"] == "same" and result["min_shared_cm2"] == 10.0


def test_cli_matrix_rejects_duplicate_scan_names(tmp_path):
    (tmp_path / "x").mkdir()
    first = _write_xyz(tmp_path / "A1.xyz", PERSON_A[:500])
    second = _write_xyz(tmp_path / "x" / "A1.xyz", PERSON_A[:500])
    with pytest.raises(SystemExit, match="unique"):
        main(["matrix", str(first), str(second), "--up-axis", "z", "--out", str(tmp_path / "out")])


def test_cli_plots_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("gudhi")
    a = _write_xyz(tmp_path / "a.xyz", PERSON_A)
    b = _write_xyz(tmp_path / "b.xyz", PERSON_B)
    out = tmp_path / "faces"
    main(["experiment", "--person", f"A={a}", "--person", f"B={b}", "--up-axis", "z", "--yaws=0,60", "--pitches",
          "0", "--example-yaw", "50", "--out", str(out)])  # fmt: skip
    for name in ("face_scores_vs_angle.png", "face_distributions.png", "face_registration.png"):
        assert (out / name).stat().st_size > 10_000


@pytest.mark.parametrize(
    "extra",
    [
        ["--person", "only-a-path.xyz", "--person", "B=b.xyz"],
        ["--person", "A=a.xyz"],
        ["--person", "A=a.xyz", "--person", "B=b.xyz", "--yaws="],
        ["--person", "A=a.xyz", "--person", "B=b.xyz", "--yaws=60:0:15"],
        ["--person", "A=a.xyz", "--person", "B=b.xyz", "--yaws=0:60"],
        ["--person", "A=a.xyz", "--person", "B=b.xyz", "--front-axis", "+z"],
        ["--person", "A=a.xyz", "--person", "B=missing.xyz"],
    ],
)
def test_cli_rejects_bad_arguments_with_a_message(tmp_path, extra):
    _write_xyz(tmp_path / "a.xyz", PERSON_A)
    _write_xyz(tmp_path / "b.xyz", PERSON_B)
    extra = [item.replace("=a.xyz", f"={tmp_path / 'a.xyz'}").replace("=b.xyz", f"={tmp_path / 'b.xyz'}")
             for item in extra]  # fmt: skip
    with pytest.raises(SystemExit) as error:
        main(["experiment", "--up-axis", "z", "--out", str(tmp_path / "out"), *extra])
    assert error.value.code not in (0, None)
