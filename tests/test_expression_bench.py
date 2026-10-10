import numpy as np
import pytest

pytest.importorskip("scipy")

from morse_lidar.expression_bench import _largest_patch, pairs_for, summarise


def test_largest_patch_keeps_the_biggest_edge_connected_group():
    # Two strips of triangles that share no edge; the second one is longer.
    faces = np.array([[0, 1, 2], [1, 3, 2], [10, 11, 12], [11, 13, 12], [12, 13, 14], [13, 15, 14]])
    chosen = np.array([True, True, True, True, True, False])
    assert _largest_patch(chosen, faces).tolist() == [False, False, True, True, True, False]


def _row(method, person, label, other, score):
    return {"method": method, "probe": f"{person}/{label}", "probe_person": person, "probe_class": label,
            "gallery": f"{other}/neutral", "gallery_person": other, "genuine": person == other,
            "geometric_mm": score, "shared_cm2": 300.0, "overlap": 0.9, "error": None}  # fmt: skip


def test_summary_gives_rank1_and_the_cost_of_each_class():
    rows = [
        _row("b0", "A", "neutral", "A", 0.4), _row("b0", "A", "neutral", "B", 2.0),
        _row("b0", "B", "neutral", "B", 0.5), _row("b0", "B", "neutral", "A", 2.1),
        _row("b0", "A", "smile_open", "A", 1.6), _row("b0", "A", "smile_open", "B", 1.9),
        _row("b0", "B", "smile_open", "B", 2.4), _row("b0", "B", "smile_open", "A", 2.2),
        {**_row("b0", "B", "frown", "A", None), "error": "ValueError: no overlap"},
    ]  # fmt: skip

    class Labels:
        rigid = brows = hair = eye_covers = np.zeros(3, dtype=bool)

    summary = summarise(rows, Labels(), {"A": {}, "B": {}})
    b0 = summary["methods"]["b0"]
    assert list(b0["classes"]) == ["neutral", "frown", "smile_open"]
    assert b0["classes"]["neutral"]["rank1"] == 1.0
    assert b0["classes"]["smile_open"]["rank1"] == 0.5
    assert b0["classes"]["smile_open"]["cost_mm"] == pytest.approx(2.0 - 0.45)
    # B/frown has no scored pair of its own: it is a miss, not left out.
    assert b0["classes"]["frown"]["rank1"] == 0.0
    assert b0["expressive_rank1"] == pytest.approx(1 / 3)
    assert b0["failed_pairs"] == 1
    assert summarise(rows, Labels(), {}, failed={"b0": 7})["methods"]["b0"]["failed_pairs"] == 7


def test_a_failed_genuine_pair_is_a_miss_and_ties_are_not_hits():
    rows = [
        {**_row("b0", "A", "frown", "A", None), "error": "ValueError: x"}, _row("b0", "A", "frown", "B", 2.0),
        _row("b0", "B", "frown", "B", 1.5), _row("b0", "B", "frown", "A", 1.5),
    ]  # fmt: skip

    class Labels:
        rigid = brows = hair = eye_covers = np.zeros(3, dtype=bool)

    entry = summarise(rows, Labels(), {})["methods"]["b0"]["classes"]["frown"]
    assert entry["rank1"] == 0.0
    assert entry["failed_rows"] == 1


def test_b2_never_meets_a_template_of_the_probe_class():
    probes = {"A": ["neutral", "smile_open", "frown"], "B": ["smile_open"]}
    galleries = {"A": ["neutral", "smile_open", "mouth_open"], "B": ["neutral", "smile_open", "mouth_open"]}
    b2 = pairs_for("b2", probes, galleries)
    assert ("A", "smile_open", "B", "smile_open") not in b2
    assert ("A", "smile_open", "A", "mouth_open") in b2
    assert ("A", "neutral", "A", "neutral") in b2
    assert len([pair for pair in b2 if pair[:2] == ("A", "frown")]) == 6
    assert {pair[3] for pair in pairs_for("b0", probes, galleries)} == {"neutral"}


def test_failed_impostor_does_not_improve_rank1():
    rows = [_row("b3", "A", "frown", "A", 0.5), _row("b3", "A", "frown", "B", 2.0),
            {**_row("b3", "A", "frown", "C", None), "error": "no forehead"}]

    class Labels:
        rigid = brows = hair = eye_covers = np.zeros(3, dtype=bool)

    result = summarise(rows, Labels(), {"A": {}, "B": {}, "C": {}})["methods"]["b3"]
    assert result["expressive_rank1"] == 0.0
    assert result["eer_is_conditional_on_success"]
    assert result["scored_rows"] == 2 and result["total_rows"] == 3
