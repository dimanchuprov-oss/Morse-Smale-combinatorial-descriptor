import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from morse_lidar import multiface
from morse_lidar.multiface import (
    VERTICES,
    DownloadError,
    Segment,
    _NoRedirect,
    _check_name,
    _get,
    fetch,
    fetch_segment,
    load_people,
    parse_checksums,
    peak_frame,
    read_segment,
    resolve,
)

TOP = "m--20180227--0000--1234567--GHS/tracked_mesh"


class Response(io.BytesIO):
    """What urlopen returns, enough for the downloader: a body, headers and the final URL."""

    def __init__(self, data: bytes, length: int | None = None, url: str = multiface.ROOT + "/x") -> None:
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.url = url

    def geturl(self) -> str:
        return self.url


def _vertices(frame: int) -> np.ndarray:
    return np.full((VERTICES, 3), float(frame), dtype="<f4")


def _archive(segment: str, frames: list[int], extra: dict[str, bytes] | None = None) -> bytes:
    """A tracked_mesh tar like the real ones: <frame>.bin, <frame>_transform.txt, <frame>.obj per frame."""
    buffer = io.BytesIO()
    pose = b"1 0 0 1\n0 1 0 2\n0 0 1 3\n"
    obj = b"v 0 0 0\n" * 3 + b"vt 0 0\n" + b"f 1/1 2/1 3/1\nf 2/1 3/1 7306/1\n"
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        members = {}
        for frame in frames:
            members[f"{TOP}/{segment}/{frame:06d}.bin"] = _vertices(frame).tobytes()
            members[f"{TOP}/{segment}/{frame:06d}_transform.txt"] = pose
            members[f"{TOP}/{segment}/{frame:06d}.obj"] = obj
        members.update(extra or {})
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_names_from_the_server_are_checked():
    assert _check_name("tracked_mesh--E001_Neutral_Eyes_Open.tar") == "tracked_mesh--E001_Neutral_Eyes_Open.tar"
    for bad in ("../CHECKSUM", "a/b", "", "-rf", "x;rm", "name with space", ".hidden"):
        with pytest.raises(ValueError):
            _check_name(bad)
    with pytest.raises(ValueError):
        _get("https://example.com/identities/1/CHECKSUM")


def test_checksum_listing_skips_malformed_lines():
    text = "\n".join([
        "85c855fb1dc46d868d52838fe8af02db  tracked_mesh--E001_Neutral_Eyes_Open.tar",
        "nothex  tracked_mesh--E002.tar",
        "4e696323b3915c6039d21f83ffdcbdb0  ../escape.tar",
        "4e696323b3915c6039d21f83ffdcbdb0  metadata.tar  extra",
    ])  # fmt: skip
    assert parse_checksums(text) == {"tracked_mesh--E001_Neutral_Eyes_Open.tar": "85c855fb1dc46d868d52838fe8af02db"}


def test_resolve_takes_the_first_segment_the_person_has():
    sums = {"tracked_mesh--E074_Blink.tar": "0" * 32, "tracked_mesh--E001_Neutral_Eyes_Open.tar": "0" * 32}
    assert resolve(sums, ["neutral", "eyes_closed", "frown"]) == {
        "neutral": "E001_Neutral_Eyes_Open",
        "eyes_closed": "E074_Blink",
    }


def test_read_segment_takes_frames_and_ignores_other_members():
    data = _archive("E001_Neutral_Eyes_Open", [105, 102], extra={
        f"{TOP}/E002_Swallow/000102.bin": _vertices(9).tobytes(),
        "../000103.bin": _vertices(9).tobytes(),
        f"{TOP}/E001_Neutral_Eyes_Open/notes.txt": b"x",
    })  # fmt: skip
    segment, faces = read_segment(io.BytesIO(data), "E001_Neutral_Eyes_Open", want_faces=True)
    assert segment.frames.tolist() == [102, 105]
    assert segment.vertices.shape == (2, VERTICES, 3)
    assert segment.vertices[1, 0, 0] == 105.0
    assert np.allclose(segment.headpose[0], [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]])
    assert faces.tolist() == [[0, 1, 2], [1, 2, 7305]]


def test_read_segment_rejects_a_frame_of_the_wrong_size():
    data = _archive("E001_Neutral_Eyes_Open", [], extra={f"{TOP}/E001_Neutral_Eyes_Open/000001.bin": b"\0" * 12})
    with pytest.raises(ValueError, match="bytes"):
        read_segment(io.BytesIO(data), "E001_Neutral_Eyes_Open")


def test_fetch_segment_checks_the_md5(monkeypatch):
    data = _archive("E001_Neutral_Eyes_Open", [102])
    monkeypatch.setattr(multiface, "_get", lambda url, timeout=0: Response(data, len(data)))
    monkeypatch.setattr(multiface.time, "sleep", lambda seconds: None)
    segment, _ = fetch_segment("1234567", "E001_Neutral_Eyes_Open", hashlib.md5(data).hexdigest())
    assert segment.frames.tolist() == [102]
    with pytest.raises(DownloadError, match="MD5"):
        fetch_segment("1234567", "E001_Neutral_Eyes_Open", "0" * 32)


def test_a_connection_closed_early_is_retried(monkeypatch):
    data = _archive("E001_Neutral_Eyes_Open", [102, 105, 108, 111])
    # Cut at a member boundary: tarfile in stream mode sees a clean end there.
    cut = 3 * 512 + 2 * (512 + (VERTICES * 12 + 511) // 512 * 512)
    answers = [Response(data[:cut], len(data)), Response(data, len(data))]
    monkeypatch.setattr(multiface, "_get", lambda url, timeout=0: answers.pop(0))
    monkeypatch.setattr(multiface.time, "sleep", lambda seconds: None)
    segment, _ = fetch_segment("1234567", "E001_Neutral_Eyes_Open", hashlib.md5(data).hexdigest())
    assert segment.frames.tolist() == [102, 105, 108, 111]
    assert answers == []


def test_a_server_sending_more_than_announced_is_stopped(monkeypatch):
    data = _archive("E001_Neutral_Eyes_Open", [102, 105])
    monkeypatch.setattr(multiface, "_get", lambda url, timeout=0: Response(data, 1024))
    monkeypatch.setattr(multiface.time, "sleep", lambda seconds: None)
    with pytest.raises(DownloadError, match="more than 1024 bytes"):
        fetch_segment("1234567", "E001_Neutral_Eyes_Open", hashlib.md5(data).hexdigest())
    assert _NoRedirect().redirect_request(None, None, 302, "Found", {}, "http://elsewhere/") is None


def test_fetch_writes_segments_faces_and_a_manifest(tmp_path, monkeypatch):
    archives = {f"tracked_mesh--{name}.tar": _archive(name, [102, 105, 108])
                for name in ("E001_Neutral_Eyes_Open", "E019_Frown")}  # fmt: skip
    sums = {name: hashlib.md5(data).hexdigest() for name, data in archives.items()}
    monkeypatch.setattr(multiface, "fetch_checksums", lambda person: sums)
    calls = []

    def get(url, timeout=0):
        calls.append(url)
        data = archives[url.rsplit("/", 1)[1]]
        return Response(data, len(data))

    monkeypatch.setattr(multiface, "_get", get)
    plan = fetch(tmp_path, ["1234567"], ["neutral", "frown", "smile_open"], workers=2)
    assert plan == {"1234567": {"neutral": "E001_Neutral_Eyes_Open", "frown": "E019_Frown"}}
    manifest = json.loads((tmp_path / "1234567" / "manifest.json").read_text())
    assert manifest["frown"]["md5"] == sums["tracked_mesh--E019_Frown.tar"]
    assert np.load(tmp_path / "faces.npy").shape == (2, 3)
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / "old copy").mkdir()
    people = load_people(tmp_path)
    assert list(people) == ["1234567"]
    assert sorted(people["1234567"]) == ["frown", "neutral"]
    assert people["1234567"]["frown"].frames.tolist() == [102, 105, 108]
    fetch(tmp_path, ["1234567"], ["neutral", "frown"])
    assert len(calls) == 2  # existing segments are not downloaded again


def test_segment_round_trip_and_peak(tmp_path):
    vertices = np.zeros((4, VERTICES, 3), dtype=np.float32)
    vertices[2, :10] = 5.0
    segment = Segment("E019_Frown", np.array([1, 2, 3, 4]), vertices, np.zeros((4, 3, 4), dtype=np.float32))
    segment.save(tmp_path / "p" / "frown.npz")
    loaded = Segment.load(tmp_path / "p" / "frown.npz")
    assert loaded.name == "E019_Frown"
    assert np.array_equal(loaded.vertices, vertices)
    assert peak_frame(loaded, np.zeros((VERTICES, 3))) == 2
    weights = np.zeros(VERTICES)
    weights[100:] = 1.0
    vertices[3, 100:] = 1.0
    assert peak_frame(Segment("x", loaded.frames, vertices, loaded.headpose), np.zeros((VERTICES, 3)), weights) == 3
