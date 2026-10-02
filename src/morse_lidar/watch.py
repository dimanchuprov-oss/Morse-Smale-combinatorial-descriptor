"""Process scans as they are exported from the scanner software.

Point Revo Scan (or any scanner app) at an export folder and run::

    morse-lidar-watch EXPORT_DIR -- --geometry delaunay --voxel-size 1 --persistence-threshold 0.5

Every new PLY/OBJ/STL/XYZ file is described with the given ``morse-lidar``
options, compared with the reference scan and with the previous scan, and
logged to ``scans.csv``. Scanner drivers are proprietary, so points are
captured by exporting from the vendor application rather than read from USB.
The session (processed files, reference, previous scan) is stored in the
results folder, so the watcher can be stopped and restarted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import signal
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

from .cli import _parser, _validate
from .cli import main as describe
from .compare import compare_reports

SCAN_SUFFIXES = {".ply", ".obj", ".stl", ".xyz", ".txt", ".csv", ".pcd", ".las", ".laz"}
LOG_FIELDS = [
    "time", "file", "sha256", "status", "point_count", "vertex_count", "persistent_min", "persistent_saddle",
    "persistent_max", "bottleneck_vs_reference", "bottleneck_vs_previous", "descriptor", "error",
]
EXCLUDED_PARAMETERS = {"input", "output", "raster_output", "periodic"}


def pipeline_parameters(options: list[str]) -> dict[str, Any]:
    """The ``parameters`` block that ``morse-lidar`` records for these options."""
    parser = _parser()
    args = parser.parse_args(["--input", "scan.ply", "--output", "scan.json", *options])
    with redirect_stderr(io.StringIO()):  # _validate may print warnings meant for a single run
        _validate(parser, args)
    return {key: value for key, value in vars(args).items() if key not in EXCLUDED_PARAMETERS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distance(first: dict[str, Any] | None, second: dict[str, Any]) -> float | None:
    if first is None:
        return None
    return compare_reports(first, second)["bottleneck_max"]


class Watcher:
    def __init__(self, directory: Path, output: Path, options: list[str], reference: Path | None, settle: float):
        self.directory, self.output, self.options, self.settle = directory, output, options, settle
        if output.resolve() == directory.resolve():
            raise SystemExit("the results folder must differ from the export folder")
        self.output.mkdir(parents=True, exist_ok=True)
        self.parameters = pipeline_parameters(options)
        self.log_path = output / "scans.csv"
        self.state_path = output / "session.json"
        state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.processed: dict[str, str] = state.get("processed", {})
        self.previous_descriptor: str | None = state.get("previous")
        self.previous = self._load(self.previous_descriptor)
        if reference is None:
            self.reference_name: str | None = state.get("reference")
            self.reference_descriptor: str | None = state.get("reference_descriptor")
            self.reference = self._load(self.reference_descriptor)
        else:
            self.reference_name, self.reference_descriptor = reference.name, None
            self.reference = self._reference_report(reference)
        if self.reference is not None and self.reference.get("parameters") != self.parameters:
            raise SystemExit(
                "the reference was computed with different morse-lidar options; "
                "rerun it with the same options or start a new results folder"
            )

    def _load(self, name: str | None) -> dict[str, Any] | None:
        if not name or not (self.output / name).exists():
            return None
        return json.loads((self.output / name).read_text(encoding="utf-8"))

    def _reference_report(self, reference: Path) -> dict[str, Any]:
        if reference.suffix.lower() == ".json":
            return json.loads(reference.read_text(encoding="utf-8"))
        report, error = self._describe(reference)
        if report is None:
            raise SystemExit(f"cannot describe reference scan {reference}: {error}")
        return report

    def _describe(self, path: Path) -> tuple[dict[str, Any] | None, str | None]:
        # The full file name keeps `scan.ply` and `scan.obj` apart.
        target = self.output / f"{path.name}.json"
        target.unlink(missing_ok=True)
        with (self.output / f"{path.name}.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            try:
                describe(["--input", str(path), "--output", str(target), *self.options])
            except SystemExit as error:  # argparse or CLI validation
                return None, f"rejected by morse-lidar (exit code {error.code})"
            except Exception as error:  # noqa: BLE001 - keep watching after a bad scan
                return None, f"{type(error).__name__}: {error}"
        try:
            return json.loads(target.read_text(encoding="utf-8")), None
        except (OSError, ValueError) as error:
            return None, f"unreadable descriptor: {error}"

    def pending(self) -> list[Path]:
        """Scan files that are new or changed and have stopped being written."""
        now = time.time()
        found: list[tuple[float, Path]] = []
        for path in self.directory.iterdir():
            if path.name.startswith(".") or path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            try:
                info = path.stat()
            except OSError:  # removed or renamed by the exporter meanwhile
                continue
            if not path.is_file() or info.st_size == 0 or now - info.st_mtime < self.settle:
                continue
            if self.processed.get(path.name) != str(info.st_mtime_ns):
                found.append((info.st_mtime, path))
        return [path for _, path in sorted(found)]

    def process(self, path: Path) -> dict[str, Any]:
        row: dict[str, Any] = {"time": datetime.now().isoformat(timespec="seconds"), "file": path.name}
        try:
            stamp = str(path.stat().st_mtime_ns)
            row["sha256"] = _sha256(path)
        except OSError as error:
            row.update(status="error", error=f"cannot read file: {error}")
            self._log(row)
            return row
        report, error = self._describe(path)
        if report is None:
            row.update(status="error", error=error)
        else:
            counts = report.get("persistent_counts") or {}
            descriptor = f"{path.name}.json"
            row.update(
                status="ok",
                point_count=report.get("point_count"),
                vertex_count=report.get("vertex_count"),
                persistent_min=counts.get("min"),
                persistent_saddle=counts.get("saddle"),
                persistent_max=counts.get("max"),
                descriptor=descriptor,
            )
            try:
                if self.reference is None:
                    self.reference, self.reference_name, self.reference_descriptor = report, path.name, descriptor
                else:
                    row["bottleneck_vs_reference"] = _distance(self.reference, report)
                row["bottleneck_vs_previous"] = _distance(self.previous, report)
            except (ValueError, RuntimeError) as compare_error:
                row["error"] = f"compare: {compare_error}"
            self.previous, self.previous_descriptor = report, descriptor
        self._log(row)
        self.processed[path.name] = stamp
        self._save_state()
        return row

    def _save_state(self) -> None:
        state = {
            "processed": self.processed,
            "reference": self.reference_name,
            "reference_descriptor": self.reference_descriptor,
            "previous": self.previous_descriptor,
        }
        self.state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")

    def _log(self, row: dict[str, Any]) -> None:
        new = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in LOG_FIELDS})

    def logged_rows(self) -> list[dict[str, str]]:
        """All rows of the session log, latest entry per file (restarts included)."""
        if not self.log_path.exists():
            return []
        with self.log_path.open(newline="", encoding="utf-8") as handle:
            latest = {row["file"]: row for row in csv.DictReader(handle)}
        return list(latest.values())

    def run_once(self) -> list[dict[str, Any]]:
        return [self.process(path) for path in self.pending()]


def _format(row: dict[str, Any], reference: str | None) -> str:
    if row["status"] != "ok":
        return f"✗ {row['file']}: {row['error']}"
    counts = f"min/saddle/max = {row['persistent_min']}/{row['persistent_saddle']}/{row['persistent_max']}"
    parts = [f"✓ {row['file']}: {row.get('point_count') or row.get('vertex_count')} точек, {counts}"]
    if row.get("bottleneck_vs_reference") is not None:
        parts.append(f"до эталона ({reference}) {row['bottleneck_vs_reference']:.4g}")
    elif row["file"] == reference:
        parts.append("это эталон")
    if row.get("bottleneck_vs_previous") is not None:
        parts.append(f"до предыдущего {row['bottleneck_vs_previous']:.4g}")
    if row.get("error"):
        parts.append(row["error"])
    return "; ".join(parts)


def write_manifest_draft(watcher: Watcher, units: str) -> Path:
    """Draft manifest for ``morse-lidar-experiment``; object labels must be checked by hand."""
    scans = [
        {"path": str((watcher.directory / row["file"]).resolve()), "object": Path(row["file"]).stem.rsplit("_", 1)[0]}
        for row in watcher.logged_rows()
        if row["status"] == "ok" and (watcher.directory / row["file"]).exists()
    ]
    target = watcher.output / "manifest.draft.json"
    target.write_text(
        json.dumps({"units": units, "parameters": watcher.parameters, "scans": scans}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    options = argv[argv.index("--") + 1 :] if "--" in argv else []
    own = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="folder the scanner software exports into")
    parser.add_argument("--out", type=Path, default=None, help="results folder (default: DIRECTORY/morse-lidar)")
    parser.add_argument("--reference", type=Path, default=None, help="reference scan or descriptor JSON (default: first scan)")
    parser.add_argument("--units", choices=("m", "cm", "mm"), default="mm", help="length units of the exports (Revo Scan: mm)")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between folder checks")
    parser.add_argument("--settle", type=float, default=3.0, help="seconds a file must be unchanged before processing")
    parser.add_argument(
        "--once", action="store_true", help="process what is there and exit (no settle wait: exports must be finished)"
    )
    args = parser.parse_args(own)
    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a folder")
    if "--input" in options or "--output" in options:
        parser.error("pass pipeline options after --, without --input/--output")
    output = args.out or args.directory / "morse-lidar"
    watcher = Watcher(args.directory, output, options, args.reference, 0.0 if args.once else args.settle)
    print(f"Слежу за {args.directory} → {output} (Ctrl+C — остановить)" if not args.once else f"Обрабатываю {args.directory}")
    if watcher.reference_name:
        print(f"Эталон: {watcher.reference_name}")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    processed = 0
    try:
        while True:
            rows = watcher.run_once()
            for row in rows:
                print(_format(row, watcher.reference_name), flush=True)
            if rows:
                processed += len(rows)
                # Kept up to date after every scan, so stopping the watcher never loses it.
                write_manifest_draft(watcher, args.units)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    if processed:
        print(f"Журнал: {watcher.log_path}\nЧерновик манифеста для morse-lidar-experiment: {output / 'manifest.draft.json'} (проверьте метки object)")


if __name__ == "__main__":
    main()
