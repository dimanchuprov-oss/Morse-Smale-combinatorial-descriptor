"""Run M4 validation on the Linux compute server, never on the Mac.

Inside the existing isolated morse-lidar-bench image, with this checkout as
the working directory:

    python tools/m4_server.py --data /data/multiface --scans /scans --out /results/m4

Stages are tests, Multiface B0/B1/B3, then the full and automatic matrices of
the 14 named real scans. Each stage has a log, and source hashes are recorded
before real-scan evaluation. Existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Run this workflow on the Linux compute server. Local runs are disabled.")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    paths = sorted(path for person in ("Ваня", "Влада", "Дима", "Олег", "Сережа")
                   for path in args.scans.glob(f"{person}[0-9].ply"))
    expected = {f"{person}{i}" for person in ("Ваня", "Влада", "Дима", "Олег", "Сережа")
                for i in range(1, 3 if person == "Дима" else 4)}
    if {p.stem for p in paths} != expected:
        raise SystemExit("Expected exactly the 14 named real scans (five people)")
    if not (args.data / "faces.npy").is_file():
        raise SystemExit("Multiface faces.npy not found; pass the downloaded dataset directory")
    args.out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "MPLBACKEND": "Agg"}
    source = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").glob("*.py")) + [Path(__file__).resolve()]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
                "python": sys.version, "jobs": args.jobs, "data": str(args.data.resolve()),
                "sources_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
                "scans_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, "stages": {}}

    def save():
        (args.out / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def stage(name, command):
        print(f"Starting {name}; log: {args.out / (name + '.log')}", flush=True)
        manifest["stages"][name] = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat()}
        save()
        with (args.out / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest["stages"][name].update(exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        if result.returncode:
            raise SystemExit(f"{name} failed ({result.returncode}); see its log. Later stages were not run.")

    stage("tests", [sys.executable, "-m", "pytest", "-q"])
    stage("multiface", [sys.executable, "-m", "morse_lidar.multiface_cli", "bench", "--data", str(args.data.resolve()),
                        "--out", str((args.out / "multiface").resolve()), "--methods", "b0,b1,b3", "--seed", "0",
                        "--jobs", str(args.jobs)])
    multiface = read(args.out / "multiface/bench_summary.json")
    if len(multiface["people"]) != 10:
        raise SystemExit("Multiface run is incomplete: expected all ten people; inspect the report before real scans")
    for method in ("b0", "b1", "b3"):
        entry = multiface["methods"][method]
        if entry["total_rows"] != 1000:
            raise SystemExit(f"Incomplete Multiface classes/pairs for {method}; expected 1000 comparisons")
    # The detector is now frozen: these very same source files evaluate real scans.
    if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != digest
           for p, digest in manifest["sources_sha256"].items()):
        raise SystemExit("Source changed during Multiface evaluation; start a fresh reproducible run")
    for region in ("full", "auto"):
        stage(f"real_{region}", [sys.executable, "-m", "morse_lidar.face_cli", "matrix", *map(lambda p: str(p.resolve()), paths),
                                  "--up-axis", "y", "--region", region, "--no-topology", "--jobs", str(args.jobs),
                                  "--out", str((args.out / f"real_{region}").resolve())])
    baseline = read(args.out / "real_full/face_matrix_summary.json")
    automatic = read(args.out / "real_auto/face_matrix_summary.json")
    base_eer = baseline["all_pairs"].get("geometric_mm", {}).get("eer")
    auto_eer = automatic["all_pairs"].get("geometric_mm", {}).get("eer")
    complete = all(report["coverage"]["scored_pairs"] == report["coverage"]["requested_pairs"] == 91
                   for report in (baseline, automatic))
    comparison = {"multiface": multiface["methods"], "real_full_eer": base_eer, "real_auto_eer": auto_eer,
                  "real_pairs_complete": complete,
                  "real_non_regression": complete and base_eer is not None and auto_eer is not None and auto_eer <= base_eer,
                  "real_full_rank1": baseline["identification"]["rank1_complete_correct"],
                  "real_auto_rank1": automatic["identification"]["rank1_complete_correct"], "real_scans": 14,
                  "note": "EER is descriptive for these five people. Missing pairs cannot pass non-regression."}
    (args.out / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
