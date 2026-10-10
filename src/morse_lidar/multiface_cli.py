"""Command line for the Multiface expression benchmark (steps M0-M2 of the Multiface plan).

Download the tracked meshes of the ten v1 people, ten expression classes each
(about 1 GB of traffic and 50 MB on disk per person). Multiface is CC-BY-NC
4.0: keep the folder out of the repository and cite arXiv 2207.11243::

    morse-lidar-multiface fetch --dest ~/data/multiface

Measure what each expression costs the face matcher (see
:mod:`morse_lidar.expression_bench`)::

    morse-lidar-multiface bench --data ~/data/multiface --out results/multiface
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .expression_bench import run
from .multiface import CLASSES, PEOPLE, fetch


def _list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="download tracked meshes")
    download.add_argument("--dest", required=True, help="data folder (outside the repository)")
    download.add_argument("--people", type=_list, default=list(PEOPLE), help="comma-separated ids (default: all v1)")
    download.add_argument("--classes", type=_list, default=list(CLASSES),
                          help=f"comma-separated classes (default: {','.join(CLASSES)})")  # fmt: skip
    download.add_argument("--workers", type=int, default=4, help="parallel downloads")
    bench = commands.add_parser("bench", help="score expressive probes against neutral templates")
    bench.add_argument("--data", required=True, help="folder filled by fetch")
    bench.add_argument("--out", required=True, help="results folder")
    bench.add_argument("--methods", type=_list, default=["b0", "b1", "b2"], help="comma-separated: b0,b1,b2,b3 (M4)")
    bench.add_argument("--people", type=_list, help="comma-separated ids (default: everybody downloaded)")
    bench.add_argument("--jobs", type=int, default=0, help="worker processes (default: CPU count - 1)")
    bench.add_argument("--seed", type=int, default=0, help="seed of the shot poses and the scanner noise")
    bench.add_argument("--no-plots", action="store_true")
    bench.add_argument("--regions-only", action="store_true", help="diagnose b3 masks without matching pairs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "fetch":
            result = fetch(Path(args.dest).expanduser(), args.people, args.classes, args.workers)
        else:
            result = run(Path(args.data).expanduser(), Path(args.out), tuple(args.methods), args.people, args.jobs,
                         args.seed, plot=not args.no_plots, regions_only=args.regions_only)  # fmt: skip
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise SystemExit(f"morse-lidar-multiface: {error}") from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
