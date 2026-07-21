#!/usr/bin/env python
"""Mirror timm's summary.csv into TensorBoard event files.

timm writes one CSV row per epoch but no TensorBoard logs. Rather than
patching timm's training loop -- which is easy to get subtly wrong and
would need redoing on every timm upgrade -- this watches the CSV and
writes events alongside it. Run it in the background while training:

    python scripts/tb_sync.py --output output &
    tensorboard --logdir output/tb --port 6006 --bind_all

Safe to start before training does, and safe to restart: it rewrites
every row it finds, so nothing is lost if it dies.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def find_summaries(output: Path) -> list[Path]:
    """timm writes output/<timestamp>-<model>-<res>/summary.csv when
    --experiment is unset, and output/train/<experiment>/ when it is set.
    Look for both."""
    return sorted(set(output.glob("*/summary.csv")) |
                  set(output.glob("train/*/summary.csv")))


def sync_once(csv_path: Path, tb_root: Path, seen: dict[Path, int]) -> int:
    """Write any rows we have not written yet. Returns rows written."""
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))

    start = seen.get(csv_path, 0)
    if len(rows) <= start:
        return 0

    run_name = csv_path.parent.name
    writer = SummaryWriter(str(tb_root / run_name), purge_step=start)
    for i, row in enumerate(rows[start:], start=start):
        step = int(float(row.get("epoch", i)))
        for key, value in row.items():
            if key == "epoch":
                continue
            try:
                writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                continue  # non-numeric column, skip
    writer.flush()
    writer.close()

    written = len(rows) - start
    seen[csv_path] = len(rows)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="output", help="timm --output dir")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds")
    ap.add_argument("--once", action="store_true", help="sync and exit")
    args = ap.parse_args()

    output = Path(args.output)
    tb_root = output / "tb"
    tb_root.mkdir(parents=True, exist_ok=True)
    print(f"watching {output}/train/*/summary.csv -> {tb_root}")

    seen: dict[Path, int] = {}
    while True:
        for csv_path in find_summaries(output):
            n = sync_once(csv_path, tb_root, seen)
            if n:
                print(f"  {csv_path.parent.name}: +{n} epoch(s), "
                      f"{seen[csv_path]} total")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
