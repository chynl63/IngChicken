# -*- coding: utf-8 -*-
"""Merge per-checkpoint perf_matrix.csv files into one matrix + plots.

Expects directories like:
  results/cl_libero_object_t00/perf_matrix.csv
  results/cl_libero_object_steps600_t00/perf_matrix.csv

Each CSV row ``task_K_...`` has SR for columns 0..K; this script collects row K
into a full (N,N) matrix.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys

import numpy as np
import yaml

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_CLM = os.path.join(_REPO_ROOT, "scripts", "evaluation", "cl_metrics.py")
_spec = importlib.util.spec_from_file_location("cl_metrics", _CLM)
_cl = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cl)
compute_nbt = _cl.compute_nbt
compute_average_sr = _cl.compute_average_sr
plot_forgetting_summary = _cl.plot_forgetting_summary
plot_performance_matrix = _cl.plot_performance_matrix
save_results_csv = _cl.save_results_csv
save_results_json = _cl.save_results_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-glob",
        type=str,
        required=True,
        help='Glob for perf_matrix.csv files, e.g. "results/cl_libero_object_steps600_t*/perf_matrix.csv"',
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help='Output directory (e.g. results/cl_libero_object_steps600_merged)',
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/continual_learning_libero_object.yaml",
    )
    args = parser.parse_args()

    repo_root = _REPO_ROOT
    if not os.path.isabs(args.csv_glob):
        pattern = os.path.join(repo_root, args.csv_glob)
    else:
        pattern = args.csv_glob

    csv_paths = sorted(glob.glob(pattern))
    if not csv_paths:
        print(f"No files matched: {pattern}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(repo_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    N = 10
    perf_global = np.full((N, N), np.nan, dtype=float)
    task_names = None

    for csv_path in csv_paths:
        parent = os.path.basename(os.path.dirname(csv_path))
        if "_t" not in parent:
            print(f"Skip (unexpected dir name): {csv_path}", file=sys.stderr)
            continue
        kk = int(parent.split("_t")[-1])

        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            print(f"Skip (empty csv): {csv_path}", file=sys.stderr)
            continue

        header = rows[0]
        if task_names is None:
            task_names = [h.split("_", 2)[2] if h.startswith("task_") else h for h in header[1:]]

        row_data = None
        for r in rows[1:]:
            if r and r[0].startswith(f"task_{kk}_"):
                row_data = r
                break
        if row_data is None:
            print(f"Warning: no row task_{kk}_ in {csv_path}", file=sys.stderr)
            continue

        for j in range(N):
            cell = row_data[j + 1] if j + 1 < len(row_data) else ""
            if cell != "":
                perf_global[kk, j] = float(cell)

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(repo_root, cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if task_names is None:
        print("Could not infer task names from CSV headers.", file=sys.stderr)
        sys.exit(1)

    if np.all(np.isnan(perf_global)):
        print(
            "Performance matrix is all NaN (missing or empty rows). "
            "Finish eval jobs first.",
            file=sys.stderr,
        )
        sys.exit(1)

    nbt = compute_nbt(perf_global)
    avg_sr = compute_average_sr(perf_global)

    save_results_json(
        perf_global, task_names, nbt, avg_sr, cfg, os.path.join(out_dir, "results.json")
    )
    save_results_csv(perf_global, task_names, os.path.join(out_dir, "perf_matrix.csv"))
    np.save(os.path.join(out_dir, "perf_matrix.npy"), perf_global)
    try:
        plot_performance_matrix(
            perf_global, task_names, os.path.join(out_dir, "heatmap.png")
        )
        plot_forgetting_summary(
            perf_global, task_names, os.path.join(out_dir, "forgetting_summary.png")
        )
    except ImportError as e:
        print(f"Skipped plots (install matplotlib if needed): {e}", file=sys.stderr)

    print(f"Merged {len(csv_paths)} CSV(s) -> {out_dir}")
    print(f"  NBT={nbt:.6f}  avg_SR_final={avg_sr:.6f}")


if __name__ == "__main__":
    main()
