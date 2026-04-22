# -*- coding: utf-8 -*-
"""Merge per-checkpoint perf_matrix.csv files into one matrix + plots.

Expects directories like:
  output/cl_libero_object_steps300_<tag>_r2_t00/perf_matrix.csv
  output/cl_libero_object_steps300_<tag>_r2_t01/perf_matrix.csv

Each CSV row ``task_K_...`` contains the evaluation row for checkpoint K.
This script rebuilds the full matrix and writes merged tables / plots.
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
_SPEC = importlib.util.spec_from_file_location("cl_metrics", _CLM)
_CL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_CL)

compute_nbt = _CL.compute_nbt
compute_average_sr = _CL.compute_average_sr
plot_forgetting_summary = _CL.plot_forgetting_summary
plot_performance_matrix = _CL.plot_performance_matrix
save_results_csv = _CL.save_results_csv
save_results_json = _CL.save_results_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-glob",
        type=str,
        required=True,
        help=(
            'Glob for perf_matrix.csv files, e.g. '
            '"output/cl_libero_object_steps300_<tag>_r2_t*/perf_matrix.csv"'
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for merged metrics and plots.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/continual_learning_libero_object_eval300.yaml",
    )
    args = parser.parse_args()

    pattern = args.csv_glob
    if not os.path.isabs(pattern):
        pattern = os.path.join(_REPO_ROOT, pattern)

    csv_paths = sorted(glob.glob(pattern))
    if not csv_paths:
        print(f"No files matched: {pattern}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_REPO_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n_tasks = 10
    perf_global = np.full((n_tasks, n_tasks), np.nan, dtype=float)
    task_names = None

    for csv_path in csv_paths:
        parent = os.path.basename(os.path.dirname(csv_path))
        if "_t" not in parent:
            print(f"Skip (unexpected dir name): {csv_path}", file=sys.stderr)
            continue

        task_idx = int(parent.split("_t")[-1])

        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            print(f"Skip (empty csv): {csv_path}", file=sys.stderr)
            continue

        header = rows[0]
        if task_names is None:
            task_names = [
                col.split("_", 2)[2] if col.startswith("task_") else col
                for col in header[1:]
            ]

        row_data = None
        for row in rows[1:]:
            if row and row[0].startswith(f"task_{task_idx}_"):
                row_data = row
                break
        if row_data is None:
            print(f"Warning: no row task_{task_idx}_ in {csv_path}", file=sys.stderr)
            continue

        for col_idx in range(n_tasks):
            cell = row_data[col_idx + 1] if col_idx + 1 < len(row_data) else ""
            if cell != "":
                perf_global[task_idx, col_idx] = float(cell)

    if task_names is None:
        print("Could not infer task names from CSV headers.", file=sys.stderr)
        sys.exit(1)

    if np.all(np.isnan(perf_global)):
        print(
            "Performance matrix is all NaN (missing or empty rows). Finish eval jobs first.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_REPO_ROOT, cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    nbt = compute_nbt(perf_global)
    avg_sr = compute_average_sr(perf_global)

    save_results_json(
        perf_global,
        task_names,
        nbt,
        avg_sr,
        cfg,
        os.path.join(out_dir, "results.json"),
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
    except ImportError as exc:
        print(f"Skipped plots (install matplotlib if needed): {exc}", file=sys.stderr)

    print(f"Merged {len(csv_paths)} CSV(s) -> {out_dir}")
    print(f"  NBT={nbt:.6f}  avg_SR_final={avg_sr:.6f}")


if __name__ == "__main__":
    main()
