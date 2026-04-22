#!/bin/bash
set -euo pipefail

TARGET_PATH="${1:-/workspace/data/libero_object}"
TARGET_BASENAME="$(basename "$TARGET_PATH")"

case "$TARGET_BASENAME" in
    libero_object|libero_spatial|libero_goal|libero_10)
        ROOT_DIR="$(dirname "$TARGET_PATH")"
        ;;
    *)
        ROOT_DIR="$TARGET_PATH"
        ;;
esac

mkdir -p "$ROOT_DIR"

echo "=== Downloading LIBERO datasets from HuggingFace ==="
echo "Root directory: $ROOT_DIR"
echo "Suites: libero_object, libero_spatial, libero_goal, libero_10"

pip install -q huggingface_hub

python3 - "$ROOT_DIR" << 'PYEOF'
import sys
import os
from huggingface_hub import hf_hub_download, list_repo_files

root_dir = os.path.abspath(sys.argv[1])
repo_id = "yifengzhu-hf/LIBERO-datasets"
requested_suites = [
    ("libero_object", ("libero_object",)),
    ("libero_spatial", ("libero_spatial",)),
    ("libero_goal", ("libero_goal",)),
    ("libero_10", ("libero_10",)),
]

print(f"Listing files in {repo_id} ...")
all_files = list_repo_files(repo_id, repo_type="dataset")
available_roots = {path.split("/", 1)[0] for path in all_files if "/" in path}

for display_name, candidates in requested_suites:
    subfolder = next((name for name in candidates if name in available_roots), None)
    if subfolder is None:
        raise FileNotFoundError(
            f"Could not find any remote folder for {display_name}. "
            f"Tried: {', '.join(candidates)}"
        )

    data_dir = os.path.join(root_dir, subfolder)
    os.makedirs(data_dir, exist_ok=True)

    hdf5_files = [
        f for f in all_files if f.startswith(subfolder + "/") and f.endswith(".hdf5")
    ]

    print(f"\n=== {display_name} ===")
    if subfolder != display_name:
        print(f"Using official dataset folder: {subfolder}")
    print(f"Target directory: {data_dir}")
    print(f"Found {len(hdf5_files)} HDF5 files to download")

    for i, filepath in enumerate(sorted(hdf5_files), start=1):
        filename = os.path.basename(filepath)
        target = os.path.join(data_dir, filename)

        if os.path.exists(target):
            print(f"  [{i:02d}/{len(hdf5_files)}] Already exists: {filename}")
            continue

        print(f"  [{i:02d}/{len(hdf5_files)}] Downloading: {filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filepath,
            repo_type="dataset",
            local_dir=root_dir,
            local_dir_use_symlinks=False,
        )

print(f"\nDone! Files saved under: {root_dir}")
PYEOF

echo ""
echo "=== Checking downloaded data ==="
HDF5_COUNT=$(find "$ROOT_DIR" -mindepth 2 -maxdepth 2 -name "*.hdf5" | wc -l)
echo "$HDF5_COUNT HDF5 files found under $ROOT_DIR:"
for suite_name in libero_object libero_spatial libero_goal libero_10; do
    suite_dir="$ROOT_DIR/$suite_name"
    if [ -d "$suite_dir" ]; then
        SUITE_COUNT=$(find "$suite_dir" -maxdepth 1 -name "*.hdf5" | wc -l)
        echo "  $suite_name: $SUITE_COUNT files"
    fi
done
echo "=== Done ==="
