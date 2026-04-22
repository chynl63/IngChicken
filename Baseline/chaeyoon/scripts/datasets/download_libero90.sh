#!/bin/bash
set -e

DATA_DIR="${1:-/workspace/data/libero90}"
mkdir -p "$DATA_DIR"

echo "=== Downloading LIBERO-90 datasets from HuggingFace ==="
echo "Target directory: $DATA_DIR"

pip install -q huggingface_hub

python3 - "$DATA_DIR" << 'PYEOF'
import sys
import os
from huggingface_hub import hf_hub_download, list_repo_files

data_dir = sys.argv[1]
repo_id = "yifengzhu-hf/LIBERO-datasets"
subfolder = "libero_90"

print(f"Listing files in {repo_id}/{subfolder} ...")
all_files = list_repo_files(repo_id, repo_type="dataset")
hdf5_files = [f for f in all_files if f.startswith(subfolder + "/") and f.endswith(".hdf5")]

print(f"Found {len(hdf5_files)} HDF5 files to download")

for i, filepath in enumerate(sorted(hdf5_files)):
    filename = os.path.basename(filepath)
    target = os.path.join(data_dir, filename)

    if os.path.exists(target):
        print(f"  [{i+1:02d}/{len(hdf5_files)}] Already exists: {filename}")
        continue

    print(f"  [{i+1:02d}/{len(hdf5_files)}] Downloading: {filename}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filepath,
        repo_type="dataset",
        local_dir=data_dir,
        local_dir_use_symlinks=False,
    )

    src = os.path.join(data_dir, filepath)
    if src != target and os.path.exists(src):
        os.rename(src, target)

print(f"\nDone! Files saved to: {data_dir}")
PYEOF

echo ""
echo "=== Checking downloaded data ==="
HDF5_COUNT=$(find "$DATA_DIR" -name "*.hdf5" | wc -l)
echo "$HDF5_COUNT HDF5 files found."
find "$DATA_DIR" -name "*.hdf5" | head -5
echo "=== Done ==="
