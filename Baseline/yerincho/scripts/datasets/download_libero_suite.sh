#!/bin/bash
set -euo pipefail

SUITE_NAME="${1:-}"
TARGET_DIR="${2:-}"

if [[ -z "${SUITE_NAME}" ]]; then
  echo "Usage: bash scripts/datasets/download_libero_suite.sh <suite_name> [target_dir]"
  echo ""
  echo "Supported suite_name values:"
  echo "  libero_object"
  echo "  libero_goal"
  echo "  libero_spatial"
  echo "  libero_10    # LIBERO-Long"
  echo "  libero_90"
  exit 1
fi

case "${SUITE_NAME}" in
  libero_object|libero_goal|libero_spatial|libero_10|libero_90)
    ;;
  *)
    echo "Unsupported suite_name: ${SUITE_NAME}" >&2
    echo "Expected one of: libero_object, libero_goal, libero_spatial, libero_10, libero_90" >&2
    exit 1
    ;;
esac

if [[ -z "${TARGET_DIR}" ]]; then
  TARGET_DIR="/workspace/data/${SUITE_NAME}"
fi

PARENT_DIR="$(dirname "${TARGET_DIR}")"
mkdir -p "${TARGET_DIR}"

echo "=== Downloading ${SUITE_NAME} dataset from HuggingFace ==="
echo "Target directory: ${TARGET_DIR}"

python3 -m pip install -q huggingface_hub

python3 - "${SUITE_NAME}" "${TARGET_DIR}" "${PARENT_DIR}" << 'PYEOF'
import os
import sys
from huggingface_hub import hf_hub_download, list_repo_files

suite_name = sys.argv[1]
target_dir = sys.argv[2]
parent_dir = sys.argv[3]

repo_id = "yifengzhu-hf/LIBERO-datasets"

print(f"Listing files in {repo_id}/{suite_name} ...")
all_files = list_repo_files(repo_id, repo_type="dataset")
hdf5_files = [
    f for f in all_files if f.startswith(suite_name + "/") and f.endswith(".hdf5")
]

if not hdf5_files:
    raise RuntimeError(
        f"No .hdf5 files found for subfolder '{suite_name}' in {repo_id}. "
        "Check the suite name and dataset availability."
    )

print(f"Found {len(hdf5_files)} HDF5 files to download")

for i, filepath in enumerate(sorted(hdf5_files), start=1):
    filename = os.path.basename(filepath)
    target = os.path.join(target_dir, filename)

    if os.path.exists(target):
        print(f"  [{i:02d}/{len(hdf5_files)}] Already exists: {filename}")
        continue

    print(f"  [{i:02d}/{len(hdf5_files)}] Downloading: {filename}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filepath,
        repo_type="dataset",
        local_dir=parent_dir,
        local_dir_use_symlinks=False,
    )

    src = os.path.join(parent_dir, filepath)
    if src != target and os.path.exists(src):
        os.replace(src, target)
    elif downloaded != target and os.path.exists(downloaded) and not os.path.exists(target):
        os.replace(downloaded, target)

print(f"\nDone! Files saved to: {target_dir}")
PYEOF

echo ""
echo "=== Checking downloaded data ==="
HDF5_COUNT=$(find "${TARGET_DIR}" -name "*.hdf5" | wc -l)
echo "${HDF5_COUNT} HDF5 files found in ${TARGET_DIR}:"
find "${TARGET_DIR}" -name "*.hdf5" -exec basename {} \; | sort
echo "=== Done ==="
