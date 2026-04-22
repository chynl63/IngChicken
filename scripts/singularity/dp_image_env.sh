# shellcheck shell=bash
# Source this only inside dp_libero.sif (or a compatible image).
#
# The stack under /opt/conda/envs/dp is how this image was built; you are not
# using Conda on the login node. We avoid `conda activate` and just put that
# tree on PATH so logs show a plain interpreter path instead of Conda steps.
DP_IMAGE_PYTHON_ROOT="${DP_IMAGE_PYTHON_ROOT:-/opt/conda/envs/dp}"
export PATH="${DP_IMAGE_PYTHON_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${DP_IMAGE_PYTHON_ROOT}/lib:${LD_LIBRARY_PATH:-}"
PY="${DP_IMAGE_PYTHON_ROOT}/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Expected image Python at $PY (override with DP_IMAGE_PYTHON_ROOT=...)" >&2
  exit 1
fi
