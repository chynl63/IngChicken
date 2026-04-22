FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    LIBERO_CONFIG_PATH=/workspace/.libero

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY repos/LIBERO /workspace/repos/LIBERO

# `robomimic` pulls in `egl_probe`, which builds a native extension via CMake.
# Install LIBERO from the local checkout under `/workspace/repos/LIBERO` so the
# container follows the same editable workflow used during development.
# Debian's python/pip here writes legacy editable metadata under
# `/usr/lib/python3.8/site-packages`, but the runtime python import path comes
# from `/usr/local/lib/python3.8/dist-packages`. We therefore also register the
# repo root via a `.pth` file in the runtime-visible site directory.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    build-essential \
    cmake \
    git \
    tmux \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglfw3 \
    libglew2.1 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/local/bin/python \
 && python -m pip install --no-cache-dir --upgrade "pip<25" "setuptools<70" wheel \
 && python -m pip install --no-cache-dir -r /workspace/requirements.txt \
 && python -c 'from pathlib import Path; Path("/workspace/repos/LIBERO/libero/__init__.py").write_text("# Added during Docker build to make local LIBERO editable install work.\n", encoding="utf-8")' \
 && mkdir -p "${LIBERO_CONFIG_PATH}" \
 && python -c 'from pathlib import Path; import os, yaml; root = Path("/workspace/repos/LIBERO/libero/libero"); cfg = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"), "init_states": str(root / "init_files"), "datasets": str(root.parent / "datasets"), "assets": str(root / "assets")}; Path(os.environ["LIBERO_CONFIG_PATH"]).joinpath("config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")' \
 && python -c 'import site; from pathlib import Path; runtime_site = next(Path(p) for p in site.getsitepackages() if p.startswith("/usr/local")); runtime_site.joinpath("libero_local.pth").write_text("/workspace/repos/LIBERO\n", encoding="utf-8")'
  
RUN cd /workspace/repos/LIBERO \
 && python -m pip install --no-cache-dir -e . \
 && cd /workspace \
 && python -c 'import libero; from libero.libero.benchmark import get_benchmark; benchmark = get_benchmark("libero_object")(task_order_index=0); print(f"LIBERO install ok ({benchmark.get_num_tasks()} tasks)")'

CMD ["bash"]
