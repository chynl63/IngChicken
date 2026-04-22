# Diffusion Policy + LIBERO (Continual Learning 실험)

LIBERO 벤치마크에서 **Diffusion Policy**를 학습·평가하는 코드입니다.  
두 가지 사용 시나리오를 지원합니다.

| 시나리오 | 설명 | 설정 파일 |
|----------|------|-----------|
| **LIBERO-90 단일 학습** | 90개 task 데모를 per-task 균등 샘플링으로 한 번에 학습 | `configs/diffusion_policy_libero90.yaml` |
| **순차 CL (LIBERO-Object)** | Object 벤치마크 task를 순서대로 학습하며 forgetting 지표 측정 | `configs/continual_learning_libero_object.yaml` |

## 레포지토리 구조

```
configs/          # YAML 설정
scripts/
  train.py                    # LIBERO-90 단일 학습
  train_sequential.py         # 순차 CL 학습 (+ 선택적 매 task 평가)
  datasets/                   # HDF5 데이터로더
  model/                      # DiffusionPolicy, U-Net, 비전 인코더
  evaluation/                 # 롤아웃 평가, NBT/heatmap 등
Singularity.def               # (선택) 컨테이너 빌드 정의
run_sequential_singularity.sh
run_evaluate_singularity.sh
```

## 환경 요구사항

- Python 3.x, PyTorch (CUDA 권장)
- `libero`, `robosuite`, MuJoCo — **평가(시뮬레이션 롤아웃)** 시 필요
- `h5py`, `pyyaml`, `numpy`, `tqdm`, `torchvision` 등

실제 의존성은 사용 중인 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) / Diffusion Policy 환경과 맞추는 것이 안전합니다. 클러스터에서는 제공하는 **Singularity 이미지**(`dp_libero.sif`, 로컬에서 빌드·배포) 사용을 권장합니다. 이 레포에는 대용량 이미지/데이터는 포함하지 않습니다 (`.gitignore` 참고).

## 데이터·경로 설정

- **CL (Object)**: `configs/continual_learning_libero_object.yaml`의 `benchmark.data_root`를 LIBERO Object 데모 HDF5가 있는 디렉터리로 맞춥니다 (예: 컨테이너 내부 `/workspace/data`).
- **LIBERO-90**: `configs/diffusion_policy_libero90.yaml`의 `data.data_dir`를 90 task HDF5가 모인 디렉터리로 설정합니다.

체크포인트·결과 디렉터리는 각 YAML의 `logging.checkpoint_dir`, `logging.results_dir`에서 지정합니다.

## 실행 방법 (저장소 루트에서)

모든 `python -m scripts....` 명령은 **프로젝트 루트**(`dp_forgetting_libero/`)를 현재 디렉터리로 두고 실행하세요.

### 1) LIBERO-90 단일 학습

```bash
python -m scripts.train --config configs/diffusion_policy_libero90.yaml
```

### 2) 순차 CL 학습 (LIBERO-Object)

```bash
python -m scripts.train_sequential \
  --config configs/continual_learning_libero_object.yaml
```

평가를 생략하고 체크포인트만 저장:

```bash
python -m scripts.train_sequential \
  --config configs/continual_learning_libero_object.yaml \
  --skip-eval
```

### 3) 저장된 CL 체크포인트만 일괄 평가

```bash
python -m scripts.evaluation.evaluate_checkpoints \
  --config configs/continual_learning_libero_object.yaml
```

선택 인자:

- `--ckpt-dir` — 체크포인트 디렉터리 (미지정 시 설정의 `logging.checkpoint_dir`)
- `--results-dir` — 결과 저장 위치
- `--ckpt-pattern` — 예: `after_task_05.pt` 또는 `after_task_0[0-4].pt`

## Singularity (권장: 루트 전체 바인드)

`run_sequential_singularity.sh` / `run_evaluate_singularity.sh`는 호스트의 **프로젝트 루트**를 `/workspace`에 마운트하고, `python -m scripts....`로 실행합니다.  
환경 변수 `SIF_IMAGE`로 이미지 경로를 바꿀 수 있습니다 (기본: `./dp_libero.sif`).

```bash
# 학습 (--skip-eval 권장 시 스크립트 내 설정 확인)
GPU_DEVICE=0 bash run_sequential_singularity.sh

# 체크포인트 평가
GPU_DEVICE=0 bash run_evaluate_singularity.sh
```

평가 스크립트는 컨테이너 안에서 `LIBERO_CONFIG_PATH` 등을 설정합니다. 데이터·체크포인트는 호스트 `data/`, `checkpoints/` 등에 두면 바인드된 `/workspace` 아래에서 동일 경로로 보입니다.

## 지표·산출물

순차 CL에서 평가를 켜 두면 `results/` 에 다음이 생성될 수 있습니다 (설정에 따름):

- `results.json`, `perf_matrix.csv`, `perf_matrix.npy`
- `heatmap.png`, `forgetting_summary.png`
- 단계별 `training_log.json` 등

**Negative Backward Transfer (NBT)** 등은 `scripts/evaluation/cl_metrics.py`에 정의되어 있습니다.

## Git

대용량 파일(`.sif`, `.tar`, `data/`, `checkpoints/`, `logs/`, `results/` 등)은 커밋하지 않도록 `.gitignore`에 포함했습니다. 원격 저장소를 만든 뒤 `git remote add` / `git push` 하시면 됩니다.

## 라이선스

프로젝트에 명시된 라이선스가 없다면, 사용 코드(LIBERO, robosuite 등)의 라이선스를 각각 준수하세요.
