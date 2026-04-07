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
  evaluation/                 # 롤아웃 평가, NBT/heatmap, merge_perf_matrices 등
Singularity.def               # (선택) 컨테이너 빌드 정의
run_sequential_singularity.sh
run_evaluate_singularity.sh
submit_*_a5000.sh             # (선택) Slurm 제출 예시
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

**한 번에 전체 체크포인트를 돌릴 때** 결과·로그를 실행 단위로 묶으려면 `--run-tag` 또는 `--run-tag-auto`를 씁니다.

- `--run-tag-auto` — `logging.results_dir` 아래에 `YYYYMMDD_HHMMSS` 하위 폴더를 만들고, 같은 이름으로 레포 루트 **`logs/<tag>/evaluate_checkpoints.log`** 에 표준 출력을 복제 저장합니다.
- `--run-tag NAME` — 위와 같되 폴더 이름을 직접 지정합니다.
- `--results-dir`와 `--run-tag` / `--run-tag-auto`는 **동시에 쓰지 마세요** (고정 경로 vs 설정 기준 하위 경로).

**Slurm 등으로 체크포인트를 쪼개 돌린 뒤** 행렬만 합치고 그때 히트맵을 그리는 경우, 각 잡에서는 PNG를 생략할 수 있습니다.

- `--no-plots` — `heatmap.png`, `forgetting_summary.png` 생성 생략 (테이블·`eval_log.json` 등은 유지).
- YAML `evaluation.save_plots: true|false` — 기본은 `true`. `--no-plots`가 있으면 항상 플롯을 끕니다.
- 통합 시각화: `python -m scripts.evaluation.merge_perf_matrices --csv-glob '...' --out-dir ...` (merged 디렉터리에서 플롯 생성).

그 밖의 선택 인자:

- `--ckpt-dir` — 체크포인트 디렉터리 (미지정 시 설정의 `logging.checkpoint_dir`)
- `--results-dir` — 결과 저장 위치 (**전체 경로**; `--run-tag*`와 배타적)
- `--ckpt-pattern` — 예: `after_task_05.pt` 또는 `after_task_0[0-4].pt`

#### 롤아웃 비디오 저장 (선택)

`evaluation` 블록에서 끄고 켤 수 있습니다 (기본 `save_video: false`이면 기존과 동일).

- `save_video`, `video_fps`, `num_videos_per_task`
- `video_episode_policy`: `first_k` (기본, 앞에서 `num_videos_per_task`개 에피만 녹화) 또는 `balanced_two` (ep0 즉시 저장·ep1은 전부 같은 라벨이면 태스크 끝에 저장, mixed면 해당 에피만 추가 저장 후 캡처 중단). `balanced_two`일 때는 `num_videos_per_task`는 무시되며, 진행 로그에 `    [video balanced_two] ...` 한 줄 디버그가 붙습니다 (디스크 flush: `ep## is flushed to the disk`, RAM만: `버퍼에 유지`, scratch 폐기: `버퍼에서 제거, 디스크 저장 안 됨` 등).
- 사람이 보기 좋게만 후처리: `video_rotate_180`, `video_crop_bottom_frac` (정책 입력 관측은 그대로)
- 저장 위치: `results_dir/videos/<checkpoint_stem>/` 아래 mp4 (`imageio` / `imageio-ffmpeg` 필요).

Sanity 전용 설정 예: `configs/sanity_eval_video.yaml`. Slurm 제출: `submit_sanity_video_a5000.sh`가 **`sbatch`로 GPU 노드에서 `singularity exec --nv`만 호출**하고, 실제 평가는 **`dp_libero.sif` 안의** `scripts/evaluation/singularity_eval_video_sanity.sh`가 수행합니다 (호스트 레포 전체를 `/workspace`에 바인드).

## Singularity (권장: 루트 전체 바인드)

`run_sequential_singularity.sh` / `run_evaluate_singularity.sh`는 호스트의 **프로젝트 루트**를 `/workspace`에 마운트하고, `python -m scripts....`로 실행합니다.  
환경 변수 `SIF_IMAGE`로 이미지 경로를 바꿀 수 있습니다 (기본: `./dp_libero.sif`).

```bash
# 학습 (--skip-eval 권장 시 스크립트 내 설정 확인)
GPU_DEVICE=0 bash run_sequential_singularity.sh

# 체크포인트 일괄 평가 (--run-tag-auto: results·logs를 동일 타임스탬프 하위에 정리)
GPU_DEVICE=0 bash run_evaluate_singularity.sh
```

`run_evaluate_singularity.sh`는 컨테이너 안에서 `evaluate_checkpoints`에 **`--run-tag-auto`** 를 넘깁니다. 결과는 대략 `results/cl_libero_object/<타임스탬프>/`, 로그 텍스트는 `logs/<타임스탬프>/evaluate_checkpoints.log`에 쌓입니다.

**체크포인트별 Slurm 잡** (`scripts/evaluation/singularity_eval_one_ckpt.sh` 등)은 호스트에서 `--results-dir`로 **디렉터리 단위**로 마운트하는 전제이므로, 스크립트 안에서는 **`--no-plots`** 를 붙여 두었습니다. 최종 히트맵은 `merge_perf_matrices`로 만드는 것을 권장합니다.

평가·학습 공통으로 컨테이너 안에서는 `HOME`/`TORCH_HOME`을 `/tmp` 하위로 두어 ResNet 등 사전학습 가중치 캐시가 쓰기 가능한 경로를 쓰도록 스크립트를 맞춰 두었습니다. `LIBERO_CONFIG_PATH` 등은 각 래퍼 스크립트에서 설정합니다. 데이터·체크포인트는 호스트 `data/`, `checkpoints/` 등에 두면 바인드된 `/workspace` 아래에서 동일 경로로 보입니다.

## 지표·산출물

순차 CL에서 평가를 켜 두면 `results/` 에 다음이 생성될 수 있습니다 (설정·CLI에 따름):

- `results.json`, `perf_matrix.csv`, `perf_matrix.npy`, `eval_log.json` (및 선택적 `perf_matrix_intermediate.npy`)
- `heatmap.png`, `forgetting_summary.png` — `--no-plots` 또는 `evaluation.save_plots: false` 이면 생략
- 비디오 옵션 사용 시: `videos/<checkpoint_stem>/*.mp4`
- 단계별 `training_log.json` 등

run 태그를 쓰면 위 파일들이 **`logging.results_dir/<run_tag>/`** 한 곳에 모입니다. 동일 `<run_tag>`로 **`logs/<run_tag>/evaluate_checkpoints.log`** 에 콘솔 출력이 복사됩니다 (Slurm `%j.out`과 별개).

**Negative Backward Transfer (NBT)** 등은 `scripts/evaluation/cl_metrics.py`에 정의되어 있습니다.

## Git

대용량 파일(`.sif`, `.tar`, `data/`, `checkpoints/`, `logs/`, `results/` 등)은 커밋하지 않도록 `.gitignore`에 포함했습니다. 원격 저장소를 만든 뒤 `git remote add` / `git push` 하시면 됩니다.

## 라이선스

프로젝트에 명시된 라이선스가 없다면, 사용 코드(LIBERO, robosuite 등)의 라이선스를 각각 준수하세요.
