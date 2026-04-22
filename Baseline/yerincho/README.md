# Diffusion Policy + LIBERO Continual Learning 실험

이 저장소는 **LIBERO benchmark task suite**에서 **Diffusion Policy**를 순차적으로 학습시키고,  
**Experience Replay (ER)** 로 forgetting을 얼마나 줄일 수 있는지 확인하기 위한 실험 코드입니다.

현재 기준 핵심 실험 설정은
[`configs/continual_learning_libero_object.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_object.yaml)에 정리되어 있습니다.

- 모델: Diffusion Policy
- 지원 benchmark: `libero_object`, `libero_goal`, `libero_spatial`, `libero_10`
- 학습 방식: task 0 → task 9 순차 학습
- replay: `enabled: true`
- replay buffer size: `1000`
- mix ratio: `0.5` (`current : replay = 1 : 1`)
- 기본 워크플로우: 학습 중 eval 생략(`--skip-eval`) → checkpoint 저장 → 별도 rollout evaluation

이제 코드 구조와 실험 파이프라인은 `cl_diffusion_libero-object-sub` 쪽과 거의 같은 형태를 따릅니다.
즉:

- replay는 **global-capacity replay memory** 기반으로 동작하고
- checkpoint는 `raw`와 `EMA`를 모두 저장하며
- `exp_name`, `run_dir`, TensorBoard, `weights_dir` finetuning 등 최신 실험 편의 기능을 지원합니다

단, **의도적으로 유지한 실험 차이점**은 `replay.buffer_size: 1000` 입니다.
따라서 `cl_diffusion_libero-object-sub`의 ER 예시 설정(`buffer_size: 5000`)과는 결과가 같지 않을 수 있습니다.

## 실험 목적

이 실험의 목적은 다음과 같습니다.

1. Diffusion Policy를 LIBERO task suite에 대해 순차적으로 학습시킨다.
2. ER를 사용했을 때 이전 task 성능이 얼마나 유지되는지 확인한다.
3. 최종적으로 `heatmap.png`, `forgetting_summary.png`, NBT 등을 통해 forgetting 패턴을 해석한다.

즉, 이 코드는 **“Diffusion Policy에서 ER가 forgetting을 얼마나 억제하는가?”** 를 보기 위한 baseline 실험입니다.

## 지원 benchmark

현재 이 레포에서 순차 CL 실험용으로 바로 실행할 수 있는 benchmark config는 다음과 같습니다.

- `libero_object`
- `libero_goal`
- `libero_spatial`
- `libero_10`

각 benchmark는 대응되는 config 파일을 따로 제공합니다.

- object: [configs/continual_learning_libero_object.yaml](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_object.yaml)
- goal: [configs/continual_learning_libero_goal.yaml](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_goal.yaml)
- spatial: [configs/continual_learning_libero_spatial.yaml](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_spatial.yaml)
- libero-10: [configs/continual_learning_libero_10.yaml](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_10.yaml)

평가 전용 300-step config도 각각 제공합니다.

## 현재 ER 설정 요약

현재 구현은 online RL buffer라기보다,  
**이전 task 데이터셋의 windowed sample을 재사용하는 offline replay 방식**입니다.

### 1) Buffer Size

- `replay.buffer_size: 1000`
- 의미: **이전 task 전체를 합쳐 최대 1000개 replay sample**을 유지
- 즉, 이 값은 **per-task capacity가 아니라 total replay budget** 입니다

예를 들어:

- task 1 직후에는 이전 task가 1개뿐이므로 거의 `1000`개를 그 task에서 사용
- task 5 학습 시에는 이전 task가 5개이므로 대략 task당 `200`개 수준으로 재분배
- task 9 학습 시에는 이전 task가 9개이므로 대략 task당 `111`개 수준으로 재분배

정확한 분배는 `ReplayMemory._rebalance()`가 이전 task 수에 따라 다시 계산합니다.

### 2) Sampling Ratio

- `replay.mix_ratio: 0.5`
- 의미: 전체 batch를 current / replay로 나눌 때 replay 비중이 약 50%

현재 구현에서는:

- current task loader를 따로 만들고
- replay memory loader를 따로 만든 다음
- 각 training step에서 두 batch를 concat하여 loss를 계산합니다

즉, 예전처럼 `ConcatDataset + WeightedRandomSampler`를 쓰는 방식이 아니라,  
**current batch와 replay batch를 분리 구성한 뒤 step 단위로 merge하는 방식**입니다.

### 3) Storage Unit

Replay에 저장하고 학습에 사용하는 단위는 full trajectory나 single transition이 아니라  
**chunk / windowed sample**입니다.

각 HDF5 demo에서:

- 최근 `obs_horizon`
- 앞으로의 `action_horizon`

을 잘라 하나의 학습 sample로 사용합니다.

### 4) Update 방식

Replay는 매 환경 step마다 online으로 push/pop되는 구조가 아닙니다.

- 각 task 학습이 끝난 뒤
- 해당 task의 sample index를 replay memory에 등록하고
- global capacity 안에서 이전 task들에 대한 보유 샘플 수를 다시 나눕니다

즉, **task boundary마다 replay memory가 재균형(rebalance)** 됩니다.

### 5) Training Schedule

Replay는 특정 시점에만 들어가는 것이 아니라, replay가 켜진 stage에서는  
**매 optimizer step마다 current task와 함께 섞여 들어갑니다.**

즉:

- current loader는 현재 task sample만 제공
- replay loader는 이전 task sample만 제공
- optimizer update마다 두 batch를 합쳐 gradient에 반영

## 레포지토리 구조

```text
configs/          # YAML 설정
scripts/
  train.py                    # LIBERO-90 단일 학습
  train_sequential.py         # LIBERO benchmark 순차 CL 학습
  utils_er.py                 # replay memory / batch merge 유틸
  datasets/                   # HDF5 데이터로더
  model/                      # DiffusionPolicy, U-Net, vision encoder
  evaluation/                 # rollout evaluation, NBT, heatmap, merge tools
run_sequential_singularity.sh
run_evaluate_singularity.sh
submit_*_a5000.sh            # Slurm 예시 스크립트
```

## 주요 설정 파일

- 메인 실험 설정:
  [`configs/continual_learning_libero_object.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_object.yaml)
  [`configs/continual_learning_libero_goal.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_goal.yaml)
  [`configs/continual_learning_libero_spatial.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_spatial.yaml)
  [`configs/continual_learning_libero_10.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_10.yaml)
- 빠른 평가용(rollout horizon 300):
  [`configs/continual_learning_libero_object_eval300.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_object_eval300.yaml)
  [`configs/continual_learning_libero_goal_eval300.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_goal_eval300.yaml)
  [`configs/continual_learning_libero_spatial_eval300.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_spatial_eval300.yaml)
  [`configs/continual_learning_libero_10_eval300.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/continual_learning_libero_10_eval300.yaml)
- sanity video 평가용:
  [`configs/sanity_eval_video.yaml`](/home/yerincho04/ing-chicken/cl_diffusion_ER/configs/sanity_eval_video.yaml)

현재 메인 설정의 중요한 값들은 다음과 같습니다.

```yaml
weights_dir: ""

benchmark:
  name: "libero_object"  # or libero_goal / libero_spatial / libero_10

continual_learning:
  epochs_per_task: 50

replay:
  enabled: true
  buffer_size: 1000
  mix_ratio: 0.5

evaluation:
  num_episodes: 20
  max_steps_per_episode: 300

logging:
  exp_name: "cl_libero_object"
  checkpoint_dir: "/workspace/checkpoints/cl_libero_object"
  results_dir: "/workspace/results/cl_libero_object"
  use_tensorboard: true
```

## 데이터 위치

현재 실험은 LIBERO dataset 데모가 아래 경로에 있다고 가정합니다.

- 컨테이너 내부: `/workspace/data`
- 호스트 기준: 프로젝트 루트의 `data/`

즉 실제 HDF5 파일은 예를 들어 다음 위치에 있어야 합니다.

- 예: [`data/libero_object/`](/home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_object)
- 예: `data/libero_goal/`, `data/libero_spatial/`, `data/libero_10/`

### 데이터 다운로드

Hugging Face의 `yifengzhu-hf/LIBERO-datasets`에서 각 suite를 직접 다운로드할 수 있습니다.

개별 wrapper:

```bash
bash scripts/datasets/download_libero_object.sh /home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_object
bash scripts/datasets/download_libero_goal.sh /home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_goal
bash scripts/datasets/download_libero_spatial.sh /home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_spatial
bash scripts/datasets/download_libero_10.sh /home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_10
```

generic downloader:

```bash
bash scripts/datasets/download_libero_suite.sh libero_object /home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_object
bash scripts/datasets/download_libero_suite.sh libero_goal /home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_goal
bash scripts/datasets/download_libero_suite.sh libero_spatial /home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_spatial
bash scripts/datasets/download_libero_suite.sh libero_10 /home/yerincho04/ing-chicken/cl_diffusion_ER/data/libero_10
```

참고:

- Hugging Face 표기상의 `LIBERO-Long`은 CLI name이 `libero_10` 입니다
- LIBERO-90은 `libero_90` subfolder를 사용합니다

## 실행 방법

모든 명령은 저장소 루트에서 실행합니다.

```bash
cd /home/yerincho04/ing-chicken/cl_diffusion_ER
```

### 1) 순차 학습

현재 주 실험은 보통 **학습 중 eval을 생략**하고 checkpoint만 저장하는 방식입니다.

Object:

```bash
python -m scripts.train_sequential \
  --config configs/continual_learning_libero_object.yaml \
  --skip-eval
```

Goal:

```bash
python -m scripts.train_sequential \
  --config configs/continual_learning_libero_goal.yaml \
  --skip-eval
```

Spatial:

```bash
python -m scripts.train_sequential \
  --config configs/continual_learning_libero_spatial.yaml \
  --skip-eval
```

LIBERO-10:

```bash
python -m scripts.train_sequential \
  --config configs/continual_learning_libero_10.yaml \
  --skip-eval
```

학습과 동시에 stage별 rollout eval도 돌리고 싶다면:

```bash
python -m scripts.train_sequential \
  --config configs/continual_learning_libero_goal.yaml
```

### 2) 저장된 체크포인트 일괄 평가

```bash
python -m scripts.evaluation.evaluate_checkpoints \
  --config configs/continual_learning_libero_goal.yaml
```

선택 인자:

- `--ckpt-dir`: config의 checkpoint 경로 대신 다른 경로 사용
- `--results-dir`: 결과 저장 위치 override
- `--ckpt-pattern`: 특정 checkpoint만 선택 평가
- `--no-plots`: heatmap / forgetting plot 생략

예:

```bash
python -m scripts.evaluation.evaluate_checkpoints \
  --config configs/continual_learning_libero_spatial.yaml \
  --ckpt-pattern "after_task_05.pt"
```

이 평가는:

- `after_task_00.pt` ~ `after_task_09.pt`를 읽고
- 각 checkpoint를 그 시점까지의 모든 task에 rollout 평가한 뒤
- performance matrix와 forgetting 지표를 저장합니다

기본적으로 `after_task_XX.pt`만 평가하고 `after_task_XX_ema.pt`는 자동 제외합니다.

### 3) LIBERO-90 단일 학습

이 레포에는 LIBERO-90 단일 학습 코드도 포함되어 있습니다.

```bash
python -m scripts.train \
  --config configs/diffusion_policy_libero90.yaml
```

여기서는:

- `samples_per_epoch`를 주면 per-task uniform sampling
- 생략하면 full shuffled epoch

방식으로 동작합니다.

## Logging / Output

현재 버전은 `cl_diffusion_libero-object-sub` 스타일의 실험 편의 기능을 따릅니다.

### 1) Run Directory

`logging.exp_name`이 설정되어 있으면 실행 시 자동으로:

```text
output/<exp_name>_<timestamp>/
```

가 생성되고, 그 아래에:

- `checkpoints/`
- `results/`
- `tensorboard/`
- `config_resolved.yaml`

이 정리됩니다.

### 2) Checkpoints

순차 학습 시 각 stage마다 다음 파일이 저장됩니다.

- `after_task_00.pt`
- `after_task_00_ema.pt`
- ...

즉:

- `.pt` 는 raw model weights
- `_ema.pt` 는 EMA weights

입니다.

### 3) TensorBoard

`logging.use_tensorboard: true` 이면 TensorBoard log도 같이 저장됩니다.

예:

```bash
tensorboard --logdir output
```

### 4) Finetuning / Warm Start

`weights_dir`에 checkpoint 파일 또는 checkpoint가 들어 있는 디렉터리를 주면
compatible parameter만 골라 로드한 뒤 finetuning할 수 있습니다.

예:

```yaml
weights_dir: "/workspace/output/previous_run/checkpoints/best_ema.pt"
```

또는:

```yaml
weights_dir: "/workspace/output/previous_run"
```

## Singularity 실행

실제 실험은 주로 Singularity 컨테이너에서 실행합니다.

### 학습

```bash
cd /home/yerincho04/ing-chicken/cl_diffusion_ER
CONFIG_PATH=/workspace/configs/continual_learning_libero_goal.yaml \
SIF_IMAGE=/scratch2/cyhoaoen/simg/dp_libero.sif \
GPU_DEVICE=0 \
bash run_sequential_singularity.sh
```

### 평가

```bash
cd /home/yerincho04/ing-chicken/cl_diffusion_ER
CONFIG_PATH=/workspace/configs/continual_learning_libero_goal.yaml \
SIF_IMAGE=/scratch2/cyhoaoen/simg/dp_libero.sif \
GPU_DEVICE=0 \
bash run_evaluate_singularity.sh
```

현재 wrapper script들은 다음을 자동으로 처리합니다.

- 프로젝트 루트를 `/workspace`에 바인드
- `CONFIG_PATH` 전달
- `LIBERO_CONFIG_PATH`를 자동 생성해 interactive prompt 방지
- runtime용 `HOME`, `TORCH_HOME`, `.runtime/` 설정
- `numpy<2`, `h5py<3.12` 등 호환성 보정

즉 현재 버전은 **non-interactive batch 실행**을 염두에 두고 수정되어 있습니다.

## Slurm + sbatch 예시

훈련을 서버에서 background로 돌리고 싶다면:

```bash
sbatch --partition=suma_a6000 --gres=gpu:1 --mem=64G --time=14:00:00 \
  --mail-type=BEGIN,END,FAIL \
  --mail-user=your_email@example.com \
  --wrap="cd /home/yerincho04/ing-chicken/cl_diffusion_ER && CONFIG_PATH=/workspace/configs/continual_learning_libero_goal.yaml SIF_IMAGE=/scratch2/cyhoaoen/simg/dp_libero.sif GPU_DEVICE=0 bash run_sequential_singularity.sh"
```

평가는:

```bash
sbatch --partition=suma_a6000 --gres=gpu:1 --mem=64G --time=14:00:00 \
  --mail-type=BEGIN,END,FAIL \
  --mail-user=your_email@example.com \
  --wrap="cd /home/yerincho04/ing-chicken/cl_diffusion_ER && CONFIG_PATH=/workspace/configs/continual_learning_libero_goal.yaml SIF_IMAGE=/scratch2/cyhoaoen/simg/dp_libero.sif GPU_DEVICE=0 bash run_evaluate_singularity.sh"
```

로그 확인:

```bash
tail -f /home/yerincho04/slurm-<jobid>.out
```

## 평가 산출물

평가가 끝나면 `output/<run_name>/results/` 또는 config에 지정한 `results_dir/` 아래에 다음이 생성됩니다.

- `results.json`
- `perf_matrix.csv`
- `perf_matrix.npy`
- `eval_log.json`
- `heatmap.png`
- `forgetting_summary.png`
- `perf_matrix_intermediate.npy`

video 저장을 켜면:

- `videos/`

도 함께 생성됩니다.

### heatmap 해석

- 행(row): `어느 task까지 학습했는가`
- 열(column): `어떤 task를 평가했는가`
- 값: 해당 시점 checkpoint의 task별 success rate

이 그림을 통해:

- 이전 task 성능이 시간이 지나며 얼마나 유지되는지
- forgetting이 특정 task에서 집중되는지
- ER가 전반적으로 retention에 효과적인지

를 확인할 수 있습니다.

### forgetting_summary 해석

이 그림은 보통 세 가지 정보를 함께 보여줍니다.

- task별 forgetting
- 학습 stage별 평균 success rate
- 막 배운 직후 성능 vs 모든 task 학습 후 성능

즉:

- 어떤 task가 특히 많이 잊히는지
- 시간이 갈수록 평균 성능이 어떻게 변하는지
- retention이 전반적으로 유지되는지

를 한 번에 볼 수 있습니다.

## 주의: 예전 README와 달라진 점

이 문서 기준 현재 구현은 **예전 ER 설명과 다르게** 동작합니다.

예전 설명:

- 각 이전 task당 최대 1000개 replay sample
- `ConcatDataset + WeightedRandomSampler`

현재 구현:

- 이전 task 전체를 합쳐 최대 1000개 replay sample
- `ReplayMemory`가 task별 sample index를 관리
- current batch / replay batch를 따로 만든 뒤 step마다 merge

따라서 예전 README를 기준으로 실험을 해석하면 replay budget을 잘못 이해할 수 있으니 주의하세요.
