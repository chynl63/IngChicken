# IngChicken

Continual Learning for Diffusion Policy on LIBERO — team workspace.

This repository is organized by experiment track (top-level) and by team member
(sub-folder). Each member keeps their own working codebase inside their folder
so that different approaches can evolve independently.

## Directory layout

```
IngChicken/
├── Baseline/
│   └── chaeyoon/          # Baseline sequential CL pipeline (LIBERO-Object / -Spatial)
│       ├── configs/
│       ├── scripts/       # train / eval / datasets / model / singularity helpers
│       ├── run_*.sh       # entrypoints (run training / evaluation)
│       ├── submit_*.sh    # SLURM submission scripts
│       └── README.md      # chaeyoon's own usage docs
└── SDFT/
    └── MSE/               # Self-Distillation Fine-Tuning (MSE variant)
        └── sdft.py        # on-policy obs collection + MSE distillation loss
```

### Tracks

- **Baseline**: vanilla sequential fine-tuning and evaluation harness for the
  continual learning benchmark. This is the shared infrastructure (datasets,
  model, rollout evaluator, metrics).
- **SDFT**: Self-Distillation Fine-Tuning experiments. Currently contains the
  `MSE` variant. The baseline `train_sequential.py` optionally pulls in
  `SDFT.MSE.sdft` when `sdft.use_sdft=true` in the config.

## Team members

- `Baseline/chaeyoon/` — chaeyoon
- `SDFT/MSE/` — MSE

New members / tracks: create your own folder (e.g. `SDFT/KL/`, `Baseline/other/`)
and push from your own branch. Merge to `main` once the experiment stabilizes.

## Getting started

Each sub-folder is self-contained and documents its own setup. See
[`Baseline/chaeyoon/README.md`](Baseline/chaeyoon/README.md) for the current
CL training / evaluation workflow (Singularity-based).

When running the baseline trainer with SDFT enabled, the working directory
should be `Baseline/chaeyoon/` and the Python path will automatically pick up
`SDFT/MSE/sdft.py` from the repository root (handled inside
`scripts/train_sequential.py`).

## Notes

- Large artifacts (`*.sif`, checkpoints, datasets, logs, wandb runs) are
  gitignored at the repo root; keep them under each member's folder locally.
- History for files in `Baseline/chaeyoon/` was migrated from the earlier
  `cl_diffusion_libero-object` repository via `git mv`, so `git log --follow`
  still works across the move.
