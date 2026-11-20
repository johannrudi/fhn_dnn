# AGENTS.md

Parameter estimation for ODEs (e.g., FitzHugh-Nagumo, Lorenz-63) with neural networks that map time-series data to ODE parameters. The PyTorch code in `pytorch/` is the primary implementation; see `pytorch/AGENTS.md`. `tensorflow/` is a legacy implementation.

## Environment

Always pass `--no-sync` to `uv run`. A bare `uv run` re-syncs the default groups, which resolves `torch` from PyPI and replaces the accelerator build installed for this machine (breaking CUDA).

To (re)install, name exactly one accelerator group matching the local driver (`cpu`, `cu126`, `cu128`, `cu130`, `cu132`, `xpu`), and add `dev` in the same command so the accelerator group is not dropped:

```bash
uv sync --group cu126 --group dev
```

## Data

Datasets live outside the repository and are read through their record files (`*_record*.json`) in the config's `data.data_dir`; paths are relative to `pytorch/`. The current FHN dataset is `../../generated_data/fhn/2026-10-11`; `data/2020-12-09` is a small dataset with hand-written records.
