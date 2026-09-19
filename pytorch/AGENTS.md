# AGENTS.md

## Commands

```bash
# Format and lint (run from this directory)
uv run --no-sync isort --check-only .    # check import order
uv run --no-sync black --check .         # check formatting
uv run --no-sync isort . && uv run --no-sync black .   # auto-format
uv run --no-sync basedpyright .          # type-check

# Run training / evaluation (preferred split entry points)
uv run --no-sync python train.py --params configs/params_dnn.yaml
uv run --no-sync python evaluate.py --params configs/params_dnn.yaml
uv run --no-sync python run.py --params configs/params_dnn.yaml  # train then evaluate
uv run --no-sync python run.py --params configs/params_dnn.yaml --json_params '{"data_train": {"Ntrain": 1024}}'  # inline overrides
```

Always pass `--no-sync`. A bare `uv run` re-syncs the default groups, which resolves `torch` from PyPI and replaces the accelerator build installed for a specific machine (breaking CUDA). To (re)install deliberately, name the group matching the local driver, e.g. `uv sync --group cu126`.

## Architecture

**Entry points:**
- `train.py` — training only (`train`, `train_profile`); writes checkpoints under `save_dir`.
- `evaluate.py` — evaluates `train`/`validate` at every checkpoint under `save_dir`; evaluates `test` at one checkpoint (`runconfig.load_checkpoint` when set, else the latest under `save_dir`).
- `run.py` — chains `train.run_train` then `evaluate.run_evaluate` (default `train_eval`); one shared log-file set.
- `common.py` — shared setup: `initialize_run`, `load_and_preprocess_data`, `find_all_checkpoints(pattern="*.pt")`, `find_latest_checkpoint(pattern="*.pt")`.

**`data.py`** — loads numpy/memmap arrays, splits into train/validate/test, normalizes, optionally applies FFT or routes through a pre-trained autoencoder encoder. Feature types: `TIME`, `TIME_NOISE`, `ODE_STATS`/`RATE_DURATION`, `NOISE`. Target types: `ODE`, `ODE_NOISE`, `NOISE`.

**`nets.py`** — factory (`create_network`, `create_ae`) for six architectures: `MLPNet`, `MLPResNet`, `ConvNet`, `ConvResNet`, `EfficientNet`, `TransformerNet`. The ResNet variants support residual blocks and optional attention. Convolutional architectures are 1D (time-series input).

**`plot_utils.py`** — training loss curves, scatter plots (truth vs. prediction), error plots, and metrics-vs-checkpoint curves.

## Configuration

All configs are YAML in `configs/`. The key sections are:

- `data` — shared settings: data directory, normalization flags, memmap sizes, derived shapes
- `data_train` — train feature/target knobs, Ntrain/Nvalidate, train batch size
- `data_evaluate` — eval feature/target knobs, Ntest, eval batch size
- `net` — network type and architecture hyperparameters
- `optimizer` — type, learning rate, betas, weight decay, optional LR scheduler
- `training` — number of epochs, optional `autocast_dtype`, and `torch.compile` settings (`compile`, `compile_mode`, `compile_fullgraph`, `compile_dynamic`, `profile_warmup_steps`)
- `runconfig` — save dir, optional `load_checkpoint`, checkpoint frequency, debug flag

`params_dnn_2025.yaml` is the current DNN default. The `params_m*.yaml` files are scalability study variants (varying network width M=4/16/64/256/1024).

## Outputs

All outputs go to `runs/dnn/` (or the configured `save_dir`):
- `params.yaml` — exact config used (for reproducibility)
- `net.txt` — architecture summary and parameter counts
- `checkpoints/` — model weights saved every N epochs
- `loss.txt` / `loss.pdf` — training loss curve
- `checkpoints_eval/<checkpoint stem>/data_vs_predict_{train,validate}.pdf` — per-checkpoint scatter plots
- `checkpoints_eval/<checkpoint stem>/predict_error_{train,validate}.pdf` — per-checkpoint error analysis
- `metrics_vs_checkpoint.pdf` — overall MSE/MAE/R2 vs checkpoint epoch (train + validate)
- `data_vs_predict_test.pdf` — test scatter plot (single checkpoint)
- `predict_error_test.pdf` — test error analysis (single checkpoint)
