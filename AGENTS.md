# AGENTS.md

## Project Overview

This repository implements parameter estimation for the FitzHugh-Nagumo ODE using Deep Neural Networks (DNNs). The codebase demonstrates neural network-based inverse maps that estimate ODE parameters from time-series data.

## Build & Run Commands

### PyTorch Implementation

Navigate to `pytorch/` directory for all PyTorch commands.

Always pass `--no-sync`. A bare `uv run` re-syncs the default groups, which resolves `torch` from PyPI and replaces the accelerator build installed for a specific machine (breaking CUDA). To (re)install deliberately, name the group matching the local driver, e.g. `uv sync --group cu126`.

**Install dependencies:**
```bash
# Sync the accelerator group matching the local driver: cpu, cu126, cu128,
# cu130, cu132, or xpu. Without a group, `torch` resolves from PyPI and you
# get a build that may not match the driver.
uv sync --group cu126

# Add the dev tools (black, isort, basedpyright) in the same command, so the
# accelerator group is not dropped from the resolution:
uv sync --group cu126 --group dev
```

**Train a model:**
```bash
uv run --no-sync python train.py --params configs/params_dnn.yaml
# legacy:
uv run --no-sync python run_dnn.py --params configs/params_dnn.yaml --mode train
```

**Evaluate a model:**
```bash
uv run --no-sync python evaluate.py --params configs/params_dnn.yaml
# legacy:
uv run --no-sync python run_dnn.py --params configs/params_dnn.yaml --mode eval
```

**Train then evaluate (combined):**
```bash
uv run --no-sync python run.py --params configs/params_dnn.yaml
```

**Run autoencoder:**
```bash
python run_ae.py --params configs/params_ae.yaml
```

**Run U-Net:**
```bash
python run_unet.py --params configs/params_unet.yaml
```

### TensorFlow Implementation

Navigate to `tensorflow/` directory for TensorFlow commands.

**Install dependencies:**
```bash
cd tensorflow
pip install -r requirements.txt
```

**Run training:**
```bash
python run_dnn.py
```

## Architecture Overview

### Directory Structure

- `pytorch/` - PyTorch implementation (primary)
- `tensorflow/` - TensorFlow implementation (alternative)
- `utils/` - Shared utilities across implementations
- `data/` - Training/evaluation datasets (limited version control)

### Key Modules

**pytorch/data.py**
- Handles data loading for FitzHugh-Nagumo ODE outputs through the dataset's record files (`load_records`, `load_arrays`, `load_data`)
- Supports multiple feature types: `TIME`, `TIME_NOISE`, `ODE_STATS`, `RATE_DURATION`, `NOISE`
- Implements custom PyTorch `Dataset` (`FHN_Dataset`) with data augmentation
- Loads arrays memory-mapped; the 2020 dataset (one file per column) has its own loader
- Preprocessing: scaling with the scales stored in the records, noise injection

**pytorch/nets.py**
- Neural network architectures for inverse maps
- Supported architectures:
  - `MLPNet` - Multi-layer perceptron
  - `MLPResNet` - MLP with residual blocks
  - `ConvNet` - 1D convolutional network
  - `EfficientNet` - EfficientNet-1D
  - `TransformerNet` - Transformer-based architecture
- Also includes autoencoder (`Autoencoder`) and U-Net components

**pytorch/run_dnn.py**
- Legacy monolithic training/evaluation script
- Orchestrates: data loading → preprocessing → network creation → training → evaluation
- Generates plots (loss curves, predictions vs. ground truth, error plots)
- Checkpointing support

**pytorch/common.py**, **pytorch/train.py**, **pytorch/evaluate.py**, **pytorch/run.py**
- Split entry points: shared setup in `common.py`; `train.py` trains and checkpoints; `evaluate.py` loads a checkpoint (or auto-discovers under `save_dir`) and runs predict/eval; `run.py` chains train then evaluate

**utils/utils.py**
- Shared enumerators (`ModeKeys`, `NetworkType`)
- Parameter loading/saving (YAML)
- Plotting utilities (`plot_loss`, `plot_data_vs_predict`, `plot_data_vs_predict_error`)

### External Dependencies

The codebase depends on `dlkit` (deep-learning-toolkit), which is referenced via relative path `../../dl-kit` in `run_dnn.py`. This toolkit provides:
- Logging utilities (`dlkit.log.log_util`)
- Network utilities (`dlkit.nets.*`)
- Training utilities (`dlkit.opt.train`)
- Learning rate schedulers

## Configuration

All experiments are configured via YAML files in `pytorch/configs/`:
- `params_dnn.yaml` - DNN inverse map configuration
- `params_ae.yaml` - Autoencoder configuration
- `params_unet.yaml` - U-Net configuration

**Key configuration sections:**
- `data`: options shared by training and evaluation: `data_dir`, feature/target types, channel and length ranges, sub-interval length and step, normalization flags, subset sizes `Ntrain`/`Nvalidation`/`Ntest`
- `data_train`, `data_evaluate`: only what differs between the modes: batch sizes, `features_sub_begin_random`/`features_sub_begin_sequence`, `features_additive_noise_std`. `load_data` raises if a shared key appears here, and if a config still uses the old name `Nvalidate`.
- `net`: network architecture, layer sizes, activation functions
- `optimizer`: optimizer type, learning rate, scheduler settings
- `training`: number of epochs
- `runconfig`: mode (train/eval), save/load directories, checkpointing

## Data Pipeline

1. **Load**: Read the record files in `data_dir` (`fhn_record*.json`; `ar1_record*.json` when noise is used) and load the arrays they list, memory-mapped. The current dataset is `../../generated_data/fhn/2026-10-11`; `data/2020-12-09` has hand-written records. There is no `file_names` option.
2. **Split**: Cut the train/validation/test subsets: the first `Ntrain` and last `Nvalidation` rows of the train split, the first `Ntest` rows of the test split
3. **Preprocess**: Scale with the record's scales when `features_normalize`/`targets_normalize` are set, and apply optional transformations
4. **Augment**: Add noise, subsample time series (optional)
5. **DataLoader**: Create PyTorch DataLoader with batching

**Feature types:**
- `TIME` - Raw time-series data from ODE
- `ODE_STATS` / `RATE_DURATION` - Spike rate and duration statistics
- `NOISE` - Correlated noise data
- `TIME_NOISE` - Time series + noise

**Target types:**
- `ODE` - ODE parameters (e.g., FitzHugh-Nagumo parameters)
- `NOISE` - Noise parameters (correlation, std dev)
- `ODE_NOISE` - Combined ODE + noise parameters

## Development Notes

- The codebase uses `torch.device` for GPU/CPU selection
- Random seeds can be set via `data.random_seed` in config files
- Training checkpoints are saved in `runs/{model_name}/checkpoints/`
- All plots and metrics are saved to the `save_dir` specified in config
- Reference output is available in `pytorch/run_dnn_ref.out` for validation
