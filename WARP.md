# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This repository implements parameter estimation for the FitzHugh-Nagumo ODE using Deep Neural Networks (DNNs). The codebase demonstrates neural network-based inverse maps that estimate ODE parameters from time-series data.

**Reference Paper**: *Parameter Estimation with Dense and Convolutional Neural Networks Applied to the FitzHugh-Nagumo ODE* by Johann Rudi, Julie Bessac, Amanda Lenzi (2021). URL: https://arxiv.org/abs/2012.06691

## Build & Run Commands

### PyTorch Implementation

Navigate to `pytorch/` directory for all PyTorch commands.

**Install dependencies:**
```bash
cd pytorch
pip install -r requirements.txt
```

**Train a model:**
```bash
python run_dnn.py --params configs/params_dnn.yaml --mode train
```

**Evaluate a model:**
```bash
python run_dnn.py --params configs/params_dnn.yaml --mode eval
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
- Handles data loading for FitzHugh-Nagumo ODE outputs
- Supports multiple feature types: `TIME`, `ODE_STATS`, `RATE_DURATION`, `NOISE`
- Implements custom PyTorch `Dataset` (`FHN_Dataset`) with data augmentation
- Handles both memmap and numpy array loading for large datasets
- Preprocessing: normalization, scaling, noise injection

**pytorch/nets.py**
- Neural network architectures for inverse maps
- Supported architectures:
  - `MLPNet` - Multi-layer perceptron
  - `MLPResNet` - MLP with residual blocks
  - `ConvNet` - 1D convolutional network
  - `EfficientNet` - EfficientNet-1D
  - `TransformerNet` - Transformer-based architecture
- Also includes autoencoder (`Autoencoder`), U-Net, and GAN components

**pytorch/run_dnn.py**
- Main training/evaluation script
- Orchestrates: data loading → preprocessing → network creation → training → evaluation
- Generates plots (loss curves, predictions vs. ground truth, error plots)
- Checkpointing support

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
- `data`: dataset paths, feature/target types, normalization, batch sizes
- `net`: network architecture, layer sizes, activation functions
- `optimizer`: optimizer type, learning rate, scheduler settings
- `training`: number of epochs
- `runconfig`: mode (train/eval), save/load directories, checkpointing

## Data Pipeline

1. **Load**: Load numpy arrays or memmap files from `data/` directory
2. **Split**: Separate into train/validate/test sets
3. **Preprocess**: Apply normalization and optional transformations
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
