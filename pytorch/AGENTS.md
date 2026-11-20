# AGENTS.md

Run all commands from this directory, always with `uv run --no-sync` (see `../AGENTS.md`).

## Commands

```bash
# format, lint, type-check
uv run --no-sync isort . && uv run --no-sync black .
uv run --no-sync basedpyright .

# train, evaluate, or both
uv run --no-sync python train.py --params configs/dnn.yaml
uv run --no-sync python evaluate.py --params configs/dnn.yaml
uv run --no-sync python run.py --params configs/dnn.yaml

# override config values inline
uv run --no-sync python run.py --params configs/dnn.yaml --json-params '{"training": {"epochs": 3}}'
```

- `train.py` trains and writes checkpoints under `runconfig.save_dir`.
- `evaluate.py` evaluates the train and validation subsets at every checkpoint under `save_dir`, and the test subset at one checkpoint: `runconfig.load_checkpoint` if set, else the latest.
- `run.py` chains both, with one set of log files.
- `--json-params` ignores keys that are not in the config (with a warning) unless `--allow-new-params` is passed. `--help` lists the other options.

## Configuration

- The key `data` holds the options shared by training and evaluation; `data_train` and `data_evaluate` hold only what differs between the modes. `load_data` raises if a shared key is set in `data_train` or `data_evaluate`, or if a removed key is used.
- All outputs (params, logs, checkpoints, plots, metrics) go under `runconfig.save_dir`.
