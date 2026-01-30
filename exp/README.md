# Experiment Scripts

This directory contains shell scripts for running benchmarks and ablation studies with DSoftKI and baseline methods.

## Benchmark Scripts

These scripts run full comparisons across multiple methods and datasets:

### `run_synthetic.sh`
Benchmark on **synthetic test functions**.
- **Datasets**: branin, six-hump-camel, styblinski-tang, hartmann, welch
- **Dimensions**: 2D to 20D
- **Models**: DSoftKI, SoftKI, SVGP, DDSVGP, DSVGP

### `run_md22.sh`
Benchmark on **MD22 molecular dynamics datasets**.
- **Datasets**: 7 molecular systems (Ac-Ala3-NHMe, DHA, AT-AT, stachyose, buckyball-catcher, AT-AT-CG-CG, double-walled-nanotube)
- **Dimensions**: 126 to 1110 (atomic coordinates)
- **Models**: DSoftKI, SoftKI, SVGP, DDSVGP


## Additional Datasets

### `run_nbody.sh`
Benchmark on **N-body gravitational Hamiltonian systems**.
- **Datasets**: N-body with 2, 4, 6, 8, 10 particles in 3D space
- **State dimension**: 2 × n_particles × 3 (positions + momenta)
- **Models**: DSoftKI, DDSVGP
- **Purpose**: Test on physics-based systems with exact gradient observations

### `run_uci.sh`
Benchmark on **UCI regression datasets with gradient support**.
- **Datasets**: pol, elevators, bike, kin40k, protein, keggdirected, ctslices, keggundirected, 3droad, song, buzz
- **Dimensions**: 3D to 385D
- **Models**: DSoftKI, DDSVGP
- **Gradients**: Noisy cradients computed via k-nearest neighbors



## Ablation Study Scripts

These scripts test specific model components:

### `ablate_mll_vs_hutch.sh`
**Ablation**: Compare **exact MLL vs. Hutchinson trace estimator**.
- **Datasets**: Synthetic test functions
- **Model**: DSoftKI only

### `run_dsoftki_10x.sh`
**Ablation**: Test DSoftKI with **10x the noise**.
- **Datasets**: Synthetic test functions
- **Model**: DSoftKI only
- **Purpose**: Evaluate value noise vs. gradient noise

### `run_no_perT.sh`
**Ablation**: Test DSoftKI **without per-interpolation-point temperature** (`per_interp_T=false`).
- **Datasets**: Synthetic test functions
- **Model**: DSoftKI only
- **Purpose**: Evaluate importance of per-point temperature parameters


## Deep Kernel Learning

### `run_md22_dkl.sh`
Benchmark **Deep Kernel Learning (DKL)** on MD22 datasets.
- **Datasets**: Same 7 MD22 molecular systems
- **Models**: DSoftKI with learned embeddings (`embed_dim` != -1)
- **Embeddings**: 24-dimensional learned representations
- **Purpose**: Test dimensionality reduction with neural network embeddings

## Usage

All scripts support a DEBUG mode for quick testing:

```bash
# Full run (takes hours)
./exp/run_md22.sh

# Quick debug run (1 epoch, 1 seed)
DEBUG=true ./exp/run_md22.sh
```

Each script is configured with:
- Dataset-specific hyperparameters (learning rates, batch sizes, noise levels)
- Multiple random seeds for statistical significance
- Wandb logging for experiment tracking


## Configuration

Scripts use these common parameters:
- `NUM_INDUCING`: Number of inducing/interpolation points (typically 512)
- `EPOCHS`: Training epochs (typically 50)
- `TRAIN_FRAC`: Fraction of data for training (varies by dataset)
- `DEVICE`: GPU device (default: "cuda:0")
- `SEEDS`: Random seeds used


## Output

Results are logged to Weights & Biases (wandb) with:
- Test RMSE and NLL metrics
- Hyperparameter values
- Training curves
- Random seed information
