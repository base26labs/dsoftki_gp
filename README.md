# DSoftKI: Scalable GP Regression with Derivatives

This repository contains an implementation of DSoftKI: scalable GP regression with derivative observations.



## Quick Start

1. Create environment with pyenv

```bash
pyenv install 3.12
pyenv virtualenv 3.12 dsoftki
pyenv activate dsoftki
pip install -e .
```

2. Get data

```bash
./download_data.sh
```

3. Run `dsoftki` on `branin` dataset.

```
python run.py \
    model=dsoftki \
    gp.dsoft_ki.model.interp_init=kmeans \
    gp.dsoft_ki.model.num_interp=512 \
    gp.dsoft_ki.model.noise=0.01 \
    gp.dsoft_ki.model.deriv_noise=0.02 \
    gp.dsoft_ki.model.mll_approx=hutchinson_fallback \
    gp.dsoft_ki.model.device=cuda:0 \
    gp.dsoft_ki.model.per_interp_T=true \
    gp.dsoft_ki.model.use_qr=true \
    gp.dsoft_ki.model.use_ard=true \
    gp.dsoft_ki.model.learn_noise=true \
    gp.dsoft_ki.model.use_scale=true \
    gp.dsoft_ki.model.kernel._target_=RBFKernel \
    gp.dsoft_ki.training.seed=6535 \
    gp.dsoft_ki.training.epochs=50 \
    gp.dsoft_ki.training.batch_size=1024 \
    gp.dsoft_ki.training.learning_rate=0.02 \
    synthetic.N=20000 \
    data_dir=data/synthetic \
    dataset.name=branin \
    dataset.train_frac=0.9 \
    dataset.val_frac=0
```


### Variables / Arguments Explanation

#### General Options
| Name | Description |
| :------------ |  :----------- |
| `model` | Model to use: `dsoftki`, `softki`, `svgp`, `dsvgp`, `ddsvgp`, `dexact_gp` |
| `data_dir` | Path to data directory (e.g., `data/synthetic`, `data/md22`) |
| `dataset.name` | Dataset name (e.g., `branin`, `AT-AT`, `pol`, `nbody`) |
| `dataset.train_frac` | Fraction of data for training (default: 0.9) |
| `dataset.val_frac` | Fraction of data for validation (default: 0.0) |

#### Model Configuration
| Name | Description |
| :------------ |  :----------- |
| `gp.dsoft_ki.model.num_interp` | Number of inducing/interpolation points (default: 512) |
| `gp.dsoft_ki.model.interp_init` | Initialization method: `kmeans` or `random` |
| `gp.dsoft_ki.model.noise` | Observation noise for function values |
| `gp.dsoft_ki.model.deriv_noise` | Observation noise for gradient observations |
| `gp.dsoft_ki.model.learn_noise` | Learn noise parameters during training (default: true) |
| `gp.dsoft_ki.model.use_qr` | Use QR decomposition for stable solving (default: true) |
| `gp.dsoft_ki.model.use_ard` | Use Automatic Relevance Determination (ARD) lengthscales |
| `gp.dsoft_ki.model.use_scale` | Use output scale parameter in kernel |
| `gp.dsoft_ki.model.per_interp_T` | Use per-interpolation-point temperature parameters |
| `gp.dsoft_ki.model.min_T` | Minimum temperature value (default: 5e-5) |
| `gp.dsoft_ki.model.mll_approx` | MLL approximation: `exact`, `hutchinson`, `hutchinson_fallback` |
| `gp.dsoft_ki.model.fit_chunk_size` | Batch size for fitting (default: 256) |
| `gp.dsoft_ki.model.solver` | Linear solver: `solve`, `cg` (default: cg) |
| `gp.dsoft_ki.model.cg_tolerance` | Conjugate gradient tolerance (default: 1e-5) |
| `gp.dsoft_ki.model.device` | Device for inference (e.g., `cuda:0`, `cpu`) |
| `gp.dsoft_ki.model.fit_device` | Device for fitting (e.g., `cuda:0`, `cpu`) |
| `gp.dsoft_ki.model.dtype` | Data type: `float32` or `float64` |
| `gp.dsoft_ki.model.grad_only` | Use only gradient observations, no function values |
| `gp.dsoft_ki.model.skip_nll` | Skip NLL computation during training |

#### Kernel Configuration
| Name | Description |
| :------------ |  :----------- |
| `gp.dsoft_ki.model.kernel._target_` | Kernel type: `RBFKernel`, `MaternKernel` |
| `gp.dsoft_ki.model.kernel.nu` | Matern kernel smoothness (1.5, 2.5) |
| `gp.dsoft_ki.model.lengthscale` | Initial lengthscale value |

#### Deep Kernel Learning (DKL)
| Name | Description |
| :------------ |  :----------- |
| `gp.dsoft_ki.model.embed_dim` | Embedding dimension (-1 to disable, >0 to enable DKL) |
| `gp.dsoft_ki.model.hidden_dim` | Hidden layer size for embedding network (default: 64) |
| `gp.dsoft_ki.model.use_dot` | Use dot-product attention in embeddings |
| `gp.dsoft_ki.training.embed_lr` | Learning rate for embedding network (default: 1e-3) |

#### Training Configuration
| Name | Description |
| :------------ |  :----------- |
| `gp.dsoft_ki.training.seed` | Random seed for reproducibility |
| `gp.dsoft_ki.training.epochs` | Number of training epochs |
| `gp.dsoft_ki.training.batch_size` | Training batch size |
| `gp.dsoft_ki.training.learning_rate` | Learning rate for hyperparameter optimization |
| `gp.dsoft_ki.training.weight_decay` | Weight decay for regularization (default: 1e-4) |

#### Dataset-Specific Options
| Name | Description |
| :------------ |  :----------- |
| `synthetic.N` | Number of samples for synthetic datasets |
| `uci.get_forces` | Compute gradients via k-NN for UCI datasets |
| `uci.n_neighbors` | Number of neighbors for gradient approximation (default: 3) |
| `nbody.n_particles` | Number of particles in N-body system |
| `nbody.n_dims` | Spatial dimensions for N-body (default: 3) |

