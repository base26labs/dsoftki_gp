"""Loader for the DSVGP/DDSVGP baseline sources.

Method reference:

    Padidar, Zhu, Huang, Gardner & Bindel. "Scaling Gaussian Processes with
    Derivative Information Using Variational Inference." NeurIPS 2021.
    https://proceedings.neurips.cc/paper/2021/file/32bbf7b2bc4ed14eb1e9c2580056a989-Paper.pdf
"""

import importlib.util
import sys
from pathlib import Path

#: Repository-root-relative location of the submodule and the package inside it.
SUBMODULE_PATH = Path(__file__).resolve().parents[1] / "third_party" / "GP-Derivatives-Variational-Inference"
UPSTREAM_PACKAGE = SUBMODULE_PATH / "directionalvi"

_INIT_HINT = (
    f"The DSVGP/DDSVGP baselines are provided by a git submodule.\n\n"
    f"Fetch it with:\n\n"
    f"    git submodule update --init --recursive\n\n"
    f"Expected location: {UPSTREAM_PACKAGE}"
)


def _install_compat_modules() -> None:
    """Re-expose GPyTorch modules the upstream sources import under old paths.

    GPyTorch moved its lazy-tensor implementations into ``linear_operator`` and
    dropped the old private module paths. ``RBFKernelDirectionalGrad`` imports
    ``gpytorch.lazy.kronecker_product_lazy_tensor`` at module scope (without
    actually using it), which is enough to break the import. Aliasing the old
    path to the current symbol lets the upstream file load unmodified.
    """
    import types

    import gpytorch.lazy

    alias = "gpytorch.lazy.kronecker_product_lazy_tensor"
    if alias not in sys.modules:
        shim = types.ModuleType(alias)
        shim.KroneckerProductLazyTensor = gpytorch.lazy.KroneckerProductLazyTensor
        sys.modules[alias] = shim


def load(module_name: str):
    """Import ``<submodule>/directionalvi/<module_name>.py`` as a standalone module.

    The upstream ``directionalvi`` directory has no ``__init__.py``, and the two
    modules we need import only from ``torch``/``gpytorch``, so they load cleanly
    by file path without putting the whole upstream tree on ``sys.path``.
    """
    cache_key = f"_dsoftki_upstream_{module_name}"
    if cache_key in sys.modules:
        return sys.modules[cache_key]

    source = UPSTREAM_PACKAGE / f"{module_name}.py"
    if not source.is_file():
        raise ImportError(f"Could not find {source}.\n\n{_INIT_HINT}")

    _install_compat_modules()

    spec = importlib.util.spec_from_file_location(cache_key, source)
    module = importlib.util.module_from_spec(spec)
    # Register before executing so any self-referential import resolves.
    sys.modules[cache_key] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[cache_key]
        raise
    return module
