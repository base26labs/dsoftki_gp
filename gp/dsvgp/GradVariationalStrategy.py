"""Gradient variational strategy for the DSVGP baseline.
"""

from gp._gp_derivatives_variational_inference import load

try:
    _upstream = load("GradVariationalStrategy")
except ImportError as err:
    # The submodule has not been initialized. Stay importable so run.py can be
    # started for any other model, and raise only if DSVGP is actually used.
    _load_error = err

    class GradVariationalStrategy:
        def __init__(self, *args, **kwargs):
            raise _load_error

else:
    GradVariationalStrategy = _upstream.GradVariationalStrategy


__all__ = ["GradVariationalStrategy"]
