"""
petra -- from the LISA global fit to catalogs of single sources.

The public entry points are the three ``make_catalog_*`` relabelers, plus
:func:`load_samples` and :class:`PosteriorChain`.  Their
hyperparameters are grouped into the frozen settings objects of
:mod:`petra.options` -- :class:`CopulaFlowFit` and
:class:`Initialization` -- each of whose fields is
also still accepted as a plain keyword.

Importing this package is deliberately cheap: the flow-based entry points pull
in ``jax``/``flowjax``, which cost seconds to import, so they are resolved
lazily on first attribute access (PEP 562). ``import petra`` on its own does
not import ``jax``.

Conventions
-----------
* A chain is an array of shape ``(n_samples, n_sources, n_params_per_source)``,
  and a NaN row means "this source is absent from this sample" -- that is how a
  trans-dimensional chain is stored in a fixed-shape array.  The rule is
  all-or-nothing: a source row is either entirely NaN or entirely finite.
  :func:`petra.utils.source_present` is the one implementation of it -- every
  fit, cost matrix and inclusion probability decides presence through that
  function -- and :class:`PosteriorChain` rejects a row that violates it on
  construction, rather than let the readers disagree about it.
* Every ``make_catalog_*`` entry point takes ``posterior_chain`` and
  ``max_num_sources`` positionally and everything else keyword-only, sharing the
  keyword names, defaults and meanings of the leading block so the methods can
  be swapped without rewriting the call.  Renamed keywords are still accepted
  for one release, with a :class:`DeprecationWarning`; unknown ones raise
  :class:`TypeError` rather than being silently ignored.
* Nothing in the package prints.  Diagnostics go to the ``petra`` logger (see
  :func:`petra.utils.get_logger`), and ``progress=False`` silences every
  progress bar.

Examples
--------
>>> import petra
>>> petra.__version__ is not None
True
>>> 'make_catalog_copula_flows' in petra.__all__
True
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("petra-catalogs")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

# Cheap, dependency-light entry points: petra.posterior_chain and
# petra.samples_io only need numpy/pandas, so they are imported eagerly.
# (samples_io used to be lazy because petra.utils imported jax.numpy at module
# scope; that import is function-local now, so the coupling is gone.)
from .posterior_chain import PosteriorChain  # noqa: E402
from .samples_io import load_samples  # noqa: E402

# petra.options imports nothing from petra and nothing outside the stdlib, so
# the grouped settings objects cost nothing to expose eagerly -- and they have
# to be, since `petra.CopulaFlowFit` is what a caller types before ever touching an
# entry point, and a lazy attribute that only ever resolves a stdlib import
# would be indirection for its own sake.
from .options import CopulaFlowFit, Initialization  # noqa: E402

# Heavy entry points, mapped name -> submodule. Resolved on first access by
# __getattr__ below so that `import petra` does not drag in jax/flowjax.
_LAZY_ATTRS = {
    "make_catalog_mv_normal": "petra.make_catalog",
    "make_catalog_copula_flows": "petra.copula_flows",
    "make_catalog_bayesian_gaussian": "petra.bayesian_gaussian",
}

__all__ = [
    "CopulaFlowFit",
    "Initialization",
    "PosteriorChain",
    "load_samples",
    "make_catalog_mv_normal",
    "make_catalog_copula_flows",
    "make_catalog_bayesian_gaussian",
    "__version__",
]


def __getattr__(name: str) -> object:
    """
    Resolve the heavy entry points on first access (PEP 562).

    Parameters
    ----------
    name : str
        Attribute being looked up on the ``petra`` package.

    Returns
    -------
    object
        The requested entry point, also cached into the module namespace so
        subsequent lookups skip this function entirely.

    Raises
    ------
    AttributeError
        If `name` is not a known petra entry point.
    """
    try:
        module_name = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value  # cache: __getattr__ is only consulted on a miss
    return value


def __dir__() -> list[str]:
    """
    List the package's public names, including the not-yet-imported ones.

    Returns
    -------
    list of str
        Sorted attribute names, so that tab completion and ``dir(petra)``
        show the lazy entry points before they have been touched.
    """
    return sorted(set(globals()) | set(_LAZY_ATTRS))
