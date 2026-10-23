"""petra: from the LISA global fit to catalogs of single sources.

The public entry points are the two ``make_catalog_*`` relabelers, :func:`load_samples` and :class:`PosteriorChain`.
``import petra`` is cheap: the entry points are resolved on first access (PEP 562), so jax is not imported until a
flow method is used. The package is typed (``py.typed``), and the lazy names are also imported under
``TYPE_CHECKING`` so a type checker sees their real signatures.
A chain is ``(n_samples, n_sources, n_params_per_source)`` and an all-NaN source row means the source is absent.
Nothing prints: diagnostics go to the ``petra`` logger, and ``progress=False`` silences every progress bar.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _dist_version
from typing import TYPE_CHECKING

try:
    __version__ = _dist_version("petra-catalogs")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

from .posterior_chain import PosteriorChain  # noqa: E402
from .samples_io import load_samples  # noqa: E402

_LAZY_ATTRS = {
    "make_catalog_mv_normal": "petra.make_catalog",
    "make_catalog_copula_flows": "petra.copula_flows",
}

if TYPE_CHECKING:  # pragma: no cover - read by type checkers, never executed
    from .copula_flows import make_catalog_copula_flows
    from .make_catalog import make_catalog_mv_normal

__all__ = [
    "PosteriorChain",
    "load_samples",
    "make_catalog_mv_normal",
    "make_catalog_copula_flows",
    "__version__",
]


def __getattr__(name: str) -> object:
    """Resolve a heavy entry point on first access and cache it in the module namespace (PEP 562)."""
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Cached in globals(): __getattr__ is only consulted on a miss.
    value = globals()[name] = getattr(import_module(_LAZY_ATTRS[name]), name)
    return value


def __dir__() -> list[str]:
    """List the package's names, including the lazy entry points not yet imported."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))
