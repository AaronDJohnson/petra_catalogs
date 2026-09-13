"""Sphinx configuration for petra-catalogs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "petra_catalogs"
copyright = (
    "2025, Aaron D. Johnson, Javier Roulet, Katerina Chatziioannou, "
    "Michele Vallisneri, Kyle Gersbach, Chris Trejo"
)
author = (
    "Aaron D. Johnson, Javier Roulet, Katerina Chatziioannou, "
    "Michele Vallisneri, Kyle Gersbach, Chris Trejo"
)

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc can inspect the public API without importing the optional plotting
# stack or initializing JAX during a documentation build.
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "coppuccino",
    "matplotlib",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "none"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True

html_theme = "alabaster"
html_static_path = ["_static"]
