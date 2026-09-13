"""
Shared pytest configuration for petra's test suite.

The only thing that has to happen before anything else is selecting a
non-interactive matplotlib backend: a couple of tests exercise petra's
diagnostic plotting, and on a developer machine the default backend is a GUI one
that opens windows (and, on some platforms, needs a main thread it does not
have).  ``matplotlib.use`` must be called before ``pyplot`` creates its first
figure, so it lives here rather than in the test modules.
"""

import matplotlib

matplotlib.use("Agg")
