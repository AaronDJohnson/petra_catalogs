petra-catalogs documentation
============================

``petra`` converts label-degenerate posterior samples from a LISA global fit
into catalogs whose source slots are consistent across samples.

The two catalog methods fit multivariate Gaussians or copula flows trained
by ``coppuccino``. Copula
training is configured with the keywords ``knots``, ``flow_layers``,
``max_epochs``, ``max_patience``, ``learning_rate`` and ``log_prob_floor``, and
the optional Gaussian initialization with ``init_with_mv_normal`` and
``init_num_iterations``.

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
