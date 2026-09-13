petra-catalogs documentation
============================

``petra`` converts label-degenerate posterior samples from a LISA global fit
into catalogs whose source slots are consistent across samples.

The three catalog methods fit multivariate Gaussians, Bayesian Gaussian
posterior predictives, or copula flows trained by ``coppuccino``. Configure
copula training with :class:`petra.options.CopulaFlowFit` and optional Gaussian
initialization with :class:`petra.options.Initialization`.

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
