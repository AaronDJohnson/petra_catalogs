# Retired experiments

`retired-experiments-2026-09-11.tar.gz` preserves the source of the retired MCMC relabeler, source merger, standalone bijections, associated experimental examples, and dedicated tests. It is excluded from package distributions.

These are historical research materials, not supported or validated implementations. MCMC scoring omitted source-presence probabilities; merging could discard overlapping observations; the standalone transforms could produce invalid densities. The notebooks included unfinished code and machine-specific input paths. Notebook outputs were cleared to avoid retaining generated figures and stale tracebacks.

The supported alternatives are the Gaussian catalog builders and the coppuccino-backed copula-flow builder in `petra`. Reintroducing an experiment requires correcting its statistical contract and adding regression tests; extracting this archive alone does not make it compatible with the current API.

`standardized-flows-2026-09-12.tar.gz` preserves the separately retired direct FlowJAX backend, with its original settings, example, notebook, and test context. It was removed to keep all supported flow methods on coppuccino. These historical files are also excluded from package distributions.
