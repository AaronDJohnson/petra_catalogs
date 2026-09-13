"""Self-checking coppuccino copula-flow catalog example.

The synthetic posterior contains two well-separated sources whose labels are
permuted independently in every sample. Small copula flows trained by coppuccino
recover the two clusters. The labels themselves are arbitrary, so the
check matches recovered means to the generating means before comparing them.

Run with::

    python examples/simple_flows_example.py
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from petra import CopulaFlowFit, Initialization, PosteriorChain
from petra import make_catalog_copula_flows


def matched_means(chain: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return recovered source means ordered to match the generating means."""
    recovered = np.nanmean(chain, axis=0)
    distances = np.linalg.norm(
        recovered[:, np.newaxis, :] - truth[np.newaxis, :, :],
        axis=-1,
    )
    recovered_indices, truth_indices = linear_sum_assignment(distances)
    matched = np.empty_like(recovered)
    matched[truth_indices] = recovered[recovered_indices]
    return matched


def main() -> None:
    """Generate a shuffled chain, train flows, and verify cluster recovery."""
    rng = np.random.default_rng(42)
    truth = np.array([[-2.0, -1.0], [2.0, 1.0]])
    chain = rng.normal(loc=truth, scale=0.2, size=(48, 2, 2))

    for sample in chain:
        rng.shuffle(sample, axis=0)

    posterior = PosteriorChain(
        chain,
        num_sources=2,
        num_params_per_source=2,
    )
    catalog = make_catalog_copula_flows(
        posterior,
        max_num_sources=2,
        num_iterations=1,
        rng_seed=42,
        threshold_samples=10,
        initialization=Initialization(num_iterations=5),
        flow_fit=CopulaFlowFit(
            knots=4, flow_layers=1, max_epochs=3, max_patience=2,
        ),
        progress=False,
    )

    recovered = matched_means(catalog.chain, truth)
    max_error = float(np.max(np.abs(recovered - truth)))
    print("Recovered means:")
    print(np.round(recovered, 3))
    print(f"Maximum coordinate error: {max_error:.3f}")
    np.testing.assert_allclose(recovered, truth, atol=0.15)


if __name__ == "__main__":
    main()
