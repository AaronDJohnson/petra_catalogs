"""Extract one frequency window of galactic-binary samples from an erebor global-fit store into a petra Feather file.

Reads the cold chain of the ``gb`` branch for the last ``--rows`` stored iterations and every walker, keeps the leaves
whose frequency lies within ``--half-bins`` of ``--f0-mhz``, converts erebor's sampler basis
``[dist kpc, f0 mHz, Mc Msun, phi0, cos_iota, psi, alpha, sin_delta, fdot_astro_ratio]`` to gbgpu's physical
parameters ``[A, f0 Hz, fdot Hz/s, fddot, phi0, iota, psi, lam, beta]`` with ``petra.gb_waveforms.erebor_to_gbgpu``
and writes a (n_samples, n_leaves, 9) chain.
Needs h5py, which petra does not depend on.

Example (the shipped ``cdl1_9mo_gb_20p38mHz.feather``)::

    python examples/erebor_gb_window.py gf_prod_9mo_testing.h5 --f0-mhz 20.38037629 --out cdl1_9mo_gb_20p38mHz.feather
"""

import argparse

import h5py
import numpy as np

from petra import PosteriorChain
from petra.gb_waveforms import erebor_to_gbgpu

# The name scripts importing this example used before the conversion moved into petra.
to_physical = erebor_to_gbgpu


def main() -> None:
    """Parse the arguments, extract the window and write the Feather file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", help="erebor HDF5 store (global_fit/chain/gb)")
    parser.add_argument("--f0-mhz", type=float, required=True, help="window centre in mHz")
    parser.add_argument("--half-bins", type=float, default=300, help="half width in bins of 1/Tobs")
    parser.add_argument("--rows", type=int, default=50, help="stored iterations to take from the end")
    parser.add_argument("--out", required=True, help="output Feather path")
    args = parser.parse_args()

    with h5py.File(args.store, "r") as store:
        group = store["global_fit"]
        log_like = group["log_like"][:, 0, 0, :]
        last = int(np.flatnonzero(np.any(log_like != 0, axis=1)).max())
        tobs = float(group["domain_settings/kwargs"].attrs["max_time"]) + float(group["domain_settings/kwargs"].attrs["min_time"])
        half = args.half_bins / tobs * 1e3
        samples = []
        for row in range(last - args.rows + 1, last + 1):
            present = group["inds/gb"][row, 0, 0]
            chain = group["chain/gb"][row, 0, 0]
            for walker in range(present.shape[0]):
                leaves = chain[walker][present[walker]]
                samples.append(leaves[np.abs(leaves[:, 1] - args.f0_mhz) < half])
    width = max(len(s) for s in samples)
    packed = np.full((len(samples), width, 9), np.nan)
    for i, leaves in enumerate(samples):
        packed[i, : len(leaves)] = leaves
    chain = erebor_to_gbgpu(packed)
    PosteriorChain(chain, width, 9, trans_dimensional=any(len(s) < width for s in samples)).to_feather(args.out)
    print(f"wrote {args.out}: {chain.shape}, Tobs={tobs:.0f} s, rows {last - args.rows + 1}-{last}")


if __name__ == "__main__":
    main()
