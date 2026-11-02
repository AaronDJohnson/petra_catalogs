"""The optional GBGPU adapter on the CDL1 example, and the waveform-match refinement end to end.

Every test that generates a waveform skips without gbgpu (the ``match`` extra), or when importing it crashes the
interpreter (see ``importorskip_native``); the argument checks and the install hint run everywhere.
"""

import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from petra.gb_waveforms import GBGPU_COLUMNS, erebor_to_gbgpu, gbgpu_waveforms
from petra.match import bank_matches, cluster_by_match, greedy_clusters, refine_catalog
from petra.posterior_chain import PosteriorChain
from tests.helpers import LISA_EXAMPLE, importorskip_native

#: The settings of the CDL1 run the example files come from: 9 months at dt = 2.5 s, TDI-2, gbgpu oversample 4.
RUN = {"T": 23328000.0, "dt": 2.5, "tdi2": True, "oversample": 4}
#: What the erebor preset needs: the run's orbits, labelled ICRS, and its start time (the CDL1 data start).
EREBOR_RUN = {"orbits": SimpleNamespace(frame="icrs"), "t0": 97729089.327664}


@pytest.fixture(scope="module")
def run_waveforms():
    """The adapter at the CDL1 run's settings; skipped without gbgpu."""
    importorskip_native("gbgpu.gbgpu")
    return gbgpu_waveforms(**RUN)


@pytest.fixture(scope="module")
def cdl1_15p63():
    """The (200, 9) rows of the CDL1 15.63 mHz source, 110,909 bins below the 20.38 mHz one."""
    path = LISA_EXAMPLE.parent / "cdl1_9mo_gb_15p63mHz.feather"
    if not path.is_file():
        pytest.skip(f"the CDL1 example {path} is missing")
    return PosteriorChain.read_feather(str(path)).chain[:, 0]


def to_erebor(rows: np.ndarray, chirp_mass: float = 0.3) -> np.ndarray:
    """Return GBGPU rows (fddot 0) in erebor's sampler basis, every source given `chirp_mass` [Msun]; needs gbgpu."""
    from gbgpu.utils.utility import get_amplitude, get_fdot

    amplitude, f0, fdot, _, phi0, iota, psi, lam, beta = rows.T
    mass = chirp_mass * 2.0 ** 0.2
    # The amplitude is proportional to 1 / distance.
    distance = get_amplitude(mass, mass, f0, 1.0) / amplitude
    return np.column_stack([distance, f0 * 1e3, np.full(len(rows), chirp_mass), phi0, np.cos(iota), psi, lam,
                            np.sin(beta), fdot / get_fdot(f=f0, Mc=chirp_mass) - 1])


def sorted_rows(chain: np.ndarray) -> list:
    """Return each sample's rows in sorted order, absent rows included, to compare samples up to a permutation."""
    return [sorted(map(tuple, np.nan_to_num(sample, nan=-np.inf))) for sample in chain]


def test_missing_packages_raise_the_install_hint(monkeypatch):
    for name in ("gbgpu", "gbgpu.gbgpu"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"petra-catalogs\[match\].*dev"):
        gbgpu_waveforms(**RUN)


@pytest.mark.parametrize("change, message", [
    ({"T": 0.0}, "T and dt"), ({"T": np.nan}, "T and dt"), ({"dt": -2.5}, "T and dt"),
    ({"convention": "ldc"}, "convention"), ({"phase_convention": "plus"}, "phase_convention"),
    ({"run_options": {"oversample": 2}}, "petra sets itself"), ({"gbgpu_options": {"flip_ref_phase": True}}, "itself"),
    ({"convention": "erebor", "columns": GBGPU_COLUMNS}, "columns cannot"), ({"batch_size": 0}, "batch_size"),
    ({"convention": "erebor"}, "ICRS frame"),
    ({"convention": "erebor", "gbgpu_options": {"t0": 0.0, "orbits": SimpleNamespace(frame="ecliptic")}}, "ICRS"),
    ({"convention": "erebor", "gbgpu_options": {"t0": 0.0, "orbits": object()}}, "ICRS"),
    ({"convention": "erebor", "gbgpu_options": {"orbits": SimpleNamespace(frame="icrs")}}, "start time"),
])
def test_invalid_arguments_raise(change, message):
    with pytest.raises(ValueError, match=message):
        gbgpu_waveforms(**{**RUN, **change})


def test_empty_input_gives_an_empty_bank(run_waveforms):
    bank = run_waveforms(np.empty((0, 9)))
    assert bank.start.shape == (0,) and bank.data.shape == (0, 2, 0)


def test_the_psi_phi0_degeneracy_is_exact(run_waveforms, cdl1_20p38):
    rows = np.repeat(cdl1_20p38.chain[:1, 0], 3, axis=0)
    rows[1, [4, 6]] += [np.pi, np.pi / 2]  # (phi0 + pi, psi + pi/2): the same waveform
    rows[2, 4] += np.pi  # phi0 + pi alone: the negated waveform
    np.testing.assert_allclose(bank_matches(run_waveforms(rows), np.array([0]), np.array([1, 2])), [[1.0, -1.0]],
                               atol=1e-9)


def test_the_posterior_samples_of_one_source_match_as_one_cluster(run_waveforms, cdl1_20p38):
    matches = bank_matches(run_waveforms(cdl1_20p38.chain[:, 0]), np.arange(200))
    off_diagonal = matches[~np.eye(200, dtype=bool)]
    assert off_diagonal.min() > 0.75 and np.median(off_diagonal) > 0.97
    assert [cluster.size for cluster in greedy_clusters(matches, 0.8)] == [200]


def test_sources_far_apart_in_frequency_match_exactly_zero(run_waveforms, cdl1_20p38, cdl1_15p63):
    bank = run_waveforms(np.vstack([cdl1_20p38.chain[:, 0], cdl1_15p63]))
    assert np.all(bank_matches(bank, np.arange(200), np.arange(200, 400)) == 0)


def test_other_column_layouts_give_the_same_bank(run_waveforms, cdl1_20p38):
    rows = cdl1_20p38.chain[:, 0]
    amplitude, f0, fdot, _, phi0, iota, psi, lam, beta = rows.T
    gbmcmc = np.column_stack([f0, fdot, amplitude, lam, np.sin(beta), np.cos(iota), psi, -phi0])
    # fddot left out, so read as 0, which the file stores.
    reversed_columns = {name: 8 - index for name, index in GBGPU_COLUMNS.items() if name != "fddot"}
    expected = run_waveforms(rows)
    for bank in (gbgpu_waveforms(**RUN, convention="gbmcmc")(gbmcmc),
                 gbgpu_waveforms(**RUN, columns=reversed_columns)(rows[:, ::-1])):
        np.testing.assert_array_equal(bank.start, expected.start)
        np.testing.assert_allclose(bank.data, expected.data, rtol=1e-12)


def test_erebor_to_gbgpu_inverts_the_sampler_basis(cdl1_20p38):
    importorskip_native("gbgpu.gbgpu")
    rows = cdl1_20p38.chain[:, 0]
    np.testing.assert_allclose(erebor_to_gbgpu(to_erebor(rows)), rows, rtol=1e-12, atol=0)
    padded = np.full((200, 2, 9), np.nan)
    padded[:, 0] = to_erebor(rows)
    snapshot = padded.copy()
    converted = erebor_to_gbgpu(padded)
    np.testing.assert_array_equal(padded, snapshot)  # gbgpu's get_fdot scales its Mc in place
    np.testing.assert_allclose(converted[:, 0], rows, rtol=1e-12, atol=0)
    assert np.isnan(converted[:, 1]).all()


def test_erebor_to_gbgpu_is_erebors_own_transform(cdl1_20p38):
    transforms = importorskip_native("lisatools.globalfit.stock.erebor.transforms")
    erebor = to_erebor(cdl1_20p38.chain[:, 0])
    expected = transforms.make_gb_transform_container(
        use_chirp_mass=True, use_fdot_astro=True, use_distance=True).both_transforms(erebor.copy())
    expected[:, 4] *= -1  # erebor's LDC phi0, which GBGPU (e^{+i phi0}) reads as the sampled value
    np.testing.assert_array_equal(erebor_to_gbgpu(erebor), expected)


def test_the_erebor_convention_generates_on_the_runs_orbits_and_start(monkeypatch, run_waveforms, cdl1_20p38):
    from gbgpu import gbgpu

    made = []

    class DevGBGPU(gbgpu.GBGPU):
        """The dev constructor's orbits, t0 and flip_ref_phase, recorded; the PyPI orbits generate."""

        def __init__(self, force_backend: str, orbits=None, t0=None, flip_ref_phase: bool = False) -> None:
            super().__init__(force_backend=force_backend)
            made.append((orbits, t0, flip_ref_phase))

    monkeypatch.setattr(gbgpu, "GBGPU", DevGBGPU)
    erebor = to_erebor(cdl1_20p38.chain[:, 0])
    bank = gbgpu_waveforms(**RUN, convention="erebor", gbgpu_options=EREBOR_RUN)(erebor)
    expected = run_waveforms(erebor_to_gbgpu(erebor))
    np.testing.assert_array_equal(bank.start, expected.start)
    np.testing.assert_array_equal(bank.data, expected.data)
    assert made == [(EREBOR_RUN["orbits"], EREBOR_RUN["t0"], False)]


def test_the_erebor_convention_refuses_a_gbgpu_without_t0(run_waveforms):
    from gbgpu.gbgpu import GBGPU

    if "t0" in inspect.signature(GBGPU).parameters:
        pytest.skip("this GBGPU takes t0")
    with pytest.raises(ImportError, match="dev stack"):
        gbgpu_waveforms(**RUN, convention="erebor", gbgpu_options=EREBOR_RUN)


@pytest.mark.parametrize("run_options", [{}, {"N": 512}])
def test_batches_give_the_bank_of_one_call(run_waveforms, cdl1_20p38, cdl1_15p63, run_options):
    rows = np.vstack([cdl1_20p38.chain[:, 0], cdl1_15p63])
    expected = gbgpu_waveforms(**RUN, run_options=run_options)(rows)
    bank = gbgpu_waveforms(**RUN, run_options=run_options, batch_size=64)(rows)
    np.testing.assert_array_equal(bank.start, expected.start)
    np.testing.assert_array_equal(bank.data, expected.data)
    assert bank.df == expected.df
    np.testing.assert_array_equal(gbgpu_waveforms(**RUN, batch_size=400)(rows).data, run_waveforms(rows).data)


def test_cluster_by_match_gives_each_source_its_label(run_waveforms, cdl1_20p38, cdl1_15p63):
    chain = np.stack([cdl1_15p63, cdl1_20p38.chain[:, 0]], axis=1)
    catalog = cluster_by_match(PosteriorChain(chain[:, ::-1].copy(), 2, 9), run_waveforms, threshold=0.8)
    np.testing.assert_array_equal(catalog.prob_in_model, [1.0, 1.0])
    f0 = catalog.chain[..., 1]
    assert all(np.ptp(f0[:, label]) < 1e-6 for label in range(2))
    assert sorted_rows(catalog.chain) == sorted_rows(chain)


def test_an_ldc_phase_uses_flip_ref_phase_where_gbgpu_has_it(monkeypatch, run_waveforms, cdl1_20p38):
    from gbgpu import gbgpu

    flips = []

    class DevGBGPU(gbgpu.GBGPU):
        """The dev constructor's flip_ref_phase, which negates phi0 before generating."""

        def __init__(self, force_backend: str, flip_ref_phase: bool = False) -> None:
            super().__init__(force_backend=force_backend)
            flips.append(flip_ref_phase)
            self.sign = -1.0 if flip_ref_phase else 1.0

        def run_wave(self, amp, f0, fdot, fddot, phi0, *args, **kwargs):
            return super().run_wave(amp, f0, fdot, fddot, self.sign * phi0, *args, **kwargs)

    monkeypatch.setattr(gbgpu, "GBGPU", DevGBGPU)
    rows = cdl1_20p38.chain[:5, 0].copy()
    expected = run_waveforms(rows)
    rows[:, 4] *= -1
    np.testing.assert_allclose(gbgpu_waveforms(**RUN, phase_convention="ldc")(rows).data, expected.data, rtol=1e-12)
    assert flips == [True]


def test_refine_merges_two_labels_that_hold_one_source(run_waveforms, cdl1_20p38):
    rows = cdl1_20p38.chain[:, 0]
    high = rows[:, 6] > np.pi / 2  # the two psi/phi0 modes, never in one sample
    chain = np.full((200, 2, 9), np.nan)
    chain[~high, 0], chain[high, 1] = rows[~high], rows[high]
    result = refine_catalog(PosteriorChain(chain, 2, 9, trans_dimensional=True), run_waveforms, threshold=0.8)
    assert [(a.kind, a.source_label, a.target_label, a.rows) for a in result.actions] == [("merge", 1, 0, high.sum())]
    np.testing.assert_array_equal(result.catalog.prob_in_model, [1.0, 0.0])
    assert sorted_rows(result.catalog.chain) == sorted_rows(chain)


def test_refine_splits_a_label_that_holds_two_sources(run_waveforms, cdl1_20p38, cdl1_15p63):
    chain = np.full((200, 2, 9), np.nan)
    chain[:100, 0], chain[100:, 0] = cdl1_20p38.chain[:100, 0], cdl1_15p63[:100]
    result = refine_catalog(PosteriorChain(chain, 2, 9, trans_dimensional=True), run_waveforms, threshold=0.8)
    assert [(a.kind, a.source_label, a.target_label, a.rows) for a in result.actions] == [("split", 0, 1, 100)]
    f0 = result.catalog.chain[..., 1]
    assert all(np.ptp(f0[~np.isnan(f0[:, label]), label]) < 1e-6 for label in range(2))
    assert sorted_rows(result.catalog.chain) == sorted_rows(chain)
