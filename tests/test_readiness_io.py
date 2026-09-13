"""Regression checks for preserving sample boundaries and valid chain values."""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from petra.posterior_chain import FEATHER_METADATA_KEY, PosteriorChain
from petra.samples_io import (load_samples, load_samples_fixed_num_sources,
                              load_samples_product_space, load_samples_ucbmcmc)
from petra.utils import find_prob_in_model


def test_fixed_loader_rejects_rows_with_incomplete_source_vectors(tmp_path):
    path = tmp_path / "fixed.txt"
    # Reshaping all six data values would invent a third sample and join rows.
    np.savetxt(path, [[1, 2, 3, 0, 0, 0, 0], [4, 5, 6, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="complete source vectors"):
        load_samples_fixed_num_sources(str(path), num_params_per_source=2)


@pytest.mark.parametrize("legacy", [False, True])
def test_feather_reader_rejects_width_that_would_mix_samples(tmp_path, legacy):
    path = tmp_path / "wrong_width.feather"
    frame = pd.DataFrame({"0": [1.0, 4.0], "1": [2.0, 5.0], "2": [3.0, 6.0]})
    if legacy:
        frame["num_sources"] = 1
        frame["num_params_per_source"] = 2
        frame["transdimensional"] = False
        frame.to_feather(path)
    else:
        metadata = {"version": 1, "num_sources": 1, "num_params_per_source": 2,
                    "trans_dimensional": False, "prob_in_model": None}
        table = pa.Table.from_pandas(frame, preserve_index=False)
        table = table.replace_schema_metadata({FEATHER_METADATA_KEY: json.dumps(metadata).encode()})
        feather.write_feather(table, path)

    with pytest.raises(ValueError, match="columns"):
        PosteriorChain.read_feather(str(path))


@pytest.mark.parametrize("validate_nan_convention", [False, True])
@pytest.mark.parametrize("value", [np.inf, -np.inf, 1 + 2j])
def test_invalid_chain_values_are_rejected_even_with_nan_validation_disabled(
        value, validate_nan_convention):
    with pytest.raises(ValueError, match="infinite|float or integer"):
        PosteriorChain(np.array([[[value]]]), 1, 1,
                       validate_nan_convention=validate_nan_convention)


def test_empty_chain_round_trips_with_dimensions_and_optional_metadata(tmp_path):
    path = tmp_path / "empty.feather"
    original = PosteriorChain(np.empty((0, 2, 3)), 2, 3, trans_dimensional=True)
    original.to_feather(str(path))
    loaded = PosteriorChain.read_feather(str(path))
    assert loaded.shape == (0, 2, 3)
    assert loaded.trans_dimensional is True
    assert loaded.prob_in_model is None
    assert loaded.cost_dict == {}


@pytest.mark.parametrize("fill_value", [0, -999.0, np.inf, None, "nan"])
@pytest.mark.parametrize("loader", [load_samples_product_space, load_samples_ucbmcmc])
def test_transdimensional_loaders_reject_non_nan_padding(tmp_path, loader, fill_value):
    np.savetxt(tmp_path / "dimension_chain.dat.1", [[0.25, 1.5]])
    np.savetxt(tmp_path / "dimension_chain.dat.2", [[2.25, 3.5], [4.25, 5.5]])
    path = tmp_path
    if loader is load_samples_product_space:
        path = tmp_path / "product.txt"
        np.savetxt(path, [[0.25, 1.5, 0, 0, 0, 0, 0, 0, 0],
                          [2.25, 3.5, 4.25, 5.5, 1, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="fill_value must be NaN"):
        loader(str(path), num_params_per_source=2, fill_value=fill_value)


def test_ucbmcmc_default_padding_preserves_fractional_data_and_presence(tmp_path):
    np.savetxt(tmp_path / "dimension_chain.dat.1", [[0.25, 1.5]])
    np.savetxt(tmp_path / "dimension_chain.dat.2", [[2.25, 3.5], [4.25, 5.5]])
    loaded = load_samples_ucbmcmc(str(tmp_path), num_params_per_source=2)
    np.testing.assert_array_equal(loaded.chain[:, 0], [[0.25, 1.5], [2.25, 3.5]])
    np.testing.assert_array_equal(find_prob_in_model(loaded.chain, 2, eps=0), [1, 0.5])


def test_loader_inference_ignores_inline_comments(tmp_path):
    path = tmp_path / "fixed.txt"
    path.write_text("1 2 3 4 5 6 0 0 0 0 # comment\n"
                    "7 8 9 10 11 12 0 0 0 0 # comment\n")
    loaded = load_samples(str(path), num_params_per_source=3)
    np.testing.assert_array_equal(loaded.chain, np.arange(1, 13).reshape(2, 2, 3))
