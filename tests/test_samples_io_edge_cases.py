"""
Failure and edge-case behaviour of the loaders in :mod:`petra.samples_io`.

``tests/test_posterior_chain_io.py`` covers the happy paths -- burn and thin per
file, forwarded ``num_params_per_source``, trailing newlines.  This file covers
what happens when the input is malformed or empty, since a loader that guesses
instead of complaining is how a whole analysis silently ends up on the wrong
data.
"""

import numpy as np
import pytest

from petra.samples_io import (_count_data_columns, load_samples,
                              load_samples_product_space, load_samples_ucbmcmc)


def write(path, rows):
    """Write `rows` (a list of lists) as a whitespace-delimited text file."""
    path.write_text("\n".join(" ".join(str(value) for value in row) for row in rows) + "\n")
    return str(path)


def test_count_data_columns_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "chain.txt"
    path.write_text("# a header\n\n   \n1.0 2.0 3.0\n4.0 5.0 6.0\n")
    assert _count_data_columns(str(path)) == 3


def test_count_data_columns_rejects_a_file_with_no_data(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("# only a comment\n\n")
    with pytest.raises(ValueError, match="contains no data rows"):
        _count_data_columns(str(path))


def test_load_samples_rejects_a_layout_it_cannot_infer(tmp_path):
    # Three columns is too few for either the 5-column product-space trailer or
    # the 4-column fixed-num-sources one.
    path = write(tmp_path / "chain.txt", [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="Cannot infer loader"):
        load_samples(path, num_params_per_source=2)


def test_product_space_rejects_a_burn_that_eats_the_whole_chain(tmp_path):
    # 2 params per source x 2 sources + 5 metadata columns.
    rows = [[0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0] for _ in range(3)]
    path = write(tmp_path / "chain.ps.txt", rows)
    with pytest.raises(ValueError, match="No samples left"):
        load_samples_product_space(path, num_params_per_source=2, burn=10)


def test_ucbmcmc_skips_the_zero_source_file(tmp_path):
    write(tmp_path / "dimension_chain.dat.0", [[0.0, 0.0]])
    write(tmp_path / "dimension_chain.dat.1", [[1.0, 2.0], [3.0, 4.0]])
    pc = load_samples_ucbmcmc(str(tmp_path), num_params_per_source=2)
    assert pc.chain.shape == (2, 1, 2)


def test_ucbmcmc_rejects_a_wrong_column_count(tmp_path):
    write(tmp_path / "dimension_chain.dat.1", [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="columns, but num_params_per_source"):
        load_samples_ucbmcmc(str(tmp_path), num_params_per_source=2)


def test_ucbmcmc_rejects_rows_that_are_not_a_whole_number_of_samples(tmp_path):
    # 3 rows cannot be a whole number of 2-source samples.
    write(tmp_path / "dimension_chain.dat.2", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError, match="not a multiple of the"):
        load_samples_ucbmcmc(str(tmp_path), num_params_per_source=2)


def test_ucbmcmc_remove_low_numbers_drops_short_files(tmp_path):
    write(tmp_path / "dimension_chain.dat.1", [[float(i), float(i)] for i in range(4)])
    write(tmp_path / "dimension_chain.dat.2",
          [[float(i), float(i)] for i in range(40)])
    pc = load_samples_ucbmcmc(str(tmp_path), num_params_per_source=2,
                              remove_low_numbers=True)
    # Only the 20-sample, 2-source file survives.
    assert pc.chain.shape == (20, 2, 2)


def test_ucbmcmc_skips_a_file_emptied_by_burn(tmp_path):
    write(tmp_path / "dimension_chain.dat.1", [[1.0, 2.0], [3.0, 4.0]])
    write(tmp_path / "dimension_chain.dat.2",
          [[float(i), float(i)] for i in range(20)])
    pc = load_samples_ucbmcmc(str(tmp_path), num_params_per_source=2, burn=5)
    # The 2-sample file is gone; 10 - 5 = 5 samples are left of the other one.
    assert pc.chain.shape == (5, 2, 2)
    assert np.isnan(pc.chain).sum() == 0


def test_ucbmcmc_rejects_a_directory_with_nothing_usable(tmp_path):
    write(tmp_path / "dimension_chain.dat.0", [[0.0, 0.0]])
    with pytest.raises(ValueError, match="No usable chain files found"):
        load_samples_ucbmcmc(str(tmp_path), num_params_per_source=2)
