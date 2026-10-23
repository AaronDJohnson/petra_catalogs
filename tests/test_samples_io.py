"""The text and UCBMCMC loaders in petra.samples_io: layout inference, argument checks and malformed input."""

import os
import re
import warnings

import numpy as np
import pytest

from petra.samples_io import (load_samples, load_samples_fixed_num_sources, load_samples_product_space,
                              load_samples_ucbmcmc)
from petra.utils import find_prob_in_model, source_present


def _write(path, rows) -> str:
    """Write `rows` (a list of lists) as a whitespace-delimited text file and return the path as a str."""
    path.write_text("\n".join(" ".join(str(value) for value in row) for row in rows) + "\n")
    return str(path)


def _pairs(n_rows: int, start: int = 0) -> list:
    """Return `n_rows` two-column rows ``[i, i]``, so a value tells which row it came from."""
    return [[float(i), float(i)] for i in range(start, start + n_rows)]


def _names(message: str, path) -> bool:
    """Whether `message` quotes `path` as a plain str, not as a ``Path(...)`` repr."""
    return repr(os.fspath(path)) in message and "Path(" not in message


_FIXED_DATA = np.arange(20.0).reshape(5, 4) + 0.5
_PRODUCT_DATA = np.arange(20.0).reshape(5, 4) + 0.25
_PRODUCT_EXPECTED = _PRODUCT_DATA.reshape(5, 2, 2).copy()
_PRODUCT_EXPECTED[[1, 4], 1, :] = np.nan

INFERRED = {
    "fixed": ("\n".join(" ".join(map(str, row)) for row in np.hstack([_FIXED_DATA, np.zeros((5, 4))])), 2,
              load_samples_fixed_num_sources, _FIXED_DATA.reshape(5, 2, 2), False),
    "product-space": ("\n".join(" ".join(map(str, row)) for row in
                                np.hstack([_PRODUCT_DATA, [[1.0], [0.0], [1.0], [1.0], [0.0]], np.zeros((5, 4))])),
                      2, load_samples_product_space, _PRODUCT_EXPECTED, True),
    "single-row": (" ".join(map(str, range(8))), 2, load_samples_fixed_num_sources,
                   np.arange(4.0).reshape(1, 2, 2), False),
    "comments-and-blank-lines": ("# a header\n\n   \n1 2 3 4 5 6 0 0 0 0 # comment\n7 8 9 10 11 12 0 0 0 0 # comment",
                                 3, load_samples_fixed_num_sources, np.arange(1.0, 13.0).reshape(2, 2, 3), False),
}


@pytest.mark.parametrize("text, num_params, reader, expected, trans_dimensional", INFERRED.values(),
                         ids=INFERRED.keys())
def test_load_samples_infers_the_layout_from_the_first_data_row(tmp_path, text, num_params, reader, expected,
                                                                trans_dimensional):
    """load_samples gives what the matching explicit reader gives, from a Path as from a str."""
    path = tmp_path / "chain.dat"
    path.write_text(text + "\n")

    from_path = load_samples(path, num_params)
    from_str = load_samples(str(path), num_params)
    explicit = reader(path, num_params)

    for pc in (from_path, from_str, explicit):
        np.testing.assert_array_equal(pc.chain, expected)
        assert pc.trans_dimensional is trans_dimensional


def test_load_samples_explains_a_layout_it_cannot_infer_or_that_is_ambiguous(tmp_path):
    """An uninferable layout states the rule and the values that fit; an ambiguous one names both readers."""
    with pytest.raises(ValueError, match="Cannot infer loader"):
        load_samples(_write(tmp_path / "three.txt", [[1.0, 2.0, 3.0]]), num_params_per_source=2)

    with pytest.raises(ValueError, match="Cannot infer loader") as excinfo:
        load_samples(_write(tmp_path / "ten.txt", [[float(i) for i in range(10)]]), num_params_per_source=4)
    message = str(excinfo.value)
    for fragment in ("num_params_per_source=4", "4 metadata columns", "in 5", "[1, 2, 3, 6] (fixed-num-sources)",
                     "[1, 5] (product-space)", "dimension_chain.dat"):
        assert fragment in message, fragment

    one_parameter = tmp_path / "ambiguous.dat"
    np.savetxt(one_parameter, [[1, 2, 0, 0, 0, 0], [3, 4, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="ambiguous") as excinfo:
        load_samples(one_parameter, num_params_per_source=1)
    assert "load_samples_fixed_num_sources" in str(excinfo.value)
    assert "load_samples_product_space" in str(excinfo.value)
    np.testing.assert_array_equal(load_samples_fixed_num_sources(one_parameter, 1).chain[:, :, 0], [[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        load_samples_product_space(one_parameter, 1)

    product = tmp_path / "product.dat"
    np.savetxt(product, [[1, 2, 1, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(load_samples_product_space(product, 1).chain[:, :, 0], [[1, 2], [3, np.nan]])


def test_product_space_width_follows_the_kept_source_counts(tmp_path):
    """The width is the largest count kept after burn and thin, absent sources are NaN rows, and a zero count too."""
    widest_unused = _write(tmp_path / "unused.ps.txt", [[1, 0, 0, 0, 0, 0, 0, 0], [2, 3, 0, 1, 0, 0, 0, 0]])
    pc = load_samples_product_space(widest_unused, num_params_per_source=1)
    np.testing.assert_array_equal(pc.chain[:, :, 0], [[1, np.nan], [2, 3]])
    assert pc.trans_dimensional is True

    burned = _write(tmp_path / "burned.ps.txt", [[1, 2, 3, 2, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0, 0]])
    assert load_samples_product_space(burned, num_params_per_source=1).shape == (2, 3, 1)
    assert load_samples_product_space(burned, num_params_per_source=1, burn=1).shape == (1, 1, 1)

    zero_sources = _write(tmp_path / "zero.ps.txt", [[0, -1, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(load_samples_product_space(zero_sources, 1).chain[:, :, 0], [[np.nan], [5]])

    data = np.arange(20 * 6, dtype=float).reshape(20, 6)
    metadata = np.zeros((20, 5))
    metadata[:, 0] = 1
    path = tmp_path / "thinned.ps.txt"
    np.savetxt(path, np.hstack([data, metadata]))
    full = load_samples_product_space(path, 3)
    thinned = load_samples(path, num_params_per_source=3, burn=10, thin=2)
    assert full.shape == (20, 2, 3) and thinned.shape == (5, 2, 3)
    np.testing.assert_array_equal(thinned.chain, full.chain[10::2])


_STRAY_NAN = np.arange(32, dtype=float).reshape(4, 8)
_STRAY_NAN[2, 5] = np.nan      # sample 1, source 0 of a two-source dimension file

MALFORMED = {
    "fractional-count": (load_samples_product_space, "1 2 0.5 0 0 0 0", 1, {}, None, False),
    "negative-count": (load_samples_product_space, "1 2 -2 0 0 0 0", 1, {}, None, False),
    "too-large-count": (load_samples_product_space, "1 2 2 0 0 0 0", 1, {}, None, False),
    "product-columns": (load_samples_product_space, "1 2 3 0 0 0 0 0", 2, {}, None, False),
    "no-kept-source": (load_samples, "1 2 3 4 1 0 0 0 0\n" * 3 + "9 9 9 9 -1 0 0 0 0\n" * 5, 2,
                       {"burn": 3, "thin": 2}, "burn=3 and thin=2", True),
    "fixed-columns": (load_samples_fixed_num_sources, "1 2 3 0 0 0 0\n4 5 6 0 0 0 0", 2, {}, None, False),
    "comment-only-fixed": (load_samples_fixed_num_sources, "# header only\n", 2, {}, "contains no data rows", True),
    "comment-only-product": (load_samples_product_space, "# header only\n", 2, {}, "contains no data rows", True),
    "comment-only-load_samples": (load_samples, "# header only\n", 2, {}, "contains no data rows", True),
    "stray-nan": (load_samples_ucbmcmc, "\n".join(" ".join(map(str, row)) for row in _STRAY_NAN), 8, {},
                  "sample 1, source 0", False),
}


@pytest.mark.parametrize("loader, text, num_params, kwargs, anchor, names_file", MALFORMED.values(),
                         ids=MALFORMED.keys())
def test_text_readers_refuse_malformed_files(tmp_path, loader, text, num_params, kwargs, anchor, names_file):
    """A malformed file raises ValueError, naming the file as a str where the error is about the file."""
    path = tmp_path / ("dimension_chain.dat.2" if loader is load_samples_ucbmcmc else "chain.txt")
    path.write_text(text + "\n")
    target = tmp_path if loader is load_samples_ucbmcmc else path

    with pytest.raises(ValueError, match=anchor) as excinfo:
        loader(target, num_params, **kwargs)

    if names_file:
        assert _names(str(excinfo.value), path), str(excinfo.value)


INVALID_ARGUMENTS = ([("burn", value) for value in (-1, 1.5, True, None)]
                     + [("thin", value) for value in (0, True, None)]
                     + [("num_params_per_source", value) for value in (0, True)])


def _small_input(tmp_path, loader) -> str:
    """Write seven samples whose first value is the sample index, in the layout `loader` reads."""
    if loader is load_samples_ucbmcmc:
        _write(tmp_path / "dimension_chain.dat.1", [[float(i), 0.0] for i in range(7)])
        return str(tmp_path)
    if loader is load_samples_product_space:
        return _write(tmp_path / "chain.txt", [[float(i), 0.0, 0, 0, 0, 0, 0] for i in range(7)])
    return _write(tmp_path / "chain.txt", [[float(i), 0.0, 0, 0, 0, 0] for i in range(7)])


@pytest.mark.parametrize("loader", [load_samples, load_samples_product_space, load_samples_fixed_num_sources,
                                    load_samples_ucbmcmc], ids=lambda loader: loader.__name__)
def test_arguments_are_checked_before_reading(tmp_path, loader):
    """Bad burn, thin, num_params_per_source or fill_value raise ValueError before the missing path is opened."""
    missing = str(tmp_path / "missing")
    invalid = list(INVALID_ARGUMENTS)
    if loader is not load_samples_fixed_num_sources:
        invalid.append(("fill_value", 0.0))

    for name, value in invalid:
        anchor = "fill_value must be NaN" if name == "fill_value" else f"{name} must be an integer >="
        with pytest.raises(ValueError, match=anchor):
            loader(missing, **{"num_params_per_source": 2, name: value})

    pc = loader(_small_input(tmp_path, loader), num_params_per_source=2, burn=np.int64(1), thin=np.int32(3))
    np.testing.assert_array_equal(pc.chain[:, 0, 0], [1.0, 4.0])


@pytest.mark.parametrize("loader, row", [
    (load_samples_fixed_num_sources, [0.0, 1.0, 2.0, 3.0, 0, 0, 0, 0]),
    (load_samples_product_space, [0.0, 1.0, 2.0, 3.0, 1, 0, 0, 0, 0]),
], ids=["fixed", "product"])
def test_a_burn_that_empties_a_text_file_is_refused_and_one_short_keeps_one_sample(tmp_path, loader, row):
    """burn equal to the sample count raises; one less keeps the last sample, through either entry."""
    path = _write(tmp_path / "chain.txt", [[float(i)] + row[1:] for i in range(5)])
    for read in (loader, load_samples):
        with pytest.raises(ValueError, match="No samples left in"):
            read(path, num_params_per_source=2, burn=5)
        np.testing.assert_array_equal(read(path, num_params_per_source=2, burn=4).chain[:, 0, 0], [4.0])


def test_text_is_read_as_utf8_with_an_optional_bom(tmp_path):
    """A byte-order mark is ignored, a UTF-8 comment is fine, and a Latin-1 byte raises UnicodeDecodeError."""
    bom = b"\xef\xbb\xbf"
    inputs = [(load_samples, "fixed.txt", b"0.125 1 2 3 0 0 0 0\n4 5 6 7 0 0 0 0\n"),
              (load_samples_fixed_num_sources, "fixed.txt", b"0.125 1 2 3 0 0 0 0\n4 5 6 7 0 0 0 0\n"),
              (load_samples_product_space, "product.txt", b"0.125 1 2 3 1 0 0 0 0\n"),
              (load_samples_ucbmcmc, "dimension_chain.dat.1", b"0.125 1\n")]
    for loader, name, content in inputs:
        chains = []
        for prefix in (b"", bom):
            folder = tmp_path / f"{loader.__name__}-{len(prefix)}"
            folder.mkdir()
            (folder / name).write_bytes(prefix + content)
            chains.append(loader(folder if loader is load_samples_ucbmcmc else folder / name, 2).chain)
        np.testing.assert_array_equal(chains[1], chains[0])
        assert chains[1][0, 0, 0] == 0.125

    commented = tmp_path / "commented.txt"
    commented.write_text("# résumé – run 3\n1 2 3 4 0 0 0 0\n", encoding="utf-8")
    assert load_samples(commented, 2).shape == (1, 2, 2)

    latin1 = tmp_path / "latin1.txt"
    latin1.write_bytes("# résumé\n1 2 3 4 0 0 0 0\n".encode("latin-1"))
    for loader in (load_samples, load_samples_fixed_num_sources):
        with pytest.raises(UnicodeDecodeError):
            loader(latin1, 2)
    assert issubclass(UnicodeDecodeError, ValueError)


def test_ucbmcmc_assembles_dimension_files_ordered_by_dimension(tmp_path):
    """Files .1, .2, .3 are stacked in that order, NaN-padded, with .0 skipped and P forwarded by load_samples."""
    run = tmp_path / "run"
    run.mkdir()
    _write(run / "dimension_chain.dat.0", [[0.5, 0.5, 0.5]])
    _write(run / "dimension_chain.dat.2", [[2.25, 2.5, 2.75]] * 6)
    _write(run / "dimension_chain.dat.1", [[0.25, 1.5, 2.75]] * 4)
    _write(run / "dimension_chain.dat.3", [[3.25, 3.5, 3.75]] * 3)
    with open(run / "dimension_chain.dat.3", "a") as handle:
        handle.write("\n")

    pc = load_samples(run, num_params_per_source=3)

    assert pc.shape == (8, 3, 3) and pc.trans_dimensional is True
    np.testing.assert_array_equal(source_present(pc.chain).sum(axis=1), [1, 1, 1, 1, 2, 2, 2, 3])
    np.testing.assert_array_equal(pc.chain[0, 0], [0.25, 1.5, 2.75])
    assert np.isnan(pc.chain[0, 1:]).all()
    assert source_present(pc.get_chain(burn=4)).sum(axis=1).min() == 2
    np.testing.assert_array_equal(find_prob_in_model(pc.chain, 3, eps=0), [1.0, 0.5, 0.125])
    np.testing.assert_array_equal(load_samples_ucbmcmc(str(run), 3).chain, pc.chain)


def test_ucbmcmc_applies_burn_per_file_and_keeps_the_width(tmp_path):
    """burn applies to each file; a file it empties is warned about and keeps its slots; emptying all raises."""
    per_file = tmp_path / "per_file"
    per_file.mkdir()
    _write(per_file / "dimension_chain.dat.1", _pairs(10))
    _write(per_file / "dimension_chain.dat.2", _pairs(20, start=1000))
    assert load_samples_ucbmcmc(per_file, 2, burn=5).shape == (10, 2, 2)
    assert load_samples_ucbmcmc(per_file, 2, thin=2).shape == (10, 2, 2)

    highest_emptied = tmp_path / "highest_emptied"
    highest_emptied.mkdir()
    _write(highest_emptied / "dimension_chain.dat.1", _pairs(20))
    _write(highest_emptied / "dimension_chain.dat.2", [[0.5, 0.5]] * 4)
    with pytest.warns(UserWarning, match=re.escape("dimension_chain.dat.2")):
        pc = load_samples(highest_emptied, num_params_per_source=2, burn=5)
    assert pc.shape == (15, 2, 2)
    assert source_present(pc.chain[:, 0]).all() and not source_present(pc.chain[:, 1]).any()

    lowest_emptied = tmp_path / "lowest_emptied"
    lowest_emptied.mkdir()
    _write(lowest_emptied / "dimension_chain.dat.1", _pairs(2))
    _write(lowest_emptied / "dimension_chain.dat.2", _pairs(20))
    with pytest.warns(UserWarning, match=re.escape("dimension_chain.dat.1")):
        pc = load_samples_ucbmcmc(lowest_emptied, 2, burn=5)
    assert pc.shape == (5, 2, 2) and not np.isnan(pc.chain).any()

    all_emptied = tmp_path / "all_emptied"
    all_emptied.mkdir()
    _write(all_emptied / "dimension_chain.dat.1", _pairs(2))
    _write(all_emptied / "dimension_chain.dat.2", _pairs(6))
    with pytest.raises(ValueError, match="No samples left in") as excinfo:
        load_samples_ucbmcmc(all_emptied, 2, burn=3, thin=2)
    assert "dimension_chain.dat.1" in str(excinfo.value) and "dimension_chain.dat.2" in str(excinfo.value)


@pytest.mark.parametrize("content", ["", "\n", "# a comment and no data\n"], ids=["empty", "newline", "comment"])
def test_ucbmcmc_skips_empty_short_and_stray_files(tmp_path, content):
    """Empty and stray files are skipped with one warning each; remove_low_numbers drops files of at most 8 samples."""
    run = tmp_path / "run"
    run.mkdir()
    _write(run / "dimension_chain.dat.2", _pairs(20))
    (run / "dimension_chain.dat.1").write_text(content)
    (run / "dimension_chain.dat.3").write_text(content)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pc = load_samples_ucbmcmc(run, 2)
    assert pc.shape == (10, 2, 2) and not np.isnan(pc.chain).any()
    assert len(caught) == 1
    assert "dimension_chain.dat.1" in str(caught[0].message) and "dimension_chain.dat.3" in str(caught[0].message)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert load_samples_ucbmcmc(run, 2, remove_low_numbers=True).shape == (10, 2, 2)
    assert caught == []

    stray = tmp_path / "stray"
    stray.mkdir()
    _write(stray / "dimension_chain.dat.2", [[1.0, 1.0]] * 6)
    (stray / "dimension_chain.dat.1~").write_text("garbage\n")
    (stray / "dimension_chain.dat.2.bak").write_text("garbage\n")
    (stray / "notes.txt").write_text("not a chain file\n")
    with pytest.warns(UserWarning) as caught:
        assert load_samples_ucbmcmc(stray, 2).shape == (3, 2, 2)
    assert len(caught) == 1
    assert "dimension_chain.dat.1~" in str(caught[0].message) and "dimension_chain.dat.2.bak" in str(caught[0].message)

    all_empty = tmp_path / "all_empty"
    all_empty.mkdir()
    (all_empty / "dimension_chain.dat.1").write_text(content)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="No usable chain files"):
        load_samples_ucbmcmc(all_empty, 2)

    for n_samples, expected in ((8, (20, 2, 2)), (9, (29, 2, 2))):
        low = tmp_path / f"low{n_samples}"
        low.mkdir()
        _write(low / "dimension_chain.dat.1", _pairs(n_samples))
        _write(low / "dimension_chain.dat.2", _pairs(40))
        assert load_samples_ucbmcmc(low, 2, remove_low_numbers=True).shape == expected
    assert load_samples_ucbmcmc(tmp_path / "low9", 2, remove_low_numbers=True, burn=5).shape == (4 + 15, 2, 2)

    highest_short = tmp_path / "highest_short"
    highest_short.mkdir()
    _write(highest_short / "dimension_chain.dat.1", _pairs(20))
    _write(highest_short / "dimension_chain.dat.3", _pairs(24))
    assert load_samples_ucbmcmc(highest_short, 2, remove_low_numbers=True).shape == (20, 1, 2)


def _wrong_column_count(tmp_path):
    """A dimension file whose rows are not num_params_per_source wide is refused, named."""
    path = tmp_path / "dimension_chain.dat.1"
    _write(path, [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError) as excinfo:
        load_samples_ucbmcmc(tmp_path, 2)
    assert _names(str(excinfo.value), path), str(excinfo.value)


def _partial_sample(tmp_path):
    """A dimension file whose rows are not a whole number of n-row samples is refused, named."""
    path = tmp_path / "dimension_chain.dat.2"
    _write(path, _pairs(3))
    with pytest.raises(ValueError) as excinfo:
        load_samples_ucbmcmc(tmp_path, 2)
    assert _names(str(excinfo.value), path), str(excinfo.value)


def _missing_folder(tmp_path):
    """A missing folder raises FileNotFoundError naming it, given as a str or a Path."""
    missing = tmp_path / "no_such_run"
    for folder in (str(missing), missing):
        with pytest.raises(FileNotFoundError, match="does not exist") as excinfo:
            load_samples_ucbmcmc(folder, 2)
        assert _names(str(excinfo.value), missing), str(excinfo.value)


def _file_as_folder(tmp_path):
    """A file passed as the folder raises NotADirectoryError."""
    path = _write(tmp_path / "dimension_chain.dat.1", [[1.0, 2.0]])
    with pytest.raises(NotADirectoryError, match="is not a directory"):
        load_samples_ucbmcmc(path, 2)


def _glob_characters(tmp_path):
    """A folder named with glob metacharacters reads its own files, not its siblings'."""
    for offset, name in enumerate(("run1", "run1x", "run[1]*")):
        folder = tmp_path / name
        folder.mkdir()
        np.savetxt(folder / "dimension_chain.dat.2", np.full((6, 2), 100.0 + offset if offset < 2 else 1.0))
    for loader in (load_samples, load_samples_ucbmcmc):
        pc = loader(str(tmp_path / "run[1]*"), num_params_per_source=2)
        assert pc.shape == (3, 2, 2)
        np.testing.assert_array_equal(np.unique(pc.chain), [1.0])


BAD_FOLDERS = [_wrong_column_count, _partial_sample, _missing_folder, _file_as_folder, _glob_characters]


@pytest.mark.parametrize("case", BAD_FOLDERS, ids=[case.__name__.strip("_") for case in BAD_FOLDERS])
def test_ucbmcmc_rejects_bad_files_and_folders_and_reads_names_literally(tmp_path, case):
    """Each bad file or folder is refused as its case function states; a folder name is never a glob."""
    case(tmp_path)
