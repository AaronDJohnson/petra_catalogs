"""The PosteriorChain container and its Feather file format.

A chain is stored as native float64 ``(n_samples, num_sources, num_params_per_source)``; an all-NaN row is an
absent source. Metadata is strict JSON under :data:`FEATHER_METADATA_KEY`; legacy padded-column files still read.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from petra.utils import check_int, find_prob_in_model, source_present

logger = logging.getLogger(__name__)

FEATHER_METADATA_KEY = b"petra_posterior_chain"
FEATHER_METADATA_VERSION = 1
_LEGACY_METADATA_COLUMNS = ("num_sources", "num_params_per_source",
                            "transdimensional", "prob_in_model")
_NON_FINITE_JSON = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


def _encode_cost_dict(cost_dict: dict, require_finite: bool = True) -> list[list[Any]]:
    """Return `cost_dict` as ``[key, value]`` pairs that strict JSON stores unchanged, or raise naming the entry."""
    encoded: list[list[Any]] = []
    for key, value in cost_dict.items():
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            json_key: int | str = int(key)
        elif isinstance(key, (float, np.floating)) and float(key).is_integer():
            json_key = int(key)
        elif isinstance(key, str):
            json_key = key
        else:
            raise ValueError(
                f"cost_dict key {key!r} ({type(key).__name__}) cannot be stored unchanged: "
                f"cost_dict is keyed by number of sources, so a key must be an integer (a "
                f"whole-number float such as 3.0 is stored as 3) or a string."
            )
        if value is not None and not isinstance(value, str):
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(
                    f"cost_dict[{key!r}] holds a value of type {type(value).__name__}, which cannot be stored "
                    f"unchanged: a cost must be a real number (an integer or a float, not a bool), "
                    f"None or a string."
                )
            value = float(value)
            if require_finite and not math.isfinite(value):
                raise ValueError(
                    f"cost_dict[{key!r}] is {value}, which strict JSON cannot represent. A cost "
                    f"is the negated mean log-likelihood of a labeling, so a non-finite one means "
                    f"the fit that scored it broke down; delete or replace the entry before "
                    f"saving the chain."
                )
        encoded.append([json_key, value])
    return encoded


@dataclass
class PosteriorChain:
    """Posterior samples of a variable number of sources, with the metadata of their current labeling.

    Parameters:
        chain: (N, S, P), (N, S*P) or flat samples of any float or integer dtype, or a masked array (masked = NaN).
        num_sources: number of source slots S.
        num_params_per_source: parameters per source P.
        trans_dimensional: whether the caller considers S variable; metadata no code in petra reads.
        prob_in_model: (S,) inclusion probability per slot, or None for not computed.
        cost_dict: cost of this labeling keyed by number of sources; None becomes {}.
        copy: store a copy; otherwise a native float64 input is stored without copying.
        validate_nan_convention: refuse a row mixing NaN and numbers; derived chains inherit the setting.

    Raises:
        ValueError: a dimension, the dtype, the shape, a partial-NaN row, prob_in_model or a cost_dict entry is invalid.
    """

    chain: np.ndarray
    num_sources: int
    num_params_per_source: int
    trans_dimensional: bool = False
    prob_in_model: np.ndarray | None = None
    cost_dict: dict = field(default_factory=dict)
    copy: bool = False
    validate_nan_convention: bool = True

    def __post_init__(self) -> None:
        """Validate the fields and reshape `chain` to ``(n_samples, num_sources, num_params_per_source)``."""
        if self.cost_dict is None:
            self.cost_dict = {}
        _encode_cost_dict(self.cost_dict, require_finite=False)

        num_sources = self.num_sources = check_int("num_sources", self.num_sources)
        num_params = self.num_params_per_source = check_int("num_params_per_source", self.num_params_per_source)
        if self.prob_in_model is not None:
            shape = np.asarray(self.prob_in_model).shape
            if shape != (num_sources,):
                raise ValueError(
                    f"prob_in_model has shape {shape}, but it holds one inclusion probability "
                    f"per source slot and is indexed by source label, so a chain with "
                    f"num_sources={num_sources} needs shape ({num_sources},). Pass "
                    f"prob_in_model=None, or recompute it from the samples with "
                    f"petra.utils.find_prob_in_model(chain, {num_sources}, eps=0); a Feather "
                    f"file whose stored array is this shape is repaired that way on read."
                )

        masked = isinstance(self.chain, np.ma.MaskedArray)
        chain = self.chain if masked else np.asarray(self.chain)
        if not (np.issubdtype(chain.dtype, np.floating) or np.issubdtype(chain.dtype, np.integer)):
            raise ValueError(
                f"chain must be a numeric array, got dtype {chain.dtype!r}; expected "
                f"a float or integer array of shape (n_samples, {num_sources}, "
                f"{num_params})."
            )
        if masked:
            # np.asarray would drop the mask and keep the placeholder under it as a present source.
            chain = np.ma.filled(self.chain.astype(np.float64), np.nan)
        elif not (chain.dtype.type is np.float64 and chain.dtype.isnative):
            # dtype.type, not ==: NumPy calls a 64-bit longdouble equal to float64, and pyarrow refuses it.
            chain = chain.astype(np.float64)
        if np.isinf(chain).any():
            raise ValueError("chain must not contain infinite values; use NaN for absent source rows.")

        entries_per_sample = num_sources * num_params
        if chain.ndim == 3:
            if chain.shape[1:] != (num_sources, num_params):
                raise ValueError(
                    f"chain has shape {chain.shape}, which is inconsistent with "
                    f"num_sources={num_sources} and "
                    f"num_params_per_source={num_params}; expected shape "
                    f"(n_samples, {num_sources}, {num_params})."
                )
        elif chain.ndim == 2:
            if chain.shape[1] != entries_per_sample:
                raise ValueError(
                    f"chain has shape {chain.shape}, but a two dimensional chain must have "
                    f"num_sources * num_params_per_source = {num_sources} * "
                    f"{num_params} = {entries_per_sample} columns; expected "
                    f"shape (n_samples, {entries_per_sample}) or (n_samples, "
                    f"{num_sources}, {num_params})."
                )
        elif chain.ndim == 1:
            if chain.size % entries_per_sample != 0:
                raise ValueError(
                    f"chain has shape {chain.shape} ({chain.size} values), which is not a "
                    f"multiple of num_sources * num_params_per_source = "
                    f"{num_sources} * {num_params} = "
                    f"{entries_per_sample}; expected shape (n_samples, "
                    f"{num_sources}, {num_params})."
                )
        else:
            raise ValueError(
                f"chain must have 1, 2, or 3 dimensions, got {chain.ndim}; expected shape "
                f"(n_samples, {num_sources}, {num_params})."
            )

        chain = chain.reshape(-1, num_sources, num_params)
        if self.validate_nan_convention:
            nan_count = np.isnan(chain).sum(axis=-1)
            partial = (nan_count != 0) & (nan_count != num_params)
            if partial.any():
                sample_index, source_index = (int(i) for i in np.argwhere(partial)[0])
                num_partial = int(partial.sum())
                raise ValueError(
                    f"chain violates petra's all-or-nothing NaN convention: sample "
                    f"{sample_index}, source {source_index} has {nan_count[sample_index, source_index]} "
                    f"NaN among its {num_params} parameters, so it is neither absent (all NaN) nor "
                    f"present (all finite). {num_partial} source row"
                    f"{'' if num_partial == 1 else 's'} "
                    f"{'is' if num_partial == 1 else 'are'} like this. A NaN row means \"this source "
                    f"is absent from this sample\". A partial one is absent to the readers that test "
                    f"the whole row (find_prob_in_model, the cost matrix and the multivariate fits) "
                    f"but present to the univariate warm start (petra.fits.uni_normal_fit), which reads one "
                    f"parameter at a time; the histogram labeler petra.initialization.label_chain_by_histogram "
                    f"leaves it out with a warning. Pass "
                    f"validate_nan_convention=False to accept the chain anyway; the chains "
                    f"PosteriorChain and petra.relabel derive from it inherit the setting."
                )
        self.chain = chain.copy() if self.copy else chain

    def __repr__(self) -> str:
        """Return the underlying array's repr, so a chain prints like its data."""
        return self.chain.view().__repr__()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the stored chain, ``(n_samples, num_sources, num_params_per_source)``."""
        return self.chain.shape

    def __getitem__(self, index: Any) -> np.ndarray:
        """Index the stored chain directly; basic indexing returns a view."""
        return self.chain[index]

    def __setitem__(self, index: Any, value: Any) -> None:
        """Assign into the stored chain, which may alias the caller's array."""
        self.chain[index] = value

    def get_chain(self, burn: int = 0, thin: int = 1) -> np.ndarray:
        """Return ``chain[burn::thin]``, a view; on a UCBMCMC chain a burn removes whole dimensions.

        Parameters:
            burn: samples to discard, an integer >= 0.
            thin: keep every thin-th sample, an integer >= 1.

        Returns:
            (ceil((N - burn) / thin), S, P) view of the chain.

        Raises:
            ValueError: burn or thin is a bool, not an integer, or out of range.
        """
        burn, thin = check_int("burn", burn, 0), check_int("thin", thin, 1)
        return self.chain[burn::thin]

    def get_valid_chain_entry(self, entry_index: int, burn: int = 0, thin: int = 1) -> np.ndarray:
        """Return the samples in which one source slot is present.

        Parameters:
            entry_index: source slot to extract.
            burn: samples to discard first, as in :meth:`get_chain`.
            thin: thinning interval, as in :meth:`get_chain`.

        Returns:
            (n_present, P) rows of that slot with no NaN.

        Raises:
            ValueError: burn or thin is invalid.
        """
        chain = self.get_chain(burn=burn, thin=thin)[:, entry_index, :]
        return chain[source_present(chain)]

    def expand_chain(self, max_num_sources: int) -> "PosteriorChain":
        """Return a copy widened to `max_num_sources` slots, the new ones absent in every sample.

        Parameters:
            max_num_sources: new slot count, an integer of at least num_sources.

        Returns:
            a trans-dimensional chain with an empty cost_dict and, if this one has one, prob_in_model recomputed.

        Raises:
            ValueError: max_num_sources is not an integer >= 1 or is below num_sources.
        """
        max_num_sources = check_int("max_num_sources", max_num_sources)
        if max_num_sources < self.num_sources:
            raise ValueError(
                f"max_num_sources ({max_num_sources}) cannot be less than the number of "
                f"entries in the chain ({self.num_sources})."
            )
        expanded_chain = np.full((self.chain.shape[0], max_num_sources, self.num_params_per_source),
                                 np.nan, dtype=np.float64)
        expanded_chain[:, :self.num_sources, :] = self.chain
        # Recomputed, not padded: a clipped or legacy input array would leave the widened one inconsistent.
        prob_in_model = (None if self.prob_in_model is None
                         else find_prob_in_model(expanded_chain, max_num_sources, eps=0))
        return PosteriorChain(expanded_chain, max_num_sources, self.num_params_per_source, True,
                              prob_in_model, {},
                              validate_nan_convention=self.validate_nan_convention)

    def randomize_entries(self, seed: int | None = None) -> "PosteriorChain":
        """Return a copy with the source slots of each sample shuffled; whole rows move.

        Parameters:
            seed: seed for numpy.random.default_rng.

        Returns:
            the shuffled chain, with an empty cost_dict and, if this one has one, prob_in_model recomputed.
        """
        rng = np.random.default_rng(seed=seed)
        chain = self.chain.copy()
        for i in range(self.chain.shape[0]):
            rng.shuffle(chain[i])
        # Each sample got its own permutation, so prob_in_model is recomputed and the cost of the old labeling dropped.
        prob_in_model = (None if self.prob_in_model is None
                         else find_prob_in_model(chain, self.num_sources, eps=0))
        return PosteriorChain(chain, self.num_sources, self.num_params_per_source,
                              self.trans_dimensional, prob_in_model, {},
                              validate_nan_convention=self.validate_nan_convention)

    def to_feather(self, feather_filepath: str | os.PathLike[str]) -> None:
        """Write the chain and its metadata to a Feather file.

        Parameters:
            feather_filepath: output path.

        Raises:
            ValueError: a cost_dict entry cannot be written unchanged, such as a non-finite cost; nothing is written.
        """
        feather.write_feather(self.to_arrow_table(), feather_filepath)
        logger.debug("Wrote PosteriorChain with shape %s to %s", self.chain.shape, feather_filepath)

    def to_arrow_table(self) -> pa.Table:
        """Return the Arrow table :meth:`to_feather` writes: columns "0".."S*P-1" and strict-JSON metadata.

        Returns:
            the table, with this chain's metadata under FEATHER_METADATA_KEY.

        Raises:
            ValueError: a cost_dict entry cannot be written unchanged.
        """
        prob_in_model: list[Any] | None = None
        if self.prob_in_model is not None:
            prob_in_model = [
                value if not isinstance(value, float) or math.isfinite(value)
                else "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
                for value in np.asarray(self.prob_in_model).tolist()
            ]
        metadata = {
            "version": FEATHER_METADATA_VERSION,
            "num_sources": int(self.num_sources),
            "num_params_per_source": int(self.num_params_per_source),
            "trans_dimensional": bool(self.trans_dimensional),
            "prob_in_model": prob_in_model,
            "cost_dict": _encode_cost_dict(self.cost_dict),
        }
        encoded_metadata = json.dumps(metadata, allow_nan=False).encode("utf-8")

        flat_chain = self.chain.reshape(self.chain.shape[0], self.num_sources * self.num_params_per_source)
        df = pd.DataFrame(flat_chain, columns=[str(i) for i in range(flat_chain.shape[1])])

        table = pa.Table.from_pandas(df, preserve_index=False)
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata[FEATHER_METADATA_KEY] = encoded_metadata
        return table.replace_schema_metadata(schema_metadata)

    @staticmethod
    def read_feather(feather_filepath: str | os.PathLike[str],
                     validate_nan_convention: bool = True) -> "PosteriorChain":
        """Read a chain written by :meth:`to_feather` or by the legacy padded-column writer.

        Parameters:
            feather_filepath: path to the file.
            validate_nan_convention: passed to the constructor; the file cannot switch the check off.

        Returns:
            the chain; a stored prob_in_model of the wrong shape is recomputed from the samples with a warning.

        Raises:
            ValueError: the file is not a PosteriorChain file, its columns do not fit its dimensions,
                or it holds a partial-NaN row while validate_nan_convention is True.
        """
        feather_filepath = os.fspath(feather_filepath)
        return PosteriorChain.from_arrow_table(feather.read_table(feather_filepath),
                                               feather_filepath, validate_nan_convention)

    @staticmethod
    def from_arrow_table(table: pa.Table, source: str,
                         validate_nan_convention: bool = True) -> "PosteriorChain":
        """Rebuild a chain from a table read out of a Feather file.

        Parameters:
            table: contents of a file written by :meth:`to_feather` or the legacy writer.
            source: where the table came from, named in errors and log messages.
            validate_nan_convention: passed to the constructor.

        Returns:
            the chain, as :meth:`read_feather` returns it.

        Raises:
            ValueError: as :meth:`read_feather`.
        """
        raw_metadata = (table.schema.metadata or {}).get(FEATHER_METADATA_KEY)
        df = table.to_pandas()
        if raw_metadata is None:
            required = ("num_sources", "num_params_per_source", "transdimensional")
            missing = [name for name in required if name not in df.columns]
            if missing:
                raise ValueError(
                    f"{source!r} is not a PosteriorChain Feather file: it has no petra "
                    f"schema metadata and is missing the legacy column(s) {missing}."
                )
            logger.info("Reading %s in the legacy format; cost_dict was never stored and will be "
                        "empty.", source)
            num_sources = int(df['num_sources'][0])
            num_params_per_source = int(df['num_params_per_source'][0])
            trans_dimensional = bool(df['transdimensional'][0])
            # The legacy column is NaN-padded to the sample count; num_sources says where the array ends,
            # so a genuine trailing NaN survives.
            stored: Any = (df['prob_in_model'].to_numpy()[:num_sources]
                           if 'prob_in_model' in df.columns else None)
            cost_dict: dict = {}
            df = df.drop(columns=[name for name in _LEGACY_METADATA_COLUMNS if name in df.columns])
        else:
            metadata = json.loads(raw_metadata.decode("utf-8"))
            num_sources = int(metadata["num_sources"])
            num_params_per_source = int(metadata["num_params_per_source"])
            trans_dimensional = bool(metadata["trans_dimensional"])
            stored = metadata["prob_in_model"]
            if stored is not None:
                stored = [_NON_FINITE_JSON.get(value, value) if isinstance(value, str) else value
                          for value in stored]
            cost_dict = {key: value for key, value in (metadata.get("cost_dict") or [])}

        chain = PosteriorChain(df.to_numpy(), num_sources, num_params_per_source, trans_dimensional,
                               None, cost_dict, validate_nan_convention=validate_nan_convention)
        if stored is not None:
            stored = np.asarray(stored)
            if stored.shape != (num_sources,):
                # A file is a run already on disk: the samples are intact and prob_in_model is a function of them.
                shape, stored = stored.shape, find_prob_in_model(chain.chain, num_sources, eps=0)
                logger.warning(
                    "%s stores a prob_in_model of shape %s for a chain of %d source slots, so it "
                    "cannot be indexed by source label as it stands; a short one is what "
                    "PosteriorChain.expand_chain wrote before it learned to widen the array. The "
                    "samples in the file are intact and prob_in_model is a function of them, so it "
                    "has been recomputed from them as %s. The chain, cost_dict and every other "
                    "field are unchanged, and the file itself is not rewritten; save the chain "
                    "again to fix it on disk.",
                    source, shape, num_sources, stored)
            chain.prob_in_model = stored
        return chain
