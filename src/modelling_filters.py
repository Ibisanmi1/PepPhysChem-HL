"""Shared modelling-set filters for Table 1 / half-life loaders.

Protocol (fixed for all architectures):
  - drop sequences with length > max_length (default 100)
  - drop records with no usable sequence (null / empty / literal 'nan')

On the prepared Peplife split CSVs this yields 356 peptides
(train 247 / validation 37 / test 72).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def is_usable_sequence(seq) -> bool:
    """Return True if ``seq`` is a non-empty amino-acid string (not null/'nan')."""
    if seq is None:
        return False
    try:
        if pd.isna(seq):
            return False
    except (TypeError, ValueError):
        pass
    text = str(seq).strip()
    if not text:
        return False
    if text.lower() == "nan":
        return False
    return True


def filter_half_life_dataframe(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    max_length: int = 100,
) -> pd.DataFrame:
    """Drop unusable and over-long sequences; return a copy."""
    if sequence_col not in df.columns:
        raise KeyError(f"Missing sequence column: {sequence_col}")
    out = df.copy()
    usable = out[sequence_col].map(is_usable_sequence)
    lengths = out.loc[usable, sequence_col].astype(str).str.len()
    keep_idx = lengths[lengths <= max_length].index
    return out.loc[keep_idx].reset_index(drop=True)


def filter_sequences_and_targets(
    sequences: Sequence,
    targets: Union[Sequence, np.ndarray],
    max_length: int = 100,
) -> Tuple[List[str], np.ndarray]:
    """Filter parallel sequence/target lists with the modelling protocol."""
    kept_seq: List[str] = []
    kept_y: List[float] = []
    y_arr = np.asarray(targets)
    for seq, y in zip(sequences, y_arr):
        if not is_usable_sequence(seq):
            continue
        text = str(seq).strip().upper()
        if len(text) > max_length:
            continue
        kept_seq.append(text)
        kept_y.append(float(y))
    return kept_seq, np.asarray(kept_y, dtype=np.float32)
