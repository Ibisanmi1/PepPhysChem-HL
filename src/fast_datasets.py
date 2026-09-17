import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import sys

from .modelling_filters import filter_half_life_dataframe


class FastPeptideDataset(Dataset):
    """
    Fast dataset using simple tokenization (like IAM_ADMET_AI).
    No slow physicochemical feature extraction - just fast tokenization.
    """

    def __init__(self, csv_path: str, sequence_col: str = 'sequence',
                 label_col: str = 'target', max_length: int = 100):
        """
        Args:
            csv_path: Path to CSV file
            sequence_col: Name of sequence column
            label_col: Name of target column
            max_length: Maximum sequence length
        """

        if isinstance(csv_path, str):
            self.df = pd.read_csv(csv_path)
        elif isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.copy()
        else:
            raise ValueError("csv_path must be either a string or pandas DataFrame")

        assert sequence_col in self.df.columns and label_col in self.df.columns

        self.df = filter_half_life_dataframe(
            self.df, sequence_col=sequence_col, max_length=max_length
        )

        self.sequences = self.df[sequence_col].astype(str).str.strip().str.upper().tolist()
        self.y = self.df[label_col].values.astype(np.float32)
        self.max_length = max_length


        self.aa_to_idx = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.vocab_size = len(self.aa_to_idx) + 1

        print(f"  Loaded {len(self.sequences)} sequences", file=sys.stderr, flush=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """Get item with fast tokenization (like IAM_ADMET_AI)."""
        seq = self.sequences[idx]
        y = self.y[idx]



        input_ids = [self.aa_to_idx.get(aa, 0) + 1 for aa in seq.upper()]
        attention_mask = [1] * len(input_ids)


        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:

            input_ids.extend([0] * (self.max_length - len(input_ids)))
            attention_mask.extend([0] * (self.max_length - len(attention_mask)))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(y, dtype=torch.float32),
        }


def fast_pad_batch(batch: List[Dict]):
    """Simple batching for transformer inputs (like IAM_ADMET_AI)."""
    if not batch:
        return None


    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
    labels = torch.stack([item['labels'] for item in batch], dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }














