import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import sys

from .physicochemical_analyzer import PhysicochemicalAnalyzer


class EmbeddingPeptideDataset(Dataset):
    """
    Dataset using tokenized sequences for embedding layers.
    Returns integer indices that will be passed through an embedding layer.
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


        self.df['seq_len'] = self.df[sequence_col].astype(str).str.len()
        self.df = self.df[self.df['seq_len'] <= max_length]

        self.sequences = self.df[sequence_col].astype(str).tolist()
        self.y = self.df[label_col].values.astype(np.float32)
        self.max_length = max_length


        self.aa_to_idx = {aa: idx + 1 for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.vocab_size = len(self.aa_to_idx) + 1

        print(f"  Loaded {len(self.sequences)} sequences", file=sys.stderr, flush=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """Get item with tokenization for embedding layer."""
        seq = self.sequences[idx]
        y = self.y[idx]


        input_ids = [self.aa_to_idx.get(aa, 0) for aa in seq.upper()]
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


def embedding_collate_fn(batch: List[Dict]):
    """Collate function for embedding-based models."""
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


class EmbeddingPhysioChemDataset(Dataset):
    """
    Dataset for CNN-BiLSTM (embedding input) + physicochemical features (cnn_bilstm_physchem).
    Filters long sequences like EmbeddingPeptideDataset and precomputes
    physicochemical features for the remaining sequences.
    """

    def __init__(self, csv_path: str, sequence_col: str = 'sequence',
                 label_col: str = 'target', max_length: int = 100):
        if isinstance(csv_path, str):
            self.df = pd.read_csv(csv_path)
        elif isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.copy()
        else:
            raise ValueError("csv_path must be either a string or pandas DataFrame")

        assert sequence_col in self.df.columns and label_col in self.df.columns

        self.df['seq_len'] = self.df[sequence_col].astype(str).str.len()
        self.df = self.df[self.df['seq_len'] <= max_length].copy()

        self.sequences = self.df[sequence_col].astype(str).tolist()
        self.y = self.df[label_col].values.astype(np.float32)
        self.max_length = max_length
        self.aa_to_idx = {aa: idx + 1 for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.vocab_size = len(self.aa_to_idx) + 1

        analyzer = PhysicochemicalAnalyzer()
        self.physchem = [
            self._extract_physchem_features(analyzer, seq)
            for seq in self.sequences
        ]

        print(
            f"  Loaded {len(self.sequences)} sequences (cnn_bilstm + physchem)",
            file=sys.stderr,
            flush=True,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        y = self.y[idx]

        input_ids = [self.aa_to_idx.get(aa, 0) for aa in seq.upper()]
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
            'physchem': self.physchem[idx],
            'labels': torch.tensor(y, dtype=torch.float32),
        }

    @staticmethod
    def _extract_physchem_features(analyzer: PhysicochemicalAnalyzer, sequence: str) -> torch.Tensor:
        try:
            basic_props = analyzer.calculate_basic_properties(sequence)
            hydrophobicity = analyzer.calculate_hydrophobicity(sequence)
            charge = analyzer.calculate_charge_distribution(sequence)
            volume = analyzer.calculate_side_chain_volume(sequence)
            composition = analyzer.calculate_amino_acid_composition(sequence)
            amphipathicity = analyzer.calculate_amphipathicity(sequence)

            profile = {
                **basic_props,
                **hydrophobicity,
                **charge,
                **volume,
                **composition,
                **amphipathicity
            }

            feature_keys = [
                'length', 'molecular_weight', 'isoelectric_point',
                'aromaticity', 'instability_index', 'gravy',
                'helix_fraction', 'turn_fraction', 'sheet_fraction',
                'hydrophobicity_kd_mean', 'hydrophobicity_kd_sum',
                'percentage_hydrophobicity', 'net_charge_pH7',
                'basic_count', 'acidic_count', 'charged_count',
                'charge_density', 'side_chain_volume_total',
                'side_chain_volume_mean', 'hydrophobic_moment',
                'amphipathicity_index', 'amphipathic_patterns'
            ]

            for aa in 'ACDEFGHIKLMNPQRSTVWY':
                feature_keys.append(f'aa_{aa}_percent')

            feature_keys.extend([
                'hydrophobic_percent', 'hydrophilic_percent',
                'aromatic_percent', 'charged_percent'
            ])

            features = [profile.get(key, 0.0) for key in feature_keys]


            features.extend([0.0] * 7)

            return torch.tensor(features, dtype=torch.float32)
        except Exception:
            return torch.zeros(53, dtype=torch.float32)


def embedding_physchem_collate_fn(batch: List[Dict]):
    """Collate batches for cnn_bilstm + physchem (embedding + descriptor tensors)."""
    if not batch:
        return None

    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
    physchem = torch.stack([item['physchem'] for item in batch], dim=0)
    labels = torch.stack([item['labels'] for item in batch], dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'physchem': physchem,
        'labels': labels,
    }



