import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
import pickle
import os
import sys

from .physicochemical_analyzer import PhysicochemicalAnalyzer
from .modelling_filters import filter_sequences_and_targets


class PhysioChemDataset(Dataset):
    """
    PyTorch Dataset for physicochemical property prediction.
    Optimized for fast loading - pre-computes features upfront.
    """

    def __init__(self, sequences: List[str], targets: np.ndarray,
                 feature_type: str = 'physchem', max_length: int = 100,
                 precompute_features: bool = True):
        """
        Args:
            sequences: List of amino acid sequences
            targets: Target values (numpy array)
            feature_type: Type of features ('physchem', 'onehot', 'both')
            max_length: Maximum sequence length for padding
            precompute_features: If True, pre-compute all features upfront (faster training)
        """
        self.sequences = sequences
        self.targets = targets
        self.feature_type = feature_type
        self.max_length = max_length


        self.aa_to_idx = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.num_aa = 20


        if precompute_features and feature_type in ['physchem', 'both']:
            self.analyzer = PhysicochemicalAnalyzer()
            self._precomputed_physchem = self._precompute_physchem_features(sequences)
        else:
            self.analyzer = None
            self._precomputed_physchem = None


        if feature_type in ['onehot', 'both']:
            self._precomputed_onehot = self._precompute_onehot_features(sequences)
        else:
            self._precomputed_onehot = None

    def _precompute_physchem_features(self, sequences: List[str]) -> List[torch.Tensor]:
        """Pre-compute all physicochemical features upfront - optimized for speed."""
        features = []
        total = len(sequences)
        for i, seq in enumerate(sequences):
            if i % 100 == 0 and i > 0:
                print(f"  Progress: {i}/{total} ({100*i/total:.1f}%)", flush=True)
            try:
                physchem = self._extract_physchem_features(seq)
                features.append(physchem)
            except Exception as e:

                features.append(torch.zeros(53))
        return features

    def _precompute_onehot_features(self, sequences: List[str]) -> List[torch.Tensor]:
        """Pre-compute all one-hot encodings upfront."""

        features = []
        for seq in sequences:
            onehot = self._sequence_to_onehot(seq)
            features.append(onehot)
        return features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get item - uses pre-computed features for speed."""
        target = self.targets[idx]
        features = {}

        if self.feature_type in ['onehot', 'both']:

            if self._precomputed_onehot is not None:
                features['sequence'] = self._precomputed_onehot[idx]
            else:
                features['sequence'] = self._sequence_to_onehot(self.sequences[idx])

        if self.feature_type in ['physchem', 'both']:

            if self._precomputed_physchem is not None:
                features['physchem'] = self._precomputed_physchem[idx]
            else:

                if self.analyzer is None:
                    self.analyzer = PhysicochemicalAnalyzer()
                features['physchem'] = self._extract_physchem_features(self.sequences[idx])

        return features, torch.FloatTensor([target])

    def _sequence_to_onehot(self, sequence: str) -> torch.Tensor:
        """Convert sequence to one-hot encoding."""

        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]


        onehot = np.zeros((self.max_length, self.num_aa))
        for i, aa in enumerate(sequence):
            if aa in self.aa_to_idx:
                onehot[i, self.aa_to_idx[aa]] = 1

        return torch.FloatTensor(onehot).transpose(0, 1)

    def _extract_physchem_features(self, sequence: str) -> torch.Tensor:
        """Extract physicochemical features (optimized - skip RDKit for speed)."""
        try:


            basic_props = self.analyzer.calculate_basic_properties(sequence)
            hydrophobicity = self.analyzer.calculate_hydrophobicity(sequence)
            charge = self.analyzer.calculate_charge_distribution(sequence)
            volume = self.analyzer.calculate_side_chain_volume(sequence)
            composition = self.analyzer.calculate_amino_acid_composition(sequence)
            amphipathicity = self.analyzer.calculate_amphipathicity(sequence)


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


            features = []
            for key in feature_keys:
                if key in profile:
                    features.append(profile[key])
                else:
                    features.append(0.0)



            rdkit_keys = ['logP', 'tpsa', 'num_h_donors', 'num_h_acceptors',
                         'fraction_csp3', 'num_rotatable_bonds', 'num_aromatic_rings']
            for key in rdkit_keys:
                features.append(0.0)

            return torch.FloatTensor(features)

        except Exception as e:


            return torch.zeros(53)


class PhysioChemDataProcessor:
    """
    Data processor for physicochemical property prediction.
    Handles data loading, feature extraction, and preprocessing.
    """

    def __init__(self, feature_type: str = 'both', max_length: int = 100):
        """
        Args:
            feature_type: Type of features ('physchem', 'onehot', 'both')
            max_length: Maximum sequence length
        """
        self.feature_type = feature_type
        self.max_length = max_length
        self.analyzer = PhysicochemicalAnalyzer()
        self.scaler = None

    def load_data(self, file_path: str, sequence_col: str = 'sequence',
                  target_col: str = 'target') -> Tuple[List[str], np.ndarray]:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            sequence_col: Name of sequence column
            target_col: Name of target column

        Returns:
            Tuple of (sequences, targets)
        """
        df = pd.read_csv(file_path)

        sequences = df[sequence_col].tolist()
        targets = df[target_col].values.astype(np.float32)

        sequences, targets = filter_sequences_and_targets(
            sequences, targets, max_length=self.max_length
        )

        return sequences, targets

    def prepare_datasets(self, sequences: List[str], targets: np.ndarray,
                        test_size: float = 0.2, val_size: float = 0.1,
                        random_state: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train/val/test datasets.

        Args:
            sequences: List of sequences
            targets: Target values
            test_size: Test set size
            val_size: Validation set size (from training set)
            random_state: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """

        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, targets, test_size=test_size, random_state=random_state
        )


        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state
        )


        train_dataset = PhysioChemDataset(X_train, y_train, self.feature_type, self.max_length, precompute_features=True)
        val_dataset = PhysioChemDataset(X_val, y_val, self.feature_type, self.max_length, precompute_features=True)
        test_dataset = PhysioChemDataset(X_test, y_test, self.feature_type, self.max_length, precompute_features=True)

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset,
                          test_dataset: Dataset, batch_size: int = 32,
                          num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size
            num_workers: Number of worker processes

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader, test_loader

    def extract_features_batch(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract features for a batch of sequences.

        Args:
            sequences: List of sequences

        Returns:
            Dictionary of feature arrays
        """
        features = {
            'physchem': [],
            'onehot': []
        }

        for seq in sequences:

            try:
                profile = self.analyzer.calculate_comprehensive_profile(seq)
                physchem_features = self._profile_to_features(profile)
                features['physchem'].append(physchem_features)
            except:
                features['physchem'].append(np.zeros(50))


            onehot = self._sequence_to_onehot(seq)
            features['onehot'].append(onehot.numpy())


        features['physchem'] = np.array(features['physchem'])
        features['onehot'] = np.array(features['onehot'])

        return features

    def _profile_to_features(self, profile: Dict) -> np.ndarray:
        """Convert profile dictionary to feature array."""
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

        features = []
        for key in feature_keys:
            features.append(profile.get(key, 0.0))


        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            features.append(profile.get(f'aa_{aa}_percent', 0.0))


        features.extend([
            profile.get('hydrophobic_percent', 0.0),
            profile.get('hydrophilic_percent', 0.0),
            profile.get('aromatic_percent', 0.0),
            profile.get('charged_percent', 0.0)
        ])


        rdkit_keys = ['logP', 'tpsa', 'num_h_donors', 'num_h_acceptors',
                     'fraction_csp3', 'num_rotatable_bonds', 'num_aromatic_rings']
        for key in rdkit_keys:
            features.append(profile.get(key, 0.0))

        return np.array(features)

    def _sequence_to_onehot(self, sequence: str) -> torch.Tensor:
        """Convert sequence to one-hot encoding."""
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        aa_to_idx = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        onehot = np.zeros((self.max_length, 20))

        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                onehot[i, aa_to_idx[aa]] = 1

        return torch.FloatTensor(onehot).transpose(0, 1)


def collate_fn(batch):
    """
    Custom collate function for batching.
    Handles different feature types.
    """
    features_list, targets_list = zip(*batch)

    targets = torch.stack(targets_list)


    first_features = features_list[0]

    batched_features = {}

    if 'sequence' in first_features:

        sequences = [f['sequence'] for f in features_list]
        batched_features['sequence'] = torch.stack(sequences)

    if 'physchem' in first_features:

        physchem = [f['physchem'] for f in features_list]
        batched_features['physchem'] = torch.stack(physchem)

    return batched_features, targets


if __name__ == "__main__":

    processor = PhysioChemDataProcessor(feature_type='both')


    sequences = ["KWKLFKKIGAVLKVL", "ACDEFGHIKLMNPQRSTVWY"]
    targets = np.array([0.8, 0.5])


    train_ds, val_ds, test_ds = processor.prepare_datasets(sequences, targets)


    train_loader, val_loader, test_loader = processor.create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=2
    )

    print("Data processing test successful!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

