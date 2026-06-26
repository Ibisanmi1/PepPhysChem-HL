import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict, List, Union, Optional
import re
from collections import Counter


class PhysicochemicalAnalyzer:
    """
    Advanced physicochemical analyzer for therapeutic peptides.
    Calculates comprehensive physicochemical properties using validated scales and methods.
    """

    def __init__(self):
        """Initialize analyzer with validated physicochemical scales."""


        self.pKa = {
            'Nterm': 9.69,
            'Cterm': 2.34,
            'D': 3.65,
            'E': 4.25,
            'K': 10.53,
            'R': 12.48,
            'H': 6.00,
            'C': 8.18,
            'Y': 10.07
        }


        self.hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }


        self.eisenberg_scale = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }


        self.side_chain_volumes = {
            'A': 88.6,  'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
            'Q': 143.8, 'E': 138.4, 'G': 60.1,  'H': 153.2, 'I': 166.7,
            'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
            'S': 89.0,  'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
        }


        self.hydrophobic_aa = {'A', 'I', 'L', 'M', 'F', 'W', 'V', 'C'}
        self.hydrophilic_aa = {'R', 'N', 'D', 'Q', 'E', 'G', 'H', 'K', 'P', 'S', 'T', 'Y'}
        self.charged_aa = {'D', 'E', 'K', 'R', 'H'}
        self.aromatic_aa = {'F', 'W', 'Y', 'H'}
        self.basic_aa = {'K', 'R', 'H'}
        self.acidic_aa = {'D', 'E'}


        self.aa_to_smiles = {
            'A': 'C[C@H](N)C(=O)O',
            'R': 'N=C(N)NCCC[C@H](N)C(=O)O',
            'N': 'N[C@@H](CC(=O)N)C(=O)O',
            'D': 'N[C@@H](CC(=O)O)C(=O)O',
            'C': 'N[C@@H](CS)C(=O)O',
            'Q': 'N[C@@H](CCC(=O)N)C(=O)O',
            'E': 'N[C@@H](CCC(=O)O)C(=O)O',
            'G': 'NCC(=O)O',
            'H': 'N[C@@H](CC1=CN=CN1)C(=O)O',
            'I': 'CC[C@H](C)[C@H](N)C(=O)O',
            'L': 'CC(C)C[C@H](N)C(=O)O',
            'K': 'NCCCC[C@H](N)C(=O)O',
            'M': 'CSCC[C@H](N)C(=O)O',
            'F': 'N[C@@H](CC1=CC=CC=C1)C(=O)O',
            'P': 'N1[C@H](CCC1)C(=O)O',
            'S': 'N[C@@H](CO)C(=O)O',
            'T': 'C[C@H](O)[C@H](N)C(=O)O',
            'W': 'N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)O',
            'Y': 'N[C@@H](CC1=CC=C(O)C=C1)C(=O)O',
            'V': 'CC(C)[C@H](N)C(=O)O'
        }

    def _validate_sequence(self, sequence: str) -> str:
        """Validate and clean amino acid sequence."""
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence must be a non-empty string")

        sequence = sequence.strip().upper()
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_aa = set(sequence) - valid_aa

        if invalid_aa:
            raise ValueError(f"Invalid amino acids found: {invalid_aa}")

        return sequence

    def calculate_net_charge(self, sequence: str, ph: float = 7.0) -> float:
        """
        Calculate net charge at specified pH using Henderson-Hasselbalch equation.

        Args:
            sequence: Amino acid sequence
            ph: pH value (default: 7.0)

        Returns:
            Net charge at specified pH
        """
        sequence = self._validate_sequence(sequence)
        charge = 0.0


        fraction_protonated_nterm = 10**(self.pKa['Nterm']) / (10**(self.pKa['Nterm']) + 10**ph)
        charge += fraction_protonated_nterm


        fraction_deprotonated_cterm = 10**ph / (10**(self.pKa['Cterm']) + 10**ph)
        charge -= fraction_deprotonated_cterm


        for aa in sequence:
            if aa in self.pKa:
                if aa in ['D', 'E']:
                    fraction_deprotonated = 10**ph / (10**(self.pKa[aa]) + 10**ph)
                    charge -= fraction_deprotonated
                elif aa in ['K', 'R']:
                    fraction_protonated = 10**(self.pKa[aa]) / (10**(self.pKa[aa]) + 10**ph)
                    charge += fraction_protonated
                elif aa == 'H':
                    fraction_protonated = 10**(self.pKa[aa]) / (10**(self.pKa[aa]) + 10**ph)
                    charge += fraction_protonated
                elif aa in ['C', 'Y']:
                    fraction_deprotonated = 10**ph / (10**(self.pKa[aa]) + 10**ph)
                    charge -= fraction_deprotonated

        return charge

    def calculate_hydrophobicity(self, sequence: str) -> Dict[str, float]:
        """
        Calculate hydrophobicity metrics using multiple scales.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of hydrophobicity metrics
        """
        sequence = self._validate_sequence(sequence)


        kd_values = [self.hydrophobicity_scale[aa] for aa in sequence]
        kd_mean = np.mean(kd_values)
        kd_sum = np.sum(kd_values)


        hydrophobic_count = sum(1 for aa in sequence if aa in self.hydrophobic_aa)
        percentage_hydrophobicity = (hydrophobic_count / len(sequence) * 100) if len(sequence) > 0 else 0.0


        gravy = kd_mean

        return {
            'hydrophobicity_kd_mean': kd_mean,
            'hydrophobicity_kd_sum': kd_sum,
            'percentage_hydrophobicity': percentage_hydrophobicity,
            'gravy': gravy,
            'hydrophobic_count': hydrophobic_count
        }

    def calculate_hydrophobic_moment(self, sequence: str, angle_per_residue: float = 100.0) -> float:
        """
        Calculate hydrophobic moment using Eisenberg scale.
        Useful for detecting amphipathic helices.

        Args:
            sequence: Amino acid sequence
            angle_per_residue: Angle per residue in degrees (100° for alpha helix)

        Returns:
            Hydrophobic moment
        """
        sequence = self._validate_sequence(sequence)

        moment = 0.0
        for i, aa in enumerate(sequence):
            angle = np.radians(angle_per_residue * i)
            hydrophobicity = self.eisenberg_scale.get(aa, 0.0)
            moment += hydrophobicity * np.cos(angle)

        return moment / len(sequence) if len(sequence) > 0 else 0.0

    def calculate_side_chain_volume(self, sequence: str) -> Dict[str, float]:
        """
        Calculate side chain volume metrics.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of volume metrics
        """
        sequence = self._validate_sequence(sequence)

        volumes = [self.side_chain_volumes[aa] for aa in sequence]
        total_volume = np.sum(volumes)
        mean_volume = np.mean(volumes)

        return {
            'side_chain_volume_total': total_volume,
            'side_chain_volume_mean': mean_volume
        }

    def calculate_charge_distribution(self, sequence: str) -> Dict[str, float]:
        """
        Calculate charge distribution metrics.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of charge metrics
        """
        sequence = self._validate_sequence(sequence)

        basic_count = sum(1 for aa in sequence if aa in self.basic_aa)
        acidic_count = sum(1 for aa in sequence if aa in self.acidic_aa)
        charged_count = sum(1 for aa in sequence if aa in self.charged_aa)

        net_charge = self.calculate_net_charge(sequence)

        return {
            'basic_count': basic_count,
            'acidic_count': acidic_count,
            'charged_count': charged_count,
            'net_charge_pH7': net_charge,
            'charge_density': net_charge / len(sequence) if len(sequence) > 0 else 0.0
        }

    def calculate_amino_acid_composition(self, sequence: str) -> Dict[str, float]:
        """
        Calculate amino acid composition percentages.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of amino acid compositions
        """
        sequence = self._validate_sequence(sequence)

        total = len(sequence)
        if total == 0:
            return {}

        composition = {}
        aa_counts = Counter(sequence)

        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            composition[f'aa_{aa}_count'] = aa_counts.get(aa, 0)
            composition[f'aa_{aa}_percent'] = (aa_counts.get(aa, 0) / total) * 100


        composition['hydrophobic_percent'] = sum(composition[f'aa_{aa}_percent'] for aa in self.hydrophobic_aa)
        composition['hydrophilic_percent'] = sum(composition[f'aa_{aa}_percent'] for aa in self.hydrophilic_aa)
        composition['aromatic_percent'] = sum(composition[f'aa_{aa}_percent'] for aa in self.aromatic_aa)
        composition['charged_percent'] = sum(composition[f'aa_{aa}_percent'] for aa in self.charged_aa)

        return composition

    def calculate_basic_properties(self, sequence: str) -> Dict[str, float]:
        """
        Calculate basic physicochemical properties using BioPython.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of basic properties
        """
        sequence = self._validate_sequence(sequence)

        try:
            analysis = ProteinAnalysis(sequence)

            properties = {
                'length': len(sequence),
                'molecular_weight': analysis.molecular_weight(),
                'isoelectric_point': analysis.isoelectric_point(),
                'aromaticity': analysis.aromaticity(),
                'instability_index': analysis.instability_index(),
                'gravy': analysis.gravy(),
            }


            ss_fraction = analysis.secondary_structure_fraction()
            properties['helix_fraction'] = ss_fraction[0]
            properties['turn_fraction'] = ss_fraction[1]
            properties['sheet_fraction'] = ss_fraction[2]

            return properties

        except Exception as e:
            raise RuntimeError(f"Error calculating basic properties: {str(e)}")

    def calculate_rdkit_properties(self, sequence: str) -> Optional[Dict[str, float]]:
        """
        Calculate molecular properties using RDKit.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of RDKit properties or None if calculation fails
        """
        sequence = self._validate_sequence(sequence)

        try:

            smiles_parts = []
            for aa in sequence:
                if aa in self.aa_to_smiles:
                    smiles_parts.append(self.aa_to_smiles[aa])
                else:
                    return None

            smiles = '.'.join(smiles_parts)
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                return None

            properties = {
                'logP': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_h_donors': Descriptors.NumHDonors(mol),
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),
                'fraction_csp3': Descriptors.FractionCSP3(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol)
            }

            return properties

        except Exception as e:
            return None

    def calculate_amphipathicity(self, sequence: str) -> Dict[str, float]:
        """
        Calculate amphipathicity metrics.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of amphipathicity metrics
        """
        sequence = self._validate_sequence(sequence)

        hydrophobic_moment = self.calculate_hydrophobic_moment(sequence)


        hydrophobic_count = sum(1 for aa in sequence if aa in self.hydrophobic_aa)
        hydrophilic_count = sum(1 for aa in sequence if aa in self.hydrophilic_aa)
        amphipathicity_index = (hydrophobic_count - hydrophilic_count) / len(sequence) if len(sequence) > 0 else 0.0


        amphipathic_patterns = self._detect_amphipathic_patterns(sequence)

        return {
            'hydrophobic_moment': hydrophobic_moment,
            'amphipathicity_index': amphipathicity_index,
            'amphipathic_patterns': amphipathic_patterns
        }

    def _detect_amphipathic_patterns(self, sequence: str, window_size: int = 7) -> int:
        """Detect potential amphipathic patterns in the sequence."""
        pattern_count = 0
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            hydrophobic_count = sum(1 for aa in window if aa in self.hydrophobic_aa)
            hydrophilic_count = sum(1 for aa in window if aa in self.hydrophilic_aa)
            if hydrophobic_count >= 3 and hydrophilic_count >= 3:
                pattern_count += 1
        return pattern_count

    def calculate_comprehensive_profile(self, sequence: str) -> Dict:
        """
        Calculate comprehensive physicochemical profile.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary containing all physicochemical properties
        """
        sequence = self._validate_sequence(sequence)

        profile = {
            'sequence': sequence,
        }


        basic_props = self.calculate_basic_properties(sequence)
        profile.update(basic_props)


        hydrophobicity = self.calculate_hydrophobicity(sequence)
        profile.update(hydrophobicity)


        charge = self.calculate_charge_distribution(sequence)
        profile.update(charge)


        volume = self.calculate_side_chain_volume(sequence)
        profile.update(volume)


        composition = self.calculate_amino_acid_composition(sequence)
        profile.update(composition)


        amphipathicity = self.calculate_amphipathicity(sequence)
        profile.update(amphipathicity)


        rdkit_props = self.calculate_rdkit_properties(sequence)
        if rdkit_props:
            profile.update(rdkit_props)

        return profile

    def analyze_batch(self, sequences: List[str], progress: bool = True) -> pd.DataFrame:
        """
        Analyze multiple sequences with progress tracking.

        Args:
            sequences: List of amino acid sequences
            progress: Whether to show progress updates

        Returns:
            DataFrame containing physicochemical profiles
        """
        results = []
        failed_sequences = []

        if progress:
            print(f"Processing {len(sequences)} sequences...")

        for i, seq in enumerate(sequences):
            try:
                if progress and i % 100 == 0:
                    print(f"Processed {i}/{len(sequences)} sequences...")

                profile = self.calculate_comprehensive_profile(seq)
                results.append(profile)

            except Exception as e:
                if progress:
                    print(f"Failed to process sequence {i+1}: {str(e)}")
                failed_sequences.append((i, seq, str(e)))
                continue

        if failed_sequences and progress:
            print(f"Failed to process {len(failed_sequences)} sequences")

        return pd.DataFrame(results)


def main():
    """Example usage"""
    analyzer = PhysicochemicalAnalyzer()


    sequence = "KWKLFKKIGAVLKVL"

    profile = analyzer.calculate_comprehensive_profile(sequence)

    print("Comprehensive Physicochemical Profile:")
    print("=" * 50)
    for key, value in profile.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

