from __future__ import annotations



from typing import Dict, List, Optional



from rdkit import Chem

from rdkit.Chem import Descriptors



AA_TO_SMILES = {

"A": "C[C@@H](C(=O)O)N",

"R": "C(CCNC(=N)N)C(C(=O)O)N",

"N": "C([C@@H](C(=O)O)N)C(=O)N",

"D": "C(C(=O)O)C(C(=O)O)N",

"C": "C(C(C(=O)O)N)S",

"Q": "C(CC(=O)N)C(C(=O)O)N",

"E": "C(CC(=O)O)C(C(=O)O)N",

"G": "C(C(=O)O)N",

"H": "C1=C(NC=N1)C(C(C(=O)O)N)C",

"I": "CC[C@H](C)C(C(=O)O)N",

"L": "CC(C)C[C@@H](C(=O)O)N",

"K": "C(CCN)C(C(=O)O)N",

"M": "CSCC[C@H](C(=O)O)N",

"F": "C1=CC=C(C=C1)C[C@@H](C(=O)O)N",

"P": "C1CCNC1C(=O)O",

"S": "C([C@@H](C(=O)O)N)O",

"T": "C[C@H](C(C(=O)O)N)O",

"W": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",

"Y": "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O",

"V": "CC(C)C[C@H](C(=O)O)N",

}



RDKIT_PHYSCHEM_KEYS: List[str] = [

"logP",

"rdkit_mol_wt",

"tpsa",

"formal_charge",

"num_h_donors",

"num_h_acceptors",

"num_rotatable_bonds",

]



PHYSCHEM_BASE_KEYS: List[str] = [

"length",

"molecular_weight",

"isoelectric_point",

"aromaticity",

"instability_index",

"gravy",

"helix_fraction",

"turn_fraction",

"sheet_fraction",

"hydrophobicity_kd_mean",

"hydrophobicity_kd_sum",

"percentage_hydrophobicity",

"net_charge_pH7",

"basic_count",

"acidic_count",

"charged_count",

"charge_density",

"side_chain_volume_total",

"side_chain_volume_mean",

"hydrophobic_moment",

"amphipathicity_index",

"amphipathic_patterns",

] + [f"aa_{aa}_percent" for aa in "ACDEFGHIKLMNPQRSTVWY"] + [

"hydrophobic_percent",

"hydrophilic_percent",

"aromatic_percent",

"charged_percent",

]



PHYSCHEM_VECTOR_DIM = len(PHYSCHEM_BASE_KEYS) + len(RDKIT_PHYSCHEM_KEYS)





def sequence_to_mol(sequence: str) -> Optional[Chem.Mol]:

    sequence = sequence.strip().upper()

parts = []

for aa in sequence:

        if aa not in AA_TO_SMILES:

            return None

parts.append(AA_TO_SMILES[aa])

return Chem.MolFromSmiles(".".join(parts))





def compute_rdkit_descriptor_dict(sequence: str) -> Dict[str, float]:

    zeros = {k: 0.0 for k in RDKIT_PHYSCHEM_KEYS}

mol = sequence_to_mol(sequence)

if mol is None:

        return zeros

try:

        return {

"logP": float(Descriptors.MolLogP(mol)),

"rdkit_mol_wt": float(Descriptors.MolWt(mol)),

"tpsa": float(Descriptors.TPSA(mol)),

"formal_charge": float(Chem.GetFormalCharge(mol)),

"num_h_donors": float(Descriptors.NumHDonors(mol)),

"num_h_acceptors": float(Descriptors.NumHAcceptors(mol)),

"num_rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),

}

except Exception:

        return zeros





def rdkit_vector_as_list(sequence: str) -> List[float]:

    d = compute_rdkit_descriptor_dict(sequence)

return [d[k] for k in RDKIT_PHYSCHEM_KEYS]





def build_physchem_vector_from_profile(profile: dict, sequence: str) -> List[float]:

    base = [float(profile.get(k, 0.0)) for k in PHYSCHEM_BASE_KEYS]

rdkit = rdkit_vector_as_list(sequence)

return base + rdkit

