import re
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit.Chem import MolFromSmiles, Draw, rdFMCS
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.ChemUtils.AlignDepict import AlignDepict
from rdkit.DataStructs import DiceSimilarity
from rdkit.Chem import AllChem
import molvs


input_path = pathlib.Path(__file__).resolve().parent / 'input'

tautomer_canonicalizer = molvs.tautomer.TautomerCanonicalizer(max_tautomers=20)
fragments = molvs.fragment.REMOVE_FRAGMENTS + (
    molvs.fragment.FragmentPattern('tartrate', 'O=C(O)C(O)C(O)C(=O)O'),
)
fragment_remover = molvs.fragment.FragmentRemover(fragments)

def mol_to_smiles(mol):
    return AllChem.MolToSmiles(mol, isomericSmiles=True)

def get_fingerprint(mol):
    return AllChem.GetMorganFingerprint(mol, 2)


old_data = pd.read_csv(input_path / 'lsp_compounds_20180503.csv')
old_data.smiles.fillna('', inplace=True)
old_data['smiles_original'] = old_data.smiles.copy()
old_data['mol'] = old_data.smiles.map(MolFromSmiles)
old_data.mol = old_data.mol.map(tautomer_canonicalizer)
old_data.smiles = old_data.mol.map(mol_to_smiles)

new_data = pd.read_csv(input_path / 'UNC_smiles_codes.csv')
new_data.columns = new_data.columns.str.lower()
new_data['smiles_original'] = new_data.smiles.copy()
new_data.smiles.fillna('', inplace=True)
new_data['mol'] = new_data.smiles.map(MolFromSmiles)
new_data.mol = new_data.mol.map(fragment_remover)
new_data.mol = new_data.mol.map(tautomer_canonicalizer)
new_data.smiles = new_data.mol.map(mol_to_smiles)

merge_identical = pd.merge(new_data, old_data, on='smiles')

old_data['fingerprint'] = old_data.mol.map(get_fingerprint)
new_data['fingerprint'] = new_data.mol.map(get_fingerprint)
similarity = np.empty((len(old_data), len(new_data)))
for i, fp1 in enumerate(old_data.fingerprint):
    similarity[i] = new_data.fingerprint.map(
        lambda fp2: DiceSimilarity(fp1, fp2) if fp1 and fp2 else 0
    )

new_data['similarity'] = similarity.max(axis=0)
max_sim_idx = similarity.argmax(axis=0)
new_data['most_similar_mol'] = old_data.mol.iloc[max_sim_idx].values

plt.hist(new_data.similarity, bins=30)
