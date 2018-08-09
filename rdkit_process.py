from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import DiceSimilarity
import numpy as np
import matplotlib.pyplot as plt


sanitize_flags = Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES

input_path = '/home/jmuhlich/Dropbox (HMS)/Shimada DTP overlap/input/Chem2D_Jun2016.sdf'

# Change this to SmilesMolSupplier or whatever you need for your data.
mol_supplier = Chem.SDMolSupplier(input_path, sanitize=False)

fingerprints = []
for mol in mol_supplier:
    try:
        Chem.SanitizeMol(mol, sanitize_flags)
    except ValueError as err:
        smiles = Chem.MolToSmiles(mol)
        print("ERROR! skipping molecule due to error: %s < %s >" % (err, smiles))
        pass
    fp = AllChem.GetMorganFingerprint(mol, 2)
    fingerprints.append(fp)

compare_fp = fingerprints[0]
similarity = [DiceSimilarity(compare_fp, fp2) for fp2 in fingerprints]

plt.hist(similarity, bins=200, log=True)
plt.show()
