import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# Read SMILES strings from .smi file
with open('qm9_5k.smi', 'r') as f:
    smiles_list = f.read().splitlines()

# Generate molecular descriptors using RDKit
descriptors_list = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    descriptors = np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2))
    descriptors_list.append(descriptors)

# Convert list of descriptors to numpy array
descriptors_array = np.array(descriptors_list)

# Save numpy array to .npy file
np.save('qm9_5k.npy', descriptors_array)
