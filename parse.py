from rdkit import Chem

supplier = Chem.SDMolSupplier("Compound_000000001_000500000.sdf")
mols = [m for m in supplier if m is not None]

for mol in mols[:5]:
    print(Chem.MolToSmiles(mol))    
