"""How often do two BindingDB InChI Keys render as the same canonical SMILES?
A model only ever sees the SMILES, so two such keys are the same input and must
not be treated as different drugs.
Usage: python diag_identity.py TARGET_DIR"""
import sys
import pandas as pd
d = pd.read_csv(f"{sys.argv[1]}/drugs.tsv", sep="\t")
p = pd.read_csv(f"{sys.argv[1]}/pairs.tsv", sep="\t")
g = d.groupby("canonical_smiles")
sizes = g.size()
dup = sizes[sizes > 1]
print(f"drug keys {len(d)}, distinct canonical SMILES {len(sizes)}")
print(f"SMILES shared by more than one key: {len(dup)} (covering {int(dup.sum())} keys)")
mixed = g["eligible"].nunique()
print(f"  of those, SMILES whose keys disagree on eligibility: {int((mixed > 1).sum())}")
key2smi = dict(zip(d["drug_key"], d["canonical_smiles"]))
p = p[p["region"] != "outside"].copy()
p["smi"] = p["drug_key"].map(key2smi)
n = len(p)
p["pid2"] = p["smi"] + "|" + p["protein_hash"]
collide = p.groupby("pid2").agg(keys=("drug_key", "nunique"), labels=("label", "nunique"))
c = collide[collide["keys"] > 1]
print(f"scored pairs {n}; (SMILES, protein) cells reached by >1 key: {len(c)}; "
      f"of those with conflicting labels: {int((c['labels'] > 1).sum())}")
reg = p.groupby("pid2")["region"].nunique()
print(f"  cells spanning both O and U: {int((reg > 1).sum())}")
