"""Are the benchmark's SMILES RDKit-canonical, and how do the U pool's differ?
Usage: python check_smiles_form.py FULL_CSV U_POOL_CSV"""
import csv, sys
from collections import Counter
from rdkit import Chem, RDLogger
from excl_sets import u_sample
RDLogger.DisableLog("rdApp.*")

def report(name, smiles):
    n = len(smiles); cx = sum(1 for s in smiles if " " in s.strip())
    canon = same = 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        canon += 1
        if Chem.MolToSmiles(m) == s.strip(): same += 1
    print(f"{name}: {n} distinct SMILES, with CXSMILES extension {cx} ({cx/n:.1%}), "
          f"parseable {canon}, already RDKit-canonical {same} ({same/n:.1%})")

full = {r["SMILES"] for r in csv.DictReader(open(sys.argv[1], newline=""))}
report("benchmark full.csv", sorted(full))
u = u_sample(sys.argv[2], 20000)
report("U sample", sorted(set(u["SMILES"])))
