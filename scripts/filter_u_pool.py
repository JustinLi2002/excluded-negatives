"""Drop U-pool drugs whose InChIKey connectivity block (first 14 chars) matches any
benchmark drug, so stereo/salt variants of benchmark drugs cannot enter U.
Usage: python filter_u_pool.py FULL_CSV U_POOL_CSV OUT_CSV"""
import csv, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
full, upool, out = sys.argv[1:4]
blocks = set()
for r in csv.DictReader(open(full, newline="")):
    blocks.add(Chem.MolToInchiKey(Chem.MolFromSmiles(r["SMILES"]))[:14])
n_in = n_out = 0
drugs_out = set()
with open(upool, newline="") as fi, open(out, "w", newline="") as fo:
    rd = csv.DictReader(fi); w = csv.DictWriter(fo, fieldnames=rd.fieldnames); w.writeheader()
    for r in rd:
        n_in += 1
        if r["ikey"][:14] in blocks:
            continue
        w.writerow(r); n_out += 1; drugs_out.add(r["ikey"])
print(f"benchmark connectivity blocks {len(blocks)}; U pairs {n_in} -> {n_out}; U drugs kept {len(drugs_out)}")
