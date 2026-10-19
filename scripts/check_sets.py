"""Check that excl_sets.build reproduces the seed-0 sets used for the DrugBAN predictions.
Usage: python check_sets.py FULL_CSV U_POOL_CSV PREDS_TSV SEED"""
import csv, sys
from collections import Counter
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from excl_sets import build
full, upool, preds, seed = sys.argv[1:5]
sets = build(full, upool, int(seed), 20000)
got = {k: Counter(zip(sets[k]["SMILES"], sets[k]["Y"])) for k in ("val", "test", "U")}
old = {k: Counter() for k in got}
for r in csv.DictReader(open(preds), delimiter="\t"):
    old[r["set"]][(r["SMILES"], int(r["Y"]))] += 1
for k in got:
    print(k, "identical" if got[k] == old[k] else "DIFFERENT", sum(got[k].values()), sum(old[k].values()))
