"""Support check for pair-labelled DTI benchmarks (DrugBAN / MolTrans releases).

For each CSV set (SMILES, Protein, Y): count entities, ask whether every entity
that appears in a negative also appears in a positive (an implicit T = 1 rule),
and tabulate negatives per entity against positives per entity.

Usage: python -I dti_support.py NAME file1.csv [file2.csv ...]
"""
import csv
import sys
from collections import Counter


def load(paths):
    rows = set()
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                d = r.get("DrugBank ID") or r["SMILES"]
                t = r.get("Gene") or r.get("Protein") or r["Target Sequence"]
                y = r.get("Y") if "Y" in r else r["Label"]
                rows.add((d, t, int(float(y))))
    return rows


def bins(x):
    if x == 0:
        return "0"
    if x == 1:
        return "1"
    if x <= 3:
        return "2-3"
    if x <= 10:
        return "4-10"
    return ">10"


def report(name, rows):
    pairs = {(d, t) for d, t, _ in rows}
    labels = Counter((d, t) for d, t, _ in rows)
    conflict = sum(1 for k, v in labels.items() if v > 1)
    pos = {(d, t) for d, t, y in rows if y == 1}
    neg = {(d, t) for d, t, y in rows if y == 0}
    print(f"### {name}")
    print(f"pairs {len(pairs)}  pos {len(pos)}  neg {len(neg)}  conflicting-label pairs {conflict}")
    for side, idx in (("drug", 0), ("target", 1)):
        ent_pos = Counter(p[idx] for p in pos)
        ent_neg = Counter(p[idx] for p in neg)
        allent = set(ent_pos) | set(ent_neg)
        only_neg = [e for e in allent if e not in ent_pos]
        only_pos = [e for e in allent if e not in ent_neg]
        print(f"  {side}: total {len(allent)}  with>=1 pos {len(ent_pos)}  "
              f"neg-only {len(only_neg)}  pos-only {len(only_pos)}")
        # negatives per entity by positive-count bin
        table = {}
        for e in allent:
            b = bins(ent_pos.get(e, 0))
            n, s, z = table.get(b, (0, 0, 0))
            table[b] = (n + 1, s + ent_neg.get(e, 0), z + (ent_neg.get(e, 0) == 0))
        for b in ("0", "1", "2-3", "4-10", ">10"):
            if b in table:
                n, s, z = table[b]
                print(f"    pos={b:>5}: entities {n:6d}  mean neg {s / n:7.2f}  no neg {z / n:6.1%}")
    # degree-matched? correlation of pos and neg counts per drug
    for side, idx in (("drug", 0), ("target", 1)):
        ent_pos = Counter(p[idx] for p in pos)
        ent_neg = Counter(p[idx] for p in neg)
        allent = sorted(set(ent_pos) | set(ent_neg))
        equal = sum(1 for e in allent if ent_pos.get(e, 0) == ent_neg.get(e, 0))
        print(f"  {side}: entities with neg count == pos count {equal}/{len(allent)}")


if __name__ == "__main__":
    report(sys.argv[1], load(sys.argv[2:]))
