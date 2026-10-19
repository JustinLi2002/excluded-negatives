"""Put the U pool's SMILES in the same form as the benchmark's.

The benchmark's own SMILES carry no CXSMILES extension and are 98.4% RDKit
canonical; the pool's come straight from BindingDB's Ligand SMILES column, so
15.3% carry a ` |r|`-style extension and only 33.8% are canonical. For a model
that reads the SMILES string (MolTrans, and BioT5 via SELFIES) that difference
is a property of the file, not of the excluded region, so it would show up as a
spurious part of any O-U gap. Rewriting each molecule as RDKit canonical SMILES
removes the extension and the formatting difference while keeping the molecule.

Exact duplicate (SMILES, Protein) pairs created by the rewrite are dropped, and
so are the few molecules SELFIES cannot represent (hypervalent iodine and the
like), so that every model scores exactly the same excluded negatives.

Usage: python canonicalise_u.py IN_CSV OUT_CSV
"""
import csv
import sys

from rdkit import Chem, RDLogger

from biot5_input import selfies_of

RDLogger.DisableLog("rdApp.*")


def main(src, dst):
    cache = {}
    seen = set()
    n = dropped = unparsed = dup = no_selfies = 0
    with open(src, newline="") as fi, open(dst, "w", newline="") as fo:
        rd = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=rd.fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rd:
            n += 1
            smi = r["SMILES"]
            if smi not in cache:
                m = Chem.MolFromSmiles(smi)
                cache[smi] = Chem.MolToSmiles(m) if m is not None else None
            canon = cache[smi]
            if canon is None:
                unparsed += 1
                dropped += 1
                continue
            if selfies_of(canon) is None:
                no_selfies += 1
                dropped += 1
                continue
            key = (canon, r["Protein"])
            if key in seen:
                dup += 1
                dropped += 1
                continue
            seen.add(key)
            r["SMILES"] = canon
            w.writerow(r)
    changed = sum(1 for k, v in cache.items() if v is not None and v != k)
    print(f"rows in {n}, out {n - dropped} (unparsed {unparsed}, "
          f"not SELFIES-encodable {no_selfies}, duplicate after rewrite {dup})")
    print(f"distinct input SMILES {len(cache)}, rewritten to a different string {changed}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
