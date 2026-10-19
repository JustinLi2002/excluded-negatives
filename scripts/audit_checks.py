"""Independent checks of the 2026-10-17 audit's two main counts.

A) MolTrans's drug encoder falls back to a single token 0 for the WHOLE molecule
   when any subword is out of vocabulary (stream.py drug2emb_encoder's bare
   except). Counts how many pairs of the shared U sample that hits, for the old
   pool (raw BindingDB SMILES) and the canonical one.
B) The drug-disjoint split partitions on the SMILES string, so one molecule
   written two ways can land on both sides. Counts train/test overlap per seed
   at full InChIKey and at connectivity-block level.

Usage: run from the MolTrans checkout (./ESPF must resolve):
  python audit_checks.py FULL_CSV U_OLD_CSV U_CANON_CSV SCRIPTS_DIR
"""
import os
import sys

sys.path.insert(0, os.getcwd())      # the MolTrans checkout, for stream.py
sys.path.insert(1, sys.argv[4])      # our own scripts
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from excl_sets import split, u_sample
from stream import drug2emb_encoder

RDLogger.DisableLog("rdApp.*")


def falls_back(smi):
    i, mask = drug2emb_encoder(smi)
    return int(np.sum(mask)) == 1 and int(i[0]) == 0


def check_a(u_csv, tag):
    u = u_sample(u_csv, 20000)
    drugs = sorted(set(u["SMILES"]))
    bad = {s for s in drugs if falls_back(s)}
    pairs = sum(1 for s in u["SMILES"] if s in bad)
    print(f"A) {tag}: {len(u)} pairs, {len(drugs)} distinct drugs; "
          f"drugs hitting the fallback {len(bad)} ({len(bad)/len(drugs):.2%}), "
          f"pairs affected {pairs} ({pairs/len(u):.3%})")


def check_b(full_csv):
    full = pd.read_csv(full_csv)[["SMILES", "Protein", "Y"]]
    smis = sorted(full["SMILES"].unique())
    key, blk = {}, {}
    for s in smis:
        m = Chem.MolFromSmiles(s)
        k = Chem.MolToInchiKey(m) if m is not None else None
        key[s], blk[s] = k, (k[:14] if k else None)
    print(f"B) {len(smis)} distinct SMILES -> {len(set(key.values()))} InChIKeys, "
          f"{len(set(blk.values()))} connectivity blocks")
    for seed in range(5):
        sets = split(full, seed)
        tr, te = set(sets["train"]["SMILES"]), set(sets["test"]["SMILES"])
        ki = {key[s] for s in tr} & {key[s] for s in te}
        bi = {blk[s] for s in tr} & {blk[s] for s in te}
        print(f"   seed {seed}: train/test shared InChIKeys {len(ki)}, "
              f"shared connectivity blocks {len(bi)}")


if __name__ == "__main__":
    check_a(sys.argv[2], "old pool (raw SMILES)")
    check_a(sys.argv[3], "canonical pool")
    check_b(sys.argv[1])
