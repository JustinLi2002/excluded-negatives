"""Acceptance checks on the declared target population's evaluation sets.

Run before training anything on it. Every check is a property the design claims,
so a failure here means the claim is wrong, not that a model is bad.

Usage: python accept_target_sets.py TARGET_DIR ESPF_DIR
"""
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger

from biot5_input import selfies_of
from excl_sets import MAX_NODES, build_target
from moltrans_vocab import MolTransVocab

RDLogger.DisableLog("rdApp.*")
OK, BAD = "ok  ", "FAIL"
fails = 0


def check(name, cond, detail=""):
    global fails
    fails += not cond
    print(f"{OK if cond else BAD} {name}{'  ' + detail if detail else ''}")


def main(target_dir, espf_dir):
    sets = build_target(target_dir)
    summary = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(target_dir, "summary.tsv")))

    print("set sizes")
    for k in ("train", "val", "test", "U"):
        d = sets[k]
        print(f"     {k:5s} {len(d):7d} pairs  {int(d['Y'].sum()):7d} positive  "
              f"{d['drug_block'].nunique():7d} blocks  {d['protein_hash'].nunique():5d} proteins")

    blocks = {k: set(sets[k]["drug_block"]) for k in ("train", "val", "test", "U")}
    check("train/val blocks disjoint", not blocks["train"] & blocks["val"],
          f"{len(blocks['train'] & blocks['val'])} shared")
    check("train/test blocks disjoint", not blocks["train"] & blocks["test"],
          f"{len(blocks['train'] & blocks['test'])} shared")
    check("val/test blocks disjoint", not blocks["val"] & blocks["test"],
          f"{len(blocks['val'] & blocks['test'])} shared")

    u_neg = sets["U"][sets["U"]["Y"] == 0]
    u_blocks = set(u_neg["drug_block"])
    check("U blocks never seen in training", not u_blocks & blocks["train"],
          f"{len(u_blocks & blocks['train'])} shared")
    check("U blocks disjoint from the test blocks too", not u_blocks & blocks["test"],
          f"{len(u_blocks & blocks['test'])} shared")

    test_pos = set(sets["test"][sets["test"]["Y"] == 1]["pair_id"])
    u_pos = set(sets["U"][sets["U"]["Y"] == 1]["pair_id"])
    check("U reuses exactly the test positives", test_pos == u_pos,
          f"{len(test_pos)} vs {len(u_pos)}")
    check("no pair is both an O and a U negative",
          not set(sets["test"][sets["test"]["Y"] == 0]["pair_id"]) & set(u_neg["pair_id"]))
    for k in ("train", "val", "test", "U"):
        check(f"{k} pair ids unique", sets[k]["pair_id"].is_unique)

    check("U negatives match the declared sample size",
          len(u_neg) == int(summary["U_sample_size"]), f"{len(u_neg)}")
    check("drug ids resolve to one SMILES each",
          all(sets[k].groupby("drug_id")["SMILES"].nunique().eq(1).all()
              for k in ("train", "val", "test", "U")))
    check("U proteins are a subset of the declared protein set",
          set(u_neg["protein_hash"]) <= set(pd.read_csv(
              os.path.join(target_dir, "proteins.tsv"), sep="\t")["protein_hash"]))

    smis = sorted({s for k in ("train", "val", "test", "U") for s in sets[k]["SMILES"]})
    mols = [Chem.MolFromSmiles(s) for s in smis]
    check("every SMILES parses", all(m is not None for m in mols))
    check("every molecule within DrugBAN's node limit",
          all(m is not None and 0 < m.GetNumHeavyAtoms() <= MAX_NODES for m in mols))
    check("every molecule SELFIES-encodable", all(selfies_of(s) is not None for s in smis))
    vocab = MolTransVocab(espf_dir)
    check("every molecule inside MolTrans's vocabulary",
          all(vocab.drug_ok(s) for s in smis),
          "an out-of-vocabulary input is silently blanked for that model only")
    prots = sorted({p for k in ("train", "val", "test", "U") for p in sets[k]["Protein"]})
    check("every protein inside MolTrans's vocabulary", all(vocab.protein_ok(p) for p in prots))
    check("SMILES are canonical", all(m is not None and Chem.MolToSmiles(m) == s
                                      for s, m in zip(smis, mols)))

    w = float(summary["W_primary_block_disjoint_U"])
    o_neg_all = int(summary["negatives_O"])
    u_neg_all = int(summary["negatives_U_block_disjoint"])
    check("W equals the declared mass split",
          abs(w - u_neg_all / (o_neg_all + u_neg_all)) < 5e-5, f"W = {w}")
    print(f"\n{'all checks passed' if not fails else str(fails) + ' CHECK(S) FAILED'}; "
          f"W = {w}, O negatives {o_neg_all}, U negatives {u_neg_all}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
