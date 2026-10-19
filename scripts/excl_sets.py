"""Evaluation sets shared by every model in the BindingDB excluded-region experiment.

Two designs, chosen with `mode`:

  drugdisjoint  benchmark DRUGS shuffled with random.Random(seed), 70/10/20 into
                train / val / test pairs. Test drugs are unseen, so O and U are
                both out-of-sample and the O-U gap isolates the excluded region.
  random        the benchmark's own published pair-level split
                (datasets/bindingdb/random/{train,val,test}.csv), which is the
                protocol every published AUROC on this benchmark is measured
                under. Test drugs DO appear in training while U drugs never do,
                so an O-U gap here also contains a drug-memorisation component;
                the drug-only baseline measures how large that component can be.

u_sample(): the first n_u usable rows of the U pool after a fixed permutation
(numpy default_rng(0)), so every model, seed and design sees the same excluded
negatives. The U set is the test positives plus that sample.
"""
import csv
import hashlib
import os
import random

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
MAX_NODES = 290  # DrugBAN's max_drug_nodes; applied to every model so U is identical


def usable(smi):
    m = Chem.MolFromSmiles(smi)
    return m is not None and 0 < m.GetNumAtoms() <= MAX_NODES


def split(full, seed):
    drugs = sorted(full["SMILES"].unique())
    rng = random.Random(seed)
    rng.shuffle(drugs)
    n = len(drugs)
    tr, va = set(drugs[: int(0.7 * n)]), set(drugs[int(0.7 * n): int(0.8 * n)])
    part = full["SMILES"].map(lambda s: "train" if s in tr else "val" if s in va else "test")
    return {k: full[part == k].reset_index(drop=True) for k in ("train", "val", "test")}


def u_sample(u_csv, n_u):
    pool = pd.read_csv(u_csv)
    pool = pool.iloc[np.random.default_rng(0).permutation(len(pool))]
    keep = []
    for i, smi in zip(pool.index, pool["SMILES"]):
        if usable(smi):
            keep.append(i)
            if len(keep) >= n_u:
                break
    return pool.loc[keep, ["SMILES", "Protein", "Y"]].reset_index(drop=True)


def add_u(sets, u_csv, n_u):
    test_pos = sets["test"][sets["test"]["Y"] == 1]
    sets["U"] = pd.concat([test_pos, u_sample(u_csv, n_u)], ignore_index=True)
    return sets


def build(full_csv, u_csv, seed, n_u):
    full = pd.read_csv(full_csv)[["SMILES", "Protein", "Y"]]
    return add_u(split(full, seed), u_csv, n_u)


def build_random(full_csv, u_csv, n_u):
    """The benchmark's published random split; full_csv only locates its folder."""
    d = os.path.join(os.path.dirname(os.path.abspath(full_csv)), "random")
    sets = {k: pd.read_csv(os.path.join(d, f"{k}.csv"))[["SMILES", "Protein", "Y"]]
            for k in ("train", "val", "test")}
    return add_u(sets, u_csv, n_u)


def build_target(target_dir):
    """Sets from the declared target population built by build_target.py.

    O is the eligible drugs' pairs, already split 70/10/20 by connectivity block
    inside pairs.tsv; U is the fixed uniform sample of excluded negatives. The
    split is part of the declared design, so a model seed changes initialisation
    and batch order only. Returns frames carrying pair_id, drug_id, drug_block
    and protein_hash alongside the model inputs.
    """
    cols = ["pair_id", "drug_id", "drug_block", "protein_hash", "SMILES", "Protein", "Y"]
    drugs = pd.read_csv(os.path.join(target_dir, "drugs.tsv"), sep="\t").set_index("drug_id")
    prot = pd.read_csv(os.path.join(target_dir, "proteins.tsv"), sep="\t").set_index("protein_hash")
    pairs = pd.read_csv(os.path.join(target_dir, "pairs.tsv"), sep="\t")
    keep = (pairs["region"] == "O") | (pairs["split"] == "U_sample")
    pairs = pairs[keep].copy()                      # map sequences only for rows we score
    pairs["SMILES"] = pairs["drug_id"].map(drugs["canonical_smiles"])
    pairs["drug_block"] = pairs["drug_id"].map(drugs["drug_block"])
    pairs["Protein"] = pairs["protein_hash"].map(prot["sequence"])
    pairs["Y"] = pairs["label"]
    assert not pairs[["SMILES", "Protein"]].isna().any().any(), "a pair id did not resolve"
    o = pairs[pairs["region"] == "O"]
    sets = {k: o[o["split"] == k][cols].reset_index(drop=True) for k in ("train", "val", "test")}
    u_neg = pairs[pairs["split"] == "U_sample"][cols]
    sets["U"] = pd.concat([sets["test"][sets["test"]["Y"] == 1], u_neg], ignore_index=True)
    return sets


def build_sets(full_csv, u_csv, seed, n_u, mode="drugdisjoint"):
    if mode == "target":
        return build_target(full_csv)               # full_csv is the target directory here
    if mode == "random":
        return build_random(full_csv, u_csv, n_u)
    if mode == "drugdisjoint":
        return build(full_csv, u_csv, seed, n_u)
    raise ValueError(f"unknown split mode {mode!r}")


def write_preds(path, frames):
    """One prediction format for every model and design.

    Scores are written at full double precision, and each row carries stable ids
    so a prediction file can be joined back to the target tables and audited
    without trusting the row order. For the older designs, which have no id
    tables, the ids are derived from the strings so the format still holds.
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["set", "pair_id", "drug_id", "drug_block", "protein_hash",
                    "protein_len", "Y", "score"])
        for name, df, scores in frames:
            has = set(df.columns)
            for i, (smi, prot, y, sc) in enumerate(zip(df["SMILES"], df["Protein"],
                                                       df["Y"], scores)):
                if "pair_id" in has:
                    pid, dkey = df["pair_id"].iat[i], df["drug_id"].iat[i]
                    blk, phash = df["drug_block"].iat[i], df["protein_hash"].iat[i]
                else:
                    pid = hashlib.sha1(f"{smi}|{prot}".encode()).hexdigest()[:16]
                    dkey, blk = "", smi
                    phash = hashlib.sha1(prot.encode()).hexdigest()[:12]
                w.writerow([name, pid, dkey, blk, phash, len(prot), int(y), f"{sc:.17g}"])
