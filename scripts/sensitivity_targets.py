"""The two pre-registered protein sensitivity analyses.

Fixing the protein LIST did not fix the protein DISTRIBUTION: in E2, 17% of the
excluded negatives sit on proteins absent from the O test negatives and 2% on
proteins training never saw. So an O-U difference could partly be about which
proteins each region covers. These two analyses hold the protein side still, in
two different ways, and are reported alongside the headline rather than instead
of it.

A) Seen proteins only. Restrict BOTH regions to proteins that appear in the
   training set, then repeat the headline comparison. Same estimand as the
   headline, on a sub-population; the retained share of each region is reported.

B) Within-protein, pooled with equal protein weights. For every protein that
   carries at least M positives, M O negatives and M U negatives, compute each
   model's AUROC inside that protein separately for O and U, then average over
   proteins with equal weight. This is a DIFFERENT estimand - it asks whether a
   model ranks better within a target, and it drops protein composition
   entirely - so it is reported separately, with the share of the excluded
   region's negative mass that the qualifying proteins carry. The bootstrap unit
   is the protein, and per-protein AUROCs do not change when proteins are
   resampled, so they are computed once.

No mixture is reported for either analysis: the W of the headline refers to the
full declared population, and restricting the population changes that weight in
a way that is a property of the restriction, not of the rule.

Usage: python sensitivity_targets.py PRED_DIR MODE [M=3] [N_BOOT=1000]
  MODE  target | random | drugdisjoint   (how the training proteins are found)
"""
import glob
import hashlib
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_excl import boot, point, prepare
from excl_sets import build_sets, build_target

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "external", "DrugBAN", "datasets", "bindingdb", "full.csv")
U_POOL = os.path.join(ROOT, "results", "bindingdb", "U_pool_canon.csv")
TARGET = os.path.join(ROOT, "results", "target")


def phash(seq):
    return hashlib.sha1(seq.encode()).hexdigest()[:12]


def train_proteins(mode, seed):
    if mode == "target":
        return set(build_target(TARGET)["train"]["protein_hash"])
    sets = build_sets(BENCH, U_POOL, seed, 20000, mode)
    return {phash(p) for p in sets["train"]["Protein"]}


def load_rows(path):
    """(protein_hash, cluster, label, score) per set, keeping the protein."""
    import csv
    rows = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows[r["set"]].append((r["protein_hash"], r["drug_block"],
                                   int(r["Y"]), float(r["score"])))
    return rows


def restrict(rows, keep):
    out = {}
    for k, rs in rows.items():
        out[k] = [(c, y, s) for p, c, y, s in rs if p in keep]
    return out


def strip(rows):
    return {k: [(c, y, s) for _, c, y, s in rs] for k, rs in rows.items()}


def per_protein(rows, m):
    """Per-protein AUROC on O and on U, for proteins with enough of everything."""
    pos, neg_o, neg_u = defaultdict(list), defaultdict(list), defaultdict(list)
    for p, _, y, s in rows["test"]:
        (pos if y == 1 else neg_o)[p].append(s)
    for p, _, y, s in rows["U"]:
        if y == 0:
            neg_u[p].append(s)
    out, mass = {}, 0
    for p in sorted(set(pos) & set(neg_o) & set(neg_u)):
        if min(len(pos[p]), len(neg_o[p]), len(neg_u[p])) < m:
            continue
        y_o = np.r_[np.ones(len(pos[p])), np.zeros(len(neg_o[p]))]
        y_u = np.r_[np.ones(len(pos[p])), np.zeros(len(neg_u[p]))]
        out[p] = (roc_auc_score(y_o, np.r_[pos[p], neg_o[p]]),
                  roc_auc_score(y_u, np.r_[pos[p], neg_u[p]]))
        mass += len(neg_u[p])
    total_u = sum(len(v) for v in neg_u.values())
    return out, mass / total_u if total_u else 0.0


def main(pred_dir, mode, m=3, n_boot=1000):
    m, n_boot = int(m), int(n_boot)
    rng = np.random.default_rng(20261018)
    by_seed = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(pred_dir, "preds_*_seed*.tsv"))):
        g = re.search(r"preds_(\w+)_seed(\d+)\.tsv", os.path.basename(f))
        by_seed[int(g.group(2))][g.group(1)] = load_rows(f)

    print(f"=== {os.path.basename(pred_dir)}  mode={mode}  min per protein={m}")
    print("\nA) seen proteins only")
    print("seed\tmodel\tAUROC_O\tAUROC_U\tO_kept\tU_kept")
    kept_a = defaultdict(dict)
    for seed, models in sorted(by_seed.items()):
        keep = train_proteins(mode, seed)
        for name, rows in sorted(models.items()):
            full = prepare(strip(rows))
            sub = prepare(restrict(rows, keep))
            kept_a[seed][name] = sub
            o_frac = len(sub["s_o"]) / len(full["s_o"])
            u_frac = len(sub["s_u"]) / len(full["s_u"])
            a = point(sub, 0.0)
            print(f"{seed}\t{name}\t{a[0]:.4f}\t{a[1]:.4f}\t{o_frac:.1%}\t{u_frac:.1%}")
    print("\nseed\tpair\tDelta_O\tDelta_U\tgap_grows_on_U")
    for seed, models in sorted(kept_a.items()):
        order = sorted(models, key=lambda n: -point(models[n], 0.0)[0])
        for a_name, b_name in combinations(order, 2):
            pa, pb = models[a_name], models[b_name]
            d = [point(pa, 0.0)[i] - point(pb, 0.0)[i] for i in range(2)]
            b = boot(pa, pb, 0.0, n_boot, rng)
            lo, hi = np.percentile(b[:, :2], [2.5, 97.5], axis=0)
            print(f"{seed}\t{a_name}-{b_name}\t{d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]"
                  f"\t{d[1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]\t{d[1] > d[0]}")

    print(f"\nB) within-protein, equal protein weights (a different estimand)")
    print("seed\tmodel\tmean_AUROC_O\tmean_AUROC_U\tproteins\tU_mass_covered")
    per = defaultdict(dict)
    for seed, models in sorted(by_seed.items()):
        for name, rows in sorted(models.items()):
            tbl, mass = per_protein(rows, m)
            per[seed][name] = tbl
            o = np.array([v[0] for v in tbl.values()])
            u = np.array([v[1] for v in tbl.values()])
            print(f"{seed}\t{name}\t{o.mean():.4f}\t{u.mean():.4f}\t{len(tbl)}\t{mass:.1%}")
    print("\nseed\tpair\tDelta_O\tDelta_U\tgap_grows_on_U")
    for seed, models in sorted(per.items()):
        order = sorted(models, key=lambda n: -np.mean([v[0] for v in models[n].values()]))
        for a_name, b_name in combinations(order, 2):
            ta, tb = models[a_name], models[b_name]
            keys = sorted(set(ta) & set(tb))
            do = np.array([ta[k][0] - tb[k][0] for k in keys])
            du = np.array([ta[k][1] - tb[k][1] for k in keys])
            idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
            lo_o, hi_o = np.percentile(do[idx].mean(1), [2.5, 97.5])
            lo_u, hi_u = np.percentile(du[idx].mean(1), [2.5, 97.5])
            print(f"{seed}\t{a_name}-{b_name}\t{do.mean():+.4f} [{lo_o:+.4f}, {hi_o:+.4f}]"
                  f"\t{du.mean():+.4f} [{lo_u:+.4f}, {hi_u:+.4f}]\t{du.mean() > do.mean()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
