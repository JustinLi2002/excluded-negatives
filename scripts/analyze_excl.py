"""Summarise excl_experiment.py / excl_moltrans.py / excl_biot5.py predictions.

Per model and seed: AUROC_O (the benchmark test set's positives vs its own
negatives), AUROC_U (the same positives vs the excluded negatives), and the
W-weighted mixture (1 - W) * AUROC_O + W * AUROC_U. The mixture is exact
arithmetic for AUROC once the positive reference is fixed, but it is a mixture
of these two empirical samples, not an estimate of deployment performance: W
comes from a current rebuild of BindingDB over the benchmark's proteins while O
is the old benchmark's test subset and U is additionally filtered, so the three
are not tied to one declared target population. Report it as a scenario.

Pairs are all pairs of models present, oriented so that Delta_O > 0. With
Delta_O > 0 the mixture changes sign exactly when Delta_U < -(1-W)/W * Delta_O,
which is the flip condition reported; c = |Delta_U - Delta_O| exceeding
c* = |Delta_O| / W is necessary for that but not sufficient, since c can also be
large because the excluded region favours the leading model.

Bootstrap: a drug-cluster bootstrap. One benchmark test drug is one cluster and
is resampled with ALL of its positive and negative pairs at once, so the
correlation between a drug's own positives and negatives is kept; the excluded
region's drugs (disjoint from the benchmark's by construction) are resampled as
their own clusters. Both models of a pair are scored on the same resample.
Weighted AUROC is computed from one presorted score vector per model, so the
cost per replicate is a few array operations. What this does NOT cover:
training randomness, uncertainty in W, the choice of U, and generalisation to
other proteins or assays. It is conditional on the trained models in hand.

Usage: python analyze_excl.py PRED_DIR W [N_BOOT=1000]
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score


def load(path):
    """(cluster, label, score) per set. The cluster is the drug's connectivity
    block when the file carries one, which is the unit the target population is
    split on; older files without ids fall back to the SMILES string."""
    rows = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            cluster = r.get("drug_block") or r.get("SMILES")
            rows[r["set"]].append((cluster, int(r["Y"]), float(r["score"])))
    return rows


class WeightedAUC:
    """AUROC of one fixed score vector under per-cluster integer weights.

    Scores are sorted once; a replicate only changes the weights. Ties get the
    usual half credit, so this agrees with sklearn's roc_auc_score at unit
    weights (checked in selftest()).
    """

    def __init__(self, pos_scores, pos_cluster, neg_scores, neg_cluster):
        s = np.concatenate([pos_scores, neg_scores])
        isp = np.concatenate([np.ones(len(pos_scores), bool), np.zeros(len(neg_scores), bool)])
        cl = np.concatenate([pos_cluster, neg_cluster])
        order = np.argsort(s, kind="mergesort")
        s, self.isp, cl = s[order], isp[order], cl[order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
        # positives and negatives index different weight vectors, so gather apart
        self.pi, self.ni = np.flatnonzero(self.isp), np.flatnonzero(~self.isp)
        self.pc, self.nc = cl[self.pi], cl[self.ni]
        self.n = len(s)

    def __call__(self, w_pos, w_neg):
        w = np.empty(self.n)
        w[self.pi] = w_pos[self.pc]
        w[self.ni] = w_neg[self.nc]
        gp = np.add.reduceat(np.where(self.isp, w, 0.0), self.starts)
        gn = np.add.reduceat(np.where(self.isp, 0.0, w), self.starts)
        tp, tn = gp.sum(), gn.sum()
        if tp == 0 or tn == 0:
            return np.nan
        below = np.r_[0.0, np.cumsum(gn)[:-1]]
        return float(np.sum(gp * (below + 0.5 * gn)) / (tp * tn))


def prepare(rows):
    """Flatten one model's predictions into cluster-indexed arrays."""
    o_rows = rows["test"]
    u_rows = [r for r in rows["U"] if r[1] == 0]
    o_drugs = sorted({d for d, _, _ in o_rows})
    u_drugs = sorted({d for d, _, _ in u_rows})
    oi = {d: i for i, d in enumerate(o_drugs)}
    ui = {d: i for i, d in enumerate(u_drugs)}
    pos = [(s, oi[d]) for d, y, s in o_rows if y == 1]
    neg_o = [(s, oi[d]) for d, y, s in o_rows if y == 0]
    neg_u = [(s, ui[d]) for d, _, s in u_rows]

    def arr(xs):
        return np.array([x[0] for x in xs]), np.array([x[1] for x in xs], dtype=np.intp)

    ps, pc = arr(pos)
    nos, noc = arr(neg_o)
    nus, nuc = arr(neg_u)
    return {
        "n_o": len(o_drugs), "n_u": len(u_drugs),
        # identity, not just counts: two models must have scored the same rows
        "o_drugs": tuple(o_drugs), "u_drugs": tuple(u_drugs),
        "labels_o": tuple(y for _, y, _ in o_rows), "labels_u": tuple(y for _, y, _ in u_rows),
        "auc_o": WeightedAUC(ps, pc, nos, noc),
        "auc_u": WeightedAUC(ps, pc, nus, nuc),
        "y_o": np.r_[np.ones(len(ps)), np.zeros(len(nos))], "s_o": np.r_[ps, nos],
        "y_u": np.r_[np.ones(len(ps)), np.zeros(len(nus))], "s_u": np.r_[ps, nus],
    }


def point(p, w):
    one_o, one_u = np.ones(p["n_o"]), np.ones(p["n_u"])
    a_o = p["auc_o"](one_o, one_o)
    a_u = p["auc_u"](one_o, one_u)
    return a_o, a_u, (1 - w) * a_o + w * a_u


def selftest(p):
    """The weighted AUROC at unit weights must equal sklearn's."""
    for tag in ("o", "u"):
        mine = point(p, 0.0)[0 if tag == "o" else 1]
        ref = roc_auc_score(p[f"y_{tag}"], p[f"s_{tag}"])
        assert abs(mine - ref) < 1e-9, (tag, mine, ref)


def boot(pa, pb, w, n, rng):
    """Resample drug clusters; score both models on the same resample."""
    n_o, n_u = pa["n_o"], pa["n_u"]
    assert (n_o, n_u) == (pb["n_o"], pb["n_u"]), "the two models scored different sets"
    for key in ("o_drugs", "u_drugs", "labels_o", "labels_u"):
        assert pa[key] == pb[key], f"the two models disagree on {key}, so they are not comparable"
    out = np.empty((n, 3))
    for k in range(n):
        wo = rng.multinomial(n_o, np.full(n_o, 1.0 / n_o)).astype(float)
        wu = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u)).astype(float)
        va, vb = [], []
        for p, v in ((pa, va), (pb, vb)):
            a_o = p["auc_o"](wo, wo)
            a_u = p["auc_u"](wo, wu)
            v += [a_o, a_u, (1 - w) * a_o + w * a_u]
        out[k] = [va[i] - vb[i] for i in range(3)]
    return out


def main(pred_dir, w, n_boot=1000):
    w, n_boot = float(w), int(n_boot)
    by_seed = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(pred_dir, "preds_*_seed*.tsv"))):
        m = re.search(r"preds_(\w+)_seed(\d+)\.tsv", os.path.basename(f))
        by_seed[int(m.group(2))][m.group(1)] = prepare(load(f))
    rng = np.random.default_rng(20261017)
    checked = False
    print(f"W = {w}   (mixture is a scenario over these empirical O and U samples)")
    print("seed\tmodel\tAUROC_O\tAUROC_U\tAUROC_mix\tO_drugs\tU_drugs")
    per_model, summary = defaultdict(list), []
    for seed, models in sorted(by_seed.items()):
        for name, p in sorted(models.items()):
            if not checked:
                selftest(p)
                checked = True
            a = point(p, w)
            per_model[name].append(a)
            print(f"{seed}\t{name}\t{a[0]:.4f}\t{a[1]:.4f}\t{a[2]:.4f}\t{p['n_o']}\t{p['n_u']}")
        order = sorted(models, key=lambda m: -point(models[m], w)[0])
        for a_name, b_name in combinations(order, 2):
            pa, pb = models[a_name], models[b_name]
            d = [point(pa, w)[i] - point(pb, w)[i] for i in range(3)]
            b = boot(pa, pb, w, n_boot, rng)
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            summary.append((seed, f"{a_name}-{b_name}", d, lo, hi))
    print("\nmean over seeds\tmodel\tAUROC_O\tAUROC_U\tAUROC_mix\tn_seeds")
    for name, vals in sorted(per_model.items()):
        m = np.mean(vals, axis=0)
        print(f"\t{name}\t{m[0]:.4f}\t{m[1]:.4f}\t{m[2]:.4f}\t{len(vals)}")
    print("\nDelta = A - B, 95% drug-cluster bootstrap (clusters carry a drug's own"
          " positives and negatives); c = |Delta_U - Delta_O|, c* = |Delta_O| / W;"
          " flips when (1-W)*Delta_O + W*Delta_U changes sign")
    print("seed\tpair\tDelta_O\tDelta_U\tDelta_mix\tc\tc*\tmix_flips")
    for seed, pair, d, lo, hi in summary:
        c, cs = abs(d[1] - d[0]), abs(d[0]) / w
        flips = np.sign(d[2]) != np.sign(d[0])
        print(f"{seed}\t{pair}\t" + "\t".join(f"{d[i]:+.4f} [{lo[i]:+.4f}, {hi[i]:+.4f}]"
                                             for i in range(3))
              + f"\t{c:.4f}\t{cs:.4f}\t{flips}")


if __name__ == "__main__":
    main(*sys.argv[1:])
