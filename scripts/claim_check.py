"""The statistics the paper's claims actually rest on, which the first pass did not compute.

Three gaps the 2026-10-18 outline review identified:

1. "the gap grows on the excluded region" was reported as a point-estimate boolean
   (Delta_U > Delta_O). The claim needs the PAIRED quantity Delta_U - Delta_O with its
   own interval, from the same bootstrap replicate, for the pooled and the within-protein
   estimands.
2. a single released checkpoint (BioT5) could only be paired against seed 0 of the trained
   models, because pairs are formed within a seed. Comparing it against every seed is what
   the claim about it needs.
3. the within-protein protein count and U-mass coverage vary by seed wherever the split does,
   so they must be carried per seed, not collapsed.
4. the within-protein estimand changes two things at once, the protein SUBSET and the
   WEIGHTING. A pooled analysis restricted to exactly the same subset isolates the second,
   so the three numbers together say which of the two moves the result.

Usage: python claim_check.py PRED_DIR MODE [M=3] [N_BOOT=2000]
"""
import glob
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score

from analyze_excl import prepare, point
from sensitivity_targets import load_rows, per_protein, strip


def restrict_prot(rows, keep):
    """Keep only the pairs whose protein is in `keep`, dropping the protein column."""
    return {k: [(c, y, sc) for p, c, y, sc in rs if p in keep] for k, rs in rows.items()}


def ci(x, lo=2.5, hi=97.5):
    a, b = np.percentile(x, [lo, hi])
    return f"[{a:+.4f}, {b:+.4f}]", (a > 0) or (b < 0)


def pooled_paired(pa, pb, n, rng):
    """Delta_O, Delta_U and their difference from one shared resample of drug clusters."""
    n_o, n_u = pa["n_o"], pa["n_u"]
    out = np.empty((n, 3))
    for k in range(n):
        wo = rng.multinomial(n_o, np.full(n_o, 1.0 / n_o)).astype(float)
        wu = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u)).astype(float)
        d_o = pa["auc_o"](wo, wo) - pb["auc_o"](wo, wo)
        d_u = pa["auc_u"](wo, wu) - pb["auc_u"](wo, wu)
        out[k] = [d_o, d_u, d_u - d_o]
    return out


def main(pred_dir, mode, m=3, n_boot=2000):
    m, n_boot = int(m), int(n_boot)
    rng = np.random.default_rng(20261018)
    by_seed = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(pred_dir, "preds_*_seed*.tsv"))):
        g = re.search(r"preds_(\w+)_seed(\d+)\.tsv", os.path.basename(f))
        by_seed[int(g.group(2))][g.group(1)] = f

    tag = os.path.basename(pred_dir)
    print(f"=== {tag}  (paired statistics, {n_boot} replicates)")

    # ---- 1 and 2: pooled, every seed, and single-checkpoint models against every seed
    rows = {s: {n: load_rows(p) for n, p in ms.items()} for s, ms in by_seed.items()}
    single = sorted({n for s, ms in rows.items() for n in ms} -
                    {n for n in rows.get(1, {})}) if 1 in rows else []
    print(f"models present per seed: "
          f"{ {s: len(ms) for s, ms in sorted(rows.items())} }; single-checkpoint: {single}")

    print("\npooled, paired Delta_U - Delta_O")
    print("pair\tseed\tDelta_O\tDelta_U\tDelta_U-Delta_O\tdiff_excludes_0")
    pooled_rows = defaultdict(list)
    for seed in sorted(rows):
        prepped = {n: prepare(strip(r)) for n, r in rows[seed].items()}
        names = sorted(prepped, key=lambda n: -point(prepped[n], 0.0)[0])
        for a, b in combinations(names, 2):
            if a.endswith("sm") or b.endswith("sm"):
                continue
            boots = pooled_paired(prepped[a], prepped[b], n_boot, rng)
            d = [point(prepped[a], 0.0)[i] - point(prepped[b], 0.0)[i] for i in range(2)]
            s_o, _ = ci(boots[:, 0])
            s_u, _ = ci(boots[:, 1])
            s_d, excl = ci(boots[:, 2])
            pooled_rows[f"{a}-{b}"].append(excl)
            print(f"{a}-{b}\t{seed}\t{d[0]:+.4f} {s_o}\t{d[1]:+.4f} {s_u}\t"
                  f"{d[1] - d[0]:+.4f} {s_d}\t{excl}")

    # single checkpoint against every seed of the trained models
    for sc in single:
        if sc.endswith("sm"):
            continue
        base = prepare(strip(rows[0][sc]))
        print(f"\n{sc} (one checkpoint) against every seed")
        print("pair\tseed\tDelta_O\tDelta_U\tDelta_U-Delta_O\tO_excludes_0\tdiff_excludes_0")
        for seed in sorted(rows):
            for other in sorted(rows[seed]):
                if other in single or other.endswith("sm"):
                    continue
                p_other = prepare(strip(rows[seed][other]))
                boots = pooled_paired(base, p_other, n_boot, rng)
                d = [point(base, 0.0)[i] - point(p_other, 0.0)[i] for i in range(2)]
                s_o, e_o = ci(boots[:, 0])
                s_u, _ = ci(boots[:, 1])
                s_d, e_d = ci(boots[:, 2])
                print(f"{sc}-{other}\t{seed}\t{d[0]:+.4f} {s_o}\t{d[1]:+.4f} {s_u}\t"
                      f"{d[1] - d[0]:+.4f} {s_d}\t{e_o}\t{e_d}")

    # ---- 4: pooled, but restricted to the within-protein subset, to separate subset
    #         from weighting
    print("\npooled on the within-protein subset (same proteins, pooled weights)")
    print("pair\tseed\tproteins\tDelta_O\tDelta_U\tDelta_U-Delta_O\tdiff_excludes_0")
    for seed in sorted(rows):
        tbl = {n: per_protein(rows[seed][n], m)[0] for n in rows[seed]}
        names = sorted(tbl, key=lambda n: -np.mean([v[0] for v in tbl[n].values()]))
        keep = set.intersection(*[set(t) for t in tbl.values()])
        sub = {n: prepare(restrict_prot(rows[seed][n], keep)) for n in rows[seed]}
        for a, b in combinations(names, 2):
            if a.endswith("sm") or b.endswith("sm"):
                continue
            boots = pooled_paired(sub[a], sub[b], n_boot, rng)
            d = [point(sub[a], 0.0)[i] - point(sub[b], 0.0)[i] for i in range(2)]
            s_o, _ = ci(boots[:, 0])
            s_u, _ = ci(boots[:, 1])
            s_d, excl = ci(boots[:, 2])
            print(f"{a}-{b}	{seed}	{len(keep)}	{d[0]:+.4f} {s_o}	{d[1]:+.4f} {s_u}	"
                  f"{d[1] - d[0]:+.4f} {s_d}	{excl}")

    # ---- 3: within-protein, per seed, with its own coverage
    print("\nwithin-protein, paired, per seed with its own protein set")
    print("pair\tseed\tproteins\tU_mass\tDelta_O\tDelta_U\tDelta_U-Delta_O\tdiff_excludes_0")
    for seed in sorted(rows):
        tbl = {n: per_protein(r, m) for n, r in rows[seed].items()}
        names = sorted(tbl, key=lambda n: -np.mean([v[0] for v in tbl[n][0].values()]))
        for a, b in combinations(names, 2):
            if a.endswith("sm") or b.endswith("sm"):
                continue
            ta, tb = tbl[a][0], tbl[b][0]
            keys = sorted(set(ta) & set(tb))
            do = np.array([ta[k][0] - tb[k][0] for k in keys])
            du = np.array([ta[k][1] - tb[k][1] for k in keys])
            idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
            s_o, _ = ci(do[idx].mean(1))
            s_u, _ = ci(du[idx].mean(1))
            s_d, excl = ci((du - do)[idx].mean(1))
            print(f"{a}-{b}\t{seed}\t{len(keys)}\t{tbl[a][1]:.1%}\t{do.mean():+.4f} {s_o}\t"
                  f"{du.mean():+.4f} {s_u}\t{(du - do).mean():+.4f} {s_d}\t{excl}")




def decompose(rows, m):
    """Exact split of pooled AUROC into its within-protein and cross-protein halves.

    Pooled AUROC is a weighted mean of AUROC over every (positive protein, negative
    protein) cell, with weight proportional to n_pos(p) * n_neg(q). The diagonal cells
    are the within-protein comparisons, the off-diagonal ones compare a positive of one
    protein against a negative of another. Concordance counts add, so the cross term
    follows from the total minus the diagonal without touching the off-diagonal cells.
    Returns, for O and for U, (within AUROC, cross AUROC, diagonal share of the weight).
    """
    from collections import defaultdict as dd
    pos, neg = dd(list), {"test": dd(list), "U": dd(list)}
    for p, _, y, s in rows["test"]:
        (pos if y == 1 else neg["test"])[p].append(s)
    for p, _, y, s in rows["U"]:
        if y == 0:
            neg["U"][p].append(s)
    allpos = np.concatenate([np.array(v) for v in pos.values()])
    out = {}
    for region, key in (("O", "test"), ("U", "U")):
        allneg = np.concatenate([np.array(v) for v in neg[key].values()])
        n_p, n_n = len(allpos), len(allneg)
        total = roc_auc_score(np.r_[np.ones(n_p), np.zeros(n_n)],
                              np.r_[allpos, allneg]) * n_p * n_n
        diag_c, diag_w = 0.0, 0
        for p in pos:
            if p not in neg[key]:
                continue
            a, b = np.array(pos[p]), np.array(neg[key][p])
            w = len(a) * len(b)
            diag_c += roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))],
                                    np.r_[a, b]) * w
            diag_w += w
        cross_w = n_p * n_n - diag_w
        out[region] = (diag_c / diag_w, (total - diag_c) / cross_w, diag_w / (n_p * n_n))
    return out


def main_decompose(pred_dir, m=3):
    import glob as _g
    by_seed = defaultdict(dict)
    for f in sorted(_g.glob(os.path.join(pred_dir, "preds_*_seed*.tsv"))):
        g = re.search(r"preds_(\w+)_seed(\d+)\.tsv", os.path.basename(f))
        by_seed[int(g.group(2))][g.group(1)] = f
    print(f"=== {os.path.basename(pred_dir)}: pooled AUROC split into within- and cross-protein")
    print("seed\tmodel\tO_within\tO_cross\tU_within\tU_cross\tdiag_share_O\tdiag_share_U")
    vals = defaultdict(dict)
    for seed in sorted(by_seed):
        for name in sorted(by_seed[seed]):
            if name.endswith("sm"):
                continue
            d = decompose(load_rows(by_seed[seed][name]), m)
            vals[seed][name] = d
            print(f"{seed}\t{name}\t{d['O'][0]:.4f}\t{d['O'][1]:.4f}\t{d['U'][0]:.4f}\t"
                  f"{d['U'][1]:.4f}\t{d['O'][2]:.1%}\t{d['U'][2]:.1%}")
    print("\nwhere the gap growth lives (Delta_U - Delta_O, by component)")
    print("pair\tseed\twithin\tcross")
    for seed in sorted(vals):
        names = sorted(vals[seed], key=lambda n: -vals[seed][n]["O"][1])
        for a, b in combinations(names, 2):
            da, db = vals[seed][a], vals[seed][b]
            win = (da["U"][0] - db["U"][0]) - (da["O"][0] - db["O"][0])
            cro = (da["U"][1] - db["U"][1]) - (da["O"][1] - db["O"][1])
            print(f"{a}-{b}\t{seed}\t{win:+.4f}\t{cro:+.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "decompose":
        main_decompose(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
