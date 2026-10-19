"""Every statistic the manuscript cites, as keyed records in one table.

The outline review (2026-10-18) showed why "the number appears somewhere in an output
file" is not a check: a generated table paired seed 0's AUROCs with seed 4's protein
count, and every number in it still appeared somewhere. Each record here carries the
whole key its value depends on:

  design     E2 | E1_random | E1_disjoint
  estimand   pooled          all test positives against all negatives of a region
             pooled_subset   the same, restricted to the within_protein proteins
             within_pairweighted  per-protein AUROC on the same proteins, averaged with
                             weights n_pos * n_neg of each region, i.e. the within-protein
                             cells of pooled_subset with the cross-protein cells dropped
             within_protein  per-protein AUROC, proteins with >= M positives and >= M
                             negatives in both regions, averaged with equal weights
             Read left to right, pooled -> pooled_subset -> within_pairweighted ->
             within_protein changes one thing per step: the protein subset, then the
             cross-protein comparisons, then the protein weights.
             decomp_diagonal the within-protein cells of pooled AUROC (pair-count weights)
             decomp_cross    the cross-protein cells of pooled AUROC
             population      counts of the declared target population (E2 only)
             inference_repeat  MolTrans re-scored by one checkpoint (dropout stays on)
  subject    a model (drugban) or an oriented pair (drugban-moltrans = A minus B)
  seed       training seed. The released BioT5 checkpoint has no seeds; in a pair with a
             trained model the seed is the trained model's
  quantity   AUROC_O, AUROC_U, Delta_O, Delta_U, G (= Delta_U - Delta_O), n_proteins,
             UminusO (one model's AUROC_U - AUROC_O, resampled like the pairs),
             G_minus_equal_weights (within_pairweighted G minus within_protein G on one
             protein replicate: the effect of the weighting alone),
             U_coverage (share of sampled U negatives on the qualified proteins),
             diag_share_O, diag_share_U, and population counts
  value, lo, hi   point estimate and 95% percentile interval (blank when not resampled)

The positive reference is the same in every record: the test split's positives, scored
against O's negatives for *_O and against U's negatives for *_U.

Intervals: pooled and pooled_subset resample drug clusters (a cluster carries a drug's
own positives and negatives) and score both models on the same replicate, so Delta_O,
Delta_U and G come from one draw; within_protein resamples qualified proteins. Every
bootstrap draws from its own RNG stream derived from its key, so adding records never
moves existing ones. Intervals are conditional on the saved scores: they include neither
training-seed variation (the five seeds are reported separately) nor MolTrans's inference
dropout (inference_repeat).

This table supersedes the intervals printed by claim_check.py and the *_summary /
*_sensitivity files, which used other RNG streams; point estimates agree exactly.

Usage: python records.py [N_BOOT=2000] [WORKERS=8]   ->  results/records.tsv
"""
import csv
import os
import re
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_excl import point, prepare, selftest  # noqa: E402
from claim_check import decompose, restrict_prot  # noqa: E402
from sensitivity_targets import load_rows, per_protein, strip  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
DESIGNS = {"E2": "hrp_e2", "E1_random": "hrp_excl_random", "E1_disjoint": "hrp_excl_canon"}
SEEDS = range(5)
TRAINED = ("drugban", "moltrans", "drugonly")
PAIRS = (("drugban", "moltrans"), ("drugban", "drugonly"), ("moltrans", "drugonly"))
SINGLE = ("biot5", "biot5sm")  # released checkpoint, E1_random only, stored as seed 0
M = 3
ESTIMANDS = ("pooled", "pooled_subset", "within_pairweighted", "within_protein")
HEADER = ("design", "estimand", "subject", "seed", "quantity", "value", "lo", "hi")


@lru_cache(maxsize=None)
def rows(design, model, seed):
    return load_rows(os.path.join(RES, DESIGNS[design], f"preds_{model}_seed{seed}.tsv"))


@lru_cache(maxsize=None)
def proteins(design, model, seed):
    return per_protein(rows(design, model, seed), M)


@lru_cache(maxsize=None)
def protein_weights(design, model, seed):
    """n_pos * n_neg behind each qualified protein's AUROC on O and on U (labels only)."""
    r = rows(design, model, seed)
    pos, neg_o, neg_u = Counter(), Counter(), Counter()
    for p, _, y, _ in r["test"]:
        (pos if y == 1 else neg_o)[p] += 1
    for p, _, y, _ in r["U"]:
        if y == 0:
            neg_u[p] += 1
    return {p: (pos[p] * neg_o[p], pos[p] * neg_u[p]) for p in proteins(design, model, seed)[0]}


def rec(design, estimand, subject, seed, quantity, value, lo="", hi=""):
    return (design, estimand, subject, str(seed), quantity, value, lo, hi)


def model_records(design, model, seed):
    r = rows(design, model, seed)
    p = prepare(strip(r))
    selftest(p)
    a_o, a_u, _ = point(p, 0.0)
    out = [rec(design, "pooled", model, seed, "AUROC_O", a_o),
           rec(design, "pooled", model, seed, "AUROC_U", a_u)]
    tbl, cov = proteins(design, model, seed)
    out += [rec(design, "within_protein", model, seed, "AUROC_O",
                float(np.mean([v[0] for v in tbl.values()]))),
            rec(design, "within_protein", model, seed, "AUROC_U",
                float(np.mean([v[1] for v in tbl.values()]))),
            rec(design, "within_protein", model, seed, "n_proteins", len(tbl)),
            rec(design, "within_protein", model, seed, "U_coverage", cov)]
    pw = protein_weights(design, model, seed)
    for i, q in ((0, "AUROC_O"), (1, "AUROC_U")):
        w = np.array([pw[k][i] for k in sorted(tbl)], float)
        x = np.array([tbl[k][i] for k in sorted(tbl)])
        out.append(rec(design, "within_pairweighted", model, seed, q, float((w * x).sum() / w.sum())))
    d = decompose(r, M)
    for est, i in (("decomp_diagonal", 0), ("decomp_cross", 1)):
        out += [rec(design, est, model, seed, "AUROC_O", d["O"][i]),
                rec(design, est, model, seed, "AUROC_U", d["U"][i])]
    out += [rec(design, "decomp_diagonal", model, seed, "diag_share_O", d["O"][2]),
            rec(design, "decomp_diagonal", model, seed, "diag_share_U", d["U"][2])]
    return out


def drop_records(design, model, seed, estimand, n_boot):
    """One model's AUROC_U - AUROC_O, resampled the same way as the pairs."""
    rng = np.random.default_rng([20261018, zlib.crc32(f"{design}|{estimand}|{model}|{seed}|drop".encode())])
    out = []
    if estimand in ("pooled", "pooled_subset"):
        r = rows(design, model, seed)
        p = prepare(strip(r) if estimand == "pooled" else
                    restrict_prot(r, set(proteins(design, model, seed)[0])))
        a_o, a_u, _ = point(p, 0.0)
        if estimand == "pooled_subset":
            out += [rec(design, estimand, model, seed, "AUROC_O", a_o),
                    rec(design, estimand, model, seed, "AUROC_U", a_u)]
        n_o, n_u = p["n_o"], p["n_u"]
        boots = np.empty(n_boot)
        for k in range(n_boot):
            wo = rng.multinomial(n_o, np.full(n_o, 1.0 / n_o)).astype(float)
            wu = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u)).astype(float)
            boots[k] = p["auc_u"](wo, wu) - p["auc_o"](wo, wo)
        pt = a_u - a_o
    else:
        tbl = proteins(design, model, seed)[0]
        keys = sorted(tbl)
        x_o = np.array([tbl[k][0] for k in keys])
        x_u = np.array([tbl[k][1] for k in keys])
        if estimand == "within_protein":
            wo = wu = np.ones(len(keys))
        else:
            pw = protein_weights(design, model, seed)
            wo = np.array([pw[k][0] for k in keys], float)
            wu = np.array([pw[k][1] for k in keys], float)
        idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
        boots = ((x_u[idx] * wu[idx]).sum(1) / wu[idx].sum(1)
                 - (x_o[idx] * wo[idx]).sum(1) / wo[idx].sum(1))
        pt = (x_u * wu).sum() / wu.sum() - (x_o * wo).sum() / wo.sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out.append(rec(design, estimand, model, seed, "UminusO", float(pt), float(lo), float(hi)))
    return out


def paired(pa, pb, n, rng):
    """Delta_O, Delta_U and G from one shared resample of drug clusters."""
    for key in ("o_drugs", "u_drugs", "labels_o", "labels_u"):
        assert pa[key] == pb[key], f"the two models disagree on {key}"
    n_o, n_u = pa["n_o"], pa["n_u"]
    out = np.empty((n, 3))
    for k in range(n):
        wo = rng.multinomial(n_o, np.full(n_o, 1.0 / n_o)).astype(float)
        wu = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u)).astype(float)
        d_o = pa["auc_o"](wo, wo) - pb["auc_o"](wo, wo)
        d_u = pa["auc_u"](wo, wu) - pb["auc_u"](wo, wu)
        out[k] = (d_o, d_u, d_u - d_o)
    return out


def pair_records(design, a, sa, b, sb, estimand, n_boot):
    subject, seed = f"{a}-{b}", sb
    rng = np.random.default_rng([20261018, zlib.crc32(f"{design}|{estimand}|{subject}|{seed}".encode())])
    out = []
    if estimand != "pooled":
        (ta, cov), (tb, cov_b) = proteins(design, a, sa), proteins(design, b, sb)
        assert set(ta) == set(tb) and cov == cov_b, "the two models qualified different proteins"
        out += [rec(design, estimand, subject, seed, "n_proteins", len(ta)),
                rec(design, estimand, subject, seed, "U_coverage", cov)]
    if estimand in ("within_protein", "within_pairweighted"):
        keys = sorted(ta)
        do = np.array([ta[k][0] - tb[k][0] for k in keys])
        du = np.array([ta[k][1] - tb[k][1] for k in keys])
        if estimand == "within_protein":
            wo = wu = np.ones(len(keys))
        else:
            pw = protein_weights(design, a, sa)
            assert pw == protein_weights(design, b, sb), "the two models saw different labels"
            wo = np.array([pw[k][0] for k in keys], float)
            wu = np.array([pw[k][1] for k in keys], float)
        idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
        b_o = (do[idx] * wo[idx]).sum(1) / wo[idx].sum(1)
        b_u = (du[idx] * wu[idx]).sum(1) / wu[idx].sum(1)
        p_o, p_u = (do * wo).sum() / wo.sum(), (du * wu).sum() / wu.sum()
        pt = (p_o, p_u, p_u - p_o)
        boots = np.column_stack([b_o, b_u, b_u - b_o])
        if estimand == "within_pairweighted":
            # the weighting itself: same proteins, same per-protein AUROCs, same replicate
            c = (p_u - p_o) - (du - do).mean()
            cb = (b_u - b_o) - (du - do)[idx].mean(1)
            lo, hi = np.percentile(cb, [2.5, 97.5])
            out.append(rec(design, estimand, subject, seed, "G_minus_equal_weights",
                           float(c), float(lo), float(hi)))
    else:
        if estimand == "pooled_subset":
            keep = set(ta)
            pa, pb = (prepare(restrict_prot(rows(design, a, sa), keep)),
                      prepare(restrict_prot(rows(design, b, sb), keep)))
        else:
            pa, pb = prepare(strip(rows(design, a, sa))), prepare(strip(rows(design, b, sb)))
        xa, xb = point(pa, 0.0), point(pb, 0.0)
        pt = (xa[0] - xb[0], xa[1] - xb[1], (xa[1] - xb[1]) - (xa[0] - xb[0]))
        boots = paired(pa, pb, n_boot, rng)
    for i, q in enumerate(("Delta_O", "Delta_U", "G")):
        lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
        out.append(rec(design, estimand, subject, seed, q, float(pt[i]), float(lo), float(hi)))
    return out


def decomp_pairs(model_recs):
    """G split into the diagonal and cross-protein parts of pooled AUROC (points only)."""
    v = {(r[0], r[1], r[2], r[3], r[4]): r[5] for r in model_recs}
    out = []
    for design in DESIGNS:
        for seed in SEEDS:
            pairs = [(a, b) for a, b in PAIRS]
            if design == "E1_random":
                pairs += [("biot5", "drugban"), ("biot5sm", "drugban")]
            for a, b in pairs:
                sa = 0 if a in SINGLE else seed
                for est in ("decomp_diagonal", "decomp_cross"):
                    get = lambda m, s, q: v[(design, est, m, str(s), q)]  # noqa: E731
                    d_o = get(a, sa, "AUROC_O") - get(b, seed, "AUROC_O")
                    d_u = get(a, sa, "AUROC_U") - get(b, seed, "AUROC_U")
                    out += [rec(design, est, f"{a}-{b}", seed, "Delta_O", d_o),
                            rec(design, est, f"{a}-{b}", seed, "Delta_U", d_u),
                            rec(design, est, f"{a}-{b}", seed, "G", d_u - d_o)]
    return out


def population_records():
    out = []
    with open(os.path.join(RES, "target", "summary.tsv"), newline="") as fh:
        for k, v in csv.reader(fh, delimiter="\t"):
            out.append(rec("E2", "population", "target", "", k, int(v) if v.isdigit() else float(v)))
    n_o = next(r[5] for r in out if r[4] == "negatives_O")
    n_u = next(r[5] for r in out if r[4] == "negatives_U_block_disjoint")
    n_test = next(r[5] for r in out if r[4] == "O_pairs_in_test_split")
    out += [rec("E2", "population", "target", "", "negatives_N_block_disjoint", n_o + n_u),
            rec("E2", "population", "target", "", "W_rule", n_u / (n_o + n_u)),
            rec("E2", "population", "target", "", "W_eval", n_u / (n_u + n_test))]
    return out


def noise_records():
    path = os.path.join(RES, "moltrans_noise.log")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            g = re.match(r"(AUROC_O|AUROC_U|U-O): mean (\S+)\s+sd (\S+)\s+range (\S+)", line)
            if g:
                q = "UminusO" if g.group(1) == "U-O" else g.group(1)
                for stat, val in zip(("mean", "sd", "range"), g.groups()[1:]):
                    out.append(rec("E2", "inference_repeat", "moltrans", 0, f"{q}_{stat}", float(val)))
            g = re.match(r"pass (\d+): AUROC_O (\S+)\s+AUROC_U (\S+)", line)
            if g:
                out.append(rec("E2", "inference_repeat", "moltrans", 0, f"pass{g.group(1)}_AUROC_O", float(g.group(2))))
                out.append(rec("E2", "inference_repeat", "moltrans", 0, f"pass{g.group(1)}_AUROC_U", float(g.group(3))))
    return out


def tasks(n_boot):
    model_t, pair_t = [], []
    for design in DESIGNS:
        for seed in SEEDS:
            for m in TRAINED:
                model_t.append((design, m, seed))
            for a, b in PAIRS:
                for est in ESTIMANDS:
                    pair_t.append((design, a, seed, b, seed, est, n_boot))
    for m in SINGLE:
        model_t.append(("E1_random", m, 0))
    for seed in SEEDS:
        for a in SINGLE:
            for est in ESTIMANDS:
                pair_t.append(("E1_random", a, 0, "drugban", seed, est, n_boot))
    for est in ("pooled", "within_protein"):
        pair_t.append(("E1_random", "biot5sm", 0, "biot5", 0, est, n_boot))
    drop_t = [(d, m, s, est, n_boot) for d, m, s in model_t for est in ESTIMANDS]
    return model_t, pair_t, drop_t


def fmt(x):
    return "" if x == "" else (f"{x:.6f}" if isinstance(x, float) else str(x))


def main(n_boot=2000, workers=8):
    n_boot, workers = int(n_boot), int(workers)
    model_t, pair_t, drop_t = tasks(n_boot)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        model_out = [r for rs in ex.map(model_records, *zip(*model_t)) for r in rs]
        pair_out = [r for rs in ex.map(pair_records, *zip(*pair_t)) for r in rs]
        drop_out = [r for rs in ex.map(drop_records, *zip(*drop_t)) for r in rs]
    out = (population_records() + model_out + pair_out + drop_out + decomp_pairs(model_out)
           + noise_records())
    order = {d: i for i, d in enumerate(DESIGNS)}
    out.sort(key=lambda r: (order[r[0]], r[1], r[2], r[3], r[4]))
    keys = [r[:5] for r in out]
    assert len(keys) == len(set(keys)), "duplicate record keys"
    dest = os.path.join(RES, "records.tsv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(HEADER)
        for r in out:
            w.writerow([*r[:5], *(fmt(x) for x in r[5:])])
    print(f"{len(out)} records -> {dest}  ({len(model_t)} model, {len(pair_t)} pair and "
          f"{len(drop_t)} drop tasks, "
          f"{n_boot} replicates)")


if __name__ == "__main__":
    main(*sys.argv[1:])
