"""Excluded candidate mass in BEELINE's scRNA-seq evaluations.

Rebuilds each input gene set the way BEELINE's utils/generateExpInputs.py does
(TFs with Bonferroni-corrected p < 0.01, plus the top-N non-TF genes by
variance), filters the reference network to that set, and compares two
candidate universes with BEELINE's own:

  AUROC/AUPRC: all ordered non-self pairs among genes that touch a reference
               edge (BLEval computeAUC, 2020; AUPRC.py, 2026)
  EPR:         reference-edge sources x reference-edge genes, non-self
               (BLEval computeEarlyPrec, 2020)

Two corrections after the 2026-10-17 audit. (1) The EPR universe's sources are the
reference network's edge sources, a few of which are NOT in the input TF set, so the
universe did not sit inside the declared target; it is now intersected with the input
TFs. (2) Paper 2's W is a share of candidate NEGATIVES, so positives are now subtracted
from both the universe and the target. The uncorrected candidate-pair fraction is kept
alongside, because that is what the earlier pilot table reported.

Target populations (declared here, not by BEELINE):
  all-pairs:   ordered non-self pairs among the input genes
  TF-pairs:    input TFs x input genes, non-self

Gene names are upper-cased throughout (mESC and mHSC expression files use
mixed case; the networks and TF lists are upper case).

W = 1 - |universe| / |target|, uniform weight on candidate pairs. Positives
are inside both universes by construction, so W is the excluded share of the
target's candidate pairs (positives are a vanishing share of either count).

Usage: python -I beeline_scope.py BEELINE_ZENODO_DIR [N=500]
"""
import csv
import os
import sys

DATASETS = {
    "hESC": ("human", ["hESC-ChIP-seq-network.csv", "Non-specific-ChIP-seq-network.csv", "STRING-network.csv"]),
    "hHep": ("human", ["HepG2-ChIP-seq-network.csv", "Non-specific-ChIP-seq-network.csv", "STRING-network.csv"]),
    "mDC": ("mouse", ["mDC-ChIP-seq-network.csv", "Non-Specific-ChIP-seq-network.csv", "STRING-network.csv"]),
    "mESC": ("mouse", ["mESC-ChIP-seq-network.csv", "Non-Specific-ChIP-seq-network.csv", "STRING-network.csv", "mESC-lofgof-network.csv"]),
    "mHSC-E": ("mouse", ["mHSC-ChIP-seq-network.csv", "Non-Specific-ChIP-seq-network.csv", "STRING-network.csv"]),
    "mHSC-GM": ("mouse", ["mHSC-ChIP-seq-network.csv", "Non-Specific-ChIP-seq-network.csv", "STRING-network.csv"]),
    "mHSC-L": ("mouse", ["mHSC-ChIP-seq-network.csv", "Non-Specific-ChIP-seq-network.csv", "STRING-network.csv"]),
}


def expression_genes(path):
    genes = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            genes.append(row[0].upper())
    return set(genes)


def gene_ordering(path):
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        return [(r[0].upper(), float(r[1]), float(r[2])) for r in reader]


def input_genes(root, name, species, n):
    d = os.path.join(root, "BEELINE-data", "inputs", "scRNA-Seq", name)
    expr = expression_genes(os.path.join(d, "ExpressionData.csv"))
    order = [g for g in gene_ordering(os.path.join(d, "GeneOrdering.csv")) if g[0] in expr]
    cutoff = 0.01 / len(order)
    sig = [g for g in order if g[1] < cutoff]
    with open(os.path.join(root, f"{species}-tfs.csv"), newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        tfs_all = {r[0].upper() for r in reader}
    tfs = {g[0] for g in sig if g[0] in tfs_all}
    rest = sorted((g for g in sig if g[0] not in tfs), key=lambda g: -g[2])
    return tfs, tfs | {g[0] for g in rest[:n]}


def network(root, species, fname, genes):
    edges = set()
    with open(os.path.join(root, "Networks", species, fname), newline="") as fh:
        for r in csv.DictReader(fh):
            a, b = r["Gene1"].upper(), r["Gene2"].upper()
            if a in genes and b in genes and a != b:
                edges.add((a, b))
    return edges


def main(root, n):
    print("dataset\tnetwork\tinput_genes\tinput_TFs\tgt_genes\tgt_sources\tgt_sources_in_input_TFs\tedges\tdensity_BEELINE"
          "\tW_allpairs_AUROC_candidates\tW_TFpairs_EPR_candidates\tW_TFpairs_EPR_negative_mass\tinput_TFs_without_targets")
    for name, (species, nets) in DATASETS.items():
        tfs, genes = input_genes(root, name, species, n)
        g = len(genes)
        for net in nets:
            e = network(root, species, net, genes)
            nodes = {a for a, _ in e} | {b for _, b in e}
            sources = {a for a, _ in e}
            density = len(e) / (len(sources) * len(nodes) - len(sources)) if sources else float("nan")
            u_auc = len(nodes) * (len(nodes) - 1)
            t_all = g * (g - 1)
            # EPR universe, intersected with the declared TF set so it sits inside the target
            src_in = sources & tfs
            u_epr = len(src_in) * len(nodes) - len(src_in & nodes)
            t_tf = len(tfs) * g - len(tfs)
            # negative mass: Paper 2's W is a share of candidate NEGATIVES, so drop positives
            pos_u = len({(a, b) for a, b in e if a in src_in and b in nodes})
            pos_t = len({(a, b) for a, b in e if a in tfs})
            w_cand = 1 - u_epr / t_tf
            w_neg = 1 - (u_epr - pos_u) / (t_tf - pos_t)
            print(f"{name}	{net.replace('-network.csv', '')}	{g}	{len(tfs)}	{len(nodes)}"
                  f"	{len(sources)}	{len(src_in)}	{len(e)}	{density:.3f}"
                  f"	{1 - u_auc / t_all:.4f}	{w_cand:.4f}	{w_neg:.4f}	{len(tfs - sources)}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500)
