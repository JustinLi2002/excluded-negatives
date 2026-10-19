"""DREAM5 network inference: rebuild the scored negatives and their excluded mass.

The official scorer (matlab/edgelist_2_D5C4_gs.m) enumerates negatives as
  regulators with >= 1 gold-standard edge  x  genes touching >= 1 gold-standard edge,
minus positives and self-pairs; predictions outside that grid are removed
(remove_edges_not_in_gs). This script (1) rebuilds that grid from the
edges-only gold standard and checks it pair for pair against the released
complete gold standard (edges AND non-edges), and (2) compares it with a
target declared here: every input TF x every gene in the expression data,
non-self, uniformly weighted.

Usage: python -I dream5_scope.py DREAM5_network_inference_challenge_DIR
"""
import csv
import os
import sys

NETS = {"1": "in silico", "3": "E. coli", "4": "S. cerevisiae"}


def read_tsv(path):
    with open(path, newline="") as fh:
        return [r for r in csv.reader(fh, delimiter="\t") if r]


def main(root):
    ev = os.path.join(root, "Evaluation scripts", "INPUT")
    print("network\torganism\tinput_TFs\tinput_genes\tgs_regulators\tgs_genes\tpositives"
          "\tnegatives_released\trebuilt_equals_released\tW_candidates\tW_negative_mass"
          "\tinput_TFs_without_targets"
          "\tinput_genes_outside_gs")
    for k, org in NETS.items():
        pos = {(a, b) for a, b, *_ in read_tsv(os.path.join(
            ev, "gold_standard_edges_only", f"DREAM5_NetworkInference_Edges_Network{k}.tsv"))}
        complete = read_tsv(os.path.join(
            ev, "gold_standard_complete", f"DREAM5_NetworkInference_GoldStandard_Network{k}.tsv"))
        rel_pos = {(a, b) for a, b, y in complete if y == "1"}
        rel_neg = {(a, b) for a, b, y in complete if y == "0"}
        regs = {a for a, _ in pos}
        genes = {a for a, _ in pos} | {b for _, b in pos}
        rebuilt = {(r, g) for r in regs for g in genes if r != g} - pos
        same = rebuilt == rel_neg and rel_pos == pos

        d = os.path.join(root, f"Network{k}", "input data")
        tfs = {r[0] for r in read_tsv(os.path.join(d, f"net{k}_transcription_factors.tsv"))}
        expr_header = read_tsv(os.path.join(d, f"net{k}_expression_data.tsv"))[0]
        allg = set(expr_header)
        target = len(tfs) * len(allg) - len(tfs & allg)
        universe = len(rebuilt) + len(pos)
        w_cand = 1 - universe / target
        # Paper 2's W is a share of candidate NEGATIVES, so drop the positives from both
        w_neg = 1 - len(rebuilt) / (target - len(pos))
        print(f"{k}\t{org}\t{len(tfs)}\t{len(allg)}\t{len(regs)}\t{len(genes)}\t{len(pos)}"
              f"\t{len(rel_neg)}\t{same}\t{w_cand:.4f}\t{w_neg:.4f}"
              f"\t{len(tfs - regs)}\t{len(allg - genes)}")
        assert regs <= tfs, "a gold-standard regulator is not an input TF"


if __name__ == "__main__":
    main(sys.argv[1])
