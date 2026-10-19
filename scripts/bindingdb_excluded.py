"""Excluded region of the DrugBAN low-bias BindingDB benchmark.

Bai et al. 2021 (and DrugBAN, Supplementary S2) build the benchmark from
BindingDB IC50 records: positive if IC50 < 100 nM, negative if IC50 > 10,000 nM,
then remove every drug that has only one label type. This script rebuilds that
binary set from a current BindingDB_All dump and measures what the second step
removes from the negative population.

Choices made here (Bai et al. do not state them):
  * single-chain targets only (the benchmark takes one protein sequence);
  * qualifiers: '<x' with x <= 100 is positive, '>x' with x >= 10,000 is
    negative, anything else that does not clear a threshold is dropped;
  * drug identity = BindingDB 'Ligand InChI Key'; protein identity = chain
    sequence; a pair with both labels is dropped as conflicting.

Target population for W: all binary negative pairs (uniform), and the same
restricted to the benchmark's 2,623 proteins. The excluded part is the
negatives of drugs with no positive ("negative-only" drugs).

Outputs (OUT_DIR):
  bindingdb_excluded_summary.tsv  counts and W
  U_pool.csv                      negative-only drugs x benchmark proteins,
                                  drugs absent from the benchmark (SMILES, Protein, Y=0, ikey)

Usage: python bindingdb_excluded.py BindingDB_All_tsv.zip drugban_bindingdb_full.csv OUT_DIR
Needs rdkit (benchmark SMILES -> InChIKey).
"""
import csv
import io
import os
import sys
import zipfile
from collections import defaultdict

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10**9)


def parse_ic50(raw):
    s = raw.strip().replace(" ", "")
    if not s:
        return None
    q = ""
    if s[0] in "<>":
        q, s = s[0], s[1:]
        if s.startswith("="):
            s = s[1:]
    try:
        v = float(s)
    except ValueError:
        return None
    if q == "":
        return 1 if v < 100 else 0 if v > 10000 else None
    if q == "<":
        return 1 if v <= 100 else None
    return 0 if v >= 10000 else None


def main(zpath, bench_csv, out):
    os.makedirs(out, exist_ok=True)
    zf = zipfile.ZipFile(zpath)
    name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
    fh = io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace", newline="")
    header = fh.readline().rstrip("\r\n").split("\t")
    col = {h: i for i, h in enumerate(header)}
    i_smi = col["Ligand SMILES"]
    i_ikey = col["Ligand InChI Key"]
    i_ic50 = col["IC50 (nM)"]
    i_nch = [i for h, i in col.items() if h.startswith("Number of Protein Chains")][0]
    i_seq = [i for h, i in col.items() if h.startswith("BindingDB Target Chain Sequence")][0]

    seq_id, seqs = {}, []
    labels = defaultdict(set)          # (ikey, sid) -> {0,1}
    smiles = {}
    n_rows = n_ic50 = n_lab = 0
    last = max(i_smi, i_ikey, i_ic50, i_nch, i_seq)
    for line in fh:
        n_rows += 1
        row = line.split("\t", last + 1)
        if len(row) <= max(i_seq, i_ic50, i_nch):
            continue
        y = parse_ic50(row[i_ic50])
        if row[i_ic50].strip():
            n_ic50 += 1
        if y is None:
            continue
        if row[i_nch].strip() != "1":
            continue
        seq = row[i_seq].strip().upper()
        ikey = row[i_ikey].strip()
        if not seq or not ikey:
            continue
        sid = seq_id.setdefault(seq, len(seqs))
        if sid == len(seqs):
            seqs.append(seq)
        labels[(ikey, sid)].add(y)
        smiles.setdefault(ikey, row[i_smi].strip())
        n_lab += 1

    pairs = {k: next(iter(v)) for k, v in labels.items() if len(v) == 1}
    n_conflict = sum(1 for v in labels.values() if len(v) > 1)
    dpos, dneg = defaultdict(int), defaultdict(int)
    for (d, _), y in pairs.items():
        (dpos if y else dneg)[d] += 1
    drugs = set(dpos) | set(dneg)
    both = {d for d in drugs if dpos[d] and dneg[d]}
    neg_only = {d for d in drugs if not dpos[d]}
    pos_only = {d for d in drugs if not dneg[d]}

    # benchmark
    bench = list(csv.DictReader(open(bench_csv, newline="")))
    b_ikeys, bad = set(), 0
    for r in bench:
        m = Chem.MolFromSmiles(r["SMILES"])
        if m is None:
            bad += 1
            continue
        b_ikeys.add(Chem.MolToInchiKey(m))
    b_seqs = {r["Protein"].strip().upper() for r in bench}
    b_sids = {seq_id[s] for s in b_seqs if s in seq_id}

    negs = [(d, s) for (d, s), y in pairs.items() if y == 0]
    w_all = sum(1 for d, _ in negs if d in neg_only) / len(negs)
    negs_b = [(d, s) for d, s in negs if s in b_sids]
    w_b = sum(1 for d, _ in negs_b if d in neg_only) / len(negs_b)

    rows = [
        ("rows_in_dump", n_rows), ("rows_with_IC50", n_ic50),
        ("rows_labelled_single_chain", n_lab), ("pairs_labelled", len(labels)),
        ("pairs_conflicting_dropped", n_conflict), ("pairs_clean", len(pairs)),
        ("positive_pairs", sum(pairs.values())), ("negative_pairs", len(negs)),
        ("drugs", len(drugs)), ("drugs_both_labels", len(both)),
        ("drugs_positive_only", len(pos_only)), ("drugs_negative_only", len(neg_only)),
        ("share_drugs_one_label_type", round(1 - len(both) / len(drugs), 4)),
        ("W_all_negatives", round(w_all, 4)),
        ("benchmark_drugs", len(b_ikeys)), ("benchmark_smiles_unparsed", bad),
        ("benchmark_drugs_found_in_dump", len(b_ikeys & drugs)),
        ("benchmark_drugs_classed_both_now", len(b_ikeys & both)),
        ("benchmark_proteins", len(b_seqs)), ("benchmark_proteins_found_in_dump", len(b_sids)),
        ("negative_pairs_on_benchmark_proteins", len(negs_b)),
        ("W_negatives_on_benchmark_proteins", round(w_b, 4)),
    ]
    # excluded pool: negative-only drugs, benchmark proteins, not a benchmark drug
    n_u = 0
    with open(os.path.join(out, "U_pool.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["SMILES", "Protein", "Y", "ikey"])
        for d, s in negs_b:
            if d in neg_only and d not in b_ikeys:
                w.writerow([smiles[d], seqs[s], 0, d])
                n_u += 1
    rows.append(("U_pool_pairs", n_u))
    rows.append(("U_pool_drugs", len({d for d, s in negs_b if d in neg_only and d not in b_ikeys})))
    with open(os.path.join(out, "bindingdb_excluded_summary.tsv"), "w") as fo:
        for k, v in rows:
            fo.write(f"{k}\t{v}\n")
            print(f"{k}\t{v}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
