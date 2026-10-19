"""One declared target population for the BindingDB excluded-region experiment.

The earlier runs mixed three sources: W came from a 2026-10 rebuild of BindingDB,
O came from the old benchmark's test files, and U was filtered further. The
mixture (1-W)*AUROC_O + W*AUROC_U was then arithmetic over samples of different
populations. This script defines all of it once, from one snapshot, so that O and
U are a PARTITION of one declared negative population and W is their mass split
by construction.

Declared target
---------------
Snapshot      BindingDB_All 2026-10 (the caller passes the zip; its sha256 is
              pinned in expected/files.sha256).
Proteins      the benchmark's own protein sequences that appear in the snapshot
              as a single-chain target. Holding the protein set fixed keeps the
              O-U comparison about the drug eligibility rule instead of about
              new protein families.
Labels        the benchmark's own IC50 thresholds, via parse_ic50 from
              bindingdb_excluded.py: positive below 100 nM, negative above
              10,000 nM, anything that does not clear a threshold dropped.
Drug identity the canonical SMILES, because that is what every model actually
              reads: two of the snapshot's InChI Keys that render as the same
              SMILES are the same input and must not count as two drugs. Keying
              on the snapshot's key instead left 191 SMILES shared by two keys,
              3 of them disagreeing on eligibility, which would have made the
              claim "O and U are disjoint drug sets" an accident rather than a
              property. drug_id hashes the SMILES; the InChI Key computed FROM
              the SMILES gives drug_block (its first 14 characters), used only
              for splitting.
Cleaning      one rule for every pair: single-chain target, protein in the
              declared set, InChI Key present in the snapshot, a representative
              SMILES that RDKit parses, at most 290 heavy atoms (DrugBAN's graph
              limit), SELFIES-encodable (BioT5's input) and inside MolTrans's
              ESPF vocabulary for both the molecule and the protein, label
              conflicts dropped both per snapshot key and again after merging
              identical inputs. The vocabulary clause is there because
              MolTrans's encoders silently replace an out-of-vocabulary input
              with a single token: without it, 8 drugs and 1 protein would be
              blanked for that model only, and "every model reads every pair"
              would be false.
Negatives     N = every cleaned negative pair. The rule under study - a drug is
              eligible when it carries at least one positive and at least one
              negative - partitions N into
                  O = negatives of eligible drugs
                  U = negatives of ineligible drugs
              so W = |U| / |N| with uniform weight over N, and O and U cannot
              come from different populations.
Splitting     eligible drug BLOCKS are split 70/10/20, so test drugs are novel
              at connectivity level. U is restricted to blocks that no eligible
              drug uses, which holds U to the same novelty standard as the test
              split rather than a stricter one.
Estimand      AUROC_target = (1-W)*AUROC(O) + W*AUROC(U) with the positive
              reference fixed to the test split's positives. AUROC(U) is
              estimated on a uniform sample of U, which is unbiased for U;
              AUROC(O) is estimated on the held-out blocks of O, which is
              unbiased for O only if blocks are exchangeable. Both are
              conditional on the positive reference, the protein set and the
              trained model.

Outputs (OUT_DIR), all joinable on the ids and stable across reruns:
  proteins.tsv  protein_hash, length, sequence
  drugs.tsv     drug_id, canonical_smiles, inchikey, drug_block, heavy_atoms,
                n_pos, n_neg, eligible, bdb_keys (the snapshot keys merged into it)
  pairs.tsv     pair_id, drug_id, protein_hash, label, region, split
  summary.tsv   every count, W, and the sensitivity of the rule to the way a
                drug is identified

Usage: python build_target.py SNAPSHOT_ZIP BENCHMARK_FULL_CSV OUT_DIR ESPF_DIR [N_U=20000] [SEED=0]
"""
import csv
import hashlib
import io
import os
import random
import sys
import zipfile
from collections import defaultdict

from rdkit import Chem, RDLogger

from bindingdb_excluded import parse_ic50
from biot5_input import selfies_of
from moltrans_vocab import MolTransVocab

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10 ** 9)
MAX_HEAVY_ATOMS = 290


def h(text, n=16):
    return hashlib.sha1(text.encode()).hexdigest()[:n]


def declared_proteins(bench_csv, vocab):
    """The benchmark's sequences, minus any that MolTrans cannot tokenise."""
    seqs = {r["Protein"].strip().upper() for r in csv.DictReader(open(bench_csv, newline=""))}
    keep = {s for s in seqs if vocab.protein_ok(s)}
    return {s: h(s, 12) for s in sorted(keep)}, len(seqs) - len(keep)


def read_snapshot(zpath, prot_hash):
    """Labelled single-chain pairs on the declared proteins, keyed as in the file."""
    zf = zipfile.ZipFile(zpath)
    name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
    fh = io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace", newline="")
    header = fh.readline().rstrip("\r\n").split("\t")
    col = {c: i for i, c in enumerate(header)}
    i_smi, i_key, i_ic50 = col["Ligand SMILES"], col["Ligand InChI Key"], col["IC50 (nM)"]
    i_nch = next(i for c, i in col.items() if c.startswith("Number of Protein Chains"))
    i_seq = next(i for c, i in col.items() if c.startswith("BindingDB Target Chain Sequence"))
    last = max(i_smi, i_key, i_ic50, i_nch, i_seq)
    stats = defaultdict(int)
    labels = defaultdict(set)          # (snapshot key, protein_hash) -> {0,1}
    smiles = defaultdict(set)          # snapshot key -> raw SMILES seen
    for line in fh:
        stats["rows"] += 1
        row = line.split("\t", last + 1)
        if len(row) <= last:
            continue
        y = parse_ic50(row[i_ic50])
        if y is None:
            stats["no_usable_ic50"] += 1
            continue
        if row[i_nch].strip() != "1":
            stats["multi_chain"] += 1
            continue
        ph = prot_hash.get(row[i_seq].strip().upper())
        if ph is None:
            stats["protein_outside_declared_set"] += 1
            continue
        key = row[i_key].strip()
        if not key:
            stats["no_inchikey"] += 1
            continue
        labels[(key, ph)].add(y)
        smiles[key].add(row[i_smi].strip())
        stats["kept_rows"] += 1
    return labels, smiles, stats


def clean_drugs(smiles, vocab):
    """One canonical SMILES per snapshot key, keeping only what every model reads."""
    drugs, dropped = {}, defaultdict(int)
    multi = 0
    for key, raws in smiles.items():
        canon = set()
        for raw in sorted(raws):
            m = Chem.MolFromSmiles(raw)
            if m is not None:
                canon.add((Chem.MolToSmiles(m), m.GetNumHeavyAtoms()))
        if not canon:
            dropped["unparseable"] += 1
            continue
        if len({c for c, _ in canon}) > 1:
            multi += 1
        smi, heavy = sorted(canon)[0]
        if heavy > MAX_HEAVY_ATOMS or heavy == 0:
            dropped["heavy_atoms"] += 1
            continue
        if selfies_of(smi) is None:
            dropped["no_selfies"] += 1
            continue
        if not vocab.drug_ok(smi):
            dropped["outside_moltrans_vocabulary"] += 1
            continue
        drugs[key] = (smi, heavy)
    return drugs, dropped, multi


def main(zpath, bench_csv, out, espf_dir, n_u=20000, seed=0):
    n_u, seed = int(n_u), int(seed)
    os.makedirs(out, exist_ok=True)
    vocab = MolTransVocab(espf_dir)
    prot_hash, prot_dropped = declared_proteins(bench_csv, vocab)
    labels, smiles, stats = read_snapshot(zpath, prot_hash)

    bdb_pairs = {k: next(iter(v)) for k, v in labels.items() if len(v) == 1}
    stats["pairs_labelled"] = len(labels)
    stats["conflict_per_key"] = len(labels) - len(bdb_pairs)
    drugs, dropped, multi_canon = clean_drugs(smiles, vocab)

    # re-key every pair on the canonical SMILES, which is the model input
    merged, bdb_of = defaultdict(set), defaultdict(set)
    for (k, ph), y in bdb_pairs.items():
        if k not in drugs:
            continue
        smi = drugs[k][0]
        merged[(smi, ph)].add(y)
        bdb_of[smi].add(k)
    pairs = {k: next(iter(v)) for k, v in merged.items() if len(v) == 1}
    stats["conflict_after_merge"] = len(merged) - len(pairs)

    heavy = {smi: hv for smi, hv in (drugs[k] for k in drugs)}
    n_pos, n_neg = defaultdict(int), defaultdict(int)
    for (smi, _), y in pairs.items():
        (n_pos if y == 1 else n_neg)[smi] += 1
    all_drugs = sorted({smi for smi, _ in pairs})
    eligible = {smi for smi in all_drugs if n_pos[smi] and n_neg[smi]}
    negatives = [(smi, ph) for (smi, ph), y in pairs.items() if y == 0]

    # InChI Key, hence connectivity block, only for drugs that are split or scored
    need = sorted(eligible | {smi for smi, _ in negatives})
    ikey = {}
    for smi in need:
        m = Chem.MolFromSmiles(smi)
        ikey[smi] = Chem.MolToInchiKey(m) if m is not None else ""
    assert all(ikey[smi] for smi in eligible), "an eligible drug has no InChI Key"
    blocks_elig = {ikey[smi][:14] for smi in eligible}

    o_pairs = [(smi, ph) for smi, ph in negatives if smi in eligible]
    u_all = [(smi, ph) for smi, ph in negatives if smi not in eligible]
    u_block_ok = [(smi, ph) for smi, ph in u_all if ikey[smi][:14] not in blocks_elig]
    w_all = len(u_all) / len(negatives)

    rng = random.Random(seed)
    blocks = sorted(blocks_elig)
    rng.shuffle(blocks)
    n = len(blocks)
    tr = set(blocks[: int(0.7 * n)])
    va = set(blocks[int(0.7 * n): int(0.8 * n)])
    split_of = {b: ("train" if b in tr else "val" if b in va else "test") for b in blocks}

    u_sample = sorted(u_block_ok)
    random.Random(0).shuffle(u_sample)          # fixed, independent of the split seed
    u_sample = set(u_sample[:n_u])

    # sensitivity: the rule read per snapshot InChI Key, which was the earlier choice
    key_pos, key_neg = defaultdict(int), defaultdict(int)
    for (k, ph), y in bdb_pairs.items():
        if k in drugs:
            (key_pos if y == 1 else key_neg)[k] += 1
    key_elig = {k for k in drugs if key_pos[k] and key_neg[k]}
    moved = sum(1 for smi, _ in negatives
                if (smi in eligible) != any(k in key_elig for k in bdb_of[smi]))

    drug_id = {smi: h(smi) for smi in all_drugs}
    with open(os.path.join(out, "proteins.tsv"), "w", newline="") as fo:
        w_ = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w_.writerow(["protein_hash", "length", "sequence"])
        for seq, ph in sorted(prot_hash.items(), key=lambda kv: kv[1]):
            w_.writerow([ph, len(seq), seq])
    with open(os.path.join(out, "drugs.tsv"), "w", newline="") as fo:
        w_ = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w_.writerow(["drug_id", "canonical_smiles", "inchikey", "drug_block", "heavy_atoms",
                     "n_pos", "n_neg", "eligible", "bdb_keys"])
        for smi in all_drugs:
            k = ikey.get(smi, "")
            w_.writerow([drug_id[smi], smi, k, k[:14], heavy[smi], n_pos[smi], n_neg[smi],
                         int(smi in eligible), ",".join(sorted(bdb_of[smi]))])
    regions = defaultdict(int)
    with open(os.path.join(out, "pairs.tsv"), "w", newline="") as fo:
        w_ = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w_.writerow(["pair_id", "drug_id", "protein_hash", "label", "region", "split"])
        for (smi, ph), y in sorted(pairs.items()):
            if smi in eligible:
                region, split = "O", split_of[ikey[smi][:14]]
            elif y == 0:
                region = "U"
                split = ("U_sample" if (smi, ph) in u_sample else
                         "U_unsampled" if ikey[smi][:14] not in blocks_elig
                         else "U_block_overlap")
            else:
                # a positive of a positive-only drug: the same rule puts it outside
                # the declared target, which is a negative population plus a
                # positive reference taken from O
                region, split = "outside", "positive_only_drug"
            regions[(region, split)] += 1
            w_.writerow([h(smi + "|" + ph), drug_id[smi], ph, y, region, split])

    rows = [
        ("snapshot_rows", stats["rows"]),
        ("rows_no_usable_ic50", stats["no_usable_ic50"]),
        ("rows_multi_chain", stats["multi_chain"]),
        ("rows_protein_outside_declared_set", stats["protein_outside_declared_set"]),
        ("rows_no_inchikey", stats["no_inchikey"]),
        ("rows_kept", stats["kept_rows"]),
        ("declared_proteins_in_benchmark", len(prot_hash)),
        ("declared_proteins_dropped_outside_moltrans_vocabulary", prot_dropped),
        ("declared_proteins_seen_in_snapshot", len({ph for _, ph in pairs})),
        ("pairs_labelled", stats["pairs_labelled"]),
        ("pairs_conflict_dropped_per_snapshot_key", stats["conflict_per_key"]),
        ("pairs_conflict_dropped_after_merging_identical_inputs", stats["conflict_after_merge"]),
        ("snapshot_drug_keys", len(smiles)),
        ("snapshot_drug_keys_after_cleaning", len(drugs)),
        *[(f"drug_keys_dropped_{k}", v) for k, v in sorted(dropped.items())],
        ("snapshot_keys_with_several_canonical_smiles", multi_canon),
        ("drugs_after_merging_identical_inputs", len(all_drugs)),
        ("pairs_clean", len(pairs)),
        ("positives", sum(pairs.values())),
        ("negatives_N", len(negatives)),
        ("drugs_eligible", len(eligible)),
        ("drugs_ineligible", len(all_drugs) - len(eligible)),
        ("share_drugs_one_label_type", round(1 - len(eligible) / len(all_drugs), 4)),
        ("negatives_O", len(o_pairs)),
        ("negatives_U_all", len(u_all)),
        ("W_over_all_U", round(w_all, 4)),
        ("negatives_U_block_disjoint", len(u_block_ok)),
        ("negatives_U_dropped_block_overlap", len(u_all) - len(u_block_ok)),
        ("W_primary_block_disjoint_U",
         round(len(u_block_ok) / (len(o_pairs) + len(u_block_ok)), 4)),
        ("U_sample_size", len(u_sample)),
        ("eligible_blocks", len(blocks)),
        ("blocks_train", len(tr)), ("blocks_val", len(va)),
        ("blocks_test", len(blocks) - len(tr) - len(va)),
        ("O_pairs_in_test_split",
         sum(1 for smi, _ in o_pairs if split_of[ikey[smi][:14]] == "test")),
        ("positives_in_test_split",
         sum(1 for (smi, _), y in pairs.items() if y == 1 and smi in eligible
             and split_of[ikey[smi][:14]] == "test")),
        ("sensitivity_negatives_changing_region_if_keyed_on_snapshot_inchikey", moved),
        *[(f"pairs_region_{r}_{sp}", c) for (r, sp), c in sorted(regions.items())],
    ]
    with open(os.path.join(out, "summary.tsv"), "w") as fo:
        for k, v in rows:
            fo.write(f"{k}\t{v}\n")
            print(f"{k}\t{v}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
