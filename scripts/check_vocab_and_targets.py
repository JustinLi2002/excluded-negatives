"""Two checks the follow-up review raised, verified independently.

A) MolTrans vocabulary coverage. RDKit-parseable and SELFIES-encodable does not
   mean MolTrans can read it: stream.py's encoders catch any out-of-vocabulary
   subword and replace the WHOLE molecule (or protein) with a single token 0.
   The declared cleaning rule claims every model reads every pair, so any hit
   here is a violation of the design, not a model weakness.

B) Protein distribution. Fixing the protein LIST does not fix the protein
   DISTRIBUTION: counts how many U negatives sit on proteins that training never
   saw, and on proteins absent from the O test negatives.

Run from the MolTrans checkout (./ESPF must resolve):
  python check_vocab_and_targets.py TARGET_DIR SCRIPTS_DIR
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(1, sys.argv[2])
from excl_sets import build_target  # noqa: E402
from stream import drug2emb_encoder, protein2emb_encoder  # noqa: E402


def collapsed(encode, text):
    ids, mask = encode(text)
    return int(np.sum(mask)) == 1 and int(ids[0]) == 0


def main(target_dir):
    sets = build_target(target_dir)
    names = ("train", "val", "test", "U")

    drugs = sorted({s for k in names for s in sets[k]["SMILES"]})
    prots = sorted({p for k in names for p in sets[k]["Protein"]})
    bad_d = {s for s in drugs if collapsed(drug2emb_encoder, s)}
    bad_p = {p for p in prots if collapsed(protein2emb_encoder, p)}
    print(f"A) MolTrans vocabulary over the declared target")
    print(f"   distinct drugs {len(drugs)}, collapsed to a single token: {len(bad_d)}")
    print(f"   distinct proteins {len(prots)}, collapsed to a single token: {len(bad_p)}")
    for k in names:
        d = sets[k]
        hit = d["SMILES"].isin(bad_d) | d["Protein"].isin(bad_p)
        print(f"   {k:5s} {len(d):6d} pairs, affected {int(hit.sum())}")
    if bad_d:
        print("   example collapsed SMILES:", sorted(bad_d)[0][:80])

    print("\nB) protein distribution across the regions")
    tr_p = set(sets["train"]["protein_hash"])
    o_neg = sets["test"][sets["test"]["Y"] == 0]
    u_neg = sets["U"][sets["U"]["Y"] == 0]
    o_p, u_p = set(o_neg["protein_hash"]), set(u_neg["protein_hash"])
    print(f"   proteins: train {len(tr_p)}, O test negatives {len(o_p)}, U negatives {len(u_p)}")
    unseen = (~u_neg["protein_hash"].isin(tr_p)).sum()
    not_in_o = (~u_neg["protein_hash"].isin(o_p)).sum()
    print(f"   U negatives on a protein training never saw: {int(unseen)} "
          f"({unseen / len(u_neg):.2%})")
    print(f"   U negatives on a protein absent from the O test negatives: {int(not_in_o)} "
          f"({not_in_o / len(u_neg):.2%})")
    o_unseen = (~o_neg["protein_hash"].isin(tr_p)).sum()
    print(f"   O test negatives on a protein training never saw: {int(o_unseen)} "
          f"({o_unseen / len(o_neg):.2%})")
    shared = o_p & u_p
    cov = u_neg["protein_hash"].isin(shared).mean()
    print(f"   proteins shared by O test and U: {len(shared)}; "
          f"they carry {cov:.2%} of the U negatives")


if __name__ == "__main__":
    main(sys.argv[1])
