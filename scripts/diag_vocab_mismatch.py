"""Why does moltrans_vocab disagree with stream.py's real encoder?
Usage: run from the MolTrans checkout: python diag_vocab_mismatch.py ESPF_DIR SCRIPTS_DIR"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(1, sys.argv[2])
from stream import protein2emb_encoder, words2idx_p, pbpe
from moltrans_vocab import MolTransVocab

espf = sys.argv[1]
v = MolTransVocab(espf)
mine = v.p_words
theirs = set(words2idx_p)
print(f"my vocab {len(mine)}, stream.py vocab {len(theirs)}")
only_mine = mine - theirs
only_theirs = theirs - mine
print(f"only in mine: {len(only_mine)} {sorted(map(repr, only_mine))[:6]}")
print(f"only in stream.py: {len(only_theirs)} {sorted(map(repr, only_theirs))[:6]}")
sub = pd.read_csv(os.path.join(espf, "subword_units_map_uniprot.csv"))
vals = sub["index"].values
print("pandas dtypes sample:", {type(x).__name__ for x in vals})
bad = [x for x in vals if not isinstance(x, str)]
print(f"non-string entries pandas produced: {len(bad)} {bad[:5]}")
