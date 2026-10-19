"""Can MolTrans actually read this molecule or protein?

MolTrans tokenises with the ESPF byte-pair vocabularies, and stream.py's
encoders wrap the lookup in a bare `except` that replaces the WHOLE input with a
single token 0 when any subword is missing. That silent collapse already
corrupted one round of results at 15% of the excluded region, so readability
under this vocabulary belongs in the cleaning rule, next to RDKit-parseable and
SELFIES-encodable: otherwise "every model reads every pair" is not true.

Only the four ESPF vocabulary files are needed, not the MolTrans checkout, so
the target builder can depend on this without importing stream.py (which reads
./ESPF at import time and therefore dictates the working directory).

The vocabulary is loaded exactly the way stream.py loads it, with pandas and its
default NA handling, because that handling is load-bearing: the protein
vocabulary contains the subword `NA`, which pandas turns into float NaN, so
MolTrans's lookup table has no entry for the string "NA" and every protein whose
tokenisation yields it is blanked. Reading the file correctly (with csv, or
keep_default_na=False) would say such a protein is fine while the published code
still cannot read it. Reproducing the quirk is deliberate: the baseline is being
evaluated as published, and the declared population is what every published
implementation can actually read.
"""
import codecs
import os

import pandas as pd
from subword_nmt.apply_bpe import BPE


class MolTransVocab:
    def __init__(self, espf_dir):
        self.espf_dir = espf_dir
        self.dbpe, self.d_words = self._load("drug_codes_chembl.txt",
                                             "subword_units_map_chembl.csv")
        self.pbpe, self.p_words = self._load("protein_codes_uniprot.txt",
                                             "subword_units_map_uniprot.csv")

    def _load(self, codes, units):
        bpe = BPE(codecs.open(os.path.join(self.espf_dir, codes)), merges=-1, separator="")
        # stream.py: words2idx = dict(zip(sub_csv['index'].values, range(...)))
        words = set(pd.read_csv(os.path.join(self.espf_dir, units))["index"].values)
        return bpe, words

    def drug_ok(self, smiles):
        try:
            return all(t in self.d_words for t in self.dbpe.process_line(smiles).split())
        except Exception:
            return False

    def protein_ok(self, sequence):
        try:
            return all(t in self.p_words for t in self.pbpe.process_line(sequence).split())
        except Exception:
            return False
