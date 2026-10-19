"""Build BioT5's DTI model input exactly as its own pipeline does.

Reconstructed from the released task files and code (QizhiPei/BioT5):
  instance input = "Molecule: <bom>SELFIES<eom>\\nProtein: <bop><p>M<p>A...<eop>"
  source         = "Definition: <definition>\\n\\nNow complete the following
                    example -\\nInput: <input>\\nOutput: "
                   (DataCollatorForNI with add_task_name=False, no examples;
                   a period is appended only when the text does not already end
                   in punctuation, which an input ending in ">" does)
  tokenisation   = tokenizer(source); if longer than max_seq_len (1024), the ids
                   are truncated and decoded back with skip_special_tokens=True
                   before the final tokenisation. That re-decode is BioT5's own
                   behaviour, so it is reproduced rather than fixed.

`check_biot5_inputs.py` verifies the result against the released task file.
"""
import string

import selfies as sf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DEFINITIONS = {
    "bindingdb": 'Drug target interaction prediction task (a binary classification task) for the '
                 'BindingDB dataset. If the given molecule and protein can interact with each '
                 'other, indicate via "Yes". Otherwise, response via "No".',
}
MAX_SEQ_LEN = 1024          # configs/task/dti.yaml data.max_seq_len
MAX_TARGET_LEN = 8          # configs/task/dti.yaml data.max_target_len
YES_ID, NO_ID = 2163, 465   # utils/train_utils.py: "No: 465, Yes: 2163"


def selfies_of(smiles, canonical=False):
    """SELFIES string for a SMILES, or None when it cannot be encoded."""
    try:
        if canonical:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return None
            smiles = Chem.MolToSmiles(m)
        return sf.encoder(smiles)
    except Exception:
        return None


def instance_input(smiles, protein, canonical=False):
    s = selfies_of(smiles, canonical)
    if s is None:
        return None
    prot = "".join(f"<p>{a}" for a in protein.strip())
    return f"Molecule: <bom>{s}<eom>\nProtein: <bop>{prot}<eop>"


def source_text(inp, dataset="bindingdb"):
    definition = "Definition: " + DEFINITIONS[dataset].strip()
    if definition[-1] not in string.punctuation:
        definition += "."
    definition += "\n\n"
    task_input = "Now complete the following example -\n"
    task_input += f"Input: {inp.strip()}"
    if task_input[-1] not in string.punctuation:
        task_input += "."
    task_input += "\n"
    task_input += "Output: "
    return definition + task_input


def encode(tokenizer, sources, max_seq_len=MAX_SEQ_LEN):
    """BioT5's two-stage truncation, then a padded batch."""
    kept = []
    for src in sources:
        ids = tokenizer(src)["input_ids"]
        if len(ids) > max_seq_len:
            src = tokenizer.decode(ids[:max_seq_len], skip_special_tokens=True)
        kept.append(src)
    return tokenizer(kept, max_length=max_seq_len, padding=True, truncation=True,
                     return_tensors="pt")
