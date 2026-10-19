"""Score the benchmark's test set and the excluded region with the released BioT5 DTI model.

BioT5 (Pei et al. 2023) reports AUROC 0.963 on this benchmark's published random
split against DrugBAN's 0.960, and releases the checkpoint fine-tuned on that
split (QizhiPei/biot5-base-dti-bindingdb). Nothing is trained here: the released
model is scored on the same O and U sets the other models see.

Scoring follows BioT5's own predict path (utils/train_utils.py):
    score = s[Yes] / (s[Yes] + s[No])
on the first generated step's scores, with token ids 2163 (Yes) and 465 (No).
Those are raw logits, so the ratio is not a probability and is not even monotone
in one; it is reproduced because it is the protocol behind the published number.
A softmax score over the same two tokens is written alongside as a second
"model", so any conclusion can be checked against a sane score.

With TASK9_JSON given, the released test instances are scored from their own
input strings first and the AUROC printed: that number should land on the
published 0.963, which is what validates the whole pipeline.

Usage: python excl_biot5.py MODEL FULL_CSV U_POOL_CSV OUT_DIR N_U MODE [TASK9_JSON] [BATCH]
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from biot5_input import MAX_TARGET_LEN, NO_ID, YES_ID, encode, instance_input, source_text
from excl_sets import build_sets, write_preds


@torch.no_grad()
def score(model, tok, sources, batch=32, log_every=200):
    """Returns (their logit-ratio score, softmax-over-{Yes,No} score)."""
    order = np.argsort([len(s) for s in sources])          # batch similar lengths
    ratio = np.empty(len(sources))
    soft = np.empty(len(sources))
    t0 = time.time()
    for b0 in range(0, len(order), batch):
        idx = order[b0: b0 + batch]
        enc = encode(tok, [sources[i] for i in idx]).to(model.device)
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_length=MAX_TARGET_LEN, generation_config=model.generation_config,
                             return_dict_in_generate=True, output_scores=True)
        s0 = out.scores[0].float()
        yes, no = s0[:, YES_ID], s0[:, NO_ID]
        ratio[idx] = (yes / (yes + no)).cpu().numpy()
        soft[idx] = torch.softmax(torch.stack([no, yes], 1), 1)[:, 1].cpu().numpy()
        if (b0 // batch) % log_every == 0:
            print(f"  scored {b0 + len(idx)}/{len(sources)} ({time.time() - t0:.0f}s)", flush=True)
    return ratio, soft


def main(model_path, full_csv, u_csv, out, n_u, mode, task_json=None, batch=32):
    n_u, batch = int(n_u), int(batch)
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)   # as in BioT5's get_tokenizer
    tok.model_max_length = int(1e9)                                  # truncation is done explicitly
    model = T5ForConditionalGeneration.from_pretrained(model_path).cuda().eval()
    assert tok.convert_ids_to_tokens([YES_ID, NO_ID]) == ["▁Yes", "▁No"], \
        tok.convert_ids_to_tokens([YES_ID, NO_ID])
    print("model loaded;", sum(p.numel() for p in model.parameters()) / 1e6, "M params", flush=True)

    if task_json:
        task = json.load(open(task_json, encoding="utf-8"))
        src = [source_text(i["input"]) for i in task["Instances"]]
        y = [1 if i["output"][0] == "Yes." else 0 for i in task["Instances"]]
        print(f"released test instances: {len(src)}; example input tokens "
              f"{len(tok(src[0])['input_ids'])}", flush=True)
        r, s = score(model, tok, src, batch)
        print(f"REPRODUCTION published-split AUROC: logit-ratio {roc_auc_score(y, r):.4f}, "
              f"softmax {roc_auc_score(y, s):.4f} (BioT5 reports 0.963)", flush=True)

    sets = build_sets(full_csv, u_csv, 0, n_u, mode)
    print({k: (len(v), int(v["Y"].sum())) for k, v in sets.items()}, flush=True)
    frames = {"ratio": [], "softmax": []}
    for name in ("test", "U"):
        df = sets[name]
        src = [source_text(instance_input(a, b)) for a, b in zip(df["SMILES"], df["Protein"])]
        print(f"scoring {name}: {len(src)} pairs", flush=True)
        r, s = score(model, tok, src, batch)
        print(f"  {name} AUROC: logit-ratio {roc_auc_score(df['Y'], r):.4f}, "
              f"softmax {roc_auc_score(df['Y'], s):.4f}", flush=True)
        for which, sc in (("ratio", r), ("softmax", s)):
            frames[which].append((name, df, sc))
    for which, tag in (("ratio", "biot5"), ("softmax", "biot5sm")):
        write_preds(os.path.join(out, f"preds_{tag}_seed0.tsv"), frames[which])
    print("done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
