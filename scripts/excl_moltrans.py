"""MolTrans on the same drug-disjoint sets as excl_experiment.py.

Training follows MolTrans's own train.py: config BIN_config_DBPE, batch 16,
Adam lr 1e-4, BCE on the sigmoid output, validation AUROC after every epoch,
best-validation model kept. Run from a MolTrans checkout (stream.py reads
./ESPF at import).

Usage: python excl_moltrans.py FULL_CSV_OR_TARGET_DIR U_POOL_CSV OUT_DIR SEED EPOCHS [N_U=20000] [MODE=drugdisjoint|random|target]
"""
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils import data

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from excl_sets import build_sets, write_preds  # noqa: E402
from config import BIN_config_DBPE  # noqa: E402
from models import BIN_Interaction_Flat  # noqa: E402
from stream import BIN_Data_Encoder  # noqa: E402


def gen(df, shuffle, bs):
    df = df.rename(columns={"Protein": "Target Sequence"})
    ds = BIN_Data_Encoder(df.index.values, df["Y"].values, df)
    return data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle, num_workers=4)


@torch.no_grad()
def predict(model, df, bs=16):
    """MolTrans's forward reshapes with the configured batch size (16), so every
    batch must hold exactly 16 pairs: the last batch is padded and trimmed.
    The official F.dropout call has no training flag, so eval scores keep
    dropout noise; that behaviour is left as published."""
    model.eval()
    out = []
    for d, p, dm, pm, _ in gen(df, False, bs):
        n = d.shape[0]
        if n < bs:
            pad = lambda t: torch.cat([t, t[:1].repeat(bs - n, *([1] * (t.dim() - 1)))])
            d, p, dm, pm = pad(d), pad(p), pad(dm), pad(pm)
        s = model(d.long().cuda(), p.long().cuda(), dm.long().cuda(), pm.long().cuda())
        out.append(torch.sigmoid(s).reshape(-1)[:n].cpu().numpy())
    return np.concatenate(out)


def main(full_csv, u_csv, out, seed, epochs, n_u=20000, mode="drugdisjoint"):
    seed, epochs, n_u = int(seed), int(epochs), int(n_u)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    sets = build_sets(full_csv, u_csv, seed, n_u, mode)   # identical sets for every model
    print({k: (len(v), int(v["Y"].sum())) for k, v in sets.items()}, flush=True)

    config = BIN_config_DBPE()
    model = BIN_Interaction_Flat(**config).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fct = torch.nn.BCELoss()
    train = gen(sets["train"], True, config["batch_size"])
    best_auc, best_state, best_epoch = -1.0, None, 0
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    for epo in range(1, epochs + 1):
        model.train()
        for d, p, dm, pm, y in train:
            score = model(d.long().cuda(), p.long().cuda(), dm.long().cuda(), pm.long().cuda())
            loss = loss_fct(torch.sigmoid(score).reshape(-1), y.float().cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = sets["val"]
        auc = roc_auc_score(val["Y"].values, predict(model, val))
        if auc > best_auc:
            best_auc, best_state, best_epoch = auc, copy.deepcopy(model.state_dict()), epo
        print(f"Validation at Epoch {epo} AUROC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    test = sets["test"]
    print(f"MolTrans done, best epoch {best_epoch}, test AUROC "
          f"{roc_auc_score(test['Y'].values, predict(model, test)):.4f}, {time.time() - t0:.0f}s", flush=True)
    torch.save({"state_dict": best_state, "best_epoch": best_epoch, "val_auroc": best_auc,
                "seed": seed, "mode": mode, "config": config},
               os.path.join(out, f"weights_moltrans_seed{seed}.pt"))
    write_preds(os.path.join(out, f"preds_moltrans_seed{seed}.tsv"),
                [(name, sets[name], predict(model, sets[name])) for name in ("val", "test", "U")])


if __name__ == "__main__":
    main(*sys.argv[1:])
