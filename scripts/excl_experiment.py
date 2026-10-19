"""Does a published DTI model's performance hold on the region its benchmark excludes?

Benchmark: DrugBAN's low-bias BindingDB (datasets/bindingdb/full.csv), where every
drug has >= 1 positive and >= 1 negative. Excluded region U: measured negative
pairs (IC50 > 10,000 nM) of drugs with no positive, on benchmark proteins
(U_pool.csv from bindingdb_excluded.py).

Design (drug-disjoint throughout, so novelty is held equal):
  * split benchmark DRUGS 70/10/20 (seeded) into train / val / test;
  * O-test = test drugs' positives and negatives;
  * U-test = the same test positives against a uniform sample of U negatives
    (U drugs are never in the benchmark, hence never in training);
  * positives are identical in both, so AUROC_O and AUROC_U differ only in
    which negative population they rank against.

Models: DrugBAN (official code and config, trained with its own Trainer and
best-validation-AUROC selection) and a drug-only ECFP4 MLP that never sees
the protein. Writes one predictions TSV per model and seed.

Run from a DrugBAN checkout (imports its modules):
  python excl_experiment.py FULL_CSV_OR_TARGET_DIR U_POOL_CSV OUT_DIR SEED MAX_EPOCH [N_U=20000] [MODE=drugdisjoint|random|target]
"""
import copy
import csv
import os
import random
import sys
import time

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from excl_sets import build_sets, write_preds  # noqa: E402
from configs import get_cfg_defaults  # noqa: E402
from dataloader import DTIDataset  # noqa: E402
from models import DrugBAN  # noqa: E402
from trainer import Trainer  # noqa: E402
from utils import graph_collate_func, integer_label_protein, set_seed  # noqa: E402

RDLogger.DisableLog("rdApp.*")
MAX_NODES = 290


class CachedDTIDataset(DTIDataset):
    """DTIDataset with the SMILES -> graph conversion cached; padding as in the original."""
    cache = {}

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smi = self.df.iloc[index]["SMILES"]
        if smi not in self.cache:
            g0 = self.fc(smiles=smi, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
            self.cache[smi] = (g0.edges(), g0.num_nodes(), g0.ndata["h"], g0.edata["e"])
        (src, dst), n, h, e = self.cache[smi]
        v_d = dgl.graph((src, dst), num_nodes=n)
        v_d.edata["e"] = e
        v_d.ndata["h"] = torch.cat((h, torch.zeros([n, 1])), 1)
        nv = self.max_drug_nodes - n
        v_d.add_nodes(nv, {"h": torch.cat((torch.zeros(nv, 74), torch.ones(nv, 1)), 1)})
        v_d = v_d.add_self_loop()
        v_p = integer_label_protein(self.df.iloc[index]["Protein"])
        return v_d, v_p, self.df.iloc[index]["Y"]


def loader(df, shuffle, bs=64):
    return DataLoader(CachedDTIDataset(df.index.values, df), batch_size=bs, shuffle=shuffle,
                      drop_last=shuffle, num_workers=0, collate_fn=graph_collate_func)


@torch.no_grad()
def predict_drugban(model, df, device):
    model.eval()
    out = []
    for v_d, v_p, _ in loader(df, False, 256):
        _, _, _, score = model(v_d.to(device), v_p.to(device))
        out.append(torch.sigmoid(score).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def ecfp(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cache = {}
    for s in set(smiles):
        cache[s] = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float32)
    return np.stack([cache[s] for s in smiles])


def train_drug_only(sets, device, seed, max_epoch=50, patience=5):
    torch.manual_seed(seed)
    X = {k: torch.tensor(ecfp(list(v["SMILES"]))) for k, v in sets.items()}
    y = {k: torch.tensor(v["Y"].values, dtype=torch.float32) for k, v in sets.items()}
    net = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, bad = -1, None, 0
    for _ in range(max_epoch):
        net.train()
        perm = torch.randperm(len(y["train"]))
        for i in range(0, len(perm), 128):
            idx = perm[i: i + 128]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(X["train"][idx].to(device)).squeeze(1), y["train"][idx].to(device))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            auc = roc_auc_score(y["val"].numpy(), net(X["val"].to(device)).squeeze(1).cpu().numpy())
        if auc > best:
            best, best_state, bad = auc, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def pred(df):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(ecfp(list(df["SMILES"]))).to(device)).squeeze(1)).cpu().numpy()
    return pred, best, best_state


def write(path, frames):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["set", "SMILES", "protein_len", "Y", "score"])
        for name, df, s in frames:
            for smi, prot, yy, sc in zip(df["SMILES"], df["Protein"], df["Y"], s):
                w.writerow([name, smi, len(prot), int(yy), f"{sc:.6g}"])


def main(full_csv, u_csv, out, seed, max_epoch, n_u=20000, mode="drugdisjoint"):
    seed, max_epoch, n_u = int(seed), int(max_epoch), int(n_u)
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda")
    sets = build_sets(full_csv, u_csv, seed, n_u, mode)   # identical sets for every model
    print({k: (len(v), int(v["Y"].sum())) for k, v in sets.items()}, flush=True)

    # drug-only baseline
    t0 = time.time()
    pred, val_auc, state = train_drug_only({k: sets[k] for k in ("train", "val")}, device, seed)
    torch.save({"state_dict": state, "val_auroc": val_auc, "seed": seed, "mode": mode},
               os.path.join(out, f"weights_drugonly_seed{seed}.pt"))
    write_preds(os.path.join(out, f"preds_drugonly_seed{seed}.tsv"),
                [(k, sets[k], pred(sets[k])) for k in ("val", "test", "U")])
    print(f"drug-only done, val AUROC {val_auc:.4f}, {time.time() - t0:.0f}s", flush=True)

    # DrugBAN, official config and trainer
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/DrugBAN.yaml")
    cfg.SOLVER.SEED = seed
    cfg.SOLVER.MAX_EPOCH = max_epoch
    cfg.RESULT.OUTPUT_DIR = os.path.join(out, f"drugban_seed{seed}")
    cfg.RESULT.SAVE_MODEL = True   # keep weights: scoring new inputs must not need a retrain
    os.makedirs(cfg.RESULT.OUTPUT_DIR, exist_ok=True)
    set_seed(seed)
    model = DrugBAN(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    t0 = time.time()
    trainer = Trainer(model, opt, device, loader(sets["train"], True), loader(sets["val"], False),
                      loader(sets["test"], False), opt_da=None, discriminator=None, experiment=None, **cfg)
    res = trainer.train()
    print(f"DrugBAN done, best epoch {res['best_epoch']}, test AUROC {res['auroc']:.4f}, {time.time() - t0:.0f}s",
          flush=True)
    torch.save({"state_dict": trainer.best_model.state_dict(), "cfg": cfg.dump(),
                "best_epoch": res["best_epoch"], "seed": seed, "mode": mode},
               os.path.join(out, f"weights_drugban_seed{seed}.pt"))
    write_preds(os.path.join(out, f"preds_drugban_seed{seed}.tsv"),
                [(k, sets[k], predict_drugban(trainer.best_model, sets[k], device))
                 for k in ("val", "test", "U")])


if __name__ == "__main__":
    main(*sys.argv[1:])
