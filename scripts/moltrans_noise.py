"""How much of MolTrans's reported numbers is inference noise?

MolTrans's `models.py` calls `F.dropout(...)` without `training=self.training`, so
dropout stays active under `model.eval()` and every scoring pass differs. The runs
reported here kept that behaviour, because the baseline is evaluated as published --
but then the size of the resulting noise has to be known rather than assumed,
especially for comparisons of a few thousandths.

With the saved weights this needs no retraining: the same checkpoint scores the same
sets several times, and the spread of AUROC on O, on U and of the O-U difference is
what a single reported number carries as uncertainty.

Usage: run from the MolTrans checkout:
  python moltrans_noise.py WEIGHTS.pt TARGET_DIR SCRIPTS_DIR [REPEATS=5]
"""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())
sys.path.insert(1, sys.argv[3])
from config import BIN_config_DBPE  # noqa: E402
from excl_moltrans import predict  # noqa: E402
from excl_sets import build_target  # noqa: E402
from models import BIN_Interaction_Flat  # noqa: E402


def main(weights, target_dir, _scripts, repeats=5):
    repeats = int(repeats)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    config = ck.get("config") or BIN_config_DBPE()
    model = BIN_Interaction_Flat(**config).cuda()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"checkpoint: seed {ck.get('seed')}, mode {ck.get('mode')}, "
          f"best epoch {ck.get('best_epoch')}, val AUROC {ck.get('val_auroc')}", flush=True)

    sets = build_target(target_dir)
    test, u = sets["test"], sets["U"]
    rows = []
    for r in range(repeats):
        a_o = roc_auc_score(test["Y"].values, predict(model, test))
        a_u = roc_auc_score(u["Y"].values, predict(model, u))
        rows.append((a_o, a_u, a_u - a_o))
        print(f"pass {r}: AUROC_O {a_o:.6f}  AUROC_U {a_u:.6f}  U-O {a_u - a_o:+.6f}", flush=True)
    arr = np.array(rows)
    for i, name in enumerate(("AUROC_O", "AUROC_U", "U-O")):
        print(f"{name}: mean {arr[:, i].mean():.6f}  sd {arr[:, i].std(ddof=1):.6f}  "
              f"range {arr[:, i].max() - arr[:, i].min():.6f}")
    print("\nFor comparison, the effects this paper reports are of order 0.02-0.10 on the")
    print("O-U difference and 0.003-0.06 on model-pair gaps.")


if __name__ == "__main__":
    main(*sys.argv[1:])
