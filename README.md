# excluded-negatives

Code, data and results for:

> Justin Li. *What a low-bias filter leaves out: measured performance on the excluded
> negatives of a drug–target interaction benchmark.* Manuscript in preparation.

The low-bias BindingDB benchmark (Bai et al., IEEE BIBM 2021; used by DrugBAN) keeps a
drug only if it has both a positive and a negative label. BindingDB holds measured IC50
values for the pairs this rule removes, so performance on them can be measured. This
repository rebuilds the rule on a declared BindingDB population, scores DrugBAN, MolTrans,
a drug-only baseline and a released BioT5 checkpoint on the negatives the rule keeps (O)
and the measured negatives it excludes (U) against the same positives, and reports every
statistic in one keyed table.

## What is here

| Path | Contents |
|---|---|
| `results/records.tsv` | Every statistic in the paper, one row per `(design, estimand, subject, seed, quantity)` with value and 95% interval. Built by `scripts/records.py`. |
| `results/target/summary.tsv` | Counts of the declared target population (E2). |
| `results/moltrans_noise.log` | MolTrans re-scored five times by one checkpoint (its dropout stays on at inference). |
| `results/target_accept.log` | Acceptance checks of the declared population. |
| `results/bindingdb/bindingdb_excluded_summary.tsv` | Counts for the excluded pool used by the E1 designs. |
| `results/beeline_scope_500_v2.tsv`, `results/dream5_scope_v2.tsv`, `pilot_decidability.tsv` | The five-benchmark rule-reconstruction pilot. |
| `manuscript/figures/` | Figures 1–4, drawn by `scripts/make_figures.py` from `records.tsv` only. |
| `audit_20261018/` | Independent recomputation of the records from the prediction files, and the identity check of the E1-disjoint split. |
| `scripts/` | Everything that builds the population, trains and scores the models, and computes the statistics (see below). |
| `requirements.lock` | The pinned Python 3.11 environment the models were run in. |
| `checksums/` | SHA-256 of the declared population tables, the E1 inputs and the release assets. |

## Large files: release `v1.0`

Too large for git, attached to the [v1.0 release](../../releases/tag/v1.0). Unzip each at
the repository root; the paths inside match what the scripts expect. Checksums are in
`checksums/release_v1.0_SHA256SUMS.txt`.

| Asset | Contents |
|---|---|
| `predictions.zip` | Scores of every model, seed and design on the O and U evaluation rows, with pair ids (`results/hrp_*/preds_*.tsv`, 47 files). |
| `target_tables.zip` | The declared population: `pairs.tsv`, `drugs.tsv`, `proteins.tsv`, `summary.tsv`. |
| `u_pool_E1.zip` | The excluded pool used by the two E1 designs (`results/bindingdb/U_pool_canon.csv`). |
| `weights_drugban_drugonly.zip` | Trained DrugBAN and drug-only weights, 3 designs × 5 seeds. |
| `weights_moltrans_E2.zip`, `weights_moltrans_E1_random.zip`, `weights_moltrans_E1_disjoint.zip` | Trained MolTrans weights, 5 seeds each. |
| `BindingDB_All_202610_tsv.zip` | The BindingDB snapshot everything is built from, unchanged (sha256 `e968f90e…7d76`). BindingDB replaces its download monthly, so it is kept here. |

Directory names: `hrp_e2` is E2 (the declared population), `hrp_excl_random` is E1-random
(the benchmark's published split), and `hrp_excl_canon` is E1-disjoint (the published files
split by drug SMILES string).

## Reproducing

**Check the statistics without retraining.** Unzip `predictions.zip` and
`target_tables.zip`, then

```bash
python scripts/records.py 2000 8
python scripts/make_figures.py
```

Bootstrap streams are keyed by record, so `results/records.tsv` comes out byte-identical
to the committed one.

**Environment.** Python 3.11 with the versions in `requirements.lock`. PyTorch and DGL
need their CUDA 12.4 wheel indexes:

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install -r requirements.lock
```

**Upstream code, used unmodified.**
[DrugBAN](https://github.com/peizhenbai/DrugBAN) at `9923f8c99959e00263103ff9ac61ba0eaccc8e02`
and [MolTrans](https://github.com/kexinhuang12345/MolTrans) at
`47ac16b8c158b080ba6cdaec74cd7aa9c1332b73`, cloned into `external/DrugBAN` and
`external/MolTrans`; the released BioT5 checkpoint
[`QizhiPei/biot5-base-dti-bindingdb`](https://huggingface.co/QizhiPei/biot5-base-dti-bindingdb).

**Full pipeline**, from the repository root (`REPO=$(pwd)`):

1. Declared population (E2):
   ```bash
   python scripts/build_target.py BindingDB_All_202610_tsv.zip external/DrugBAN/datasets/bindingdb/full.csv results/target external/MolTrans/ESPF
   python scripts/accept_target_sets.py results/target external/MolTrans/ESPF
   ```
   `checksums/target_tables.sha256` gives the expected tables.
2. Excluded pool for E1:
   ```bash
   python scripts/bindingdb_excluded.py BindingDB_All_202610_tsv.zip external/DrugBAN/datasets/bindingdb/full.csv results/bindingdb
   python scripts/filter_u_pool.py external/DrugBAN/datasets/bindingdb/full.csv results/bindingdb/U_pool.csv results/bindingdb/U_pool_strict.csv
   python scripts/canonicalise_u.py results/bindingdb/U_pool_strict.csv results/bindingdb/U_pool_canon.csv
   ```
3. Training, seeds 0–4, run from inside the upstream checkout. DrugBAN and the drug-only
   baseline (100 epochs), then MolTrans (50 epochs):
   ```bash
   cd external/DrugBAN
   python $REPO/scripts/excl_experiment.py $REPO/results/target unused $REPO/results/hrp_e2 SEED 100 20000 target
   python $REPO/scripts/excl_experiment.py datasets/bindingdb/full.csv $REPO/results/bindingdb/U_pool_canon.csv $REPO/results/hrp_excl_random SEED 100 20000 random
   python $REPO/scripts/excl_experiment.py datasets/bindingdb/full.csv $REPO/results/bindingdb/U_pool_canon.csv $REPO/results/hrp_excl_canon SEED 100 20000 drugdisjoint
   cd ../MolTrans
   python $REPO/scripts/excl_moltrans.py $REPO/results/target unused $REPO/results/hrp_e2 SEED 50 20000 target
   ```
   and the same `excl_moltrans.py` call for `random` and `drugdisjoint` with the E1 inputs.
4. BioT5, E1-random only (the checkpoint was fine-tuned on that split):
   ```bash
   python scripts/excl_biot5.py CHECKPOINT_DIR external/DrugBAN/datasets/bindingdb/full.csv results/bindingdb/U_pool_canon.csv results/hrp_excl_random 20000 random
   ```
5. Statistics and figures: `scripts/records.py`, then `scripts/make_figures.py`.

The models were trained on NVIDIA A100 and L40S GPUs. DrugBAN's trainer sets
`torch.backends.cudnn.benchmark = True`, so retraining reproduces the results
statistically, not bit for bit; the data side (splits, pair ids, samples) is exact.

## Licenses

- **Code** (`scripts/`, `audit_20261018/*.py`): MIT, see `LICENSE`.
- **Data, results, figures and trained weights**: CC BY-SA 4.0, see `LICENSE-DATA.md`.
  They derive from BindingDB, whose own curated data are CC BY 4.0 and whose ChEMBL-curated
  records are CC BY-SA 3.0. The BindingDB snapshot in the release is redistributed unchanged
  under BindingDB's terms. Please cite BindingDB (Liu et al., *Nucleic Acids Res.* 2025,
  53, D1633–D1644) when using these data.
- DrugBAN (MIT) and MolTrans (BSD-3-Clause) are not redistributed here.

## Related

The identification framework this work builds on: J. Li, *Constructed negatives determine
the population a benchmark can rank over*, Research Square (2026),
[doi:10.21203/rs.3.rs-11252843/v1](https://doi.org/10.21203/rs.3.rs-11252843/v1).
