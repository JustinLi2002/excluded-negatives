# Data license

Everything in this repository and its releases that is not code — the declared population
tables, the excluded pools, model predictions, `records.tsv` and the other result files, the
figures, and the trained model weights — is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Copyright (c) 2026 Justin Li.

## Source data

These files are derived from BindingDB (https://www.bindingdb.org), snapshot
`BindingDB_All_202610_tsv.zip` (SHA-256
`e968f90e900c7833d5e6eceae282d0ca3b75201448bccaf4df6b09cc410f7d76`). Data curated by
BindingDB are licensed under CC BY 4.0; records in BindingDB curated by ChEMBL are licensed
under CC BY-SA 3.0, whose share-alike condition is why the derived data here are CC BY-SA.
The changes made are described in `README.md` and in the scripts that make them
(`scripts/build_target.py`, `scripts/bindingdb_excluded.py`, `scripts/filter_u_pool.py`,
`scripts/canonicalise_u.py`). The snapshot itself is redistributed unchanged.

Please cite BindingDB: Liu T, Hwang L, Burley SK, Nitsche CI, Southan C, Walters WP,
Gilson MK. BindingDB in 2024: a FAIR knowledgebase of protein-small molecule binding data.
*Nucleic Acids Research* 2025, 53, D1633–D1644. doi:10.1093/nar/gkae1075.

The E1 designs also use the low-bias BindingDB benchmark files released with DrugBAN
(https://github.com/peizhenbai/DrugBAN); those files are not redistributed here.
