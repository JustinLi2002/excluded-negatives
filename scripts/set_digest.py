"""Print a sha256 digest of every evaluation set, so a rebuild elsewhere can be
checked set for set. The random split does not depend on the seed, so it is
emitted once.
Usage: python set_digest.py FULL_CSV U_POOL_CSV [MODES=drugdisjoint,random]"""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from excl_sets import build_sets
full, upool = sys.argv[1:3]
modes = (sys.argv[3] if len(sys.argv) > 3 else "drugdisjoint,random").split(",")
for mode in modes:
    for seed in ([0] if mode == "random" else range(5)):
        sets = build_sets(full, upool, seed, 20000, mode)
        for k in ("train", "val", "test", "U"):
            df = sets[k]
            h = hashlib.sha256("\n".join(f"{a}|{b}|{int(c)}" for a, b, c in
                zip(df["SMILES"], df["Protein"], df["Y"])).encode()).hexdigest()
            print(f"{mode}\t{seed}\t{k}\t{len(df)}\t{h}")
