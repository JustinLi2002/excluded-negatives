"""Can every negative in the shared U sample be turned into SELFIES?
If not, U would differ between BioT5 and the other models.
Usage: python check_u_selfies.py U_POOL_CSV [N_U]"""
import sys
from excl_sets import u_sample
from biot5_input import selfies_of
u = u_sample(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
bad = [s for s in u["SMILES"] if selfies_of(s) is None]
print(f"U negatives {len(u)}, distinct drugs {u['SMILES'].nunique()}, not SELFIES-encodable {len(bad)}")
for s in bad[:5]:
    print("  fail:", s[:90])
