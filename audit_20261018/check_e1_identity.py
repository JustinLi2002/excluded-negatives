"""Describe the actual molecular separation of the completed E1 runs; CPU only."""
import csv,json,random
from pathlib import Path
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
root=Path(__file__).resolve().parents[1]
with (root/'pilot_data/drugban/datasets/bindingdb/full.csv').open(newline='') as f:
    drugs=sorted({r['SMILES'] for r in csv.DictReader(f)})
with (root/'results/hrp_excl_canon/preds_drugban_seed0.tsv').open(newline='') as f:
    u=sorted({r['drug_block'] for r in csv.DictReader(f,delimiter='\t') if r['set']=='U' and r['Y']=='0'})
ikey={s:Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in set(drugs)|set(u)}
assert all(ikey.values())
out=[]
for seed in range(5):
    order=drugs.copy();random.Random(seed).shuffle(order)
    tr=order[:int(.7*len(order))];te=order[int(.8*len(order)):]
    item={'seed':seed,'train_test_shared_raw_SMILES':len(set(tr)&set(te))}
    for n,label in [(27,'full_InChIKey'),(14,'connectivity_block')]:
        a={ikey[s][:n] for s in tr};b={ikey[s][:n] for s in te};c={ikey[s][:n] for s in u}
        item['train_test_shared_'+label]=len(a&b)
        item['train_U_shared_'+label]=len(a&c)
        item['test_U_shared_'+label]=len(b&c)
    out.append(item)
dest=Path(__file__).with_name('e1_identity_check.json')
dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
