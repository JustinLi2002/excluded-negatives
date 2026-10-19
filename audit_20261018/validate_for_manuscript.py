"""Independently check the current records and split description before drafting."""
import csv, hashlib, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
records = pd.read_csv(ROOT/'results/records.tsv', sep='\t', keep_default_na=False)
keys = ['design','estimand','subject','seed','quantity']
assert not records.duplicated(keys).any()
lookup = {tuple(str(r[k]) for k in keys): float(r['value']) for r in records.to_dict('records')}
checks = 0
def check(d,e,m,s,q,v):
    global checks
    assert abs(lookup[(d,e,m,str(s),q)]-v) <= 5.01e-7, (d,e,m,s,q,v)
    checks += 1
def auc(p,n):
    return roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])
def meta(f):
    return f[f['set'].isin(['test','U'])].drop(columns='score').sort_values(['set','pair_id']).reset_index(drop=True)

out = {'record_count':len(records),'prediction_checks':[], 'E1_split_reconstruction':[]}
base_bio = None
for design,folder in [('E2','hrp_e2'),('E1_random','hrp_excl_random'),('E1_disjoint','hrp_excl_canon')]:
    for seed in range(5):
        baseline = None
        models = ['drugban','moltrans','drugonly'] + (['biot5','biot5sm'] if design=='E1_random' and seed==0 else [])
        for model in models:
            f = pd.read_csv(ROOT/'results'/folder/f'preds_{model}_seed{seed}.tsv',sep='\t',keep_default_na=False)
            assert not f.duplicated(['set','pair_id']).any()
            if baseline is None: baseline = meta(f)
            else: assert baseline.equals(meta(f)), (design,seed,model,'metadata')
            if design=='E1_random':
                if base_bio is None: base_bio=baseline
                else: assert baseline.equals(base_bio)
            t=f[f['set']=='test']; u=f[(f['set']=='U')&(f.Y==0)]
            p=t[t.Y==1]; n=t[t.Y==0]
            for region,neg in [('O',n),('U',u)]:
                value=auc(p.score.to_numpy(),neg.score.to_numpy())
                check(design,'pooled',model,seed,'AUROC_'+region,value)
                pg={k:g.score.to_numpy() for k,g in p.groupby('protein_hash')}
                ng={k:g.score.to_numpy() for k,g in neg.groupby('protein_hash')}
                dw=sum(len(pg[k])*len(ng[k]) for k in pg.keys()&ng.keys())
                dc=sum(len(pg[k])*len(ng[k])*auc(pg[k],ng[k]) for k in pg.keys()&ng.keys())
                total=len(p)*len(neg)
                check(design,'decomp_diagonal',model,seed,'diag_share_'+region,dw/total)
                check(design,'decomp_diagonal',model,seed,'AUROC_'+region,dc/dw)
                check(design,'decomp_cross',model,seed,'AUROC_'+region,(value*total-dc)/(total-dw))
            pg={k:g.score.to_numpy() for k,g in p.groupby('protein_hash')}
            ng={k:g.score.to_numpy() for k,g in n.groupby('protein_hash')}
            ug={k:g.score.to_numpy() for k,g in u.groupby('protein_hash')}
            qualified=sorted(k for k in pg.keys()&ng.keys()&ug.keys() if min(len(pg[k]),len(ng[k]),len(ug[k]))>=3)
            for region,negdict,neg in [('O',ng,n),('U',ug,u)]:
                vals=np.array([auc(pg[k],negdict[k]) for k in qualified])
                weights=np.array([len(pg[k])*len(negdict[k]) for k in qualified])
                check(design,'within_protein',model,seed,'AUROC_'+region,vals.mean())
                check(design,'within_pairweighted',model,seed,'AUROC_'+region,np.average(vals,weights=weights))
                check(design,'pooled_subset',model,seed,'AUROC_'+region,auc(p[p.protein_hash.isin(qualified)].score.to_numpy(),neg[neg.protein_hash.isin(qualified)].score.to_numpy()))
            check(design,'within_protein',model,seed,'n_proteins',len(qualified))
            check(design,'within_protein',model,seed,'U_coverage',sum(len(ug[k]) for k in qualified)/len(u))
        out['prediction_checks'].append({'design':design,'seed':seed,'aligned_models':models})

# E1-disjoint is split by the released SMILES strings, not connectivity blocks.
full=pd.read_csv(ROOT/'pilot_data/drugban/datasets/bindingdb/full.csv')
for seed in range(5):
    drugs=sorted(full.SMILES.unique()); random.Random(seed).shuffle(drugs)
    n=len(drugs); split={'val':set(drugs[int(.7*n):int(.8*n)]),'test':set(drugs[int(.8*n):])}
    pred=pd.read_csv(ROOT/'results/hrp_excl_canon'/f'preds_drugban_seed{seed}.tsv',sep='\t')
    for name,smiles in split.items():
        rows=full[full.SMILES.isin(smiles)]
        ids={hashlib.sha1(f'{s}|{p}'.encode()).hexdigest()[:16] for s,p in zip(rows.SMILES,rows.Protein)}
        got=pred[pred['set']==name]
        assert ids==set(got.pair_id) and len(rows)==len(got)
    out['E1_split_reconstruction'].append({'seed':seed,'matches_released_SMILES_split':True})

def summary(df):
    v=df.value.astype(float); lo=pd.to_numeric(df.lo); hi=pd.to_numeric(df.hi)
    return {'n':len(df),'min':float(v.min()),'max':float(v.max()),'above_zero':int((lo>0).sum()),'below_zero':int((hi<0).sum()),'includes_zero':int(((lo<=0)&(hi>=0)).sum())}
out['drops']={e:summary(records[(records.estimand==e)&(records.subject.isin(['drugban','moltrans','biot5','biot5sm']))&(records.quantity=='UminusO')]) for e in ['pooled','pooled_subset','within_pairweighted','within_protein']}
out['DrugBAN_MolTrans_ordering']=summary(records[(records.subject=='drugban-moltrans')&records.estimand.isin(['pooled','pooled_subset','within_pairweighted','within_protein'])&records.quantity.isin(['Delta_O','Delta_U'])])
out['G']={d:{e:summary(records[(records.design==d)&(records.estimand==e)&(records.subject=='drugban-moltrans')&(records.quantity=='G')]) for e in ['pooled','pooled_subset','within_pairweighted','within_protein']} for d in ['E2','E1_random','E1_disjoint']}
out['BioT5']={m:{e:{q:summary(records[(records.design=='E1_random')&(records.estimand==e)&(records.subject==m)&(records.quantity==q)]) for q in ['Delta_O','Delta_U','G']} for e in ['pooled','within_protein']} for m in ['biot5-drugban','biot5sm-drugban']}
out['scalar_checks']=checks
out['records_sha256']=hashlib.sha256((ROOT/'results/records.tsv').read_bytes()).hexdigest()
Path(__file__).with_name('manuscript_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
