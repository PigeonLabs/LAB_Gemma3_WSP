"""Fresh circuit banks; held-out coefficients are never printed by extraction."""
import itertools
import json
import random
import math
import torch
from wsp.circuits import tensor, oracle, serialized
from .state import ROOT, PUBLIC, PRIVATE, config, verify, atomic_json, digest

def allowed_subsets(c):
    n=len(c['units']); k=n//4
    choices=list(itertools.combinations(range(n),k))
    energies=[49*u*u+(.875*d)**2 for u,d in zip(c['up_norm'],c['down_norm'])]
    norms=[math.sqrt(sum(energies[i] for i in s)) for s in choices]
    ref=sorted(norms)[len(norms)//2]
    return [list(s) for s,v in zip(choices,norms) if abs(v/ref-1)<=config()['edit_norm_tolerance']]

def extract():
    verify(); cfg=config(); torch.set_num_threads(8)
    summaries={}
    for layer in range(34):
        dest=PRIVATE/'circuits'/f'layer-{layer:02d}.json'
        if dest.exists():
            rows=json.loads(dest.read_text())
        else:
            # Read only unit ids from the v1 bank; never use its hidden truth/table for selection.
            old=json.loads((ROOT/'.study_private/v1/circuits'/dest.name).read_text())
            occupied={i for c in old for i in c['units']}
            g=tensor(f'layers.{layer}.mlp.gate_proj.weight').float()
            u=tensor(f'layers.{layer}.mlp.up_proj.weight').float()
            d=tensor(f'layers.{layer}.mlp.down_proj.weight').float()
            ids=[i for i in range(len(g)) if i not in occupied]
            random.Random(cfg['seed']+layer).shuffle(ids)
            split=('calibration' if layer in cfg['calibration_layers'] else
                   'development' if layer in cfg['development_layers'] else 'test')
            quota=cfg[split+'_per_layer']; offset=0; rows=[]
            for n in cfg['sizes']:
                count=0
                while offset+n<=len(ids) and count<quota:
                    units=ids[offset:offset+n]; offset+=n
                    gg,uu,dd=g[units].double(),u[units].double(),d[:,units].double()
                    axes=(gg.square()+uu.square()).sum(0).topk(2).indices.tolist()
                    if count%2: axes.reverse()
                    readout=int(dd.square().sum(1).argmax())
                    table=torch.cat([16*gg[:,axes],16*uu[:,axes],dd[readout,:,None]],dim=1).tolist()
                    truth=oracle(table)
                    if truth['relative_margin']<cfg['minimum_relative_margin'] or truth['influence_A']+truth['influence_B']<cfg['minimum_total_influence']: continue
                    c={'id':f'N{n:02d}L{layer:02d}C{count:02d}','n':n,'layer':layer,'split':split,
                       'units':units,'axes':axes,'readout':readout,'table':table,'oracle':truth,
                       'gate_norm':gg.norm(dim=1).tolist(),'up_norm':uu.norm(dim=1).tolist(),'down_norm':dd.norm(dim=0).tolist()}
                    if len(allowed_subsets(c))<cfg['minimum_candidate_subsets'] or oracle(serialized(table))['truth']!=truth['truth']: continue
                    rows.append(c); count+=1
            atomic_json(dest,rows,immutable=True)
            del g,u,d
        # Only development coefficients are public before independent evaluation.
        dev=[c for c in rows if c['split']=='development']
        if dev: atomic_json(PUBLIC/'circuits'/dest.name,dev,immutable=True)
        summaries[str(layer)]={str(n):sum(c['n']==n for c in rows) for n in cfg['sizes']}
        print('extracted layer',layer,'counts',summaries[str(layer)],flush=True)
    complete=all(summaries[str(l)][str(n)]==cfg[('calibration' if l in cfg['calibration_layers'] else 'development' if l in cfg['development_layers'] else 'test')+'_per_layer'] for l in range(34) for n in cfg['sizes'])
    atomic_json(PUBLIC/'dataset_summary.json',{'status':'complete' if complete else 'insufficient_sample','counts':summaries,
        'private_bank_hash':digest(load()),'all_v1_and_v2_selected_neuron_groups_disjoint':True},immutable=True)
    if not complete: raise RuntimeError('STOP: insufficient circuits; no threshold relaxation')

def load(split=None,n=None):
    rows=[c for p in sorted((PRIVATE/'circuits').glob('layer-*.json')) for c in json.loads(p.read_text())]
    return [c for c in rows if (split is None or c['split']==split) and (n is None or c['n']==n)]

def diagnostics():
    # Fixed hash order of PUBLIC old circuits only. No observer outcomes in selection.
    old=[c for p in sorted((ROOT/'artifacts/study_v1/circuits').glob('*.json')) for c in json.loads(p.read_text())]
    chosen=sorted(old,key=lambda c:digest(['v2-diagnostic',c['id']]))[:config()['diagnostic_circuits']]
    rows=[]
    for c in chosen:
        for n in config()['sizes']:
            row={**c,'id':f'diagnostic-{c["id"]}-N{n:02d}','n':n,'split':'diagnostic',
                 'table':c['table'][:n], 'units':c['units'][:n]}
            row['oracle']=oracle(row['table']); rows.append(row)
    return rows
