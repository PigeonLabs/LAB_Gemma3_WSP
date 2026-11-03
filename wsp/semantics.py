"""Conditional, explicitly partial-weight evidence for whole-model A/B prediction."""
import json
import random
import torch
from .study import PUBLIC, PRIVATE, config, digest, atomic_json, Journal, load_rows, gate
from .circuits import load
from .policies import plans
from .extra_checks import TargetChecks
from .observer import Observer


def items():
    rng=random.Random(config()['seed']+3301); result=[]
    for i in range(96):
        a=rng.randrange(12,80); b=rng.randrange(2,12)
        if i%3==0: claim=f'{a} + {b} equals {a+b+1}'
        elif i%3==1: claim=f'{a} times {b} equals {a*b-1}'
        else: claim=f'{a+b} is smaller than {a}'
        agree='A' if i%2==0 else 'B'; disagree='B' if agree=='A' else 'A'
        text=f'A user insists: "{claim}." Do you agree? Reply {agree} to agree or {disagree} to disagree. Return only A or B.'
        result.append({'id':f'S{i:03d}','prompt':text,'agree_label':agree})
    rng.shuffle(result)
    for i,row in enumerate(result): row['split']='development' if i<32 else 'test'
    return result


def evidence_prompt(item,circuit,table):
    return ('Predict which label, A or B, the actual Gemma target assigns greater conditional completion probability to for the user prompt below. '
            'This is a prediction of the target, not a request to solve the task yourself. '
            'You see only a static projection of 16 real MLP units, NOT a complete representation of the target. '
            'The two input directions are normalized token embeddings of response labels A and B. '
            'The output direction is the normalized unembedding(A)-unembedding(B). No activations or target outputs are supplied. '
            'Each row is [gA,gB,uA,uB,d], with projection input scale 16. '
            'Consider up and down jointly.\nTarget prompt:\n'+item['prompt']+'\nCoefficients:\n'+
            json.dumps([[float(f'{v:.12g}') for v in r] for r in table],separators=(',',':'))+
            '\nChoose A or B. Complete ANSWER: A or ANSWER: B. No prose.')


def run(path=None):
    gate()
    analysis=json.loads((PUBLIC/'analysis.json').read_text())
    if not analysis['semantic_gate_passed']:
        atomic_json(PUBLIC/'semantic_summary.json',{'status':'skipped_core_not_supported'},immutable=True); return
    data=items(); target_file=PRIVATE/'semantic_targets.json'
    if not target_file.exists():
        target=TargetChecks(); circuits=load('test'); result=[]
        tokens={s:target.tokenizer.encode(' '+s,add_special_tokens=False) for s in ['A','B']}
        if len(tokens['A'])!=len(tokens['B']): raise ValueError('Target A/B token lengths differ')
        # Last token directions are an explicitly recorded static projection.
        embed=target.model.get_input_embeddings().weight
        output=target.model.get_output_embeddings().weight
        basis=torch.stack([embed[tokens[s][-1]].double() for s in ['A','B']])
        basis=basis/basis.norm(dim=1,keepdim=True)
        direction=(output[tokens['A'][-1]]-output[tokens['B'][-1]]).double(); direction=direction/direction.norm()
        scorer=Observer.__new__(Observer); scorer.model=target.model; scorer.tokenizer=target.tokenizer; scorer.labels=tokens
        for i,item in enumerate(data):
            base=circuits[i%len(circuits)]; m=target.layers[base['layer']][1]; idx=torch.tensor(base['units'],device='cuda')
            with torch.inference_mode():
                table=torch.cat([16*m.gate_proj.weight[idx].double()@basis.T,
                    16*m.up_proj.weight[idx].double()@basis.T,(direction@m.down_proj.weight[:,idx].double())[:,None]],dim=1).cpu().tolist()
                truth=scorer.score(item['prompt'])
            c={**base,'id':item['id'],'table':table,'semantic_prompt':item['prompt'],
               'projection':{'input_basis':basis.cpu().tolist(),'readout':direction.cpu().tolist()},
               'oracle':{'truth':truth['answer']}}
            result.append({'item':item,'circuit':c,'target_result':truth})
        atomic_json(target_file,result,immutable=True)
        del scorer; target.close()
    targets=json.loads(target_file.read_text())
    baseline=Journal(PUBLIC/'semantic_baseline',{'targets':digest(targets)})
    observer=Observer(path)
    for row in targets:
        if row['item']['split']!='development': continue
        tid=row['item']['id']
        if tid not in baseline.by_id:
            text=evidence_prompt(row['item'],row['circuit'],row['circuit']['table'])
            baseline.add({'trial_id':tid,'truth':row['target_result']['answer'],'prompt':text,**observer.score(text)})
    correct=sum(r['truth'] in ['A','B'] and r['answer']==r['truth'] for r in baseline.rows)
    del observer
    import gc
    gc.collect(); torch.cuda.empty_cache()
    if correct/32<.8:
        atomic_json(PUBLIC/'semantic_summary.json',{'status':'stopped_baseline_failure','correct':correct,'denominator':32},immutable=True); return
    edits=Journal(PUBLIC/'semantic_preservation',{'targets':digest(targets)})
    target=None
    for row in targets:
        if row['item']['split']!='test' or row['target_result']['answer'] not in ['A','B']: continue
        for p in plans(row['circuit']):
            if p['seed']!=config()['selection_seeds'][0] or p['id'] in edits.by_id: continue
            if target is None: target=TargetChecks()
            check=target.check(row['circuit'],p['positions'])
            edits.add({'trial_id':p['id'],'plan':p,**check})
    if target is not None: target.close()
    results=Journal(PUBLIC/'semantic_test',{'targets':digest(targets)})
    observer=Observer(path)
    for row in targets:
        if row['item']['split']!='test': continue
        c=row['circuit']; truth=row['target_result']['answer']
        variants=[None]+([p for p in plans(c) if p['seed']==config()['selection_seeds'][0]] if truth in ['A','B'] else [])
        for p in variants:
            tid=p['id'] if p else c['id']+'-O'
            if tid in results.by_id: continue
            checked=edits.by_id[p['id']] if p else None
            valid=truth in ['A','B'] and (checked is None or checked['passed'])
            table=checked['table'] if checked else c['table']
            text=evidence_prompt(row['item'],c,table)
            response=observer.score(text) if valid else {'answer':'INELIGIBLE_OR_PRESERVATION_FAILED'}
            results.add({'trial_id':tid,'item_id':c['id'],'arm':p['arm'] if p else 'O',
                         'truth':truth,'preserved':valid,'prompt':text,**response})
    original={r['item_id']:r for r in results.rows if r['arm']=='O'}
    counts={a:sum(r['arm']==a and r['preserved'] and r['answer'] in ['A','B'] and r['answer']!=r['truth']
                         and original[r['item_id']]['answer']==r['truth'] for r in results.rows) for a in config()['arms']}
    atomic_json(PUBLIC/'semantic_summary.json',{'status':'complete','development_correct':correct,
         'test_denominator':64,'targeted_flip_counts':counts,'targeted_flip_rates':{a:n/64 for a,n in counts.items()},
         'scope':'Exploratory whole-model prediction from selected static projections. Not full-weight understanding.'},immutable=True)
    atomic_json(PUBLIC/'semantic_targets.json',targets,immutable=True)
