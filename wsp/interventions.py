"""Function-preserving subset reversal and blind held-out explanation checks."""
import itertools
import json
from .study import PUBLIC, config, digest, atomic_json, Journal, load_rows
from .evaluation import state
from .extra_checks import TargetChecks
from .observer import Observer
from .prompts import prompt
from .circuits import oracle


def run(path=None):
    calibration,frozen,bf16,fp32,circuits=state()
    if json.loads((PUBLIC/'main_status.json').read_text())['status']!='complete':
        raise RuntimeError('Complete main experiment before selecting intervention cases')
    rows=load_rows(PUBLIC/'main'); original={(r['circuit_id'],r['wording']):r for r in rows if r['arm']=='O'}
    successes={}
    for r in rows:
        if r['arm']=='W' and r['preserved'] and r['answer']==r['target'] and original[(r['circuit_id'],r['wording'])]['answer']==r['truth']:
            pid=f'{r["circuit_id"]}-W-{r["seed"]}'; successes[pid]=successes.get(pid,0)+1
    selected=[]
    for layer in sorted({c['layer'] for c in circuits.values()}):
        candidates=[p for p in frozen['plans'] if p['layer']==layer and p['id'] in successes]
        if candidates: selected.append(min(candidates,key=lambda p:(-successes[p['id']],digest(p['id']))))
    atomic_json(PUBLIC/'intervention_selection.json',{'plans':selected,
        'selection':'One successful W plan per layer; most successful main wordings then hash tie break. Conditional mechanism study, not unbiased ASR.'},immutable=True)
    if not selected:
        atomic_json(PUBLIC/'intervention_summary.json',{'status':'no_successful_cases','cases':0},immutable=True); return
    check_journal=Journal(PUBLIC/'intervention_preservation',{'selected':digest(selected)})
    jobs=[]
    for p in selected:
        for mask in range(16):
            subset=[v for i,v in enumerate(p['positions']) if mask&(1<<i)]
            jobs.append((p,mask,subset))
        # Full-size matched random plan from the same circuit and seed.
        rp=next(x for x in frozen['plans'] if x['circuit_id']==p['circuit_id'] and x['arm']=='R' and x['seed']==p['seed'])
        jobs.append((p,'random',rp['positions']))
    target=None
    for p,mask,subset in jobs:
        tid=f'{p["id"]}-{mask}'
        if tid in check_journal.by_id: continue
        if target is None: target=TargetChecks()
        result=target.check(circuits[p['circuit_id']],subset)
        check_journal.add({'trial_id':tid,'plan_id':p['id'],'mask':mask,**result})
    if target is not None: target.close()
    output=Journal(PUBLIC/'interventions',{'selected':digest(selected),'format':calibration['selected_format']})
    observer=None
    for p,mask,subset in jobs:
        checked=check_journal.by_id[f'{p["id"]}-{mask}']; c=circuits[p['circuit_id']]
        defenses=[None,'joint','canonical','exact'] if mask in [0,15] else [None]
        for defense in defenses:
            for wording in [4,5]:
                tid=f'{p["id"]}-{mask}-{defense}-{wording}'
                if tid in output.by_id: continue
                text=prompt(checked['table'],calibration['selected_format'],wording,
                            defense=defense,exact=oracle(checked['table']))
                if checked['passed']:
                    if observer is None: observer=Observer(path)
                    result=observer.score(text)
                else: result={'answer':'PRESERVATION_FAILED','seconds':0.}
                output.add({'trial_id':tid,'plan_id':p['id'],'mask':mask,'defense':defense,'wording':wording,
                            'truth':c['oracle']['truth'],'target':p['target'],'preserved':checked['passed'],
                            'prompt':text,**result})
    conclusions=[]
    for p in selected:
        def both(mask,answer):
            matches=[r for r in output.rows if r['plan_id']==p['id'] and r['mask']==mask and r['defense'] is None]
            return len(matches)==2 and all(r['answer']==answer and r['preserved'] for r in matches)
        truth=circuits[p['circuit_id']]['oracle']['truth']
        reproduced=both(0,truth) and both(15,p['target'])
        single=[p['units'][i] for i in range(4) if reproduced and both(1<<i,p['target']) and both(15^(1<<i),truth)]
        conclusions.append({'plan_id':p['id'],'replicated_on_two_new_wordings':reproduced,
                            'individually_necessary_and_sufficient_units':single})
    atomic_json(PUBLIC/'intervention_summary.json',{'status':'complete','cases':len(selected),'conclusions':conclusions,
                'claim_scope':'parameter-rescaling effects on an external explainer; not deception inside Gemma'},immutable=True)
