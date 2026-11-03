"""Frozen holdout evaluation; no response-driven candidate selection."""
import json
import random
from .study import PUBLIC, PRIVATE, config, digest, Journal, gate, load_rows, atomic_json
from .circuits import load
from .prompts import prompt
from .observer import Observer


def state():
    calibration=gate()
    frozen=json.loads((PRIVATE/'frozen_plan.json').read_text())
    bf16={r['trial_id']:r for r in load_rows(PUBLIC/'preservation_bfloat16')}
    fp32={r['trial_id']:r for r in load_rows(PUBLIC/'preservation_float32')}
    if set(bf16)!=set(p['id'] for p in frozen['plans']) or set(fp32)!=set(frozen['fp32_ids']):
        raise RuntimeError('All declared BF16 and FP32 checks must finish before holdout evaluation')
    circuits={c['id']:c for c in load('test')}
    return calibration,frozen,bf16,fp32,circuits


def run(prose=False,path=None,limit=None):
    calibration,frozen,bf16,fp32,circuits=state(); cfg=config()
    phase='prose' if prose else 'main'
    journal=Journal(PUBLIC/phase,{'plan':digest(frozen),'format':calibration['selected_format'],
                                'phase':phase,'config':digest(cfg)})
    ids=frozen['prose_circuit_ids'] if prose else sorted(circuits)
    wordings=cfg['prose_wordings'] if prose else cfg['main_wordings']
    tasks=[]
    for cid in ids:
        c=circuits[cid]
        variants=[None]+[p for p in frozen['plans'] if p['circuit_id']==cid and (not prose or p['seed']==cfg['selection_seeds'][0])]
        for p in variants:
            for w in wordings:
                ident=f'{p["id"] if p else cid+"-O"}-{w}'
                if ident not in journal.by_id: tasks.append((ident,c,p,w))
    random.Random(cfg['seed']+(114 if prose else 113)).shuffle(tasks)
    if limit is not None: tasks=tasks[:limit]
    observer=None
    for ident,c,p,w in tasks:
        valid=(p is None or (bf16[p['id']]['passed'] and (p['id'] not in fp32 or fp32[p['id']]['passed'])))
        table=c['table'] if p is None else bf16[p['id']]['actual_edited_table']
        text=prompt(table,calibration['selected_format'],w,prose=prose)
        if valid:
            if observer is None: observer=Observer(path)
            result=observer.prose(text) if prose else observer.score(text)
        else: result={'answer':'PRESERVATION_FAILED','seconds':0.,'generated_tokens':0}
        journal.add({'trial_id':ident,'circuit_id':c['id'],'layer':c['layer'],
                     'arm':p['arm'] if p else 'O','seed':p['seed'] if p else None,'wording':w,
                     'truth':c['oracle']['truth'],'target':'B' if c['oracle']['truth']=='A' else 'A',
                     'preserved':valid,'prompt':text,'prompt_sha256':digest(text),**result})
        print(f'{phase} {ident} answer={result["answer"]}',flush=True)
    expected=960 if prose else 9984
    atomic_json(PUBLIC/f'{phase}_status.json',{'status':'complete' if len(journal.rows)==expected else 'partial',
                 'completed':len(journal.rows),'expected':expected})
    if len(journal.rows)==expected:
        # Release the holdout contents only after frozen evaluation has finished.
        atomic_json(PUBLIC/'holdout_circuits.json',list(circuits.values()),immutable=True)
        atomic_json(PUBLIC/'frozen_plan.json',frozen,immutable=True)
