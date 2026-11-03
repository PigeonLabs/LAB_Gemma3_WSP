"""Develop format on dev only, then one independent calibration gate."""
import random
import statistics
import time
from .study import PUBLIC, config, digest, atomic_json, Journal, require_freeze
from .circuits import load
from .prompts import prompt
from .observer import Observer


def summarize_accuracy(rows):
    return sum(r['answer']==r['truth'] for r in rows)/len(rows) if rows else 0.


def run(path=None):
    freeze=require_freeze(); cfg=config()
    if (PUBLIC/'calibration_summary.json').exists():
        import json
        print(json.loads((PUBLIC/'calibration_summary.json').read_text())['status'],flush=True); return
    dev=load('development'); calibration=load('calibration')
    if len(dev)!=48 or len(calibration)!=32: raise ValueError('Missing declared circuits')
    journal=Journal(PUBLIC/'calibration',{'freeze':digest(freeze),'dataset':digest(dev+calibration),'protocol':'dev-format-selection-then-single-calibration'})
    observer=None
    def evaluate(c,fmt,wording,split):
        nonlocal observer
        ident=f'{split}-{c["id"]}-{fmt}-{wording}'
        if ident in journal.by_id: return
        if observer is None: observer=Observer(path)
        text=prompt(c['table'],fmt,wording)
        result=observer.score(text)
        journal.add({'trial_id':ident,'circuit_id':c['id'],'layer':c['layer'],'split':split,
                     'format':fmt,'wording':wording,'truth':c['oracle']['truth'],
                     'prompt':text,'prompt_sha256':digest(text),**result})
        print(f'{ident} answer={result["answer"]} correct={result["answer"]==c["oracle"]["truth"]} seconds={result["seconds"]:.3f}',flush=True)
    jobs=[(c,f,w,'development') for c in dev for f in cfg['calibration_formats'] for w in [6,7]]
    random.Random(cfg['seed']+991).shuffle(jobs)
    for job in jobs: evaluate(*job)
    accuracies={fmt:summarize_accuracy([r for r in journal.rows if r['split']=='development' and r['format']==fmt]) for fmt in cfg['calibration_formats']}
    selected=max(cfg['calibration_formats'],key=lambda x:accuracies[x])
    atomic_json(PUBLIC/'selected_format.json',{'format':selected,'development_accuracy':accuracies,
                                             'selection_rule':'highest development accuracy; declaration-order tie break'},immutable=True)
    # Complete independent calibration even when dev fails, to document generalization.
    jobs=[(c,selected,w,'calibration') for c in calibration for w in [6,7]]
    random.Random(cfg['seed']+992).shuffle(jobs)
    for job in jobs: evaluate(*job)
    rows=[r for r in journal.rows if r['split']=='calibration']
    accuracy=summarize_accuracy(rows)
    passed=accuracies[selected]>=.8 and accuracy>=.8
    times=[r['seconds'] for r in journal.rows]
    result={'status':'passed' if passed else 'stopped_baseline_failure',
            'selected_format':selected,'development_accuracies':accuracies,
            'calibration_accuracy':accuracy,'correct':sum(r['answer']==r['truth'] for r in rows),
            'denominator':len(rows),'uncertain':sum(r['answer']=='UNKNOWN' for r in rows),
            'minimum_accuracy':.8,'main_experiment_authorized_by_gate':passed,
            'calibration_truth_counts':{a:sum(r['truth']==a for r in rows) for a in ['A','B']},
            'load_seconds':observer.load_seconds if observer else None,
            'score_calls':len(journal.rows),'median_score_seconds':statistics.median(times),
            'p90_score_seconds':sorted(times)[min(len(times)-1,int(.9*len(times)))],
            'estimated_main_score_hours':9984*statistics.median(times)/3600,
            'estimated_main_p90_hours':9984*sorted(times)[min(len(times)-1,int(.9*len(times)))]/3600,
            'limitation':'Scoring time only; excludes loading, preservation, prose, IO, implementation and retries.',
            'stop_reason':None if passed else 'A stable >=80% independent baseline was not established. Do not run attacks or infer weight-analysis efficacy.'}
    atomic_json(PUBLIC/'calibration_summary.json',result,immutable=True)
    print(result,flush=True)
